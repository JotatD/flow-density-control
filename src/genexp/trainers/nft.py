
import copy
from argparse import Namespace
from typing import Any

import numpy as np
import torch
from diffusiongym import DDMixin, Scheduler
from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment, Sample
from diffusiongym.rewards import Reward
from diffusiongym.types import D

from genexp.trainers.utils import StepTimer


class RewardStatTracker:
    """Normalize rewards within contiguous groups."""

    def __init__(self, advantage_group_size: int = 16):
        self.advantage_group_size = advantage_group_size

    def update(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.detach()
        batch_size = rewards.shape[0]
        group_size = self.advantage_group_size
        greatest_multiple = (batch_size // group_size) * group_size
        normalized_parts = []
        if greatest_multiple > 0:
            rewards_batch = rewards[:greatest_multiple]
            rewards_batch = rewards_batch.reshape(-1, group_size)

            rewards_mean = rewards_batch.mean(dim=1, keepdim=True)
            rewards_std = rewards_batch.std(dim=1, keepdim=True, correction=0)
            rewards_batch = (rewards_batch - rewards_mean) / (rewards_std + 1e-4)
            normalized_parts.append(rewards_batch.reshape(-1))

        # Normalize remaining incomplete group
        last_batch = rewards[greatest_multiple:]
        if last_batch.numel() > 0:
            last_batch_mean = last_batch.mean()
            last_batch_std = last_batch.std(correction=0)
            last_batch = (last_batch - last_batch_mean) / (last_batch_std + 1e-4)
            normalized_parts.append(last_batch)

        if not normalized_parts:
            return rewards

        return torch.cat(normalized_parts, dim=0)


def subsample_steps(total_steps, percentage):
    """Create a subset of time-steps for efficient computation (Appendix G2)."""
    steps_count = int(total_steps * percentage)
    samples = np.random.choice(np.arange(total_steps), size=steps_count, replace=False)
    return np.sort(samples)


def endpoint(model: BaseModel[D], vt: D, xt: D, t: torch.Tensor) -> D:
    """Recover the clean endpoint from an interpolant state and velocity."""
    scheduler: Scheduler = model.scheduler
    beta = scheduler.beta(xt, t)
    beta_dot = scheduler.beta_dot(xt, t)
    alpha = scheduler.alpha(xt, t)
    alpha_dot = scheduler.alpha_dot(xt, t)
    return (beta * vt - beta_dot * xt) / (alpha_dot * beta - alpha * beta_dot)


def velocity(model: BaseModel[D], x: D, t: torch.Tensor, **kwargs) -> D:
    """Convert an endpoint prediction to the corresponding interpolant velocity."""
    output = model.forward(x, t, **kwargs)
    scheduler: Scheduler = model.scheduler

    if model.output_type == "endpoint":
        alpha = scheduler.alpha(x, t)
        beta = scheduler.beta(x, t)
        beta_dot = scheduler.beta_dot(x, t)
        alpha_dot = scheduler.alpha_dot(x, t)

        return (beta_dot / beta) * x + (alpha_dot - alpha * beta_dot / beta) * output

    raise ValueError(f"{model.output_type} not supported")

class DiffusionNFTrainer:
    def __init__(
        self,
        config: Namespace,
        env: Environment,
        base_model: BaseModel,
        device: torch.device | None = None,
        use_valids: bool = False,
    ):
        # There are three policies: the mirror-descent base policy, the EMA exploration
        # policy used for sampling, and the trainable fine policy.
        self.config = config
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.only_valids = config.only_valids

        self.batch_size: int = config.batch_size

        self.use_valids = use_valids
        self.adv_clip_max: float = config.adv_clip_max
        self.clip_grad_norm: float = config.clip_grad_norm
        self.num_inner_epochs: int = config.num_inner_epochs
        self.advantage_group_size: int = config.advantage_group_size

        self.timestep_fraction: float = config.timestep_fraction

        self.mixing_beta: float = config.beta
        self.kl_weight: float = config.alpha
        # self.exploration_decay_type: int = config.exploration_decay_type

        self.env = env
        self.base_model = base_model
        self.base_model.to(self.device)

        self.fine_model = env.model
        self.exploration_model = copy.deepcopy(self.base_model)
        self.exploration_model.requires_grad_(False)
        self.reward_stat_tracker = RewardStatTracker(advantage_group_size=self.advantage_group_size)
        self.optimizer_steps = 0
        self.fulfill_max_attempts = config.fulfill_max_attempts
        self.exploration_decay_type = config.exploration_decay_type

        self.fulfill = config.fulfill_num_samples

        self.timer = StepTimer(
            device=self.device,
        )

        self.configure_optimizers()

    def configure_optimizers(self):
        if hasattr(self, "optimizer"):
            del self.optimizer
        self.optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=self.config.lr)

    def training_state_dict(self) -> dict[str, Any]:
        """Return all mutable trainer state needed to resume training."""
        return {
            "fine_model": self.fine_model.state_dict(),
            "base_model": self.base_model.state_dict(),
            "exploration_model": self.exploration_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_steps": self.optimizer_steps,
        }

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        """Restore mutable trainer state produced by training_state_dict."""
        self.fine_model.load_state_dict(state["fine_model"])
        self.base_model.load_state_dict(state["base_model"])
        self.exploration_model.load_state_dict(state["exploration_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer_steps = state["optimizer_steps"]

    def generate_dataset(self, reward: Reward[D]) -> tuple[list[Sample], torch.Tensor, torch.Tensor]:
        """Collect exploration-policy samples and normalize their rewards."""
        self.fine_model.eval()
        self.exploration_model.eval()
        all_samples: list[Sample] = []
        all_rewards = []
        remaining = self.config.num_samples
        self.curr_attempts = 0

        original_policy = self.env._policy
        self.env.policy = self.exploration_model
        try:
            while remaining > 0 and self.curr_attempts < self.fulfill_max_attempts:
                batch = min(self.batch_size, 5 * remaining) if self.fulfill else min(self.batch_size, remaining)
                env_sample = self.env.sample(batch, pbar=True)
                rewards, info = reward(env_sample.sample, env_sample.latent)
                self.curr_attempts += batch
                for i, sample in enumerate(env_sample):  # ty: ignore[invalid-argument-type]
                    if not self.only_valids or (self.only_valids and info["valids"][i]):
                        all_samples.append(sample)
                        all_rewards.append(rewards[i])
                        remaining -= 1
        finally:
            self.env._policy = original_policy

        if len(all_samples) == 0:
            return [], torch.tensor([]), torch.tensor([])

        all_rewards = all_rewards[: self.config.num_samples]
        all_samples = all_samples[: self.config.num_samples]

        all_rewards = torch.stack(all_rewards, dim=0)
        advantages = self.reward_stat_tracker.update(all_rewards)

        return all_samples, advantages, all_rewards

    def train_step(self, sample: Sample, advantages: torch.Tensor) -> dict[str, list[float]]:
        clean_latent: DDMixin = sample.latent.to(self.device)
        timesteps = sample.timesteps.to(self.device)
        kwargs = sample.kwargs
        timed_statistics = {
            "policy_loss": [],
            "unweighted_policy_loss": [],
            "kl_div_loss": [],
            "old_kl_div": [],
            "total_loss": [],
            "x0_norm": [],
            "x0_norm_max": [],
            "old_deviate": [],
            "old_deviate_max": [],
        }

        # idxs = np.arange(T)

        T = self.env.discretization_steps
        idxs = subsample_steps(T, self.timestep_fraction)
        adv_clipped = torch.clamp(advantages, -self.adv_clip_max, self.adv_clip_max)
        normalized_advantages_clip = 0.5 * (adv_clipped / self.adv_clip_max) + 0.5
        r = torch.clamp(normalized_advantages_clip, 0, 1).to(self.device)

        self.optimizer.zero_grad()
        num_timesteps = len(idxs)
        for idx in idxs:
            print(idx)
            t_batch = timesteps[idx].unsqueeze(0).expand(len(clean_latent))
            noise = clean_latent.randn_like().to(self.device)
            interpolant_alpha = self.base_model.scheduler.alpha(clean_latent, t_batch)
            interpolant_beta = self.base_model.scheduler.beta(clean_latent, t_batch)
            xt = interpolant_alpha * clean_latent + interpolant_beta * noise

            step_kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

            forward_prediction = velocity(self.fine_model, xt, t_batch, **step_kwargs)

            with torch.no_grad():
                old_prediction = velocity(self.exploration_model, xt, t_batch, **step_kwargs)
                ref_forward_prediction = velocity(self.base_model, xt, t_batch, **step_kwargs)

            positive_v = self.mixing_beta * forward_prediction + (1 - self.mixing_beta) * old_prediction
            positive_endpoint = endpoint(self.base_model, positive_v, xt, t_batch)

            with torch.no_grad():
                positive_weight_factor = (
                    (positive_endpoint - clean_latent).apply(torch.abs).aggregate("mean").clip(min=0.0001)
                )

            positive_loss = ((positive_endpoint - clean_latent) ** 2).aggregate("mean") / positive_weight_factor

            negative_v = -self.mixing_beta * forward_prediction + (1 + self.mixing_beta) * old_prediction
            negative_endpoint = endpoint(self.base_model, negative_v, xt, t_batch)
            with torch.no_grad():
                negative_weight_factor = (
                    (negative_endpoint - clean_latent).apply(torch.abs).aggregate("mean").clip(min=0.0001)
                )

            negative_loss = ((negative_endpoint - clean_latent) ** 2).aggregate("mean") / negative_weight_factor

            ori_policy_loss = r * (positive_loss / self.mixing_beta) + (1.0 - r) * (negative_loss / self.mixing_beta)
            policy_loss = (ori_policy_loss * self.adv_clip_max).mean()

            kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).aggregate("mean").mean()
            loss = policy_loss + self.kl_weight * kl_div_loss

            timed_statistics["policy_loss"].append(policy_loss.item())
            timed_statistics["unweighted_policy_loss"].append(ori_policy_loss.mean().item())
            timed_statistics["kl_div_loss"].append(kl_div_loss.item())
            timed_statistics["old_kl_div"].append(
                ((old_prediction - ref_forward_prediction) ** 2).aggregate("mean").mean().item()
            )
            timed_statistics["total_loss"].append(loss.item())
            timed_statistics["x0_norm"].append((clean_latent**2).aggregate("mean").mean().item())
            timed_statistics["x0_norm_max"].append((clean_latent**2).aggregate("max").mean().item())
            timed_statistics["old_deviate"].append(
                ((old_prediction - forward_prediction) ** 2).aggregate("mean").mean().item()
            )
            timed_statistics["old_deviate_max"].append(
                ((old_prediction - forward_prediction) ** 2).aggregate("max").mean().item()
            )

            if loss.isnan():
                self.optimizer.zero_grad()
                return {}

            (loss / num_timesteps).backward()

        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.fine_model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.optimizer_steps += 1

        return timed_statistics

    def finetune(self, samples: list[Sample], advantages: torch.Tensor) -> dict[str, list[float]]:
        """Finetune the fine model using the given samples and advantages."""
        self.fine_model.train()
        stats_avgs_per_batch = []
        start = 0
        while start < len(samples):
            advs = advantages[start : start + self.batch_size]
            batch_samples = Sample.concat(samples[start : start + self.batch_size])
            stats_avgs_per_batch.append(self.train_step(batch_samples, advs))
            start += self.batch_size

        timed_final_stats = stats_avgs_per_batch[0]
        for key in timed_final_stats:
            for t in range(len(timed_final_stats[key])):
                timed_final_stats[key][t] = np.mean([stats[key][t] for stats in stats_avgs_per_batch])
        return timed_final_stats


    def update_exploration_model(self) -> None:
        """Update the exploration policy from the fine policy using the configured EMA."""
        decay = exploration_decay(self.optimizer_steps, self.exploration_decay_type)
        for fine_parameter, exploration_parameter in zip(
            self.fine_model.parameters(),
            self.exploration_model.parameters(),
            strict=True,
        ):
            exploration_parameter.lerp_(fine_parameter, 1.0 - decay)
            
def exploration_decay(step: int, decay_type: int) -> float:
    """Return the exploration-policy EMA decay used by DiffusionNFT."""
    if decay_type == 0:
        flat, rate, maximum = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, rate, maximum = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, rate, maximum = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unknown exploration decay type: {decay_type}")

    if step < flat:
        return 0.0
    return min((step - flat) * rate, maximum)