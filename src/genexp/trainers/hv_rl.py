from fsspec.implementations.http import ex
import enum
from genexp.trainers.utils import StepTimer
import copy
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from diffusiongym import DDMixin, Scheduler
from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment, Sample
from diffusiongym.rewards import Reward
from diffusiongym.types import D

from genexp.mo.base import MOReward
from genexp.mo.utils import HVComputer
from genexp.trainers.adjoint_matching import create_timestep_subset


class RewardStatTracker:
    """Normalize one prompt-free sampling round like DiffusionNFT's global tracker."""

    def __init__(self):
        self.mean = None
        self.std = None

    def update(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.detach()
        self.mean = rewards.mean()
        self.std = rewards.std(correction=0)
        return (rewards - self.mean) / (self.std + 1e-4)


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


class FirstVariation(Reward[D]):
    def __init__(self, call_fn: Callable[[D, D], tuple[torch.Tensor, dict[str, Any]]]):
        super().__init__()
        self.call_fn = call_fn

    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.call_fn(sample, latent, **kwargs)


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
        self.clip_range: float = config.clip_range
        self.adv_clip_max: float = config.adv_clip_max
        self.clip_grad_norm: float = config.clip_grad_norm
        self.num_inner_epochs: int = config.num_inner_epochs
        self.timestep_fraction: float = config.timestep_fraction

        self.mixing_beta: float = config.beta
        self.kl_weight: float = config.alpha
        self.exploration_decay_type: int = config.exploration_decay_type

        self.env = env
        self.base_model = base_model
        self.base_model.to(self.device)

        self.fine_model = env.model
        self.exploration_model = copy.deepcopy(self.base_model)
        self.exploration_model.requires_grad_(False)
        self.reward_stat_tracker = RewardStatTracker()
        self.optimizer_steps = 0
        
        self.timer = StepTimer(device=self.device)

        self.configure_optimizers()

    def configure_optimizers(self):
        if hasattr(self, "optimizer"):
            del self.optimizer
        self.optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=self.config.lr)

    def generate_dataset(self, reward: Reward[D]) -> tuple[list[Sample], torch.Tensor]:
        """Collect exploration-policy samples and normalize their rewards."""
        self.fine_model.eval()
        self.exploration_model.eval()
        all_samples: list[Sample] = []
        all_rewards = []
        remaining = self.config.num_samples

        original_policy = self.env._policy
        self.env.policy = self.exploration_model
        try:
            while remaining > 0:
                batch = min(remaining, self.batch_size)
                env_sample = self.env.sample(batch, pbar=False)
                rewards, info = reward(env_sample.sample, env_sample.latent)
                for i, sample in enumerate(env_sample):  # ty: ignore[invalid-argument-type]
                    if not self.only_valids or (self.only_valids and info['valids'][i]):
                        all_samples.append(sample)
                        all_rewards.append(rewards[i])
                remaining -= batch
        finally:
            self.env._policy = original_policy

        if len(all_samples) == 0:
            return [], torch.tensor([])
        all_rewards = torch.stack(all_rewards, dim=0)
        advantages = self.reward_stat_tracker.update(all_rewards)

        return all_samples, advantages

    def train_step(self, sample: Sample, advantages: torch.Tensor) -> dict[str, list[float]]:
        clean_latent: DDMixin = sample.latent.to(self.device)
        timesteps = sample.timesteps.to(self.device)
        kwargs = sample.kwargs
        timed_statistics = {
            'policy_loss': [],
            'unweighted_policy_loss': [],
            'kl_div_loss': [],
            'kl_div': [],
            'old_kl_div': [],
            'total_loss': [],
            'x0_norm': [],
            'x0_norm_max': [],
            'old_deviate': [],
            'old_deviate_max': [],
        }

        T = self.env.discretization_steps
        if self.timestep_fraction < 1.0 and self.timestep_fraction > 0.0:
            sp = max(0.0, self.timestep_fraction - 0.25)
            idxs = create_timestep_subset(T, final_percent=0.25, sample_percent=sp)
        else:
            idxs = np.arange(T)

        adv_clipped = torch.clamp(advantages, -self.adv_clip_max, self.adv_clip_max)
        normalized_advantages_clip = 0.5 * (adv_clipped / self.adv_clip_max) + 0.5
        r = torch.clamp(normalized_advantages_clip, 0, 1).to(self.device)

        losses = []
        for idx in idxs:
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
                positive_weight_factor = (positive_endpoint - clean_latent).apply(torch.abs).aggregate("mean").clip(min=0.0001)
                
            positive_loss = ((positive_endpoint - clean_latent) ** 2).aggregate("mean") / positive_weight_factor

            negative_v = -self.mixing_beta * forward_prediction + (1 + self.mixing_beta) * old_prediction
            negative_endpoint = endpoint(self.base_model, negative_v, xt, t_batch)
            with torch.no_grad():
                negative_weight_factor = (negative_endpoint - clean_latent).apply(torch.abs).aggregate("mean").clip(min=0.0001)
                
            negative_loss = ((negative_endpoint - clean_latent) ** 2).aggregate("mean") / negative_weight_factor

            ori_policy_loss = r * (positive_loss / self.mixing_beta) + (1.0 - r) * (negative_loss / self.mixing_beta)
            policy_loss = (ori_policy_loss * self.adv_clip_max).mean()

            kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).aggregate("mean").mean()
            loss = policy_loss + self.kl_weight * kl_div_loss

            losses.append(loss)
            
            timed_statistics['policy_loss'].append(policy_loss.item())
            timed_statistics['unweighted_policy_loss'].append(ori_policy_loss.mean().item())
            timed_statistics['kl_div_loss'].append(kl_div_loss.item())
            timed_statistics['kl_div'].append(((forward_prediction - ref_forward_prediction) ** 2).aggregate("mean").mean().item())
            timed_statistics['old_kl_div'].append(((old_prediction - ref_forward_prediction) ** 2).aggregate("mean").mean().item())
            timed_statistics['total_loss'].append(loss.item())
            timed_statistics['x0_norm'].append((clean_latent**2).aggregate("mean").mean().item())
            timed_statistics['x0_norm_max'].append((clean_latent**2).aggregate("max").mean().item())
            timed_statistics['old_deviate'].append(((old_prediction - ref_forward_prediction) ** 2).aggregate("mean").mean().item())
            timed_statistics['old_deviate_max'].append(((old_prediction - ref_forward_prediction) ** 2).aggregate("max").mean().item())

        loss = torch.stack(losses).mean()
    
        if loss.isnan():
            return {}

        self.optimizer.zero_grad()
        loss.backward()
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

    @torch.no_grad()
    def update_exploration_model(self) -> None:
        """Update the exploration policy from the fine policy using the configured EMA."""
        decay = exploration_decay(self.optimizer_steps, self.exploration_decay_type)
        for fine_parameter, exploration_parameter in zip(
            self.fine_model.parameters(),
            self.exploration_model.parameters(),
            strict=True,
        ):
            exploration_parameter.lerp_(fine_parameter, 1.0 - decay)


class HVRL(DiffusionNFTrainer):
    def __init__(
        self,
        config: Namespace,
        env: Environment,
        reward: MOReward,
        hv_computer: HVComputer,
        device: torch.device | None = None,
    ):
        if device is None:
            device = env.base_model.device

        self.batch_size = config.batch_size
        self.hv_config = config
        self.env = env
        self.device = device
        self.hv_computer = hv_computer

        self.base_model = copy.deepcopy(env.base_model)

        self.reward = reward
        self.n = config.n
        self.num_rews = reward.num_rew

        self.num_p_nm1 = config.num_p_nm1

        super().__init__(
            config,
            env,
            self.base_model,
            device,
            False,
        )

    def fix_optimization_problem(self):
        with torch.no_grad():
            self.evaluations_X_ = self.sample_rewards()
            self.hypervolume_X_ = self.hv_computer(self.evaluations_X_)

    def hv_first_variation(self, sample: D, latent: D) -> tuple[torch.Tensor, dict[str, Any]]:
        obj_x, info = self.reward(sample, latent)
        inp_batch = obj_x.shape[0]
        obj_x = obj_x.reshape(inp_batch, 1, 1, self.num_rews).expand(
            inp_batch,
            self.num_p_nm1,
            1,
            self.num_rews,
        )
        expanded_obj_X_ = self.evaluations_X_.expand(inp_batch, self.num_p_nm1, self.n - 1, self.num_rews)
        complete_X = torch.cat([expanded_obj_X_, obj_x], dim=2)
        complete_hv = self.hv_computer(complete_X)
        expanded_hv_X_ = self.hypervolume_X_.expand(inp_batch, self.num_p_nm1)
        hv_improvement = complete_hv - expanded_hv_X_
        first_var = hv_improvement.mean(dim=1)

        return first_var, info

    @torch.no_grad()
    def sample_rewards(self) -> torch.Tensor:
        all_rewards = []
        remaining = self.num_p_nm1 * (self.n - 1)

        original_policy = self.env._policy
        self.env.policy = self.base_model
        try:
            while remaining > 0:
                current_batch_size = min(self.batch_size, remaining)
                env_sample = self.env.sample(current_batch_size, pbar=False)
                rews, _ = self.reward(env_sample.sample, env_sample.latent)
                all_rewards.append(rews)
                remaining -= current_batch_size
        finally:
            self.env._policy = original_policy

        rewards = torch.cat(all_rewards, dim=0)
        return rewards.reshape(self.num_p_nm1, self.n - 1, self.num_rews)

    def update_base_model(self):
        state = self.fine_model.state_dict()
        self.base_model.load_state_dict(state)
        self.env.base_model.load_state_dict(state)

    def generate_dataset_fv(self) -> tuple[list[Sample], torch.Tensor]:
        """Collect trajectories, compute global advantages, build training dataset."""
        return self.generate_dataset(FirstVariation(self.hv_first_variation))
