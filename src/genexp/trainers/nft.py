import tqdm

import copy
from argparse import Namespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusiongym.environments import Environment, Sample
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.flowmol import FlowMolBaseModel
from diffusiongym.rewards import Reward
from diffusiongym.types import D



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


def _mean_per_graph(values: torch.Tensor, batch_idxs: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Average one scalar per node or edge independently for each graph."""
    totals = values.new_zeros(batch_size)
    counts = values.new_zeros(batch_size)
    totals.index_add_(0, batch_idxs, values)
    counts.index_add_(0, batch_idxs, torch.ones_like(values))
    return totals / counts.clamp(min=1)


@torch.no_grad()
def flowmol_adaptive_error_scale(
    prediction: DDGraph,
    target: DDGraph,
    weights: dict[str, float],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute a detached per-molecule error scale for mixed FlowMol endpoints."""
    pred_graph = prediction.graph
    target_graph = target.graph
    batch_size = len(prediction)

    errors = {
        "x": _mean_per_graph(
            (pred_graph.ndata["x_t"] - target_graph.ndata["x_t"]).abs().mean(dim=-1),
            prediction.n_idx,
            batch_size,
        ),
        "a": _mean_per_graph(
            1.0
            - F.softmax(pred_graph.ndata["a_t"], dim=-1)
            .gather(-1, target_graph.ndata["a_t"].argmax(dim=-1, keepdim=True))
            .squeeze(-1),
            prediction.n_idx,
            batch_size,
        ),
        "c": _mean_per_graph(
            1.0
            - F.softmax(pred_graph.ndata["c_t"], dim=-1)
            .gather(-1, target_graph.ndata["c_t"].argmax(dim=-1, keepdim=True))
            .squeeze(-1),
            prediction.n_idx,
            batch_size,
        ),
    }

    upper_edge_mask = prediction.ue_mask
    errors["e"] = _mean_per_graph(
        1.0
        - F.softmax(pred_graph.edata["e_t"][upper_edge_mask], dim=-1)
        .gather(-1, target_graph.edata["e_t"][upper_edge_mask].argmax(dim=-1, keepdim=True))
        .squeeze(-1),
        prediction.e_idx[upper_edge_mask],
        batch_size,
    )

    total = pred_graph.ndata["x_t"].new_zeros(batch_size)
    total_weight = 0.0
    for field, field_error in errors.items():
        total = total + weights[field] * field_error
        total_weight += weights[field]
    return (total / total_weight).detach().clamp(min=eps)


def flowmol_reference_loss(prediction: DDGraph, reference: DDGraph, weights: dict[str, float]) -> torch.Tensor:
    """Compute per-molecule coordinate MSE and categorical KL to a reference endpoint."""
    pred_graph = prediction.graph
    ref_graph = reference.graph
    batch_size = len(prediction)

    losses = {
        "x": _mean_per_graph(
            ((pred_graph.ndata["x_t"] - ref_graph.ndata["x_t"]) ** 2).mean(dim=-1),
            prediction.n_idx,
            batch_size,
        ),
        "a": _mean_per_graph(
            F.kl_div(
                F.log_softmax(pred_graph.ndata["a_t"], dim=-1),
                F.softmax(ref_graph.ndata["a_t"], dim=-1),
                reduction="none",
            ).sum(dim=-1),
            prediction.n_idx,
            batch_size,
        ),
        "c": _mean_per_graph(
            F.kl_div(
                F.log_softmax(pred_graph.ndata["c_t"], dim=-1),
                F.softmax(ref_graph.ndata["c_t"], dim=-1),
                reduction="none",
            ).sum(dim=-1),
            prediction.n_idx,
            batch_size,
        ),
    }

    upper_edge_mask = prediction.ue_mask
    losses["e"] = _mean_per_graph(
        F.kl_div(
            F.log_softmax(pred_graph.edata["e_t"][upper_edge_mask], dim=-1),
            F.softmax(ref_graph.edata["e_t"][upper_edge_mask], dim=-1),
            reduction="none",
        ).sum(dim=-1),
        prediction.e_idx[upper_edge_mask],
        batch_size,
    )

    total = pred_graph.ndata["x_t"].new_zeros(batch_size)
    for field, field_loss in losses.items():
        total = total + weights[field] * field_loss
    return total


class DiffusionNFTrainer:
    def __init__(
        self,
        config: Namespace,
        env: Environment,
        base_model: FlowMolBaseModel,
        device: torch.device | None = None,
        use_valids: bool = False,
    ):
        # There are three policies: the mirror-descent base policy, the EMA exploration
        # policy used for sampling, and the trainable fine policy.
        self.config = config
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.only_valids = config.only_valids

        self.batch_size: int = config.batch_size
        self.backward_batch_size: int = config.backward_batch_size

        self.adv_clip_max: float = config.adv_clip_max
        self.clip_grad_norm: float = config.clip_grad_norm
        self.num_inner_epochs: int = config.num_inner_epochs
        self.advantage_group_size: int = config.advantage_group_size

        self.timestep_fraction: float = config.timestep_fraction

        self.mixing_beta: float = config.beta
        self.kl_weight: float = config.alpha
        self.adaptive_loss_scaling: bool = getattr(config, "adaptive_loss_scaling", True)
        self.adaptive_scale_eps: float = getattr(config, "adaptive_scale_eps", 1e-5)
        # self.exploration_decay_type: int = config.exploration_decay_type

        self.env = env
        self.base_model = base_model
        self.base_model.to(self.device)

        self.fine_model = env.model
        self.exploration_model = copy.deepcopy(self.base_model)
        self.exploration_model.requires_grad_(False)
        
        self.off_policy = copy.deepcopy(self.base_model)
        self.off_policy.requires_grad_(False)
        
        self.reward_stat_tracker = RewardStatTracker(advantage_group_size=self.advantage_group_size)
        self.optimizer_steps = 0
        self.fulfill_max_attempts = config.fulfill_max_attempts
        self.exploration_decay_type = config.exploration_decay_type

        self.fulfill = config.fulfill_num_samples

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
        clean_latent: DDGraph = sample.latent.to(self.device)
        timesteps = sample.timesteps.to(self.device)
        kwargs = sample.kwargs
        timed_statistics = {
            "policy_loss": [],
            "raw_policy_loss": [],
            "unweighted_policy_loss": [],
            "positive_error_scale": [],
            "negative_error_scale": [],
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
        loss_weights = self.base_model.model.total_loss_weights
        for idx in tqdm.tqdm(idxs):
            t_batch = timesteps[idx].unsqueeze(0).expand(len(clean_latent))
            noise = clean_latent.randn_like().to(self.device)
            interpolant_alpha = self.base_model.scheduler.alpha(clean_latent, t_batch)
            interpolant_beta = self.base_model.scheduler.beta(clean_latent, t_batch)
            xt = interpolant_alpha * clean_latent + interpolant_beta * noise

            step_kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            prediction_kwargs = {**step_kwargs, "apply_softmax": False, "remove_com": False}

            forward_endpoint = self.fine_model.forward(xt, t_batch, **prediction_kwargs)

            with torch.no_grad():
                old_endpoint = self.exploration_model.forward(xt, t_batch, **prediction_kwargs)
                ref_endpoint = self.base_model.forward(xt, t_batch, **prediction_kwargs)

            positive_endpoint = self.mixing_beta * forward_endpoint + (1 - self.mixing_beta) * old_endpoint
            raw_positive_loss = self.base_model.train_loss(clean_latent, xt=xt, t=t_batch, pred=positive_endpoint)

            negative_endpoint = -self.mixing_beta * forward_endpoint + (1 + self.mixing_beta) * old_endpoint
            raw_negative_loss = self.base_model.train_loss(clean_latent, xt=xt, t=t_batch, pred=negative_endpoint)

            positive_error_scale = torch.ones_like(raw_positive_loss)
            negative_error_scale = torch.ones_like(raw_negative_loss)
            if self.adaptive_loss_scaling:
                positive_error_scale = flowmol_adaptive_error_scale(
                    positive_endpoint, clean_latent, loss_weights, self.adaptive_scale_eps
                )
                negative_error_scale = flowmol_adaptive_error_scale(
                    negative_endpoint, clean_latent, loss_weights, self.adaptive_scale_eps
                )
            positive_loss = raw_positive_loss / positive_error_scale
            negative_loss = raw_negative_loss / negative_error_scale

            raw_ori_policy_loss = r * (raw_positive_loss / self.mixing_beta) + (1.0 - r) * (
                raw_negative_loss / self.mixing_beta
            )
            ori_policy_loss = r * (positive_loss / self.mixing_beta) + (1.0 - r) * (negative_loss / self.mixing_beta)
            policy_loss = (ori_policy_loss * self.adv_clip_max).mean()

            kl_div_loss = flowmol_reference_loss(forward_endpoint, ref_endpoint, loss_weights).mean()
            old_deviate = flowmol_reference_loss(forward_endpoint, old_endpoint, loss_weights)
            loss = policy_loss + self.kl_weight * kl_div_loss

            timed_statistics["policy_loss"].append(policy_loss.item())
            timed_statistics["raw_policy_loss"].append((raw_ori_policy_loss * self.adv_clip_max).mean().item())
            timed_statistics["unweighted_policy_loss"].append(ori_policy_loss.mean().item())
            timed_statistics["positive_error_scale"].append(positive_error_scale.mean().item())
            timed_statistics["negative_error_scale"].append(negative_error_scale.mean().item())
            timed_statistics["kl_div_loss"].append(kl_div_loss.item())
            timed_statistics["old_kl_div"].append(
                flowmol_reference_loss(old_endpoint, ref_endpoint, loss_weights).mean().item()
            )
            timed_statistics["total_loss"].append(loss.item())
            timed_statistics["x0_norm"].append((clean_latent**2).aggregate("mean").mean().item())
            timed_statistics["x0_norm_max"].append((clean_latent**2).aggregate("max").mean().item())
            timed_statistics["old_deviate"].append(old_deviate.mean().item())
            timed_statistics["old_deviate_max"].append(old_deviate.max().item())

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
            advs = advantages[start : start + self.backward_batch_size]
            batch_samples = Sample.concat(samples[start : start + self.backward_batch_size])
            stats_avgs_per_batch.append(self.train_step(batch_samples, advs))
            start += self.backward_batch_size

        timed_final_stats = stats_avgs_per_batch[0]
        for key in timed_final_stats:
            for t in range(len(timed_final_stats[key])):
                timed_final_stats[key][t] = np.mean([stats[key][t] for stats in stats_avgs_per_batch])
        return timed_final_stats


    def update_off_policy_model(self) -> None:
        """I dont want to figure a more efficient way dawg."""
        decay = exploration_decay(self.optimizer_steps, self.exploration_decay_type)
        for fine_parameter, exploration_parameter in zip(
            self.fine_model.parameters(),
            self.off_policy.parameters(),
            strict=True,
        ):
            exploration_parameter.lerp_(fine_parameter, 1.0 - decay)
            
    def update_exploration_model(self) -> None:
        self.exploration_model.load_state_dict(self.off_policy.state_dict())
            
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
