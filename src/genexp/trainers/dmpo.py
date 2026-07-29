"""Distribution Matching Policy Optimization for discrete PepTune diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from diffusiongym.types import DDTensor

from peptidesgym.model import DiscreteModel
from peptidesgym.env import DiscreteEnvironment, DiscreteSample

ScalarReward = Callable[..., Any]


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass
class DMPOSample:
    """A rollout batch and its detached DMPO importance statistics."""

    ts: torch.Tensor
    log_importance: torch.Tensor
    x_final: DDTensor
    rewards: torch.Tensor
    log_reference_minus_policy: torch.Tensor
    full_env_sample: DiscreteSample[DDTensor] | None = None

    @property
    def log_rnd(self) -> torch.Tensor:
        """Backward-compatible name for the complete log importance weight."""
        return self.log_importance

    def __post_init__(self) -> None:
        batch_size = len(self.x_final)
        for name, value in (
            ("log_importance", self.log_importance),
            ("rewards", self.rewards),
            ("log_reference_minus_policy", self.log_reference_minus_policy),
        ):
            if value.shape != (batch_size,):
                raise ValueError(
                    f"{name} must have shape ({batch_size},), got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")


class DMPODataset(Dataset[DMPOSample]):
    """A one-item dataset that preserves rollout batches for normalized weights."""

    def __init__(self, dmpo_sample: DMPOSample):
        self.sample = dmpo_sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> DMPOSample:
        if index not in (0, -1):
            raise IndexError(index)
        return self.sample


def loss_wdce(
    policy_model: DiscreteModel,
    log_rnd: torch.Tensor,
    x: DDTensor | torch.Tensor,
    num_replicates: int = 16,
    *,
    eps: float = 1e-3,
    centering: bool = False,
    uniform_weights: bool = False,
) -> torch.Tensor:
    """Weighted denoising cross entropy used by DMPO.

    Final samples are independently masked with probability ``lambda``.  The
    ``1 / lambda`` factor makes the summed masked-token likelihood an unbiased
    estimator of the full denoising objective.  Importance weights are
    detached because rollout collection is off-policy.
    """
    if num_replicates < 1:
        raise ValueError("num_replicates must be positive")
    if not 0 < eps < 1:
        raise ValueError("eps must be in (0, 1)")

    tokens = x.data if isinstance(x, DDTensor) else x
    tokens = tokens.to(policy_model.device)
    if tokens.ndim != 2:
        raise ValueError(f"x must have shape (batch, length), got {tuple(tokens.shape)}")
    log_rnd = log_rnd.to(device=tokens.device, dtype=torch.float32)
    if log_rnd.shape != (tokens.shape[0],):
        raise ValueError(
            f"log_rnd must have shape ({tokens.shape[0]},), got {tuple(log_rnd.shape)}"
        )

    batch_size = tokens.shape[0]
    if uniform_weights:
        weights = torch.ones_like(log_rnd)
    else:
        # Multiplication by B gives mean weight one and keeps loss scales stable.
        weights = torch.softmax(log_rnd.detach(), dim=0) * batch_size
    if centering:
        weights = weights - weights.mean()

    batch = tokens.repeat_interleave(num_replicates, dim=0)
    weights = weights.repeat_interleave(num_replicates)
    mask_probability = torch.rand(batch.shape[0], device=batch.device)
    mask_probability = mask_probability.clamp(min=eps, max=1 - eps)
    masked = torch.rand(batch.shape, device=batch.device) < mask_probability[:, None]
    perturbed = torch.where(masked, policy_model.mask_index, batch)

    log_probs = policy_model(DDTensor(perturbed), mask_probability).data
    selected_log_probs = log_probs.gather(-1, batch.unsqueeze(-1)).squeeze(-1)
    token_nll = -(selected_log_probs * masked.to(selected_log_probs.dtype)).sum(dim=-1)
    per_example_loss = token_nll / mask_probability
    return (per_example_loss * weights.to(per_example_loss.dtype)).mean()


class DMPOTrainer:
    """Fine-tune a discrete diffusion policy toward a scalar terminal reward.

    Rollouts come from ``fine_model``.  Their unnormalized target-to-behavior
    log density ratio is

    ``log p_reference(x) - log p_policy(x) + reward(x) / temperature``.

    The trajectory likelihoods are exact for the wrapper's chosen reverse
    discretization, rather than sequence-level ELBO approximations.
    """

    def __init__(
        self,
        config: Any,
        env: DiscreteEnvironment,
        fine_model: DiscreteModel,
        base_model: DiscreteModel,
        reward_fn: ScalarReward,
        device: torch.device | None = None,
        verbose: bool = False,
    ):
        if fine_model is base_model:
            raise ValueError("fine_model and base_model must be distinct model instances")
        self.config = config
        self.sampling_config = config.sampling
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.clip_grad_norm = config.get("clip_grad_norm", 2.0)
        self.alpha = config.get("alpha", 1.0)
        self.importance_coefficient = config.get("importance_coefficient", 1.0)
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        self.env: DiscreteEnvironment = env
        self.fine_model = fine_model.to(self.device)
        self.base_model = base_model.to(self.device)
        self.env.base_model = self.fine_model
        self.reward_fn = reward_fn
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.last_metrics: dict[str, float] = {}
        self.configure_optimizers()

    def configure_optimizers(self) -> None:
        parameters = [
            parameter
            for parameter in self.fine_model.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = torch.optim.Adam(parameters, lr=float(self.config.lr))

    def get_model(self) -> DiscreteModel:
        return self.fine_model

    def generate_dataset(self) -> ConcatDataset[DMPOSample]:
        """Collect rollout batches and compute detached importance weights."""
        num_samples = int(self.sampling_config.num_samples)
        batch_size = int(self.config.batch_size)
        if num_samples < 1:
            raise ValueError("sampling.num_samples must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        datasets: list[DMPODataset] = []
        self.fine_model.eval()
        self.base_model.eval()

        for start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - start)
            env_sample = self.env.sample(current_batch_size, pbar=False)
            env_sample = env_sample.to(self.device)
            reward_values  = self.reward_fn(env_sample)
            dmpo_sample = self.solve(
                samples=env_sample.latent,
                trajectory=env_sample.trajectory,
                log_ps=env_sample.log_p,
                timestep=env_sample.timesteps,
                rewards=reward_values,
            )
            dmpo_sample.full_env_sample = env_sample.to("cpu")
            datasets.append(DMPODataset(dmpo_sample))
        return ConcatDataset(datasets)

    @torch.no_grad()
    def solve(
        self,
        samples: DDTensor,
        trajectory: Sequence[DDTensor],
        log_ps: Sequence[DDTensor],
        timestep: torch.Tensor,
        rewards: torch.Tensor,
    ) -> DMPOSample:
        """Compute exact discretized trajectory likelihood ratios."""
        num_steps = timestep.numel() - 1
        assert len(trajectory) == num_steps + 1 and len(log_ps) == num_steps, (
            "Trajectory and log_ps lengths must match timestep length"
        )

        batch_size = len(samples)
        log_ratio = torch.zeros(batch_size, device=self.device, dtype=torch.float64)

        for step in range(num_steps):
            x = trajectory[step]
            x_next = trajectory[step + 1]
            t_value = timestep[step]
            dt = (timestep[step] - timestep[step + 1])
            t_batch = t_value.expand(batch_size)
            reference_log_p = self.base_model.transition_log_prob(x, x_next, t_batch, dt).data
            policy_log_p = log_ps[step].data
            log_ratio += (reference_log_p - policy_log_p).sum(dim=-1)

        reward_values = torch.as_tensor(
            rewards,
            dtype=log_ratio.dtype,
            device=self.device,
        )
        if reward_values.shape != (batch_size,):
            raise ValueError(
                f"reward_fn must return shape ({batch_size},), "
                f"got {tuple(reward_values.shape)}"
            )
        if not torch.isfinite(reward_values).all():
            raise ValueError("reward_fn returned non-finite values")
        log_importance = (
            log_ratio + reward_values / self.alpha
        ) * self.importance_coefficient
        return DMPOSample(
            ts=timestep.detach().cpu(),
            log_importance=log_importance.detach().cpu(),
            x_final=samples.detach().to("cpu"),
            rewards=reward_values.detach().cpu(),
            log_reference_minus_policy=log_ratio.detach().cpu(),
        )

    def train_step(self, sample: DMPOSample) -> float:
        self.fine_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_wdce(
            self.fine_model,
            sample.log_importance,
            sample.x_final,
            num_replicates=int(self.config.num_replicates),
            eps=float(self.config.get("mask_eps", 1e-3)),
            centering=bool(self.config.get("centering", False)),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DMPO loss: {loss.item()}")
        loss.backward()

        parameters_with_grad = [parameter for parameter in self.fine_model.parameters() if parameter.grad is not None]
        grad_norm = torch.linalg.vector_norm(torch.stack([parameter.grad.detach().norm() for parameter in parameters_with_grad]))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("DMPO produced non-finite gradients")
        
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters_with_grad, self.clip_grad_norm)
        self.optimizer.step()

        self.last_metrics = {
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "mean_reward": float(sample.rewards.mean()),
            "mean_log_reference_minus_policy": float(sample.log_reference_minus_policy.mean()),
            "effective_sample_size": self._effective_sample_size(sample.log_importance),
        }
        return self.last_metrics["loss"]

    @staticmethod
    def _effective_sample_size(log_weights: torch.Tensor) -> float:
        weights = torch.softmax(log_weights.float(), dim=0)
        return float(1.0 / (weights.square().sum() * weights.numel()))

    def finetune(
        self,
        dataset: Dataset[DMPOSample],
        steps: int | None = None,
        debug: bool = False,
    ) -> list[float] | float:
        breakpoint()
        indices = np.random.permutation(len(dataset))
        if steps is not None:
            indices = indices[:steps]
        
        losses = []
        for index in indices:
            print(index)
            losses.append(self.train_step(dataset[int(index)]))
        return losses if debug else float(np.mean(losses))
