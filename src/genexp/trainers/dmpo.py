"""Distribution Matching Policy Optimization for discrete PepTune diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from diffusiongym.types import DDTensor

from peptidesgym.model import DiscreteModel
from peptidesgym.env import DiscreteEnvironment, DiscreteSample


@dataclass
class DMPOSample(DiscreteSample[DDTensor]):
    """Discrete rollout trajectories and their detached DMPO statistics."""

    log_importance: torch.Tensor
    log_reference_minus_policy: torch.Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        batch_size = len(self)
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

    def __getitem__(self, index: int) -> DMPOSample:
        index = range(len(self))[index]
        sample = super().__getitem__(index)
        return DMPOSample(
            sample=sample.sample,
            latent=sample.latent,
            trajectory=sample.trajectory,
            timesteps=sample.timesteps,
            log_p=sample.log_p,
            rewards=sample.rewards,
            info=sample.info,
            kwargs=sample.kwargs,
            log_importance=self.log_importance[index : index + 1],
            log_reference_minus_policy=self.log_reference_minus_policy[index : index + 1],
        )

    @staticmethod
    def concat(samples: list[DMPOSample]) -> DMPOSample:
        sample = DiscreteSample.concat(samples)
        return DMPOSample(
            sample=sample.sample,
            latent=sample.latent,
            trajectory=sample.trajectory,
            timesteps=sample.timesteps,
            log_p=sample.log_p,
            rewards=sample.rewards,
            info=sample.info,
            kwargs=sample.kwargs,
            log_importance=torch.cat([item.log_importance for item in samples]),
            log_reference_minus_policy=torch.cat(
                [item.log_reference_minus_policy for item in samples]
            ),
        )

    def to(self, device: torch.device | str) -> DMPOSample:
        sample = super().to(device)
        return DMPOSample(
            sample=sample.sample,
            latent=sample.latent,
            trajectory=sample.trajectory,
            timesteps=sample.timesteps,
            log_p=sample.log_p,
            rewards=sample.rewards,
            info=sample.info,
            kwargs=sample.kwargs,
            log_importance=self.log_importance.to(device),
            log_reference_minus_policy=self.log_reference_minus_policy.to(device),
        )


def loss_wdce(
    policy_model: DiscreteModel,
    log_rnd: torch.Tensor,
    x: DDTensor | torch.Tensor,
    num_replicates: int = 16,
    *,
    eps: float = 1e-3,
    centering: bool = False,
    uniform_weights: bool = False,
    batch_size: int
) -> torch.Tensor:
    """Weighted denoising cross entropy used by DMPO.

    Final samples are independently masked with probability ``lambda``.  The
    ``1 / lambda`` factor makes the summed masked-token likelihood an unbiased
    estimator of the full denoising objective.  Importance weights are
    detached because rollout collection is off-policy.
    """
    tokens = x.data if isinstance(x, DDTensor) else x
    tokens = tokens.to(policy_model.device)
    if tokens.ndim != 2:
        raise ValueError(f"x must have shape (batch, length), got {tuple(tokens.shape)}")
    log_rnd = log_rnd.to(device=tokens.device, dtype=torch.float32)
    if log_rnd.shape != (tokens.shape[0],):
        raise ValueError(
            f"log_rnd must have shape ({tokens.shape[0]},), got {tuple(log_rnd.shape)}"
        )

    batch = tokens.repeat_interleave(num_replicates, dim=0)
    full_size = batch.shape[0]
    if uniform_weights:
        weights = torch.ones_like(log_rnd) / tokens.shape[0]
    else:
        weights = torch.softmax(log_rnd.detach(), dim=0)
    if centering:
        weights = weights - weights.mean()

    weights = weights.repeat_interleave(num_replicates)
    mask_probability = torch.rand(batch.shape[0], device=batch.device)
    mask_probability = mask_probability.clamp(min=eps, max=1 - eps)
    masked = torch.rand(batch.shape, device=batch.device) < mask_probability[:, None]
    perturbed = torch.where(masked, policy_model.mask_index, batch)
    log_probs = []
    for start in range(0, full_size, batch_size):
        end = min(start + batch_size, full_size)
        pertuber_batch = perturbed[start:end]
        lp = policy_model(DDTensor(pertuber_batch), mask_probability[start:end])
        log_probs.append(lp.data)
    log_probs = torch.cat(log_probs, dim=0)
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
        device: torch.device | None = None,
        verbose: bool = False,
    ):
        if fine_model is base_model:
            raise ValueError("fine_model and base_model must be distinct model instances")
        self.config = config
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

    def generate_dataset(self) -> DMPOSample:
        """Collect rollout batches and compute detached importance weights."""
        self.env.base_model.eval()
        num_samples = int(self.config.num_samples)
        batch_size = int(self.config.batch_size)
        if num_samples < 1:
            raise ValueError("sampling.num_samples must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        samples: list[DMPOSample] = []
        self.fine_model.eval()
        self.base_model.eval()

        for start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - start)
            env_sample = self.env.sample(current_batch_size, pbar=False)
            env_sample = env_sample.to(self.device)
            samples.append(self.solve(env_sample))
        return DMPOSample.concat(samples)

    @torch.no_grad()
    def solve(self, env_sample: DiscreteSample[DDTensor]) -> DMPOSample:
        """Compute exact discretized trajectory likelihood ratios."""
        self.base_model.eval()
        samples = env_sample.latent
        trajectory = env_sample.trajectory
        log_ps = env_sample.log_p
        timestep = env_sample.timesteps
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
            env_sample.rewards,
            dtype=log_ratio.dtype,
            device=self.device,
        )
        if reward_values.shape != (batch_size,):
            raise ValueError(
                f"Environment rewards must have shape ({batch_size},), "
                f"got {tuple(reward_values.shape)}"
            )
        if not torch.isfinite(reward_values).all():
            raise ValueError("Environment rewards contain non-finite values")
        log_importance = (
            log_ratio + reward_values / self.alpha
        ) * self.importance_coefficient
        sample = env_sample.to("cpu")
        return DMPOSample(
            sample=sample.sample,
            latent=sample.latent,
            trajectory=sample.trajectory,
            timesteps=sample.timesteps,
            log_p=sample.log_p,
            rewards=sample.rewards,
            info=sample.info,
            kwargs=sample.kwargs,
            log_importance=log_importance.detach().cpu(),
            log_reference_minus_policy=log_ratio.detach().cpu(),
        )

    def train_step(self, sample: DMPOSample) -> float:
        self.fine_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_wdce(
            self.fine_model,
            sample.log_importance,
            sample.latent,
            num_replicates=self.config.num_replicates,
            eps=self.config.get("mask_eps", 1e-3),
            centering=self.config.get("centering", False),
            batch_size=self.config.batch_size,
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

    def finetune(self, dataset: DMPOSample, steps: int | None = None, debug: bool = False) -> list[float] | float:
        batch_size = self.config.batch_size
        indices = np.random.permutation(len(dataset))
        if steps is not None:
            indices = indices[: steps * batch_size]

        losses: list[float] = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = DMPOSample.concat([dataset[int(index)] for index in batch_indices])
            losses.append(self.train_step(batch))
        return losses if debug else float(np.mean(losses))
