import math
import random
from collections.abc import Sequence
from typing import Any

import torch
from diffusiongym import BaseModel, D, DummyReward, EndpointEnvironment, Sample


class BlendEnvironment(EndpointEnvironment[D]):
    """Inference environment for DB-MPA and its probabilistic approximation."""

    def __init__(self, tilteds: list[BaseModel[D]], discretization_steps: int, sampling_mode: str, reward_scale: float = 1.0):
        if not tilteds:
            raise ValueError("tilteds must contain at least one model")
        if sampling_mode not in {"full", "proba"}:
            raise ValueError("sampling_mode must be either 'full' or 'proba'")

        self.tilteds = tilteds
        self.num_rews = len(tilteds)
        self.sampling_mode = sampling_mode
        super().__init__(tilteds[0], reward=DummyReward(), discretization_steps=discretization_steps, reward_scale=reward_scale)

    def _validate_blend_vector(self, blend_vector: Sequence[float]) -> list[float]:
        weights = [float(weight) for weight in blend_vector]
        if len(weights) != self.num_rews:
            raise ValueError(f"blend_vector must have {self.num_rews} entries, got {len(weights)}")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("blend_vector entries must be finite and nonnegative")
        if not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"blend_vector must sum to 1, got {sum(weights)}")
        return weights

    def drift(self, x: D, t: torch.Tensor, **kwargs: Any) -> tuple[D, torch.Tensor]:
        blend_vector = self._validate_blend_vector(kwargs.pop("blend_vector"))

        if self.sampling_mode == "full":
            action = blend_vector[0] * self.tilteds[0](x, t, **kwargs)
            for weight, model in zip(blend_vector[1:], self.tilteds[1:], strict=True):
                action += weight * model(x, t, **kwargs)
        else:
            model_idx = random.choices(range(self.num_rews), weights=blend_vector, k=1)[0]
            action = self.tilteds[model_idx](x, t, **kwargs)

        # EndpointEnvironment's drift is affine in its model output, so a simplex-weighted
        # endpoint prediction is exactly the same as the corresponding weighted drift.
        alpha = self.scheduler.alpha(x, t)
        beta = self.scheduler.beta(x, t)
        kappa = self.scheduler.kappa(x, t)
        eta = self.scheduler.eta(x, t)
        sigma = self.scheduler.sigma(x, t)
        sigma_eta = 0.5 * sigma * sigma + eta
        a = kappa - sigma_eta / (beta * beta)
        b = sigma_eta * alpha / (beta * beta)
        return a * x + b * action, torch.zeros(len(x), device=self.device, dtype=t.dtype)

    @torch.no_grad()
    def sample(self, n: int, pbar: bool = True, x0: D | None = None, blend_vector: Sequence[float] | None = None, **kwargs: Any) -> Sample[D]:
        if blend_vector is None:
            raise ValueError("blend_vector must be provided")
        return super().sample(n, pbar=pbar, x0=x0, blend_vector=self._validate_blend_vector(blend_vector), **kwargs)
