"""ZDT multi-objective rewards."""

from typing import Any, Literal

import torch

from diffusiongym import DDTensor

from genexp.mo.base import MOReward


class _ZDTTorch(MOReward[DDTensor]):
    default_input_dim = 30
    ref_point = torch.tensor([-1.1, -10.1])
    valid_input_transforms = {"clamp", "sigmoid", "none"}

    def __init__(
        self,
        input_dim: int | None = None,
        input_transform: Literal["clamp", "sigmoid", "none"] = "clamp",
        eps: float = 1e-8,
    ):
        super().__init__(num_rew=2, ref_point=self.ref_point)
        self.input_dim = self.default_input_dim if input_dim is None else input_dim
        self.input_transform = input_transform
        self.eps = eps

    def _unit_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_transform == "clamp":
            return x.clamp(min=0.0, max=1.0)
        if self.input_transform == "sigmoid":
            return torch.sigmoid(x)
        if self.input_transform == "none":
            return x
        raise RuntimeError(f"Unexpected input_transform={self.input_transform!r}")

    def _evaluate_minimization(self, x_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __call__(self, sample: DDTensor, latent: DDTensor, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        x_unit = self._unit_input(sample.data)
        f1, f2 = self._evaluate_minimization(x_unit)
        return -torch.stack([f1, f2], dim=1), {}


class ZDT1Torch(_ZDTTorch):
    """Continuous ZDT1 reward with negated objectives for maximization."""

    default_input_dim = 30
    ref_point = torch.tensor([-1.1, -10.1])

    def _evaluate_minimization(self, x_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = x_unit[:, 0]
        g = 1.0 + 9.0 * ((x_unit[:, 1:] - 0.5) ** 2).mean(dim=1) / 0.25
        h = 1.0 - torch.sqrt((f1 / g).clamp_min(self.eps))
        return f1, g * h


class ZDT2Torch(_ZDTTorch):
    """Continuous ZDT2 reward with negated objectives for maximization."""

    default_input_dim = 30
    ref_point = torch.tensor([-1.1, -10.1])

    def _evaluate_minimization(self, x_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = x_unit[:, 0]
        g = 1.0 + 9.0 * ((x_unit[:, 1:] - 0.5) ** 2).mean(dim=1) / 0.25
        h = 1.0 - (f1 / g) ** 2
        return f1, g * h


class ZDT3Torch(_ZDTTorch):
    """Continuous ZDT3 reward with disconnected Pareto front segments."""

    default_input_dim = 30
    ref_point = torch.tensor([-1.1, -20.1])

    def _evaluate_minimization(self, x_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = x_unit[:, 0]
        g = 1.0 + 9.0 * ((x_unit[:, 1:] - 0.5) ** 2).mean(dim=1) / 0.25
        ratio = f1 / g
        h = 1.0 - torch.sqrt(ratio.clamp_min(self.eps)) - ratio * torch.sin(10.0 * torch.pi * f1)
        return f1, g * h


class ZDT4Torch(_ZDTTorch):
    """Continuous ZDT4 reward with negated objectives for maximization."""

    default_input_dim = 10
    ref_point = torch.tensor([-1.1, -125.1])

    def _evaluate_minimization(self, x_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = x_unit[:, 0]
        x_tail = 20.0 * x_unit[:, 1:] - 10.0
        g = 1.0 + 10.0 * (self.input_dim - 1) + (x_tail**2 - 10.0 * torch.cos(4.0 * torch.pi * x_tail)).sum(dim=1)
        h = 1.0 - torch.sqrt((f1 / g).clamp_min(self.eps))
        return f1, g * h


class ZDT6Torch(_ZDTTorch):
    """Continuous ZDT6 reward with negated objectives for maximization."""

    default_input_dim = 10
    ref_point = torch.tensor([-1.1, -10.1])

    def _evaluate_minimization(self, x_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = x_unit[:, 0]
        f1 = 1.0 - torch.exp(-4.0 * x1) * torch.sin(6.0 * torch.pi * x1) ** 6
        g = 1.0 + 9.0 * ((x_unit[:, 1:] - 0.5) ** 2).mean(dim=1) / 0.25**0.25
        h = 1.0 - (f1 / g) ** 2
        return f1, g * h
