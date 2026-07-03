from typing import Any, Literal

import numpy as np
import torch
from diffusiongym.base_models import BaseModel
from torch import nn
from diffusiongym.schedulers import Scheduler
from diffusiongym.types import DDTensor




class TensorMLPModel(BaseModel[DDTensor]):
    def __init__(
        self,
        scheduler: Scheduler[DDTensor],
        output_type: Literal["epsilon", "velocity"],
        input_dim: int,
        device: torch.device,
    ):
        super().__init__(device)
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, 512),
            nn.SELU(),
            nn.Linear(512, 512),
            nn.SELU(),
            nn.Linear(512, 512),
            nn.SELU(),
            nn.Linear(512, 512),
            nn.SELU(),
            nn.Linear(512, input_dim),
        )
        self._scheduler = scheduler
        self.output_type = output_type
        self.input_dim = input_dim

    @property
    def scheduler(self) -> Scheduler[DDTensor]:
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[DDTensor, dict[str, Any]]:
        return DDTensor(torch.randn(n, self.input_dim, device=self.device)), kwargs

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs: Any) -> DDTensor:
        model_t = self.scheduler.model_input(t).to(device=x.data.device, dtype=x.data.dtype)
        return DDTensor(self.model(torch.cat([x.data, model_t[:, None]], dim=1)))