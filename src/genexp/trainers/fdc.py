from genexp import FlowExpansionTrainer

from omegaconf import DictConfig
from diffusiongym.environments import Environment
from typing import Optional

import torch


class FDCTrainer(FlowExpansionTrainer):
    def __init__(
        self,
        config: DictConfig,
        env: Environment,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        config.eta_coeff = 0.0
        config.traj = False
        super().__init__(config, env, device, verbose)
