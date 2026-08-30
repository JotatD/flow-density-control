from triton.language import advance
import copy
import random
from argparse import Namespace
from typing import Any, Optional

import numpy as np
import torch
from diffusiongym import BaseModel, D, DummyReward, Environment, Sample, EndpointEnvironment

from genexp.mo.base import MOReward
from genexp.trainers.nft import DiffusionNFTrainer, RewardStatTracker

class BlendEnvironment(EndpointEnvironment[D]):
    def __init__(self, tilteds: list[BaseModel[D]], discretization_steps: int, reward_scale: float = 1.0):
        self.reward = DummyReward()
        self.num_rews = len(tilteds)
        self.tilteds = tilteds
        super().__init__(tilteds[0], reward=self.reward, discretization_steps=discretization_steps, reward_scale=reward_scale)
    
    def drift(self, x: D, t: torch.Tensor, **kwargs: Any) -> tuple[D, torch.Tensor]:
        prev_policy = self.policy
        blend_vector: list[float] | None = kwargs.pop("blend_vector", None)
                 
        try:
            idx = random.choices(range(self.num_rews), weights=blend_vector, k=1)[0]
            self.policy = self.tilteds[idx]
            a, b = super().drift(x, t, **kwargs)
        finally:
            self.policy = prev_policy
        
        return a, b
    
    @torch.no_grad()
    def sample(self, n: int, pbar: bool = True, x0: D | None = None, blend_vector: list[float] | None = None, **kwargs: Any) -> Sample[D]:
        if blend_vector is None:
            raise ValueError("blend_vector must be provided in kwargs for BlendEnvironment drift computation.")
        else:
            assert len(blend_vector) == self.num_rews 
            
        kwargs.update({"blend_vector": blend_vector})
        return super().sample(n, pbar=pbar, x0=x0, **kwargs)
