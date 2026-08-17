"""Multi-objective reward helpers."""

from typing import Any, Sequence

import torch

from diffusiongym import Reward
from diffusiongym.types import D


class MOReward(Reward[D]):
    """Reward with an explicit number of reward dimensions."""

    def __init__(self, reward: Reward[D], num_rew: int = 1, ref_point: torch.Tensor | None = None):
        self.num_rew = num_rew
        if ref_point is None:
            ref_point = torch.zeros(self.num_rew)
        self.ref_point = ref_point
        assert self.ref_point.shape == (self.num_rew,), f"Expected ref_point with shape ({self.num_rew},); got {self.ref_point.shape}" 
        self.reward = reward

    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.reward(sample, latent, **kwargs)


class CombinedRewards(MOReward[D]):
    """Concatenate outputs from several multi-objective rewards."""

    def __init__(self, rewards: Sequence[MOReward[D]], ref_point: torch.Tensor | None = None):
        self.rewards = list(rewards)
        if ref_point is None:
            ref_point = torch.stack([reward.ref_point for reward in self.rewards])

    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        rewards = []
        infos = {}

        for i, reward in enumerate(self.rewards):
            reward_value, info = reward(sample, latent, **kwargs)
            if reward_value.ndim == 1:
                reward_value = reward_value.unsqueeze(1)
            rewards.append(reward_value)
            infos[f"rew_{i}"] = info

        return torch.cat(rewards, dim=1), infos
