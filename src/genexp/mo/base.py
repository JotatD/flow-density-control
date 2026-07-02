"""Multi-objective reward helpers."""

from typing import Any, Sequence

import torch

from diffusiongym import Reward
from diffusiongym.types import D


class MOReward(Reward[D]):
    """Reward with an explicit number of reward dimensions."""

    def __init__(self, num_rew: int = 1):
        self.num_rew = num_rew

    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError


class CombinedRewards(MOReward[D]):
    """Concatenate outputs from several multi-objective rewards."""

    def __init__(self, rewards: Sequence[MOReward[D]]):
        super().__init__(num_rew=sum(reward.num_rew for reward in rewards))
        self.rewards = list(rewards)

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
