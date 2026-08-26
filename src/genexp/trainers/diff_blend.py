from triton.language import advance
import copy
import random
from argparse import Namespace
from typing import Any

import numpy as np
import torch
from diffusiongym import BaseModel, D, DummyReward, Environment, Sample

from genexp.mo.base import MOReward
from genexp.trainers.nft import DiffusionNFTrainer, RewardStatTracker

class BlendEnvironment(Environment):
    def __init__(self, base_model: BaseModel[D], reward: MOReward[D], discretization_steps: int, reward_scale: float = 1.0):
        self.reward = DummyReward()
        self.num_rews = reward.num_rew
        self.tilteds = [copy.deepcopy(base_model) for _ in range(reward.num_rew)]
        super().__init__(base_model, reward=reward, discretization_steps=discretization_steps, reward_scale=reward_scale)
    
    def drift(self, x: D, t: torch.Tensor, **kwargs: Any) -> tuple[D, torch.Tensor]:
        prev_policy = self.policy
        blend_vector: list[float] | None = kwargs.pop("blend_vector", None)
        
        if blend_vector is None:
            raise ValueError("blend_vector must be provided in kwargs for BlendEnvironment drift computation.")
        try:
            idx = random.choices(range(self.num_rews), weights=blend_vector, k=1)[0]
            self.policy = self.tilteds[idx]
            a, b = super().drift(x, t, **kwargs)
        finally:
            self.policy = prev_policy
        
        return a, b
    
    
class DiffBlend:
    def __init__(
        self,
        config: Namespace,
        env: BlendEnvironment,
        reward: MOReward,
        device: torch.device | None = None,
    ):
        if device is None:
            device = env.base_model.device

        self.batch_size = config.batch_size
        self.env = env
        self.device = device

        self.base_model = copy.deepcopy(env.base_model)
        self.tilteds: list[BaseModel]= env.tilteds
        self.config = config
        self.trainers: list[DiffusionNFTrainer] = [
            DiffusionNFTrainer(
                config,
                env,
                tilted,
                device,
                False,
            )
            for tilted in self.tilteds
        ]

        self.reward = reward
        self.num_rews = reward.num_rew
        self.reward_stat_tracker = RewardStatTracker(config.advantage_group_size, config.advantage_group_stride, config.advantage_group_aggregation)


    def generate_dataset(self, reward: MOReward[D]) -> tuple[list[Sample], torch.Tensor, torch.Tensor]:
        """Collect exploration-policy samples and normalize their rewards."""
        all_samples = []
        advantages = []
        all_rewards = []
        
        for i in range(self.num_rews):
            tilted = self.tilteds[i]
            trainer = self.trainers[i]
            env_sample = self.env.sample(self.batch_size, pbar=False)
            rews, _ = reward(env_sample.sample, env_sample.latent, tradeoff=torch.eye(self.num_rews, device=self.device)[i])
            all_rewards.append(rews)
            all_samples.append(env_sample)
            advs = self.reward_stat_tracker.update(rews)
            advantages.append(advs)

        return all_samples, advantages, all_rewards

    def finetune(self, samples: list[Sample], advantages: torch.Tensor) -> list[dict[str, list[float]]]:
        """Finetune the fine model using the given samples and advantages."""
        stats_avgs_per_batch = [[] for _ in range(self.num_rews)]
        start = 0
        while start < len(samples):
            for i, tilted in enumerate(self.tilteds):
                advs = advantages[i, start : start + self.batch_size]
                tilted.train()
                trainer = self.trainers[i]
                batch_samples = Sample.concat(samples[start : start + self.batch_size])
                stats_avgs_per_batch[i].append(trainer.train_step(batch_samples, advs))
            start += self.batch_size

        timed_final_stats = [{} for _ in range(self.num_rews)]
        for i in range(self.num_rews):
            timed_final_stats[i] = stats_avgs_per_batch[i][0]
            for key in timed_final_stats[i]:
                for t in range(len(timed_final_stats[i][key])):
                    timed_final_stats[i][key][t] = np.mean([stats[key][t] for stats in stats_avgs_per_batch[i]])
        return timed_final_stats


    @torch.no_grad()
    def sample_blend(self, num_samples: int, samples_per_tradeoff: int) -> tuple[list[Sample], torch.Tensor]:
        assert num_samples % samples_per_tradeoff == 0, "num_samples must be divisible by samples_per_tradeoff"
        all_rewards = []
        all_samples = []
        remaining = num_samples
        tradeoffs = self.uniform_positive_tradeoffs(num_samples // samples_per_tradeoff)
        
        while remaining > 0:
            for tradeoff in tradeoffs:
                if remaining <= 0:
                    break
                env_sample = self.env.sample(samples_per_tradeoff, pbar=False)
                rews, _ = self.reward(env_sample.sample, env_sample.latent, tradeoff=tradeoff)
                all_rewards.append(rews)
                all_samples.append(env_sample)
                remaining -= samples_per_tradeoff
            remaining -= samples_per_tradeoff

        rewards = torch.cat(all_rewards, dim=0)
        return all_samples, rewards.reshape(num_samples, self.num_rews)
    
    def uniform_positive_tradeoffs(self, num_samples: int) -> torch.Tensor:
        """Generate uniform positive tradeoffs that sum to 1."""
        tradeoffs = torch.rand(num_samples, self.num_rews, device=self.device)
        tradeoffs /= tradeoffs.sum(dim=1, keepdim=True)
        return tradeoffs
