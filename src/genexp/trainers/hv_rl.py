import copy
import pickle as pkl
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch
from diffusiongym.environments import Environment, Sample
from diffusiongym.rewards import Reward
from diffusiongym.types import D

from genexp.mo.base import MOReward
from genexp.mo.utils import HVComputer
from genexp.trainers.nft import DiffusionNFTrainer


class FirstVariation(Reward[D]):
    def __init__(self, call_fn: Callable[[D, D], tuple[torch.Tensor, dict[str, Any]]]):
        super().__init__()
        self.call_fn = call_fn

    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.call_fn(sample, latent, **kwargs)



class HVRL(DiffusionNFTrainer):
    def __init__(
        self,
        config: Namespace,
        env: Environment,
        reward: MOReward,
        hv_computer: HVComputer,
        device: torch.device | None = None,
    ):
        if device is None:
            device = env.base_model.device

        self.batch_size = config.batch_size
        self.hv_config = config
        self.env = env
        self.device = device
        self.hv_computer = hv_computer

        self.base_model = copy.deepcopy(env.base_model)

        self.reward = reward
        self.n = config.n
        self.num_rews = reward.num_rew

        self.num_p_nm1 = config.num_p_nm1
        
        self.scalarization = config.scalarization
        super().__init__(
            config,
            env,
            self.base_model,
            device,
            False,
        )

    def fix_optimization_problem(self):
        with torch.no_grad():
            self.evaluations_X_ = self.sample_rewards()
            self.hypervolume_X_ = self.hv_computer(self.evaluations_X_)

    def training_state_dict(self) -> dict[str, Any]:
        state = super().training_state_dict()
        state["evaluations_X_"] = getattr(self, "evaluations_X_", None)
        state["hypervolume_X_"] = getattr(self, "hypervolume_X_", None)
        return state

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        super().load_training_state_dict(state)
        for name in ("evaluations_X_", "hypervolume_X_"):
            value = state.get(name)
            if value is not None:
                setattr(self, name, value)

    def hv_first_variation(self, sample: D, latent: D) -> tuple[torch.Tensor, dict[str, Any]]:
        obj_x, info = self.reward(sample, latent)
        info['obj'] = obj_x
        inp_batch = obj_x.shape[0]
        obj_x = obj_x.reshape(inp_batch, 1, 1, self.num_rews).expand(
            inp_batch,
            self.num_p_nm1,
            1,
            self.num_rews,
        )
        self.evaluations_X_ = self.evaluations_X_.to(obj_x.device)
        self.hypervolume_X_ = self.hypervolume_X_.to(obj_x.device)
        expanded_obj_X_ = self.evaluations_X_.expand(inp_batch, self.num_p_nm1, self.n - 1, self.num_rews)
        complete_X = torch.cat([expanded_obj_X_, obj_x], dim=2)
        complete_hv = self.hv_computer(complete_X)
        expanded_hv_X_ = self.hypervolume_X_.expand(inp_batch, self.num_p_nm1)
        hv_improvement = complete_hv - expanded_hv_X_
        first_var = hv_improvement.mean(dim=1)

        info['scalarization'] = first_var
        return first_var, info
    
    def sum_scalarization(self, sample: D, latent: D) -> tuple[torch.Tensor, dict[str, Any]]:
        obj_x, info = self.reward(sample, latent)
        
        scalarization = obj_x.sum(dim=1)
        info['obj'] = obj_x
        info['scalarization'] = scalarization
        return scalarization, info

    @torch.no_grad()
    def sample_rewards(self) -> torch.Tensor:
        all_rewards = []
        remaining = self.num_p_nm1 * (self.n - 1)

        original_policy = self.env._policy
        self.env.policy = self.base_model
        try:
            while remaining > 0:
                current_batch_size = min(self.batch_size, remaining)
                env_sample = self.env.sample(current_batch_size, pbar=False)
                rews, _ = self.reward(env_sample.sample, env_sample.latent)
                all_rewards.append(rews)
                remaining -= current_batch_size
        finally:
            self.env._policy = original_policy

        rewards = torch.cat(all_rewards, dim=0)
        return rewards.reshape(self.num_p_nm1, self.n - 1, self.num_rews)

    def update_base_model(self):
        state = self.fine_model.state_dict()
        self.base_model.load_state_dict(state)
        self.env.base_model.load_state_dict(state)

    def generate_dataset_fv(self) -> tuple[list[Sample], torch.Tensor, torch.Tensor]:
        """Collect trajectories, compute global advantages, build training dataset."""
        if self.scalarization == "improvement":
            samples, advantages, rewards = self.generate_dataset(FirstVariation(self.hv_first_variation))
        elif self.scalarization == "sum":
            samples, advantages, rewards = self.generate_dataset(FirstVariation(self.sum_scalarization))
        else:
            raise ValueError(f"Unknown scalarization method: {self.scalarization}")
            
        return samples, advantages, rewards
