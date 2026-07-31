import copy
from typing import Any, Optional

import torch
from omegaconf import DictConfig

from peptidesgym.model import DiscreteModel
from peptidesgym import DiscreteEnvironment
from diffusiongym.types import D

from diffusiongym import Reward
from genexp.trainers.dmpo import DMPOTrainer
from genexp.trainers.utils import _score_func
from genexp.mo.utils import HVComputer


EPS = 1e-6


class FirstVariation(Reward[D]):
    def __init__(self, call_fn: callable):
        super().__init__()
        self.call_fn = call_fn

    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.call_fn(sample, latent, **kwargs)

class HVDiscreteRL(DMPOTrainer):
    def __init__(
        self,
        config: DictConfig,
        env: DiscreteEnvironment,
        base_model: DiscreteModel,
        fine_model: DiscreteModel,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        if device is None:
            device = fine_model.device

        self.hv_config = config
        self.env = env
        self.device = device
        self.hv_computer = HVComputer(env.reward.ref_point, env.reward.num_rew)

        self.base_model = base_model.to(device)
        self.env.base_model = fine_model

        self.num_rews = env.reward.num_rew
        self.ref_point = copy.deepcopy(env.reward.ref_point).to(device)
        self.temperature = config.get("temperature", 0.001)
        self.lmbda = config.lmbda
        self.n = config.n
        self.og_problem = env.reward

        self.num_p_nm1 = config.get("num_p_nm1", 512)
        if self.num_p_nm1 % (self.n - 1) != 0:
            self.num_p_nm1 = self.num_p_nm1 - (self.num_p_nm1 % (self.n - 1))
            print(f"Warning: num_p_nm1 is not divisible by n. Adjusting num_p_nm1 to be divisible by n-1. New value: {self.num_p_nm1}")
        self.sample_p_nm1_batch_size = config.get("sample_p_nm1_batch_size", -1)
        if self.sample_p_nm1_batch_size <= 0:
            self.sample_p_nm1_batch_size = self.num_p_nm1 
            
        self.sampling_kwargs = config.get("sampling_kwargs", {})

        super().__init__(
            config.dmpo,
            env,
            fine_model,
            base_model,
            device,
            verbose,
        )
        self.fix_optimization_problem()

    def fix_optimization_problem(self):
        with torch.no_grad():
            self.env.reward = self.og_problem
            self.evaluations_X_ = self.sample_rewards()
            self.hypervolume_X_ = self.hv_computer(self.evaluations_X_)
        self.env.reward = FirstVariation(self.hv_first_variation)

    def grad_reward_fn(self, sample: D, latent: D, **kwargs) -> D:
        return self.hv_first_variation(sample, latent), {}

    def hv_first_variation(self, sample: D, latent: D) -> torch.Tensor:        
        obj_x, info = self.og_problem(sample, latent)
        inp_batch = obj_x.shape[0]
        nm1_batch = self.num_p_nm1 // (self.n - 1)
        obj_x = obj_x.reshape(inp_batch, 1, 1, self.num_rews).expand(inp_batch, nm1_batch, 1, self.num_rews)
        expanded_obj_X_ = self.evaluations_X_.expand(inp_batch, nm1_batch, self.n-1, self.num_rews) #<- i think we need a reshape here
        complete_X = torch.cat([expanded_obj_X_, obj_x], dim=2) #inp_batch, nm1_batch, n, k
        complete_hv = self.hv_computer(complete_X) #inp_batch, nm1_batch
        expanded_hv_X_ = self.hypervolume_X_.expand(inp_batch, nm1_batch)
        hv_improvement = complete_hv - expanded_hv_X_
        first_var = hv_improvement.mean(dim=1)
        
        return self.lmbda * first_var, info


    @torch.no_grad()
    def sample_rewards(self) -> torch.Tensor:
        num_samples = self.num_p_nm1 #* (self.n - 1)
        batch_size = self.sample_p_nm1_batch_size
        final_batch_size = num_samples // (self.n - 1)

        all_rewards = []
        remaining = num_samples
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            env_sample = self.env.sample(current_batch_size, pbar=False)
            rewards = env_sample.rewards.to(self.device)
            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(1)
            all_rewards.append(rewards.to(self.device))
            remaining -= current_batch_size

        rewards = torch.cat(all_rewards, dim=0)
        return rewards.reshape(final_batch_size, self.n - 1, self.num_rews)
    
    def update_base_model(self):
        state = self.fine_model.state_dict()
        self.base_model.load_state_dict(state)
        self.env.base_model.load_state_dict(state)
        self.fix_optimization_problem()
