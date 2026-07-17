import copy
from typing import Optional

import torch
from omegaconf import DictConfig

from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment, Sample
from diffusiongym.types import D

from genexp.trainers.ddpo import DDPOTrainer
from genexp.trainers.utils import _score_func
from genexp.mo.utils import HVComputer


EPS = 1e-6


class HVRL(DDPOTrainer):
    def __init__(
        self,
        config: DictConfig,
        env: Environment,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        if device is None:
            device = env.base_model.device

        self.hv_config = config
        self.env = env
        self.device = device
        self.hv_computer = HVComputer(env.reward.ref_point, env.reward.num_rew)

        fine_model = copy.deepcopy(env.base_model)
        base_model = copy.deepcopy(env.base_model)
        self.pre_trained_model: BaseModel = copy.deepcopy(env.base_model).to(device)

        self.num_rews = env.reward.num_rew
        self.ref_point = copy.deepcopy(env.reward.ref_point).to(device)
        self.temperature = config.get("temperature", 0.001)
        self.alpha_div = config.alpha_div
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
            config.ddpo,
            env,
            fine_model,
            device,
            verbose,
            False,
        )
        self.fix_optimization_problem()

    def fix_optimization_problem(self):
        with torch.no_grad():
            self.evaluations_X_ = self.sample_rewards()
            self.hypervolume_X_ = self.hv_computer(self.evaluations_X_)

    def grad_reward_fn(self, sample: D, latent: D, **kwargs) -> D:
        return self.hv_first_variation(sample, latent), {}

    def hv_first_variation(self, sample: D, latent: D) -> torch.Tensor:        
        obj_x, _ = self.og_problem(sample, latent)
        inp_batch = obj_x.shape[0]
        obj_x = obj_x.reshape(inp_batch, 1, 1, self.num_rews).expand(inp_batch, self.num_p_nm1 // (self.n - 1), 1, self.num_rews) #inp_batch, MC_times_p_n_minus_1, 1, k
        expanded_obj_X_ = self.evaluations_X_.expand(inp_batch, self.num_p_nm1 // (self.n - 1), self.n-1, self.num_rews)
        complete_X = torch.cat([expanded_obj_X_, obj_x], dim=2) #inp_batch, MC_times_p_n_minus_1, n, k
        complete_hv = self.hv_computer(complete_X) #inp_batch, MC_times_p_n_minus_1
        expanded_hv_X_ = self.hypervolume_X_.expand(inp_batch, self.num_p_nm1 // (self.n - 1))
        hv_improvement = complete_hv - expanded_hv_X_
        first_var = hv_improvement.mean(dim=1)
        
        return self.lmbda * first_var


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

    def divergence(self, latent: D) -> D:
        t = torch.zeros((len(latent),), device=latent.device, dtype=torch.float32)
        base_score = _score_func(self.base_model, latent, t.detach())
        pretrained_score = _score_func(self.pre_trained_model, latent, t.detach())
        return base_score - pretrained_score

    def update_base_model(self):
        state = self.fine_model.state_dict()
        self.base_model.load_state_dict(state)
        self.env.base_model.load_state_dict(state)
        self.fix_optimization_problem()
        
    def sample_trajectories(self) -> Sample:
        """Sample one batch of trajectories using the fine model as policy."""
        original_policy = self.env._policy
        original_base = self.env.base_model
        self.env.policy = self.fine_model
        self.env.base_model = self.fine_model
        self.env.reward = self.grad_reward_fn
        try:
            env_sample = self.env.sample(self.config.batch_size, pbar=False)
        finally:
            self.env._policy = original_policy
            self.env.base_model = original_base
            self.env.reward = self.og_problem
        return env_sample
