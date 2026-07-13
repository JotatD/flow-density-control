import copy
from typing import Optional

import torch
from omegaconf import DictConfig

from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment
from diffusiongym.types import D

from genexp.trainers.adjoint_matching import AMTrainerFlow
from genexp.trainers.utils import _score_func


EPS = 1e-6


class HVDiff(AMTrainerFlow):
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

        fine_model = copy.deepcopy(env.base_model)
        base_model = copy.deepcopy(env.base_model)
        self.pre_trained_model: BaseModel = copy.deepcopy(env.base_model).to(device)

        self.num_rews = env.reward.num_rew
        self.ref_point = copy.deepcopy(env.reward.ref_point).to(device)
        self.temperature = config.get("temperature", 0.001)
        self.alpha_div = config.alpha_div
        self.lmbda = config.lmbda
        self.n = config.n

        self.num_lambda = config.get("num_lambda", 4000)
        self.num_p_nm1 = config.get("num_p_nm1", 512)
        if self.num_p_nm1 % (self.n - 1) != 0:
            self.num_p_nm1 = self.num_p_nm1 - (self.num_p_nm1 % (self.n - 1))
            print(f"Warning: num_p_nm1 is not divisible by n. Adjusting num_p_nm1 to be divisible by n-1. New value: {self.num_p_nm1}")
        self.sample_p_nm1_batch_size = config.get("sample_p_nm1_batch_size", -1)
        if self.sample_p_nm1_batch_size <= 0:
            self.sample_p_nm1_batch_size = self.num_p_nm1 

        def grad_reward_fn(sample: D, latent: D) -> D:
            return self.lmbda * self.hv_gradient(sample, latent) - self.alpha_div * self.divergence(latent)

        super().__init__(
            config.adjoint_matching,
            env,
            fine_model,
            base_model,
            grad_reward_fn,
            None,
            device,
            verbose,
        )
        self.fix_optimization_problem()

    def fix_optimization_problem(self):
        self.lambda_ = self.sample_lambda_first_quadrant((self.num_lambda,)).to(self.device)

        with torch.no_grad():
            rewards = self.sample_rewards()
            self.max_s_lambda_X_ = self.max_s_lambda(rewards, self.lambda_)

    def hv_gradient(self, sample: D, latent: D) -> D:
        with torch.enable_grad():
            sample_local = sample.detach().requires_grad(True)
            rewards, _ = self.env.reward(sample_local, latent)
            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(1)

            s_lambda = self.s_hat_lambda(rewards, self.lambda_)
            utility = torch.relu(s_lambda.unsqueeze(1) - self.max_s_lambda_X_.unsqueeze(0)).mean()
            grad = sample_local.gradient(utility)

        return -grad

    def s_hat_lambda(self, rewards: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        rewards = rewards.unsqueeze(1)
        rewards = rewards.expand(-1, self.num_lambda, -1)

        # Compute diff_dot: ReLU((rewards - ref) / lambda)^num_rews
        diff = (rewards - self.ref_point) / lambda_
        diff_dot = torch.relu(diff**self.num_rews)

        # s_lambda: smooth min via -logsumexp(-diff_dot/temp, dim=points) * temp
        s_lambda = -torch.logsumexp(-diff_dot / self.temperature, dim=2) * self.temperature

        return s_lambda.squeeze(-1)

    def max_s_lambda(self, rewards: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        assert self.num_p_nm1 == rewards.shape[0]
        assert self.n - 1 == rewards.shape[1]
        assert self.num_rews == rewards.shape[2]

        rewards = rewards.unsqueeze(2)
        rewards = rewards.expand(-1, -1, self.num_lambda, -1)

        lambda_ = lambda_.unsqueeze(0).unsqueeze(0)
        diff = (rewards - self.ref_point) / lambda_
        diff_dot = torch.relu(diff**self.num_rews)

        s_lambda = torch.min(diff_dot, dim=3, keepdim=False).values
        s_lambda_max = s_lambda.max(dim=1, keepdim=False).values

        return s_lambda_max

    def sample_lambda_first_quadrant(self, shape=(1,)) -> torch.Tensor:
        """Sample lambda directions uniformly from the positive orthant of the unit sphere."""
        z = torch.randn(*shape, self.num_rews, dtype=torch.float32).abs().clamp_min(EPS)
        return z / z.norm(dim=-1, keepdim=True).clamp_min(EPS)

    @torch.no_grad()
    def sample_rewards(self) -> torch.Tensor:
        num_samples = self.num_p_nm1 #* (self.n - 1)
        batch_size = self.sample_p_nm1_batch_size

        all_rewards = []
        remaining = num_samples
        old_discretization_steps = self.env.discretization_steps
        self.env.discretization_steps = 1000
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            env_sample = self.env.sample(current_batch_size, pbar=False)
            sample = env_sample.sample.to(self.device)
            latent = env_sample.trajectory[-1].to(self.device)
            rewards, _ = self.env.reward(sample, latent)
            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(1)
            all_rewards.append(rewards.to(self.device))
            remaining -= current_batch_size
        self.env.discretization_steps = old_discretization_steps

        rewards = torch.cat(all_rewards, dim=0)
        return rewards.reshape(self.num_p_nm1, self.n - 1, self.num_rews)

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
