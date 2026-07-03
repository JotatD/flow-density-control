import copy
from typing import Optional

import torch
from omegaconf import DictConfig

from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment
from diffusiongym.types import D

from genexp.trainers.adjoint_matching import AMTrainerFlow
from genexp.trainers.utils import _score_func


class RewDiff(AMTrainerFlow):
    def __init__(
        self,
        config: DictConfig,
        env: Environment,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        if device is None:
            device = env.base_model.device

        num_rew = getattr(env.reward, "num_rew", 1)
        if num_rew != 1:
            raise ValueError(f"RewDiff is only intended for scalar rewards; got num_rew={num_rew}")

        self.rew_config = config
        self.env = env
        self.device = device

        fine_model = copy.deepcopy(env.base_model)
        base_model = copy.deepcopy(env.base_model)
        self.pre_trained_model: BaseModel = copy.deepcopy(env.base_model).to(device)

        self.alpha_div = config.alpha_div
        self.lmbda = config.lmbda

        def grad_reward_fn(sample: D, latent: D) -> D:
            return -self.lmbda * self.reward_gradient(sample, latent) - self.alpha_div * self.divergence(latent)

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

    def reward_gradient(self, sample: D, latent: D) -> D:
        with torch.enable_grad():
            sample_local = sample.detach().requires_grad(True)
            rewards, _ = self.env.reward(sample_local, latent)
            utility = rewards.sum()
            grad = sample_local.gradient(utility)

        return grad

    def divergence(self, latent: D) -> D:
        t = torch.zeros((len(latent),), device=latent.device, dtype=torch.float32)
        base_score = _score_func(self.base_model, latent, t.detach())
        pretrained_score = _score_func(self.pre_trained_model, latent, t.detach())
        return base_score - pretrained_score

    def update_base_model(self):
        state = self.fine_model.state_dict()
        self.base_model.load_state_dict(state)
        self.env.base_model.load_state_dict(state)
