import copy
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from diffusiongym import DDMixin, Scheduler
from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment, Sample
from diffusiongym.rewards import Reward
from diffusiongym.types import D

from genexp.mo.base import MOReward
from genexp.mo.utils import HVComputer
from genexp.trainers.adjoint_matching import create_timestep_subset

EPS = 1e-6

def endpoint(model: BaseModel[D], vt: D, xt: D, t: torch.Tensor) -> D:
    scheduler: Scheduler = model.scheduler
    beta = scheduler.beta(xt, t)
    beta_dot = scheduler.beta_dot(xt, t)
    alpha = scheduler.alpha(xt, t)
    alpha_dot = scheduler.alpha_dot(xt, t)
    return (beta * vt - beta_dot * xt) / (alpha_dot * beta - alpha * beta_dot + EPS)

def velocity(model: BaseModel[D], x: D, t: torch.Tensor, **kwargs) -> D:
    """Compute the SDE drift used by each environment type from model output.

    Mirrors the `a * x + b * action` formulas in each Environment.drift().
    No control correction term — only the policy-dependent mean.
    """
    output = model.forward(x, t, **kwargs)
    scheduler: Scheduler = model.scheduler
    
    if model.output_type == "endpoint":
        beta = scheduler.beta(x, t)
        beta_dot = scheduler.beta_dot(x,t)
        alpha_dot = scheduler.alpha_dot(x, t)
        
        return (beta_dot / beta) * x + (alpha_dot - beta_dot / beta) * output
    
    else:
        raise ValueError(f"{model.output_type} not supported")
    
class FirstVariation(Reward[D]):
    def __init__(self, call_fn: Callable[[D, D], tuple[torch.Tensor, dict[str, Any]]]):
        super().__init__()
        self.call_fn = call_fn

    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.call_fn(sample, latent, **kwargs)

class DiffusionNFTrainer:
    def __init__(
        self,
        config: Namespace,
        env: Environment,
        base_model: BaseModel,
        device: torch.device | None = None,
        use_valids: bool = False,
    ):
        # There are 3 conceptual policies
        # The base policy <-- updated every MD iteration
        # The exploration policy <--- given by MCTS in Peptune, maybe a rolling EMA in this case
        # The finetuned policy
        # The finetuned policy will be carried by the enviornment.
        
        
        self.config = config
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.batch_size: int = config.batch_size

        self.use_valids = use_valids
        self.clip_range: float = config.clip_range
        self.adv_clip_max: float = config.adv_clip_max
        self.clip_grad_norm: float = config.clip_grad_norm
        self.num_inner_epochs: int = config.num_inner_epochs
        self.timestep_fraction: float = config.timestep_fraction
        
        self.beta: float = config.beta # do not confuse with interpolant beta
        self.alpha: float = config.alpha #do not confuse with interpolant alpha

        self.env = env
        self.base_model = base_model
        self.base_model.to(self.device)
                
        self.fine_model = env.model
        
        self.exploration_model = copy.deepcopy(self.base_model)

        self.configure_optimizers()

    def configure_optimizers(self):
        if hasattr(self, "optimizer"):
            del self.optimizer
        self.optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=self.config.lr)

    def generate_dataset(self, reward: Reward[D]) -> tuple[list[Sample], torch.Tensor]:
        """Collect trajectories, compute global advantages, build training dataset."""
        self.fine_model.eval()
        all_samples: list[Sample] = []
        all_rewards = []
        remaining = self.config.num_samples
        while remaining > 0:
            batch = min(remaining, self.batch_size)
            env_sample = self.env.sample(batch, pbar=False)
            rewards, _ = reward(env_sample.sample, env_sample.latent)
            all_samples.extend([s for s in env_sample])
            all_rewards.append(rewards)
            remaining -= batch

        # TODO: insert the reward tracker
        all_rewards = torch.cat(all_rewards, dim=0)
        advantages = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

        return all_samples, advantages
    
    def train_step(self, sample: Sample, advantages: torch.Tensor) -> float:
        x0: DDMixin = sample.latent.to(self.device)
        timesteps = sample.timesteps.to(self.device)
        kwargs = sample.kwargs

        T = self.env.discretization_steps
        idxs = create_timestep_subset(T, final_percent=0.25, sample_percent=max(0.0, self.timestep_fraction - 0.25)) if self.timestep_fraction < 1.0 else np.arange(T)

        adv_clipped = torch.clamp(advantages, -self.adv_clip_max, self.adv_clip_max)
        normalized_advantages_clip = 0.5 * (adv_clipped / self.adv_clip_max) + 0.5
        r = torch.clamp(normalized_advantages_clip, 0, 1).to(self.device)

        losses = []
        for idx in idxs:
            t = timesteps[idx].unsqueeze(0).expand(len(x0))
            noise = x0.randn_like().to(self.device)
            interpolant_beta, interpolant_alpha = self.base_model.scheduler.beta(x0, t), self.base_model.scheduler.alpha(x0, t)
            
            xt = interpolant_beta * x0 + interpolant_alpha * noise
            #sigma = diffusions[idx].to(self.device)

            n = len(xt)
            t_batch = timesteps[idx].unsqueeze(0).expand(n)
            step_kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

            # Differentiable drift from the current fine model
            forward_prediction = velocity(self.fine_model, xt, t_batch, **step_kwargs)
            
            with torch.no_grad():
                old_prediction = velocity(self.exploration_model, xt, t_batch, **step_kwargs)
                ref_forward_prediction = velocity(self.base_model, xt, t_batch, **step_kwargs)
            
            positive_v = self.beta * forward_prediction + (1 - self.beta) * old_prediction
            positive_x0 = endpoint(self.base_model, positive_v, xt, t_batch)
            
            with torch.no_grad():
                positive_weight_factor = (positive_x0 - x0).apply(torch.abs).aggregate('mean').clip(0.0001)
            positive_loss = ((positive_x0 - x0)**2).aggregate('mean') / positive_weight_factor
            
            negative_v = -self.beta * forward_prediction + (1 + self.beta) * old_prediction
            negative_x0 = endpoint(self.base_model, negative_v, xt, t_batch)
            with torch.no_grad():
                negative_weight_factor = (negative_x0 - x0).apply(torch.abs).aggregate("mean").clip(0.0001)
            negative_loss = ((negative_x0 - x0) ** 2).aggregate("mean") / negative_weight_factor
            
            ori_policy_loss = r * (positive_loss / self.beta) + (1.0 - r) * (negative_loss / self.beta)
            policy_loss = (ori_policy_loss * self.adv_clip_max).mean()

            
            kl_div_loss = ((forward_prediction - ref_forward_prediction)**2).aggregate("mean").mean()
            
            loss = policy_loss + self.alpha * kl_div_loss
            
            losses.append(loss)

        if not losses:
            return float("inf")

        loss = torch.stack(losses).mean()
        if loss.isnan():
            return float("inf")

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.fine_model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        
        return loss.item()
        

    def finetune(self, samples: list[Sample], advantages: torch.Tensor) -> float:
        """Finetune the fine model using the given samples and advantages."""
        self.fine_model.train()
        start = 0
        losses = []
        while start < len(samples):
            advs = advantages[start:start + self.batch_size]
            batch_samples = Sample.concat(samples[start:start + self.batch_size])
            loss = self.train_step(batch_samples, advs)
            start += self.batch_size
            losses.append(loss)
        return sum(losses) / len(losses)

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

    def hv_first_variation(self, sample: D, latent: D) -> tuple[torch.Tensor, dict[str, Any]]:        
        obj_x, info = self.reward(sample, latent)
        inp_batch = obj_x.shape[0]
        obj_x = obj_x.reshape(inp_batch, 1, 1, self.num_rews).expand(inp_batch, self.num_p_nm1, 1, self.num_rews) #inp_batch, MC_times_p_n_minus_1, 1, k
        expanded_obj_X_ = self.evaluations_X_.expand(inp_batch, self.num_p_nm1, self.n-1, self.num_rews)
        complete_X = torch.cat([expanded_obj_X_, obj_x], dim=2) #inp_batch, MC_times_p_n_minus_1, n, k
        complete_hv = self.hv_computer(complete_X) #inp_batch, MC_times_p_n_minus_1
        expanded_hv_X_ = self.hypervolume_X_.expand(inp_batch, self.num_p_nm1)
        hv_improvement = complete_hv - expanded_hv_X_
        first_var = hv_improvement.mean(dim=1)
        
        return first_var, info


    @torch.no_grad()
    def sample_rewards(self) -> torch.Tensor:
        all_rewards = []
        remaining = self.num_p_nm1 * (self.n - 1)
        while remaining > 0:
            current_batch_size = min(self.batch_size, remaining)
            env_sample = self.env.sample(current_batch_size, pbar=False)
            rews, _ = self.reward(env_sample.sample, env_sample.latent)
            all_rewards.append(rews) 
            remaining -= current_batch_size

        rewards = torch.cat(all_rewards, dim=0)
        return rewards.reshape(self.num_p_nm1, self.n - 1, self.num_rews)

    def update_base_model(self):
        state = self.fine_model.state_dict()
        self.base_model.load_state_dict(state)
        self.env.base_model.load_state_dict(state)
        
    def generate_dataset_fv(self) -> tuple[list[Sample], torch.Tensor]:
        """Collect trajectories, compute global advantages, build training dataset."""
        return self.generate_dataset(FirstVariation(self.hv_first_variation))
        