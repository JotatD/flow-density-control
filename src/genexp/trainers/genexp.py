from omegaconf import DictConfig
from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment
from diffusiongym.types import D
from genexp.constraints import Constraint
from genexp.trainers.adjoint_matching import AMTrainerFlow
from genexp.trainers.ddpo import DDPOTrainer
from typing import Optional
from tqdm import tqdm

import torch
import copy


def _score_func(model: BaseModel[D], latent: D, t: torch.Tensor) -> D:
    """Compute the score function ∇log p_t(x) from a velocity-predicting model."""
    if model.output_type == "score":
        return model.forward(latent, t)

    elif model.output_type == "velocity":
        v = model.forward(latent, t)
        scheduler = model.scheduler
        kappa = scheduler.kappa(latent, t)
        eta = scheduler.eta(latent, t)
        return (v - kappa * latent) / eta

    elif model.output_type == "endpoint":
        x_1 = model.forward(latent, t)
        scheduler = model.scheduler
        alpha = scheduler.alpha(latent, t)
        beta = scheduler.beta(latent, t)
        return (alpha * x_1 - latent) / (beta**2)

    elif model.output_type == "epsilon":
        eps = model.forward(latent, t)
        beta = model.scheduler.beta(latent, t)
        return -eps / beta

    raise ValueError("Incorrectly specified base model")


class FlowExpansionTrainer:
    def __init__(
        self,
        config: DictConfig,
        env: Environment,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        if device is None:
            device = env.base_model.device

        model = copy.deepcopy(env.base_model)
        base_model = copy.deepcopy(env.base_model)
        constraint = env.reward if isinstance(env.reward, Constraint) else None

        self.gamma: float = config.get("gamma", 1.0)
        self.eta_coeff: float = config.get("eta", 1.0)
        self.beta: float = config.get("beta", 0.0)
        self.epsilon = torch.tensor(config.epsilon, dtype=torch.float32, device=device)
        self.device = device
        self.constraint = constraint
        self.traj: bool = config.traj
        self.base_base_model = copy.deepcopy(base_model).to(device)
        self.lmbda_schedule: str = config.get("lmbda", "const")

        grad_reward_fn, grad_f_k_fn = self._make_fns(base_model, self.base_base_model)

        self._am_trainer = AMTrainerFlow(
            config.adjoint_matching,
            env,
            model,
            base_model,
            grad_reward_fn,
            grad_f_k_fn if self.traj else None,
            device,
            verbose,
        )

        self._top_config = config
        self._ddpo_trainer: Optional[DDPOTrainer] = None
        self._constraint_am_trainer: Optional[AMTrainerFlow] = None
        ddpo_cfg = config.get("ddpo", None)
        if constraint is not None and self.eta_coeff > 0.0:
            if ddpo_cfg is not None:
                self._ddpo_trainer = DDPOTrainer(
                    ddpo_cfg,
                    env,
                    self._am_trainer.fine_model,
                    device=device,
                    verbose=verbose,
                    use_valids=True,
                )
            else:
                eta = self.eta_coeff

                def _constraint_grad_reward_fn(x, latent, _c=constraint, _e=eta):
                    with torch.enable_grad():
                        x_g = x.detach().requires_grad(True)
                        return x_g.gradient(_c(x_g, latent)[0].sum()) * _e

                self._constraint_am_trainer = AMTrainerFlow(
                    config.adjoint_matching,
                    env,
                    self._am_trainer.fine_model,
                    base_model,
                    _constraint_grad_reward_fn,
                    None,
                    device,
                    verbose,
                )

    @property
    def fine_model(self) -> BaseModel:
        return self._am_trainer.fine_model

    @property
    def base_model(self) -> BaseModel:
        return self._am_trainer.base_model

    def _lmbda(self, model, x, t):
        if self.lmbda_schedule == "variance":
            return model.scheduler.sigma(x, t)
        return 1.0

    def _combined_score(self, base_model, base_base_model, latent, t):
        return _score_func(base_model, latent, t) - self.beta * _score_func(base_base_model, latent, t)

    def _make_fns(self, base_model, base_base_model):
        eps = float(self.epsilon)
        gamma = self.gamma

        def grad_reward_fn(x, latent):
            t = torch.full((len(x),), 1.0 - eps, device=x.device)
            score = self._combined_score(base_model, base_base_model, latent, t)
            return -gamma * self._lmbda(base_model, latent, t) * score

        def grad_f_k_fn(latent, t: torch.Tensor):
            t_clip = t.clamp(max=1.0 - eps)
            score = self._combined_score(base_model, base_base_model, latent, t_clip)
            return -gamma * self._lmbda(base_model, latent, t_clip) * score

        return grad_reward_fn, grad_f_k_fn

    def expand(self):
        """Update AM reward functions to use the current base model."""
        grad_reward_fn, grad_f_k_fn = self._make_fns(self._am_trainer.base_model, self.base_base_model)
        self._am_trainer.grad_reward_fn = grad_reward_fn
        self._am_trainer.grad_f_k_fn = grad_f_k_fn if self.traj else None

    def generate_dataset(self):
        return self._am_trainer.generate_dataset()

    def finetune(self, dataset, steps=None, debug=False):
        return self._am_trainer.finetune(dataset, steps=steps, debug=debug)

    def update_base_model(self):
        state = self._am_trainer.fine_model.state_dict()
        self._am_trainer.base_model.load_state_dict(state)
        self._am_trainer.env.base_model.load_state_dict(state)

    def fit(self, num_iterations: int, pbar: bool = False) -> list[float]:
        """Run the full expand-project mirror-descent loop.

        Expand uses adjoint matching. The project step uses:
          - DDPO (validity reward) when config.ddpo is set and env.reward is a Constraint
          - AM with the constraint gradient when eta > 0 but config.ddpo is absent
          - nothing when eta == 0 or env.reward is not a Constraint

        Returns a flat list of per-round losses.
        """
        am_cfg = self._am_trainer.config
        am_iters = am_cfg.get("num_iterations", 1)
        finetune_steps = am_cfg.get("finetune_steps", None)
        losses = []

        it = tqdm(range(num_iterations)) if pbar else range(num_iterations)

        for _ in it:
            self.expand()
            for _ in range(am_iters):
                dataset = self._am_trainer.generate_dataset()
                losses.append(self._am_trainer.finetune(dataset, steps=finetune_steps))

            if self._ddpo_trainer is not None:
                ddpo_cfg = self._top_config.get("ddpo", {})
                ddpo_iters = ddpo_cfg.get("num_iterations", 1)
                ddpo_steps = ddpo_cfg.get("finetune_steps", None)
                for _ in range(ddpo_iters):
                    dataset = self._ddpo_trainer.generate_dataset()
                    losses.append(self._ddpo_trainer.finetune(dataset, steps=ddpo_steps))
            elif self._constraint_am_trainer is not None:
                cam_cfg = self._constraint_am_trainer.config
                cam_iters = cam_cfg.get("num_iterations", 1)
                cam_steps = cam_cfg.get("finetune_steps", None)
                for _ in range(cam_iters):
                    dataset = self._constraint_am_trainer.generate_dataset()
                    losses.append(self._constraint_am_trainer.finetune(dataset, steps=cam_steps))

            self.update_base_model()

        return losses
