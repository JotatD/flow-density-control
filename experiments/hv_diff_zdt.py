import argparse
from math import isfinite
from pathlib import Path
from omegaconf import OmegaConf
import optuna
import torch
from genexp.mo import ZDT1Torch, ZDT2Torch, ZDT3Torch, ZDT6Torch
from genexp.mo.utils import HVComputer
from genexp.trainers.hv_diff import HVDiff
from tqdm.auto import tqdm
from diffusiongym.schedulers import DiffusionScheduler
from genexp.base_models.mlp import TensorMLPModel
from diffusiongym.environments import EpsilonEnvironment

from genexp.wandb_log import WandbLogger
from utils import seed_everything
from genexp.mo.utils import plot_objective_points
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("zdt1", "zdt2", "zdt3", "zdt6"), default="zdt1")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--name", type=str, default="reward_zdt1_study", help="Name of the study for logging")
    parser.add_argument("--optuna_seed", type=int, default=42, help="Random seed for Optuna sampler")
    return parser.parse_args()


def build_reward(problem_name: str):
    reward_specs = {
        "zdt1": (ZDT1Torch, 30, "sigmoid"),
        "zdt2": (ZDT2Torch, 30, "sigmoid"),
        "zdt3": (ZDT3Torch, 30, "sigmoid"),
        "zdt6": (ZDT6Torch, 10, "sigmoid"),
    }
    reward_cls, input_dim, input_transform = reward_specs[problem_name]
    return reward_cls(input_dim=input_dim, input_transform=input_transform)


def build_diffusion_model(problem_name: str, input_dim: int, device, config):
    n_steps = int(config.get("diffusion_steps", 1000))
    beta_0 = float(config.get("beta_0", 0.1))
    beta_1 = float(config.get("beta_1", 12.0))
    betas = torch.linspace(beta_0 / n_steps, beta_1 / n_steps, n_steps, device=device)
    scheduler = DiffusionScheduler((1.0 - betas).cumprod(dim=0))
    
    model = TensorMLPModel(
        scheduler,
        output_type="epsilon",
        input_dim=input_dim,
        device=device,
    ).to(device)
    
    model_path = Path(f"assets/{problem_name}/pretrained_diffusion.pth")
    model.model.load_state_dict(torch.load(model_path, map_location=device))
    return model



def evaluate_hypervolume(trainer, num_samples: int, hv_computer) -> tuple[float, float, torch.Tensor]:
    """Evaluate trainer-aligned n-HV and full-set HV from exactly num_samples rewards."""
    if num_samples % trainer.n != 0:
        raise ValueError(f"num_samples={num_samples} must be a multiple of n={trainer.n}")

    rewards = []
    collected = 0
    old_discretization_steps = trainer.env.discretization_steps
    trainer.env.discretization_steps = 1000
    
    with torch.no_grad():
        while collected < num_samples:
            sample = trainer.sample_trajectories()
            batch_rewards = sample.rewards.to(trainer.device)
            if batch_rewards.ndim == 1:
                batch_rewards = batch_rewards.unsqueeze(1)

            remaining = num_samples - collected
            batch_rewards = batch_rewards[:remaining]
            rewards.append(batch_rewards)
            collected += batch_rewards.shape[0]

    reward_values = torch.cat(rewards, dim=0)
    n_objectives = reward_values.reshape(-1, trainer.n, trainer.num_rews)
    full_objectives = reward_values.reshape(1, num_samples, trainer.num_rews)

    n_hypervolume = hv_computer(n_objectives).mean().detach().cpu().item()
    full_hypervolume = hv_computer(full_objectives).detach().cpu().item()
    trainer.env.discretization_steps = old_discretization_steps
    
    return n_hypervolume, full_hypervolume, reward_values

def main(config: OmegaConf) -> None:
    problem_name = str(config.problem).lower()
    data_path = Path(f"assets/{problem_name}/data/obj.npy")
    ambient = torch.from_numpy(np.load(data_path)).float()
    
    seed_everything(int(config.seed))
    
    print(f"problem={problem_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward = build_reward(problem_name)
    model = build_diffusion_model(problem_name, reward.input_dim, device, config)
    env = EpsilonEnvironment(model, reward, discretization_steps=int(config.adjoint_matching.sampling.num_integration_steps))
    trainer = HVDiff(config, env, device=device)
    
    vol_samples = int(config.get("vol_samples", 256))
    
    log = WandbLogger(
        project_name=config.project_name,
        config=OmegaConf.to_container(config, resolve=True),
        use_wandb=config.wandb,
        run_name=config.run_name
    )
    
    md_step = log.set_step_metric(0, "md_step")
    global_step = log.set_step_metric(0, "global_step")
    n_hv = log.watch('n_hypervolume', 'md_step')
    full_hv = log.watch('full_hypervolume', 'md_step')
    obj_img = log.set_image('objective_points', 'md_step')
    
    hv_computer = HVComputer(ref_point=reward.ref_point.to(device), num_rew=reward.num_rew)
    
    n_hv.val, full_hv.val, reward_values = evaluate_hypervolume(trainer, num_samples=vol_samples, hv_computer=hv_computer)
    obj_img.val = plot_objective_points(ambient=ambient, special=reward_values)
    
    
    print(f"n_hypervolume={n_hv.val:.6f} full_hypervolume={full_hv.val:.6f} ", flush=True)
    loss = log.watch('loss', 'global_step')
    try:
        for _ in tqdm(range(config.num_md_iterations)):
            md_step += 1
            for am in range(config.adjoint_matching.num_iterations):
                global_step += 1
                am_dataset = trainer.generate_dataset()
                loss.val = trainer.finetune(am_dataset, steps=None)

            n_hv.val, full_hv.val, reward_values = evaluate_hypervolume(trainer, num_samples=vol_samples, hv_computer=hv_computer)
            obj_img.val = plot_objective_points(ambient=ambient, special=reward_values)
            
            loss_text = "nan" if not isfinite(loss.val) else f"{loss.val:.6f}"
            print(
                f"md={md_step} adjoint={am+1} "
                f"loss={loss_text} "
                f"n_hypervolume={n_hv.val:.6f} full_hypervolume={full_hv.val:.6f} ",
                flush=True,
            )
            if loss_text == "nan" or not isfinite(n_hv.val) or not isfinite(full_hv.val):
                raise ValueError("Encountered NaN or infinite values in loss or hypervolume metrics.")
            trainer.update_base_model()
    except Exception as e:
        print(f"Error occurred during training: {e}", flush=True)
    finally:
        log.finish()
        return full_hv.val

def optuna_entry(trial):
    args = parse_args()
    config = {
        "seed": 5,
        "n": trial.suggest_categorical("n", [4, 8, 16, 32, 64, 128]),
        "num_md_iterations": 50,
        "alpha_div": trial.suggest_float("alpha_div", 1e-4, 1e-2, log=True),
        "lmbda": trial.suggest_float("lmbda", 1e2, 1e-4, log=True),
        "temperature": 1e-5,
        "num_lambda": 4000,
        "num_p_nm1": 2048,
        "sample_p_nm1_batch_size": 64,
        "vol_samples": 256,
        "adjoint_matching": {
            "num_iterations": 30,
            "batch_size": 64,
            "clip_grad_norm": 2.0,
            "clip_loss": 1e5,
            "lr": 0.001,
            "sampling": {
                "num_samples": 64,
                "num_integration_steps": 40
            }
        },
        "problem": args.problem,
        "wandb": args.wandb,
        "project_name": trial.study.study_name,
        "run_name": f"trial_{trial.number}"
    }
    config = OmegaConf.create(config)
    result = main(config)
    return result

if __name__ == "__main__":
    args = parse_args()
    study = optuna.create_study(
        study_name=args.name,
        sampler=optuna.samplers.QMCSampler(seed=args.optuna_seed),
        direction="maximize",
        storage="sqlite:///optuna_store.db",
        load_if_exists=True
    )

    study.optimize(optuna_entry, n_trials=32)

    
