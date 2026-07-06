import argparse
from math import isfinite
from pathlib import Path
import torch
from genexp.mo import ZDT1Torch, ZDT2Torch, ZDT3Torch, ZDT6Torch
from genexp.mo.utils import HVComputer
from genexp.trainers.hv_diff import HVDiff
from tqdm.auto import tqdm
from diffusiongym.schedulers import DiffusionScheduler
from genexp.base_models.mlp import TensorMLPModel
from diffusiongym.environments import EpsilonEnvironment

from utils import resolve_config, seed_everything
from genexp.mo.utils import plot_objective_points
import numpy as np
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hv_diff_zdt.yaml")
    parser.add_argument("--config_idx", type=int, default=None)
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
    
    model_path = Path(config.model_dir) / problem_name / "pretrained_diffusion.pth"
    model.model.load_state_dict(torch.load(model_path, map_location=device))
    return model



def evaluate_hypervolume(trainer, num_samples: int, hv_computer) -> tuple[float, float, torch.Tensor]:
    """Evaluate trainer-aligned n-HV and full-set HV from exactly num_samples rewards."""
    if num_samples % trainer.n != 0:
        raise ValueError(f"num_samples={num_samples} must be a multiple of n={trainer.n}")

    rewards = []
    collected = 0
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

    return n_hypervolume, full_hypervolume, reward_values

def main():
    args = parse_args()

    config, _ = resolve_config(args)

    problem_name = str(config.problem).lower()
    
    data_path = Path(f"assets/{problem_name}/data/obj.npy")
    ambient = torch.from_numpy(np.load(data_path)).float()
    
    figs_output_dir = Path(f"output/{problem_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
    os.makedirs(figs_output_dir, exist_ok=True)
    
    
    seed_everything(int(config.seed))
    
    print(f"problem={problem_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward = build_reward(problem_name)
    model = build_diffusion_model(problem_name, reward.input_dim, device, config)
    env = EpsilonEnvironment(model, reward, device=device)
    trainer = HVDiff(config, env, int(config.adjoint_matching.sampling.num_integration_steps), device=device)
    
    vol_samples = int(config.get("vol_samples", 256))
    hv_computer = HVComputer(ref_point=reward.ref_point.to(device), num_rew=reward.num_rew)
    
    loaded_n_hv, loaded_full_hv, reward_values = evaluate_hypervolume(trainer, num_samples=vol_samples, hv_computer=hv_computer)
    plot_objective_points(ambient=ambient, special=reward_values, save_path=figs_output_dir / "pretrained.png")
    
    print(
        f"pretrained path {Path(config.model_dir) / problem_name / 'pretrained_diffusion.pth'} "
        f"n_hypervolume={loaded_n_hv:.6f} full_hypervolume={loaded_full_hv:.6f} ",
        flush=True,
    )

    global_step = 0
    for md_iteration in tqdm(range(config.num_md_iterations)):
        for adjoint_iteration in range(config.adjoint_matching.num_iterations):
            global_step += 1
            am_dataset = trainer.generate_dataset()
            loss = trainer.finetune(am_dataset, steps=None)

            n_hv, full_hv, reward_values = evaluate_hypervolume(trainer, num_samples=vol_samples, hv_computer=hv_computer)
            plot_objective_points(ambient=ambient, special=reward_values, save_path=figs_output_dir / f"md{md_iteration + 1}_adjoint{adjoint_iteration + 1}.png")
            
            loss_text = "nan" if not isfinite(loss) else f"{loss:.6f}"
            print(
                f"md={md_iteration + 1} adjoint={adjoint_iteration + 1} "
                f"loss={loss_text} "
                f"n_hypervolume={n_hv:.6f} full_hypervolume={full_hv:.6f} ",
                flush=True,
            )
            if loss_text == "nan" or not isfinite(n_hv) or not isfinite(full_hv):
                print("NaN detected in loss or hypervolume, stopping training.", flush=True)
                return
            
        trainer.update_base_model()


if __name__ == "__main__":
    main()
