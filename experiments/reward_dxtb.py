from math import isfinite

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from diffusiongym.environments import EndpointEnvironment
from diffusiongym.molecules.flowmol import GEOMBaseModel

from genexp.mo import DXTBDipoleL2, DXTBEnergy
from genexp.trainers.rew_diff import RewDiff
from genexp.mo.utils import plot_clipped_values
from utils import seed_everything
import argparse
from genexp.wandb_log import WandbLogger
import numpy as np
import optuna
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("energy", "dipole_l2"), default="energy")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--name", type=str, default="reward_dxtb_study", help="Name of the study for logging")
    parser.add_argument("--optuna_seed", type=int, default=42, help="Random seed for Optuna sampler")
    return parser.parse_args()


def build_environment(config, reward, device):
    base_model = GEOMBaseModel(device=device)
    env = EndpointEnvironment(
        base_model,
        reward,
        discretization_steps=config.adjoint_matching.sampling.num_integration_steps,
    )
    sample = env.sample

    def sample_fixed_num_atoms(*args, **kwargs):
        kwargs.setdefault("n_atoms", int(config.fixed_num_atoms))
        return sample(*args, **kwargs)

    env.sample = sample_fixed_num_atoms
    return env

def evaluate_median(trainer, num_samples: int):
    """Evaluate the median reward of the fine model."""
    left = num_samples
    batch_size = trainer.config.batch_size
    rewards = []
    with torch.no_grad():
        while left > 0:
            samples = trainer.sample_trajectories().rewards
            rewards.append(samples)
            left -= batch_size 
    rewards = torch.stack(rewards).reshape(-1)
    rewards, _ = torch.sort(rewards)
    return rewards.median().item(), rewards

def main(config: OmegaConf) -> None:
    seed_everything(int(config.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_cls = {"energy": DXTBEnergy, "dipole_l2": DXTBDipoleL2}[config.problem]
    reward = reward_cls(fixed_num_atoms=int(config.fixed_num_atoms))
    env = build_environment(config, reward, device)
    trainer = RewDiff(config, env, device=device)
    num_eval_samples = int(config.get("num_eval_samples", 16))
    log = WandbLogger(
        project_name=config.project_name,
        config=OmegaConf.to_container(config, resolve=True),
        use_wandb=config.wandb,
        run_name=config.run_name
    )
    
    global_step = log.set_step_metric(0, "global_step")
    problem_median = log.watch('problem_median', 'global_step')
    clip_vals_img = log.set_image('clip_vals_img', 'global_step')
    data = []
    problem_median.val, rew = evaluate_median(trainer, num_samples=num_eval_samples)
    clip_vals_img.val = plot_clipped_values(high=200, low=-30, values=rew.numpy())
    data.append(rew)
    print(
        f"problem={config.problem} problem_eval=loaded num_samples={num_eval_samples} "
        f"problem_median={problem_median.val:.6f}",
        flush=True,
    )

    loss = log.watch('loss', 'global_step')
    try:
        for md_iteration in tqdm(range(config.num_md_iterations)):
            for adjoint_iteration in range(config.adjoint_matching.num_iterations):
                global_step += 1
                am_dataset = trainer.generate_dataset()
                loss.val = trainer.finetune(am_dataset, steps=None)
                
                problem_median.val, rew = evaluate_median(trainer, num_samples=num_eval_samples)
                clip_vals_img.val = plot_clipped_values(high=200, low=-30, values=rew.numpy())
                data.append(rew)
                loss_text = "nan" if not isfinite(loss.val) else f"{loss.val:.6f}"
                print(
                    f"md={md_iteration + 1} adjoint={adjoint_iteration + 1} "
                    f"loss={loss_text} problem_median={problem_median.val:.6f}",
                    flush=True,
                )
                if loss_text == "nan":
                    raise ValueError("Loss is NaN, stopping training.")
            trainer.update_base_model()
        return problem_median.val
    except Exception as e:
        print(f"Error occurred during training: {e}", flush=True)
        return problem_median.val
    finally:
        log.finish()
        data = torch.stack(data, dim=0).numpy()
        save_path = f"output/{log.project_name}/{log.run_name}/reward.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path, data, delimiter=',')

def optuna_entry(trial: optuna.Trial):
    args = parse_args()
    config = {
        "seed": 2,
        "model_name": 'geom_gaussian',
        "fixed_num_atoms": 10,
        "num_md_iterations": 1,
        "num_eval_samples": 32,
        "alpha_div": trial.suggest_float("alpha_div", 0, 10),
        "lmbda": trial.suggest_float("lmbda", 50, 650, log=True),
        "adjoint_matching": {
            "num_iterations": 50,
            "batch_size": 32,
            "clip_grad_norm": 2.0,
            "clip_loss": 1e5,
            "lr": trial.suggest_float("lr", 8.0e-5, 2.0e-4), 
            "sampling": {
                "num_samples": 32,
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

    study.optimize(optuna_entry, n_trials=128)

