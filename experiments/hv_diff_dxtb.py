import argparse
from math import isfinite
from pathlib import Path
from omegaconf import OmegaConf
import optuna
import torch
from genexp.mo.mo_dxtb import DXTBTask
from diffusiongym.molecules.flowmol import GEOMBaseModel
from genexp.mo.utils import HVComputer
from genexp.trainers.hv_diff import HVDiff
from tqdm.auto import tqdm
from diffusiongym.environments import EndpointEnvironment, EpsilonEnvironment

from genexp.wandb_log import WandbLogger
from utils import seed_everything
from genexp.mo.utils import plot_objective_points
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--name", type=str, default="hv_dxtb_test2", help="Name of the study for logging")
    parser.add_argument("--optuna_seed", type=int, default=42, help="Random seed for Optuna sampler")
    return parser.parse_args()



def evaluate_hypervolume(trainer: HVDiff, num_samples: int, hv_computer, discretization_steps: int = 250) -> tuple[float, float, torch.Tensor]:
    """Evaluate trainer-aligned n-HV and full-set HV from exactly num_samples rewards."""
    if num_samples % trainer.n != 0:
        raise ValueError(f"num_samples={num_samples} must be a multiple of n={trainer.n}")

    rewards = []
    left = num_samples
    
    original_policy = trainer.env._policy
    trainer.env.policy = trainer.fine_model
    with torch.no_grad():
        while left > 0:
            batch = min(left, trainer.config.batch_size)
            sample = trainer.env.sample(batch, discretization_steps=discretization_steps, pbar=False)
            rewards.append(sample.rewards)
            left -= batch
    trainer.env.policy = original_policy

    reward_values = torch.cat(rewards, dim=0)
    n_objectives = reward_values.reshape(-1, trainer.n, trainer.num_rews)
    full_objectives = reward_values.reshape(1, num_samples, trainer.num_rews)

    n_hypervolume = hv_computer(n_objectives).mean().detach().cpu().item()
    full_hypervolume = hv_computer(full_objectives).detach().cpu().item()
    
    return n_hypervolume, full_hypervolume, reward_values

def main(config: OmegaConf) -> None:
    problem_name = "dxtb_10A"
    data_path = Path(f"assets/{problem_name}/data/obj.npy")
    ambient = torch.from_numpy(np.load(data_path)).float()
    
    seed_everything(int(config.seed))
    
    print(f"problem={problem_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward = DXTBTask(fixed_num_atoms=10)
    model =  GEOMBaseModel(device=device)
    env = EndpointEnvironment(model, reward, discretization_steps=int(config.adjoint_matching.sampling.num_integration_steps))
    unconstrained_sample = env.sample
    env.sample = lambda *args, **kwargs: unconstrained_sample(*args, n_atoms=10, **kwargs)
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
    most_inner_step = log.set_step_metric(0, "most_inner_step")
    
    n_hv = log.watch('n_hypervolume', 'md_step')
    full_hv = log.watch('full_hypervolume', 'md_step')
    obj_img = log.set_image('objective_points', 'md_step')
    inner_hv = log.watch('inner_hypervolume', 'global_step')
    inner_loss = log.watch('inner_loss', 'most_inner_step')
    inner_img = log.set_image('inner_objective_points', 'most_inner_step')
    
    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)
    
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
                losses = trainer.finetune(am_dataset, steps=None, debug=True)
                
                loss.val = np.array(losses).mean().item()
                dataset_rewards = torch.cat([d.rews for d in am_dataset], dim=0)
                inner_hv.val = hv_computer(dataset_rewards.unsqueeze(0)).item()
                inner_img.val = plot_objective_points(ambient=ambient, special=dataset_rewards)
                for l in losses: 
                    most_inner_step += 1
                    inner_loss.val = l
            n_hv.val, full_hv.val, reward_values = evaluate_hypervolume(trainer, num_samples=vol_samples, hv_computer=hv_computer)
            obj_img.val = plot_objective_points(ambient=ambient, special=reward_values)
            
            loss_text = "nan" if not isfinite(loss.val) else f"{loss.val:.6f}"
            print(
                f"md={md_step} adjoint={am+1} "
                f"loss={loss_text} "
                f"n_hypervolume={n_hv.val:.6f} full_hypervolume={full_hv.val:.6f} ",
                flush=True,
            )
            if not isfinite(n_hv.val) or not isfinite(full_hv.val):
                raise ValueError("Encountered NaN or infinite values in loss or hypervolume metrics.")
            trainer.update_base_model()
    except Exception as e:
        print(f"Error occurred during training: {e}", flush=True)
    finally:
        log.finish()
        return full_hv.val

def optuna_entry(trial: optuna.Trial) -> float:
    x = 16
    args = parse_args()
    config = {
        "seed": 5,
        "n": 4,
        "num_md_iterations": 15,
        "alpha_div": trial.suggest_categorical(name="alpha_div", choices=[10**x for x in range(-4, 4)]),
        "lmbda": trial.suggest_categorical(name="lmbda", choices=[10**x for x in range(-4, 4)]),
        "temperature": 1e-5,
        "num_lambda": 400,
        "num_p_nm1": 128 // x,
        "sample_p_nm1_batch_size": 64 // x,
        "vol_samples": 64 // x,
        "adjoint_matching": {
            "num_iterations": 10,
            "batch_size": 64 // x,
            "clip_grad_norm": 2.0,
            "clip_loss": 1e5,
            "lr": trial.suggest_categorical(name="lr", choices=[5e-5, 1e-4, 1e-3]),
            "sampling": {
                "num_samples": 64 // x,
                "num_integration_steps": 40
            }
        },
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
        sampler=optuna.samplers.BruteForceSampler(seed=args.optuna_seed),
        direction="maximize",
        storage="sqlite:///optuna_store.db",
        load_if_exists=True
    )

    study.optimize(optuna_entry, n_trials=147)

    
