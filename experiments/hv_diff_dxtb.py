import argparse
import pickle as pkl
import traceback
from math import isfinite
from pathlib import Path

import numpy as np
import optuna
import torch
from diffusiongym.environments import EndpointEnvironment, Sample
from diffusiongym.molecules.flowmol import GEOMBaseModel
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from utils import seed_everything

from genexp.mo.mo_dxtb import DXTBTask
from genexp.mo.utils import HVComputer, plot_objective_points
from genexp.trainers.hv_diff import HVDiff
from genexp.wandb_log import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--name", type=str, default="hv_dxtb_test2", help="Name of the study for logging")
    parser.add_argument("--optuna_seed", type=int, default=42, help="Random seed for Optuna sampler")
    return parser.parse_args()

def dump_samples(samples: list[Sample], filename: str) -> None:
    """Dump a list of Sample objects to a file using pickle."""
    with open(filename, "wb") as f:
        pkl.dump(samples, f)

def evaluate_hypervolume(trainer: HVDiff, num_samples: int, hv_computer, discretization_steps: int = 250) -> tuple[float, float, torch.Tensor, list[Sample], float]:
    """Evaluate trainer-aligned n-HV and full-set HV from exactly num_samples rewards."""
    if num_samples % trainer.n != 0:
        raise ValueError(f"num_samples={num_samples} must be a multiple of n={trainer.n}")

    rewards = []
    all_samples = []
    left = num_samples
    valids = 0
    
    original_policy = trainer.env._policy
    trainer.env.policy = trainer.fine_model
    with torch.no_grad():
        while left > 0:
            batch = min(left, trainer.config.batch_size)
            sample = trainer.env.sample(batch, discretization_steps=discretization_steps, pbar=False)
            rewards.append(sample.rewards)
            all_samples.append(sample)
            valids += sample.info["valids"].sum().item()
            left -= batch
    trainer.env.policy = original_policy

    reward_values = torch.cat(rewards, dim=0)
    n_objectives = reward_values.reshape(-1, trainer.n, trainer.num_rews)
    full_objectives = reward_values.reshape(1, num_samples, trainer.num_rews)

    n_hypervolume = hv_computer(n_objectives).mean().detach().cpu().item()
    full_hypervolume = hv_computer(full_objectives).detach().cpu().item()
    
    return n_hypervolume, full_hypervolume, reward_values, all_samples, valids / num_samples

def main(config: OmegaConf) -> None:
    problem_name = "dxtb_10A"
    data_path = Path(f"assets/{problem_name}/data/obj_lf2.npy")
    ambient = torch.from_numpy(np.load(data_path)).float()
    scaling_factor = torch.ones((2,))
    if config.scale:
        scaling_factor = ambient.mean(axis=0)
        print(scaling_factor)
        ambient = ambient / scaling_factor
    
    seed_everything(int(config.seed))
    
    print(f"problem={problem_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward = DXTBTask(fixed_num_atoms=10, scaling_factor=scaling_factor.to(device))
    model =  GEOMBaseModel(device=device)
    env = EndpointEnvironment(model, reward, discretization_steps=int(config.adjoint_matching.sampling.num_integration_steps))
    unconstrained_sample = env.sample
    env.sample = lambda *args, **kwargs: unconstrained_sample(*args, n_atoms=10, **kwargs)
    trainer = HVDiff(config, env, device=device)
    folder = Path(f"output/{config.project_name}/{config.run_name}")
    folder.mkdir(parents=True, exist_ok=True)

    
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
    dtst_hv = log.watch('dataset_hypervolume', 'global_step')
    inner_loss = log.watch('inner_loss', 'most_inner_step')
    dtst_img = log.set_image('dataset_objective_points', 'global_step')
    valid_frac = log.watch('valid_fraction', 'md_step')
    dtst_valid_frac = log.watch('dataset_valid_fraction', 'global_step')
    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)
    nm1_hv = log.watch('nm1_hypervolume', 'md_step')
    nm1_img = log.set_image('nm1_objective_points', 'md_step')
    nm1_valid_frac = log.watch('nm1_valid_fraction', 'md_step')
    
    n_hv.val, full_hv.val, reward_values, pretrained_samples, valid_frac.val = evaluate_hypervolume(trainer, num_samples=vol_samples, hv_computer=hv_computer)
    obj_img.val = plot_objective_points(ambient=ambient, special=reward_values)
    dump_samples(pretrained_samples, folder / "0.pkl")
    
    print(f"n_hypervolume={n_hv.val:.6f} full_hypervolume={full_hv.val:.6f} ", flush=True)
    loss = log.watch('loss', 'global_step')
    try:
        for _ in tqdm(range(config.num_md_iterations)):
            nm1_hv.val = hv_computer(trainer.rewards.unsqueeze(0)).item()
            nm1_img.val = plot_objective_points(ambient=ambient, special=trainer.rewards)
            nm1_valid_frac.val = trainer.valid_frac
            md_step += 1
            torch.save(trainer.base_model.state_dict(), folder / f"model_gb_{global_step}_base.pth")
            for am in range(config.adjoint_matching.num_iterations):
                global_step += 1
                
                am_dataset = trainer.generate_dataset()
                with open(folder / f"dataset_{global_step}.pkl", "wb") as f:
                    pkl.dump(am_dataset, f)
                dtst_rewards = torch.cat([d.full_env_sample.rewards for d in am_dataset], dim=0)
                torch.save(dtst_rewards, folder / f"dataset_rewards_{global_step}.pt")
                dtst_hv.val = hv_computer(dtst_rewards.unsqueeze(0)).item()
                dtst_img.val = plot_objective_points(ambient=ambient, special=dtst_rewards)
                dtst_valid_frac.val = sum([d.full_env_sample.info["valids"].sum().item() for d in am_dataset]) / sum([len(d.full_env_sample.sample) for d in am_dataset])
                
                losses = trainer.finetune(am_dataset, steps=None, debug=True)
                loss.val = np.array(losses).mean().item()
                for l in losses: 
                    most_inner_step += 1
                    inner_loss.val = l
                torch.save(trainer.fine_model.state_dict(), folder / f"model_gb_{global_step}_fine.pth")

            n_hv.val, full_hv.val, reward_values, new_samples, valid_frac.val = evaluate_hypervolume(trainer, num_samples=vol_samples, hv_computer=hv_computer)
            obj_img.val = plot_objective_points(ambient=ambient, special=reward_values)
            dump_samples(new_samples, folder / f"{md_step}.pkl")
            loss_text = "nan" if not isfinite(loss.val) else f"{loss.val:.6f}"
            print(f"md={md_step} adjoint={am+1} loss={loss_text} n_hypervolume={n_hv.val:.6f} full_hypervolume={full_hv.val:.6f} ", flush=True)
            if not isfinite(n_hv.val) or not isfinite(full_hv.val):
                raise ValueError("Encountered NaN or infinite values in loss or hypervolume metrics.")
            trainer.update_base_model()
            
            # torch.save(trainer.fine_model.state_dict(), folder / "model_last.pth")
            if full_hv.is_curr_max():
                print(f"New best hypervolume: {full_hv.val:.6f} at step {md_step}", flush=True)
                torch.save(trainer.fine_model.state_dict(), folder / "model_best.pth")                
    except Exception as e:
        traceback.print_exc()
        print(f"Error occurred during training: {e}", flush=True)
        torch.save(trainer.fine_model.state_dict(), folder / "model_last_fine.pth")
        torch.save(trainer.base_model.state_dict(), folder / "model_last_base.pth")
        with open(folder / "dataset.pkl", "wb") as f:
            pkl.dump(am_dataset, f)
    finally:
        torch.save(trainer.fine_model.state_dict(), folder / "model_last.pth")
        log.finish()
        return full_hv.val

def optuna_entry(trial: optuna.Trial) -> float:
    x = 1
    args = parse_args()
    config = {
        "seed": 5,
        "n": 4,
        "num_md_iterations": 15,
        "alpha_div": trial.suggest_categorical(name="alpha_div", choices=[10**x for x in range(-3, -1)]),
        "lmbda": trial.suggest_categorical(name="lmbda", choices=[10**x for x in range(1, 3)]),
        "temperature": 1e-5,
        "num_lambda": 400,
        "num_p_nm1": 256 // x,
        "sample_p_nm1_batch_size": 64 // x,
        "vol_samples": 256 // x,
        "adjoint_matching": {
            "num_iterations": 10,
            "batch_size": 64 // x,
            "clip_grad_norm": 2.0, # dangerous parameter?
            "clip_loss": 1e10,
            "lr": trial.suggest_categorical(name="lr", choices=[5e-5, 1e-4]),
            "sampling": {
                "num_samples": 64 // x,
                "num_integration_steps": 100
            }
        },
        "wandb": args.wandb,
        "project_name": trial.study.study_name,
        "run_name": f"{trial.number}",
        "scale": trial.suggest_categorical(name="scale", choices=[True, False]),
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

    
