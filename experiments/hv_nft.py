import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from diffusiongym.environments import EndpointEnvironment
from diffusiongym.molecules import XTBTask
from diffusiongym.molecules.flowmol import GEOMBaseModel
from diffusiongym.rewards import DummyReward
from tqdm.auto import tqdm
from utils import seed_everything

from genexp.mo.base import MOReward
from genexp.mo.utils import HVComputer
from genexp.trainers.hv_rl import HVRL
from genexp.wandb_log import WandbLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--name", type=str, default="hv_dxtb_test2", help="W&B project name")
    parser.add_argument("--project_name", type=str, default="whos_back", help="W&B project name")
    
    parser.add_argument("--run_name", type=str, default="hv_nft", help="Name of the run")
    
    parser.add_argument("--seed", type=int, default=5)
    
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=1)
    
    parser.add_argument("--num_p_nm1", type=int, default=85)

    parser.add_argument("--update_pretrained_every_n_steps", type=int, default=20)
    parser.add_argument("--resample_every_n_steps", type=int, default=20)
    parser.add_argument("--sample_nm1_every_n_steps", type=int, default=20)
    
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--adv_clip_max", type=float, default=10.0)
    parser.add_argument("--clip_grad_norm", type=float, default=2.0)
    parser.add_argument("--num_inner_epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.5)

    parser.add_argument("--timestep_fraction", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_integration_steps", type=int, default=40)
    return parser.parse_args()

def evaluate_hypervolume(
    trainer: HVRL, num_samples: int, hv_computer, reward, discretization_steps: int = 250
) -> tuple[float, float, torch.Tensor]:
    """Evaluate trainer-aligned n-HV and full-set HV from exactly num_samples rewards."""
    if num_samples % trainer.n != 0:
        raise ValueError(f"num_samples={num_samples} must be a multiple of n={trainer.n}")

    rewards = []
    left = num_samples

    trainer.env.policy = trainer.fine_model
    with torch.no_grad():
        while left > 0:
            batch = min(left, trainer.config.batch_size)
            sample = trainer.env.sample(batch, discretization_steps=discretization_steps, pbar=False)
            rew, _ = reward(sample.sample, sample.latent)
            rewards.append(rew)
            left -= batch

    reward_values = torch.cat(rewards, dim=0)
    n_objectives = reward_values.reshape(-1, trainer.n, trainer.num_rews)
    full_objectives = reward_values.reshape(1, num_samples, trainer.num_rews)

    n_hypervolume = hv_computer(n_objectives).mean().detach().cpu().item()
    full_hypervolume = hv_computer(full_objectives).detach().cpu().item()

    return n_hypervolume, full_hypervolume, reward_values


def main(config: Namespace) -> None:
    assert config.sample_nm1_every_n_steps % config.resample_every_n_steps == 0
    assert config.update_pretrained_every_n_steps % config.resample_every_n_steps == 0
    assert config.update_pretrained_every_n_steps % config.sample_nm1_every_n_steps == 0
    
    problem_name = "dxtb_10A"
    data_path = Path(f"assets/{problem_name}/data/obj.npy")
    ambient = torch.from_numpy(np.load(data_path)).float()

    seed_everything(int(config.seed))

    print(f"problem={problem_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward = MOReward(XTBTask(), num_rew=2, ref_point=torch.tensor([-1.0, -1.0], device=device))
    model = GEOMBaseModel(device=device)
    env = EndpointEnvironment(model, DummyReward(), discretization_steps=int(config.num_integration_steps))
    unconstrained_sample = env.sample    
    env.sample = lambda *args, **kwargs: unconstrained_sample(*args, n_atoms=10, **kwargs)  # ty: ignore[invalid-assignment]
    
    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)
    trainer = HVRL(config, env, reward, hv_computer=hv_computer, device=device)

    vol_samples = 8

    log = WandbLogger(
        project_name=config.project_name,
        config=vars(config),
        use_wandb=config.wandb,
        run_name=config.run_name,
    )
    
    epoch = log.set_step_metric(0, "epoch")
    
    n_hv = log.watch("n_hypervolume", "md_step")
    full_hv = log.watch("full_hypervolume", "md_step")
    obj_img = log.set_image("objective_points", "md_step")

    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)

    # n_hv.val, full_hv.val, reward_values = evaluate_hypervolume(
    #     trainer, num_samples=vol_samples, hv_computer=hv_computer, reward=reward
    # )
    # obj_img.val = plot_objective_points(ambient=ambient, special=reward_values)

    # print(f"n_hypervolume={n_hv.val:.6f} full_hypervolume={full_hv.val:.6f} ", flush=True)
    loss = log.watch("loss", "global_step")
    
    for _ in tqdm(range(config.epochs)):
        if epoch.val % config.update_pretrained_every_n_steps == 0:
            trainer.update_base_model()
            
        if epoch.val % config.sample_nm1_every_n_steps == 0:
            trainer.fix_optimization_problem()
        
        if epoch.val % config.resample_every_n_steps == 0:
            samples, advantages = trainer.generate_dataset_fv()
            
        
        loss.val = trainer.finetune(samples, advantages)            
            
        n_hv.val, full_hv.val, _ = evaluate_hypervolume(
            trainer, num_samples=vol_samples, hv_computer=hv_computer, reward=reward
        )
        epoch += 1
        
    log.finish()
if __name__ == "__main__":
    args = parse_args()
    main(args)
