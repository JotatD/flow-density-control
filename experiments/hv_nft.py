import argparse
from argparse import Namespace
from math import ceil
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
from genexp.resume import (
    load_latest_training_checkpoint,
    mark_run_complete,
    resolve_run,
    restore_rng_state,
    save_training_checkpoint,
)
from genexp.trainers.hv_rl import HVRL
from genexp.wandb_log import WandbLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--force_new_start", action="store_true")
    parser.add_argument("--name", type=str, default="hv_dxtb_test2")
    parser.add_argument("--project_name", type=str, default="whos_back")

    parser.add_argument("--run_name", type=str, default="hv_nft")

    parser.add_argument("--seed", type=int, default=5)

    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--exploration_decay_type", type=int, choices=(0, 1, 2), default=1)

    parser.add_argument("--num_p_nm1", type=int, default=85)

    parser.add_argument("--update_pretrained_every_n_steps", type=int, default=20)
    parser.add_argument("--resample_every_n_steps", type=int, default=1)
    parser.add_argument("--sample_nm1_every_n_steps", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_inner_epochs", type=int, default=1) # currently unused
    parser.add_argument("--only_valids", action="store_true", default=False)

    parser.add_argument("--timestep_fraction", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_samples", type=int, default=320)
    parser.add_argument("--num_integration_steps", type=int, default=40)
    
    parser.add_argument("--vol_samples", type=int, default=128)
    parser.add_argument("--num-time-groups", type=int, default=5)
    
    parser.add_argument("--scalarization", type=str, choices=["sum", "improvement"], default="sum")
    return parser.parse_args()


def evaluate(
    trainer: HVRL, num_samples: int, hv_computer, reward, discretization_steps: int = 128
) -> tuple[float, float, torch.Tensor, float]:
    if num_samples % trainer.n != 0:
        raise ValueError(f"num_samples={num_samples} must be a multiple of n={trainer.n}")

    rewards = []
    left = num_samples

    original_policy = trainer.env._policy
    trainer.env.policy = trainer.fine_model
    valids = 0
    try:
        with torch.no_grad():
            while left > 0:
                batch = min(left, trainer.config.batch_size)
                sample = trainer.env.sample(batch, discretization_steps=discretization_steps, pbar=False)
                rew, info = reward(sample.sample, sample.latent)
                valids += info["valids"].sum().item()
                rewards.extend([r for i, r in enumerate(rew) if info["valids"][i]])
                left -= batch
    finally:
        trainer.env._policy = original_policy
        
    if rewards:
        reward_values = torch.stack(rewards, dim=0)
        full_objectives = reward_values.reshape(1, -1, trainer.num_rews)
        full_hypervolume = hv_computer(full_objectives).detach().cpu().item()
    else:
        reward_values = torch.tensor([[0, 0]], device=trainer.device)
        full_hypervolume = 0.0
        
    as_many = reward_values.shape[0] - (reward_values.shape[0] % trainer.n)
    if as_many > 0:
        n_objectives = reward_values[:as_many].reshape(-1, trainer.n, trainer.num_rews)
        n_hypervolume = hv_computer(n_objectives).mean().detach().cpu().item()
    else: 
        n_hypervolume = 0.0

    return n_hypervolume, full_hypervolume, reward_values, valids / num_samples


def main(config: Namespace) -> None:
    assert config.sample_nm1_every_n_steps % config.resample_every_n_steps == 0
    assert config.update_pretrained_every_n_steps % config.resample_every_n_steps == 0
    assert config.update_pretrained_every_n_steps % config.sample_nm1_every_n_steps == 0

    results_root = Path("output") / config.project_name
    run_resolution = resolve_run(config, results_root, config.run_name)
    if run_resolution.completed:
        print(f"Matching run is already complete: {run_resolution.run_dir}")
        return

    print(f"run_dir={run_resolution.run_dir}")
    log = WandbLogger(
        project_name=config.project_name,
        config=vars(config),
        use_wandb=config.wandb,
        run_name=run_resolution.run_dir.name,
        id=run_resolution.wandb_run_id,
        resume="allow",
        dir=str(run_resolution.run_dir),
    )

    seed_everything(int(config.seed))

    print("problem=dxtb_10A")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward = MOReward(XTBTask(), num_rew=2, ref_point=torch.tensor([-1.0, -1.0], device=device))
    model = GEOMBaseModel(device=device)
    env = EndpointEnvironment(model, DummyReward(), discretization_steps=int(config.num_integration_steps))
    unconstrained_sample = env.sample
    env.sample = lambda *args, **kwargs: unconstrained_sample(*args, n_atoms=10, **kwargs)  # ty: ignore[invalid-assignment]

    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)
    trainer = HVRL(config, env, reward, hv_computer=hv_computer, device=device)

    resume_checkpoint = load_latest_training_checkpoint(run_resolution.run_dir, map_location=device)
    start_epoch = 0
    samples = None
    advantages = None
    if resume_checkpoint is not None:
        trainer.load_training_state_dict(resume_checkpoint["trainer_state"])
        start_epoch = resume_checkpoint["next_epoch"]
        loop_state = resume_checkpoint.get("loop_state", {})
        samples = loop_state.get("samples")
        advantages = loop_state.get("advantages")
        restore_rng_state(resume_checkpoint)

    vol_samples = config.vol_samples

    epoch = log.set_step_metric(start_epoch, "epoch")

    n_hv = log.watch("n_hypervolume", "epoch")
    full_hv = log.watch("full_hypervolume", "epoch")
    energy = log.watch("energy", "epoch")
    dipole = log.watch("dipole", "epoch")
    valid_frac = log.watch("valid_fraction", "epoch")

    group_length = ceil((config.num_integration_steps-1) / config.num_time_groups)
    for _ in tqdm(range(start_epoch, config.epochs)):
        if epoch.val % config.update_pretrained_every_n_steps == 0 and config.scalarization == "improvement":
            with trainer.timer.section("update_base_model"):
                trainer.update_base_model()

        if epoch.val % config.sample_nm1_every_n_steps == 0:
            with trainer.timer.section("sample_bg"):
                trainer.fix_optimization_problem()

        if epoch.val % config.resample_every_n_steps == 0:
            with trainer.timer.section("generate_dataset"):
                samples, advantages = trainer.generate_dataset_fv()
                
        if not samples or not advantages:
            print("No valid samples or advantages, skipping epoch.")
            epoch += 1
            continue
        
        with trainer.timer.section("finetune"):
            timed_stats = trainer.finetune(samples, advantages)
            
        group_stats = {}
        fulltime_stats = {f"full/{k}": np.mean(v) for k, v in timed_stats.items()}
        for i in range(config.num_time_groups):
            start = i * group_length
            end = min((i + 1) * group_length, config.num_integration_steps-1) 
            print(start, end)   
            for k in timed_stats:
                group_stats[f"{i}_group/{k}"] = np.mean(timed_stats[k][start:end])
            
        log.log_dict(group_stats, 'epoch')
        log.log_dict(fulltime_stats, 'epoch')
        
        trainer.update_exploration_model()
        
        rows = trainer.timer.summary()
        
        with trainer.timer.section("evaluate_hypervolume"):            
            n_hv.val, full_hv.val, rewards, valid_frac.val = evaluate(trainer, num_samples=vol_samples, hv_computer=hv_computer, reward=reward)

        energy.val, dipole.val = rewards.mean(dim=0).detach().cpu().numpy().tolist()

        print("\n=== Timing summary (by total time) ===")
        for name, cnt, total, mean, p50, p95 in rows:
            print(f"{name:30s}  n={cnt:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  "
                f"p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")

        epoch += 1
        save_training_checkpoint(
            run_resolution.run_dir,
            next_epoch=epoch.val,
            trainer_state=trainer.training_state_dict(),
            loop_state={
                "samples": samples,
                "advantages": advantages,
            },
        )

    log.finish()
    mark_run_complete(run_resolution.run_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
