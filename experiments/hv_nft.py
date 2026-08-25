import argparse
from argparse import Namespace
from math import ceil
from pathlib import Path

import numpy as np
import torch
from diffusiongym import Sample
from diffusiongym.environments import EndpointEnvironment
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.flowmol import GEOMBaseModel
from diffusiongym.rewards import DummyReward
from tqdm.auto import tqdm
from utils import seed_everything

from genexp.mo.base import MOReward
from genexp.mo.mo_mol import MolecularMetrics, TopologyMetrics
from genexp.mo.moses import diversity_metrics
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
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--exploration_decay_type", type=int, choices=(0, 1, 2), default=1)

    parser.add_argument("--num_p_nm1", type=int, default=85)

    parser.add_argument("--update_pretrained_every_n_steps", type=int, default=20)
    parser.add_argument("--resample_every_n_steps", type=int, default=20)
    parser.add_argument("--sample_nm1_every_n_steps", type=int, default=20)
    parser.add_argument("--evaluate_diversity_every_n_steps", type=int, default=100)
    parser.add_argument("--evaluate_every_n_steps", type=int, default=5)
    
    parser.add_argument("--num_diversity_samples", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_inner_epochs", type=int, default=1) # currently unused
    parser.add_argument("--only_valids", action="store_true", default=False)
    parser.add_argument("--full_bust", action="store_true")

    parser.add_argument("--timestep_fraction", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--advantage_group_size", type=int, default=16)
    parser.add_argument("--fulfill_num_samples", action="store_true")
    parser.add_argument("--fulfill_max_attempts", type=int, default=10_000)
    parser.add_argument("--num_integration_steps", type=int, default=40)
    
    parser.add_argument("--vol_samples", type=int, default=128)
    parser.add_argument("--num-time-groups", type=int, default=5)
    
    parser.add_argument("--scalarization", type=str, choices=["sum", "improvement"], default="improvement")
    parser.add_argument("--reward", type=str, choices=["molecular", "topology"], default="molecular")
    return parser.parse_args()

def sample_x(num_samples: int, trainer: HVRL, discretization_steps: int = 128) -> list[Sample[DDGraph]]:
    left = num_samples
    samples: list[Sample] = []

    original_policy = trainer.env._policy
    trainer.env.policy = trainer.fine_model
    try:
        with torch.no_grad():
            while left > 0:
                batch = min(left, trainer.config.batch_size)
                sample = trainer.env.sample(batch, discretization_steps=discretization_steps, pbar=True)
                samples.extend([s for s in sample])
                left -= batch
    finally:
        trainer.env._policy = original_policy
    
    return samples


def evaluate(samples: list[Sample], reward: MOReward, hv_computer: HVComputer, n: int) -> tuple[float, float, torch.Tensor, float]:
    samples_cat = Sample.concat(samples)
    rew, info = reward(samples_cat.sample, samples_cat.latent)
    valids = info["valids"].sum().item()
    rewards = [r for i, r in enumerate(rew) if info["valids"][i]]
        
    if rewards:
        reward_values = torch.stack(rewards, dim=0)
        full_objectives = reward_values.reshape(1, -1, reward.num_rew)
        full_hypervolume = hv_computer(full_objectives).detach().cpu().item()
    else:
        reward_values = torch.zeros((1, reward.num_rew), device=reward.ref_point.device, dtype=torch.float32)
        full_hypervolume = 0.0
        
    as_many = reward_values.shape[0] - (reward_values.shape[0] % n)
    if as_many > 0:
        n_objectives = reward_values[:as_many].reshape(-1, n, reward.num_rew)
        n_hypervolume = hv_computer(n_objectives).mean().detach().cpu().item()
    else: 
        n_hypervolume = 0.0

    return n_hypervolume, full_hypervolume, reward_values, valids / len(samples)


def main(config: Namespace) -> None:
    assert config.sample_nm1_every_n_steps % config.resample_every_n_steps == 0
    assert config.update_pretrained_every_n_steps % config.resample_every_n_steps == 0
    assert config.update_pretrained_every_n_steps % config.sample_nm1_every_n_steps == 0
    assert (not config.fulfill_num_samples and not config.only_valids) or (config.fulfill_num_samples and config.only_valids)

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
    # reward = MOReward(XTBTask(), num_rew=2, ref_point=torch.tensor([-1.0, -1.0], device=device))
    if config.reward == "molecular":
        reward = MolecularMetrics(do_relax=False, full_bust=config.full_bust)
    elif config.reward == "topology":
        reward = TopologyMetrics(do_relax=False, full_bust=config.full_bust)
    model = GEOMBaseModel(device=device)
    env = EndpointEnvironment(model, DummyReward(), discretization_steps=int(config.num_integration_steps))
    unconstrained_sample = env.sample
    env.sample = lambda *args, **kwargs: unconstrained_sample(*args, n_atoms=10, **kwargs)  # ty: ignore[invalid-assignment]

    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)
    trainer = HVRL(config, env, reward, hv_computer=hv_computer, device=device)

    resume_checkpoint = load_latest_training_checkpoint(run_resolution.run_dir, map_location=device)
    start_epoch = 0
    dataset = None
    advantages = None
    if resume_checkpoint is not None:
        trainer.load_training_state_dict(resume_checkpoint["trainer_state"])
        start_epoch = resume_checkpoint["next_epoch"]
        loop_state = resume_checkpoint.get("loop_state", {})
        dataset = loop_state.get("dataset")
        advantages = loop_state.get("advantages")
        restore_rng_state(resume_checkpoint)

    vol_samples = config.vol_samples

    epoch = log.set_step_metric(start_epoch, "epoch")

    n_hv = log.watch("n_hypervolume", "epoch")
    full_hv = log.watch("full_hypervolume", "epoch")
    # energy = log.watch("energy", "epoch")
    qed = log.watch("qed", "epoch")
    sa = log.watch("sa", "epoch")
    # dipole = log.watch("dipole", "epoch")
    valid_frac = log.watch("valid_fraction", "epoch")
    valid_div = log.watch("diversity/valid_diversity", "epoch")
    diversity = log.watch("diversity/diversity", "epoch")
    urscat = log.watch("diversity/vendi_usrcat", "epoch")
    auc = log.watch("diversity/auc_coverage", "epoch")
    first_var = log.watch(f"dataset/{config.scalarization}", "epoch")
    fulfillment = log.watch("dataset/fulfillment", "epoch")
    hypervol_X_ = log.watch("bg/hypervolume_bg", "epoch")

    samples_eval = sample_x(vol_samples, trainer, discretization_steps=config.num_integration_steps)
    n_hv.val, full_hv.val, rewards, valid_frac.val = evaluate(samples_eval, reward, hv_computer, n=config.n)
    qed.val, sa.val = rewards.mean(dim=0).detach().cpu().numpy().tolist()
    group_length = ceil((config.num_integration_steps-1) / config.num_time_groups)
    for _ in tqdm(range(start_epoch, config.epochs)):
        if epoch.val % config.evaluate_diversity_every_n_steps == 0:
            with trainer.timer.section("evaluate_diversity"):
                if epoch.val == 0:
                    # 10 
                    # valid_div.val = 0.678
                    # diversity.val = 0.67491
                    # urscat.val = 86.64319
                    # auc.val = 274.025
                    # free
                    valid_div.val = 0.334
                    diversity.val = 0.8282
                    urscat.val = 172.19049
                    auc.val = 242.065
                else:
                    samples_diversity = sample_x(config.num_diversity_samples, trainer, discretization_steps=config.num_integration_steps)
                    valid_div.val, diversity.val, urscat.val, auc.val = diversity_metrics(samples_diversity)
        
        print("a")
        if epoch.val % config.update_pretrained_every_n_steps == 0 and config.scalarization == "improvement":
            with trainer.timer.section("update_base_model"):
                trainer.update_base_model()
        print("b")
        if epoch.val % config.sample_nm1_every_n_steps == 0 and config.scalarization == "improvement":
            with trainer.timer.section("sample_bg"):
                trainer.fix_optimization_problem()
                hypervol_X_.val = trainer.hypervolume_X_.median().item()
        print("c")
        if epoch.val % config.resample_every_n_steps == 0:
            with trainer.timer.section("generate_dataset"):
                dataset, advantages, fv = trainer.generate_dataset_fv()
                
                fulfillment.val = len(dataset) / config.num_samples
    
                if dataset:
                    first_var.val = torch.median(fv).item()
        print("d")      
        save_training_checkpoint(
            run_resolution.run_dir,
            next_epoch=epoch.val+1,
            trainer_state=trainer.training_state_dict(),
            loop_state={
                "dataset": dataset,
                "advantages": advantages,
            },
        )
        
        if not dataset or advantages is None:
            print("No valid dataset or advantages, skipping epoch.")
            epoch += 1
            continue
        print("e")
        print(f"Epoch {epoch.val} starting with dataset size: {len(dataset)} and advantages size: {len(advantages)}")
        with trainer.timer.section("finetune"):
            timed_stats = trainer.finetune(dataset, advantages)
            
        print(f"Epoch {epoch.val} completed")
        
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
        
        if epoch.val % config.evaluate_every_n_steps == 0:
            with trainer.timer.section("evaluate_hypervolume"):           
                samples_eval = sample_x(vol_samples, trainer, discretization_steps=config.num_integration_steps)
                n_hv.val, full_hv.val, rewards, valid_frac.val = evaluate(samples_eval, reward, hv_computer, n=config.n)
                qed.val, sa.val = rewards.mean(dim=0).detach().cpu().numpy().tolist()

        print("\n=== Timing summary (by total time) ===")
        for name, cnt, total, mean, p50, p95 in rows:
            print(f"{name:30s}  n={cnt:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  "
                f"p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")

        epoch += 1

    log.finish()
    mark_run_complete(run_resolution.run_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
