import argparse
from argparse import Namespace
from math import ceil
from pathlib import Path

import torch
from diffusiongym import Sample
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.flowmol import GEOMBaseModel
from tqdm.auto import tqdm
from utils import seed_everything

from genexp.mo.base import MOReward
from genexp.mo.mo_mol import TopologyMetrics
from genexp.mo.moses import diversity_metrics
from genexp.mo.utils import HVComputer
from genexp.resume import (
    mark_run_complete,
    resolve_run,
)
from genexp.trainers.diff_blend import BlendEnvironment
from genexp.trainers.utils import StepTimer
from genexp.wandb_log import WandbLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    #loggin
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project_name", type=str, default="semi_bust")
    parser.add_argument("--run_name", type=str, default="hv_nft")
    parser.add_argument("--seed", type=int, default=5)
    
    parser.add_argument("--n", type=int, default=4) #only for loggin purposes

    #hyper param
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_integration_steps", type=int, default=100)
    
    #size
    parser.add_argument("--batch_size", type=int, default=320)
    
    parser.add_argument("--vol_samples", type=int, default=64)
    parser.add_argument("--samples_per_tradeoff", type=int, default=8)
    
    parser.add_argument("--num_diversity_samples", type=int, default=128)
    parser.add_argument("--evaluate_every_n_steps", type=int, default=10)
    
    
    #validity
    parser.add_argument("--validate_2d", type=str, default="none", choices=["none", "full"])
    parser.add_argument("--validate_3d", type=str, default="none", choices=["none", "fast", "full"])
    
    
    #models
    parser.add_argument("--m1_folder", type=str, default="output/qed/2023-06-19_15-30-00")
    parser.add_argument("--m2_folder", type=str, default="output/sa/2023-06-19_15-30-00")
    
    #modifiers
    parser.add_argument("--fixed_A", type=int, default=10)
    parser.add_argument("--invalid_val", type=float, default=-1.0)


    
    
    return parser.parse_args()

def sample_x(env: BlendEnvironment, tradeoffs: list[list[float]], samples_per_tradeoff:int,  discretization_steps: int = 128, batch_size: int = 320) -> list[Sample[DDGraph]]:
    samples: list[Sample] = []

    with torch.no_grad():
        for blend_vector in tradeoffs:
            left = samples_per_tradeoff
            while left > 0:
                batch = min(left, batch_size)
                sample = env.sample(batch, blend_vector=blend_vector, discretization_steps=discretization_steps, pbar=True)
                samples.extend([s for s in sample])
                left -= batch

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
    
def uniform_positive_tradeoffs(num_samples: int, num_rews: int, device: torch.device) -> list[list[float]]:
    """Generate uniform positive tradeoffs that sum to 1."""
    tradeoffs = torch.rand(num_samples, num_rews, device=device)
    tradeoffs /= tradeoffs.sum(dim=1, keepdim=True)
    return tradeoffs.detach().cpu().numpy().tolist()


def summarize_rewards(rewards: torch.Tensor) -> tuple[list[float], list[float], list[float]]:
    top_decile_count = ceil(rewards.shape[0] * 0.1)
    means = rewards.mean(dim=0)
    top_decile_means = rewards.topk(top_decile_count, dim=0).values.mean(dim=0)
    top_3_means = rewards.topk(min(3, rewards.shape[0]), dim=0).values.mean(dim=0)
    return tuple(values.detach().cpu().numpy().tolist() for values in (means, top_decile_means, top_3_means))


def main(config: Namespace) -> None:
    assert config.vol_samples % config.samples_per_tradeoff == 0, "vol_samples must be divisible by samples_per_tradeoff"
    assert (not config.fulfill_num_samples and not config.only_valids) or (config.fulfill_num_samples and config.only_valids)
    timer = StepTimer(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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
    rew_cnf = {"valid_3d": config.validate_3d, "valid_2d": config.validate_2d, "invalid_val": config.invalid_val}
    reward = TopologyMetrics(**rew_cnf)
    
    model_1 = GEOMBaseModel(device=device)
    path_1 = Path(config.m1_folder)
    model_2 = GEOMBaseModel(device=device)
    path_2 = Path(config.m2_folder)
    
    env = BlendEnvironment([model_1, model_2], discretization_steps=config.num_integration_steps)
    
    vol_samples = config.vol_samples
    tradeoffs = uniform_positive_tradeoffs(vol_samples // config.samples_per_tradeoff, reward.num_rew, device=device)
    
    
    if config.fixed_A > 0:
        unconstrained_sample = env.sample
        env.sample = lambda *args, **kwargs: unconstrained_sample(*args, n_atoms=config.fixed_A, **kwargs)  # ty: ignore[invalid-assignment]


    epoch = log.set_step_metric(0, "epoch")

    n_hv = log.watch("n_hypervolume", "epoch")
    full_hv = log.watch("full_hypervolume", "epoch")
    qed = log.watch("qed", "epoch")
    qed_td = log.watch("top_decile/qed_td", "epoch")
    qed_t3 = log.watch("top_3/qed_t3", "epoch")
    sa = log.watch("sa", "epoch")
    sa_td = log.watch("top_decile/sa_td", "epoch")
    sa_t3 = log.watch("top_3/sa_t3", "epoch")
    valid_frac = log.watch("valid_fraction", "epoch")
    # valid_2d = log.watch("diversity/validity_2d", "epoch")
    # valid_3d = log.watch("diversity/validity_3d", "epoch")
    # diversity_usrcat = log.watch("diversity/diversity_usrcat", "epoch")
    # vendi_usrcat = log.watch("diversity/vendi_usrcat", "epoch")
    # auc_usrcat = log.watch("diversity/auc_coverage_usrcat", "epoch")
    # diversity_tanimoto = log.watch("diversity/diversity_tanimoto", "epoch")
    # vendi_tanimoto = log.watch("diversity/vendi_tanimoto", "epoch")
    # auc_tanimoto = log.watch("diversity/auc_coverage_tanimoto", "epoch")
    
    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)

    for _ in tqdm(range(0, config.epochs + 1, config.evaluate_every_n_steps), desc="Epochs"):
        epoch += config.evaluate_every_n_steps
        
        with timer.section("load_state"):
            state_dict_1 = torch.load(path_1 / f"model_epoch_{epoch.val}.pt", map_location=device)
            env.tilteds[0].load_state_dict(state_dict_1)
            
            state_dict_2 = torch.load(path_2 / f"model_epoch_{epoch.val}.pt", map_location=device)
            env.tilteds[1].load_state_dict(state_dict_2)
            
        
        # with timer.section("evaluate_diversity"):
        #     samples_diversity = sample_x(config.num_diversity_samples, env, discretization_steps=config.num_integration_steps, batch_size=config.batch_size)
        #     (
        #         valid_2d.val,
        #         diversity_tanimoto.val,
        #         vendi_tanimoto.val,
        #         auc_tanimoto.val,
        #         valid_3d.val,
        #         diversity_usrcat.val,
        #         vendi_usrcat.val,
        #         auc_usrcat.val,
        #     ) = diversity_metrics(samples_diversity, full_bust=config.full_bust)
        
        
        with timer.section("evaluate_hypervolume"):           
            samples_eval = sample_x(env=env, discretization_steps=config.num_integration_steps, batch_size=config.batch_size, tradeoffs=tradeoffs, samples_per_tradeoff=config.samples_per_tradeoff)
            n_hv.val, full_hv.val, rewards, valid_frac.val = evaluate(samples_eval, reward, hv_computer=hv_computer, n=config.n)
            (qed.val, sa.val), (qed_td.val, sa_td.val), (qed_t3.val, sa_t3.val) = summarize_rewards(rewards)

        rows = timer.summary()
        print("\n=== Timing summary (by total time) ===")
        for name, cnt, total, mean, p50, p95 in rows:
            print(f"{name:30s}  n={cnt:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  "
                f"p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")

    log.finish()
    mark_run_complete(run_resolution.run_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
