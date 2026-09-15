import argparse
import pickle
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from diffusiongym.environments import EndpointEnvironment
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.flowmol import GEOMBaseModel
from diffusiongym.molecules.rewards.utils import graph_to_mols
from diffusiongym.rewards import DummyReward
from utils import seed_everything

from genexp.mo.mo_mol import RDkitReward
from genexp.mo.moses import diversity_metrics_2d
from genexp.mo.utils import HVComputer
from genexp.resume import mark_run_complete, resolve_run
from genexp.trainers.utils import StepTimer
from genexp.wandb_log import WandbLogger
from hv_nft import evaluate, jack_hv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the pretrained GEOM base model at epoch 0.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--force_new_start", action="store_true")
    parser.add_argument("--project_name", type=str, default="retro_compatibility")
    parser.add_argument("--run_name", type=str, default="geom_base_model")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=1_000)
    parser.add_argument("--evaluate_max_retries", type=int, default=10_000)
    parser.add_argument("--num_integration_steps", type=int, default=100)
    parser.add_argument("--jackknife_permutations", type=int, default=100_000)
    parser.add_argument("--validate_2d", choices=["none", "full"], default="full")
    parser.add_argument("--validate_3d", choices=["none", "fast", "full"], default="none")
    parser.add_argument("--fixed_A", type=int, default=10)
    parser.add_argument("--invalid_val", type=float, default=-1.0)
    parser.add_argument("--rewards", nargs="+", choices=["qed", "sa", "dipole"], default=["qed", "sa", "dipole"])
    return parser.parse_args()


def sample_base_model(
    env: EndpointEnvironment,
    reward: RDkitReward,
    batch_size: int,
    num_valids_requested: int,
    max_attempts: int,
    discretization_steps: int,
    fixed_atoms: int,
) -> tuple[list, torch.Tensor, torch.Tensor]:
    rewards = []
    valid_masks = []
    sampled_graphs: list[DDGraph] = []
    num_valids = 0
    attempts = 0
    sample_kwargs = {"n_atoms": fixed_atoms} if fixed_atoms > 0 else {}

    while num_valids < num_valids_requested and attempts < max_attempts:
        print(f"num_valids={num_valids}, attempts={attempts}, remaining={num_valids_requested - num_valids}")
        batch = min(batch_size, max_attempts - attempts, (num_valids_requested - num_valids) * 5)
        with torch.no_grad():
            samples = env.sample(batch, discretization_steps=discretization_steps, pbar=True, **sample_kwargs)
            batch_rewards, info = reward(samples.sample, samples.latent)
        batch_valids = info["valids"]
        remaining = num_valids_requested - num_valids

        if batch_valids.sum().item() >= remaining:
            batch = int(torch.nonzero(batch_valids, as_tuple=False)[remaining - 1].item()) + 1

        rewards.append(batch_rewards[:batch].detach().cpu())
        valid_masks.append(batch_valids[:batch].detach().cpu())
        sampled_graphs.extend(sample.sample for sample in islice(samples, batch))
        num_valids += int(batch_valids[:batch].sum().item())
        attempts += batch

    if num_valids < num_valids_requested:
        print(f"Warning: base model produced {num_valids}/{num_valids_requested} valid molecules after {attempts} attempts")

    molecules = list(graph_to_mols(DDGraph.collate(sampled_graphs)))
    return molecules, torch.cat(rewards, dim=0), torch.cat(valid_masks, dim=0)


def main(config: argparse.Namespace) -> None:
    assert config.n > 0, "n must be positive"
    assert config.batch_size > 0, "batch_size must be positive"
    assert config.num_samples > 0, "num_samples must be positive"
    assert config.evaluate_max_retries > 0, "evaluate_max_retries must be positive"
    if len(config.rewards) not in {2, 3} or len(set(config.rewards)) != len(config.rewards):
        raise ValueError("rewards must contain two or three distinct objectives")

    config.model = "geom"
    config.epochs = 0
    config.evaluation_fixed_A = config.fixed_A

    results_root = Path("output") / config.project_name
    run_resolution = resolve_run(config, results_root, config.run_name)
    if run_resolution.completed:
        print(f"Matching run is already complete: {run_resolution.run_dir}")
        return

    c_wandb = vars(config)
    c_wandb.update({'scalarization': 'base'})
    print(f"run_dir={run_resolution.run_dir}")
    log = WandbLogger(
        project_name=config.project_name,
        config=c_wandb,
        use_wandb=config.wandb,
        run_name=run_resolution.run_dir.name,
        id=run_resolution.wandb_run_id,
        resume="allow",
        dir=str(run_resolution.run_dir),
    )
    log.set_step_metric(0, "epoch")
    valid_fraction = log.watch("valid_fraction", "epoch")
    diversity_tanimoto = log.watch("diversity/diversity_tanimoto", "epoch")
    vendi_tanimoto = log.watch("diversity/vendi_tanimoto", "epoch")
    auc_tanimoto = log.watch("diversity/auc_coverage_tanimoto", "epoch")

    obj_dir = run_resolution.run_dir / "objs"
    scores_dir = run_resolution.run_dir / "scores"
    obj_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer = StepTimer(device=device)
    reward = RDkitReward(config.rewards, valid_2d=config.validate_2d, valid_3d=config.validate_3d, invalid_val=config.invalid_val)
    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)
    model = GEOMBaseModel(device=device)
    model.requires_grad_(False)
    model.eval()
    env = EndpointEnvironment(model, DummyReward(), discretization_steps=config.num_integration_steps)

    try:
        with timer.section("sample_base_model"):
            molecules, scores, validity = sample_base_model(
                env=env,
                reward=reward,
                batch_size=config.batch_size,
                num_valids_requested=config.num_samples,
                max_attempts=config.evaluate_max_retries,
                discretization_steps=config.num_integration_steps,
                fixed_atoms=config.fixed_A,
            )

        with (obj_dir / "obj_0.pkl").open("wb") as molecule_file:
            pickle.dump(molecules, molecule_file)
        np.savez(scores_dir / "score_0.npz", scores=scores.numpy(), validity=validity.numpy())

        valid_scores = scores[validity]
        score_metrics = evaluate(valid_scores, reward.reward_names, hv_computer, n=config.n)
        for jackknife_n in (100, 500):
            if valid_scores.shape[0] > jackknife_n:
                with timer.section(f"jackknife_{jackknife_n}"):
                    jack_mean, jack_low, jack_high, jack_variance = jack_hv(
                        valid_scores,
                        hv_computer,
                        n=jackknife_n,
                        permutations=config.jackknife_permutations,
                        ci=0.95,
                    )
                score_metrics.update({
                    f"jack/mean_{jackknife_n}": jack_mean,
                    f"jack/low_{jackknife_n}": jack_low,
                    f"jack/high_{jackknife_n}": jack_high,
                    f"jack/var_{jackknife_n}": jack_variance,
                })
        log.log_dict(score_metrics, "epoch")

        valid_fraction.val = validity.float().mean().item()
        clean_molecules = [molecule for molecule, valid in zip(molecules, validity.tolist()) if valid]
        _, diversity_tanimoto.val, vendi_tanimoto.val, auc_tanimoto.val = diversity_metrics_2d(clean_molecules)

        print("\n=== Timing summary (by total time) ===")
        for name, count, total, mean, p50, p95 in timer.summary():
            print(f"{name:30s}  n={count:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")
    finally:
        log.finish()

    mark_run_complete(run_resolution.run_dir)


if __name__ == "__main__":
    main(parse_args())
