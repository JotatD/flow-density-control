import argparse
import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from genexp.mo.mo_mol import TopologyMetrics
from genexp.mo.moses import diversity_metrics_2d
from genexp.mo.utils import HVComputer
from genexp.trainers.utils import StepTimer
from genexp.wandb_log import WandbLogger
from hv_nft import evaluate, jack_hv


PROJECT_DIR = Path("/home/juan.guevara/Code/flow-density-control/output/geom_10_di_sec")
WANDB_PROJECT = "retro_compatibility"
CADENCE = 50
MOLECULE_PATTERN = re.compile(r"mols_epoch_(\d+)\.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy molecule evaluations to the current output and logging format.")
    parser.add_argument("--wandb", action="store_true", help="Publish recomputed metrics to W&B")
    parser.add_argument("--jackknife_permutations", type=int, default=100_000)
    parser.add_argument("--validate_2d", choices=["none", "full"], default=None, help="Override the validation mode stored in each run manifest")
    parser.add_argument("--validate_3d", choices=["none", "fast", "full"], default=None, help="Override the validation mode stored in each run manifest")
    return parser.parse_args()


def load_configuration(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    with manifest_path.open() as manifest_file:
        manifest = json.load(manifest_file)
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Run manifest {manifest_path} does not contain a configuration object")
    return configuration


def find_molecule_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in (run_dir / "checkpoints").iterdir():
        match = MOLECULE_PATTERN.fullmatch(path.name)
        if path.is_file() and match is not None:
            epoch = int(match.group(1))
            if epoch % CADENCE == 0:
                checkpoints.append((epoch, path))
    return sorted(checkpoints)


def load_molecules(path: Path) -> list:
    with path.open("rb") as molecule_file:
        molecules = pickle.load(molecule_file)
    if not isinstance(molecules, list):
        raise ValueError(f"Expected {path} to contain a molecule list, got {type(molecules).__name__}")
    return molecules


def migrate_run(run_dir: Path, args: argparse.Namespace, timer: StepTimer) -> None:
    configuration = load_configuration(run_dir)
    print(f"Migrating run {run_dir.name} with configuration: {configuration}")
    if configuration.get("reward") != "topology":
        raise ValueError(f"Expected a topology reward in {run_dir / 'run_manifest.json'}")

    checkpoints = find_molecule_checkpoints(run_dir)
    if not checkpoints:
        raise ValueError(f"No cadence-{CADENCE} molecule checkpoints found in {run_dir / 'checkpoints'}")

    validate_2d = args.validate_2d or str(configuration.get("validate_2d", "none"))
    validate_3d = args.validate_3d or str(configuration.get("validate_3d", "none"))
    reward = TopologyMetrics(valid_2d=validate_2d, valid_3d=validate_3d, invalid_val=float(configuration.get("invalid_val", 0.0)))
    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)

    obj_dir = run_dir / "objs"
    scores_dir = run_dir / "scores"
    obj_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    wandb_config = dict(configuration)
    wandb_config.update({
        "retro_source_run": str(run_dir),
        "retro_cadence": CADENCE,
        "jackknife_permutations": args.jackknife_permutations,
        "retro_validate_2d": validate_2d,
        "retro_validate_3d": validate_3d,
    })
    log = WandbLogger(project_name=WANDB_PROJECT, config=wandb_config, use_wandb=args.wandb, run_name=run_dir.name, dir=str(run_dir))
    print(f"Logging to W&B project {WANDB_PROJECT} with run name {run_dir.name}")
    epoch_metric = log.set_step_metric(checkpoints[0][0], "epoch")
    valid_fraction = log.watch("valid_fraction", "epoch")
    diversity_tanimoto = log.watch("diversity/diversity_tanimoto", "epoch")
    vendi_tanimoto = log.watch("diversity/vendi_tanimoto", "epoch")
    auc_tanimoto = log.watch("diversity/auc_coverage_tanimoto", "epoch")

    print(f"Starting migration of {len(checkpoints)} checkpoints from {run_dir.name} to W&B project {WANDB_PROJECT}")
    try:
        for epoch, source_path in checkpoints:
            if epoch != epoch_metric.val:
                epoch_metric.val = epoch

            molecules = load_molecules(source_path)
            shutil.copy2(source_path, obj_dir / f"obj_{epoch}.pkl")

            with timer.section(f"compute_scores_epoch_{epoch}"):    
                scores, info = reward.compute(molecules, torch.device("cpu"))
            validity = info["valids"]
            np.savez(scores_dir / f"score_{epoch}.npz", scores=scores.numpy(), validity=validity.numpy())

            valid_scores = scores[validity]
            
            with timer.section(f"evaluate_scores_epoch_{epoch}"):
                score_metrics = evaluate(valid_scores, reward.reward_names, hv_computer, n=int(configuration["n"]))
            for jackknife_n in (100, 500):
                if valid_scores.shape[0] > jackknife_n:
                    with timer.section(f"jackknife_{jackknife_n}"):
                        jack_mean, jack_low, jack_high, jack_variance = jack_hv(
                            valid_scores,
                            hv_computer,
                            n=jackknife_n,
                            permutations=args.jackknife_permutations,
                            ci=0.95,
                        )
                    score_metrics.update({
                        f"jack/mean_{jackknife_n}": jack_mean,
                        f"jack/low_{jackknife_n}": jack_low,
                        f"jack/high_{jackknife_n}": jack_high,
                        f"jack/var_{jackknife_n}": jack_variance,
                    })
            log.log_dict(score_metrics, "epoch")
            valid_fraction.val = validity.float().mean().item() if molecules else 0.0

            clean_molecules = [molecule for molecule, valid in zip(molecules, validity.tolist()) if valid]
            _, diversity_tanimoto.val, vendi_tanimoto.val, auc_tanimoto.val = diversity_metrics_2d(clean_molecules)
            print(f"Migrated {run_dir.name} epoch {epoch}: {len(molecules)} molecules")
    finally:
        log.finish()


def main() -> None:
    args = parse_args()
    timer = StepTimer(device=torch.device("cpu"))
    run_dirs = sorted(path for path in PROJECT_DIR.iterdir() if path.is_dir() and (path / "run_manifest.json").is_file())
    if not run_dirs:
        raise ValueError(f"No run directories found in {PROJECT_DIR}")
    for run_dir in run_dirs:
        migrate_run(run_dir, args, timer)


if __name__ == "__main__":
    main()
