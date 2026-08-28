"""Run discovery and training-checkpoint helpers for resumable experiments."""

from __future__ import annotations

import datetime
import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

MANIFEST_NAME = "run_manifest.json"
CHECKPOINT_PREFIX = "training_state_epoch_"
CONFIG_EXCLUSIONS = {
    "epochs",
    "force_new_start",
    "evaluate_diversity_every_n_steps",
    "evaluate_every_n_steps",
    "batch_size",
    # "timestep_fraction",
    "wandb",
    # "num_integration_steps",
    "backward_batch_size",
}


@dataclass(frozen=True)
class RunResolution:
    run_dir: Path
    wandb_run_id: str
    resumed: bool
    completed: bool


def _configuration(config) -> dict[str, Any]:
    return {key: value for key, value in vars(config).items() if key not in CONFIG_EXCLUSIONS}


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def resolve_run(config, results_root: Path, run_prefix: str) -> RunResolution:
    """Find the latest matching run, or create a timestamped run directory."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(config)

    if not config.force_new_start:
        matches = []
        for manifest_path in results_root.glob(f"*/{MANIFEST_NAME}"):
            manifest = _read_manifest(manifest_path)
            if manifest is None:
                continue
            if manifest.get("configuration"):
                temp_config = manifest.get("configuration")
                candidate = {key: value for key, value in temp_config.items() if key not in CONFIG_EXCLUSIONS}
                if configuration != candidate:
                    continue
            matches.append((manifest.get("created_at", ""), manifest_path, manifest))

        if matches:
            _, manifest_path, manifest = max(matches, key=lambda item: item[0])
            completed = manifest.get("status") == "complete" and manifest.get("epochs", 0) >= config.epochs
            if not completed:
                manifest["epochs"] = config.epochs
                manifest["status"] = "running"
                manifest.pop("completed_at", None)
                _write_manifest(manifest_path, manifest)

            return RunResolution(
                run_dir=manifest_path.parent,
                wandb_run_id=manifest["wandb_run_id"],
                resumed=not completed,
                completed=completed,
            )

    created_at = datetime.datetime.now(datetime.timezone.utc)
    timestamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
    run_dir = results_root / f"{run_prefix}_{timestamp}"
    run_dir.mkdir(parents=False, exist_ok=False)
    wandb_run_id = uuid.uuid4().hex[:8]
    _write_manifest(
        run_dir / MANIFEST_NAME,
        {
            "schema_version": 1,
            "created_at": created_at.isoformat(),
            "status": "running",
            "wandb_run_id": wandb_run_id,
            "epochs": config.epochs,
            "configuration": configuration,
        },
    )
    return RunResolution(
        run_dir=run_dir,
        wandb_run_id=wandb_run_id,
        resumed=False,
        completed=False,
    )


def mark_run_complete(run_dir: Path) -> None:
    manifest_path = Path(run_dir) / MANIFEST_NAME
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        raise RuntimeError(f"Cannot read run manifest: {manifest_path}")
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)


def _checkpoint_paths(run_dir: Path) -> list[Path]:
    checkpoint_dir = Path(run_dir) / "checkpoints"
    return sorted(
        checkpoint_dir.glob(f"{CHECKPOINT_PREFIX}*.pt"),
        reverse=True,
    )


def load_latest_training_checkpoint(run_dir: Path, map_location: str | torch.device) -> dict[str, Any] | None:
    paths = _checkpoint_paths(run_dir)
    failures = []
    for path in paths:
        try:
            try:
                checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location=map_location)
            if not isinstance(checkpoint, dict) or "next_epoch" not in checkpoint:
                raise ValueError("not a training-state checkpoint")
            print(f"Resuming from {path}")
            return checkpoint
        except Exception as error:
            failures.append(f"{path.name}: {error}")

    if paths:
        details = "; ".join(failures)
        raise RuntimeError(f"No valid training checkpoint found in {run_dir}: {details}")
    return None


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])


def save_training_checkpoint(
    run_dir: Path,
    next_epoch: int,
    trainer_state: dict[str, Any],
    loop_state: dict[str, Any],
    keep: int = 3,
) -> Path:
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{CHECKPOINT_PREFIX}{next_epoch:06d}.pt"

    checkpoint = {
        "schema_version": 1,
        "next_epoch": next_epoch,
        "trainer_state": trainer_state,
        "loop_state": loop_state,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(checkpoint, checkpoint_path)

    for stale_path in _checkpoint_paths(run_dir)[keep:]:
        stale_path.unlink()
    return checkpoint_path
