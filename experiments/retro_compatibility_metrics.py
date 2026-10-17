import json
import math
from pathlib import Path

import moocore
import numpy as np
import pandas as pd
import wandb


WANDB_ENTITY = "jdg8833-new-york-university"
PROJECTS = {"ablation_n_topology": "tanimoto"}
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"

SCALARIZATIONS = ["null", "base", "sum", "improvement", "blend"]
IDENTIFIER_COLUMNS = ["run_id", "run", "scalarization", "seed", "n"]


def metric_keys(diversity: str) -> dict[str, str]:
    return {
        "jack_mean_100": "jack/mean_100",
        "valid_frac": "valid_fraction",
        f"auc_coverage_{diversity}": f"diversity/auc_coverage_{diversity}",
        f"vendi_{diversity}": f"diversity/vendi_{diversity}",
    }


def load_valid_scores(run_dir: Path, epoch: int) -> np.ndarray:
    path = run_dir / "scores" / f"score_{epoch}.npz"
    with np.load(path, allow_pickle=False) as archive:
        scores, validity = archive["scores"], archive["validity"]
    if scores.ndim != 2 or scores.shape[1] not in (2, 3):
        raise ValueError(f"{path}: expected scores with two or three objectives, got {scores.shape}")
    if validity.dtype != np.bool_ or validity.shape != (len(scores),):
        raise ValueError(f"{path}: expected a boolean validity mask with shape ({len(scores)},)")
    valid_scores = scores[validity]
    if not len(valid_scores) or not np.issubdtype(valid_scores.dtype, np.floating) or not np.isfinite(valid_scores).all():
        raise ValueError(f"{path}: valid scores must be nonempty, floating-point, and finite")
    return valid_scores


def sampled_hypervolume_mean(scores: np.ndarray, ref_point: np.ndarray, seed: int, permutations: int, n: int = 10) -> float:
    """Same subset-mean estimator as jack_hv, without its unused CI allocation."""
    if len(scores) <= n:
        raise ValueError(f"Need more than {n} valid samples for jackknife sampling, found {len(scores)}")
    if permutations <= 0:
        raise ValueError("jackknife_permutations must be positive")
    rng = np.random.RandomState(seed)
    # HVComputer casts each volume to the score dtype before jack_hv takes its mean.
    volumes = np.empty(permutations, dtype=scores.dtype)
    for index in range(permutations):
        subset = scores[rng.choice(len(scores), size=n, replace=False)]
        volumes[index] = moocore.hypervolume(subset, ref=ref_point, maximise=True)
    return float(volumes.mean())


def finite_float(value: object, label: str, run_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Run {run_id} has a non-numeric {label}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Run {run_id} has a non-finite {label}: {value!r}")
    return number


def epoch_value(value: object, run_id: str) -> int:
    number = finite_float(value, "epoch", run_id)
    if not number.is_integer():
        raise ValueError(f"Run {run_id} has a non-integer epoch: {value!r}")
    return int(number)


def scalarization_value(wandb_run) -> str:
    scalarization = wandb_run.config.get("scalarization")
    normalized = "null" if scalarization is None else str(scalarization)
    if normalized not in SCALARIZATIONS:
        raise ValueError(f"Run {wandb_run.id} has unexpected scalarization {scalarization!r}")
    return normalized


def seed_value(wandb_run) -> int:
    seed = finite_float(wandb_run.config.get("seed"), "seed", wandb_run.id)
    if not seed.is_integer():
        raise ValueError(f"Run {wandb_run.id} has a non-integer seed: {seed!r}")
    return int(seed)


def summarize_run(wandb_run, run_dir: Path, diversity: str) -> dict[str, object]:
    print(f"Summarizing run {wandb_run.id} ({wandb_run.name})...")
    history = list(wandb_run.scan_history(page_size=1_000))
    keys = metric_keys(diversity)
    jack_key = keys["jack_mean_100"]
    candidates = []
    for row in history:
        if row.get("epoch") is None or row.get(jack_key) is None:
            continue
        epoch = epoch_value(row["epoch"], wandb_run.id)
        jack_mean_100 = finite_float(row[jack_key], jack_key, wandb_run.id)
        candidates.append((epoch, jack_mean_100))

    if not candidates:
        raise ValueError(f"Run {wandb_run.id} has no finite history values for {jack_key!r}")
    best_epoch, best_jack_mean_100 = max(candidates, key=lambda item: (item[1], -item[0]))

    selected_values: dict[str, float] = {"jack_mean_100": best_jack_mean_100}
    for row in history:
        if row.get("epoch") is None or epoch_value(row["epoch"], wandb_run.id) != best_epoch:
            continue
        for column, key in {**keys, "jack_mean_10": "jack/mean_10"}.items():
            if column == "jack_mean_100" or row.get(key) is None:
                continue
            selected_values[column] = finite_float(row[key], key, wandb_run.id)

    missing = [column for column in keys if column not in selected_values]
    if missing:
        raise ValueError(f"Run {wandb_run.id} is missing metrics at epoch {best_epoch}: {missing}")

    scores = load_valid_scores(run_dir, best_epoch)
    selected_values["pareto_frac"] = float(np.mean(moocore.pareto_rank(scores, maximise=True) == 0))
    seed = seed_value(wandb_run)
    if "jack_mean_10" not in selected_values:
        reference = finite_float(wandb_run.config.get("invalid_val", -1.0), "invalid_val", wandb_run.id)
        permutations = int(wandb_run.config.get("jackknife_permutations", 100_000))
        print(f"  Epoch {best_epoch}: computing jack_mean_10 from {len(scores)} valid samples ({permutations} subsets)")
        selected_values["jack_mean_10"] = sampled_hypervolume_mean(scores, np.full(scores.shape[1], reference, dtype=scores.dtype), seed, permutations)

    return {
        "run_id": wandb_run.id,
        "run": wandb_run.name,
        "scalarization": scalarization_value(wandb_run),
        "seed": seed,
        "epoch": best_epoch,
        **selected_values,
    }


def build_individual_metrics(api: wandb.Api, project: str, diversity: str) -> pd.DataFrame:
    runs = list(api.runs(f"{WANDB_ENTITY}/{project}"))
    if not runs:
        raise ValueError(f"No runs found in {WANDB_ENTITY}/{project}")

    output_dir = OUTPUT_ROOT / project
    directories_by_id = {}
    for manifest_path in output_dir.glob("*/run_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("wandb_run_id"):
            directories_by_id[manifest["wandb_run_id"]] = manifest_path.parent
    rows = []
    for wandb_run in runs:
        run_dir = directories_by_id.get(wandb_run.id, output_dir / wandb_run.name)
        try:
            metadata = {}
            if project == "ablation_n_topology":
                n = finite_float(wandb_run.config.get("n"), "n", wandb_run.id)
                if not n.is_integer() or n <= 0:
                    raise ValueError(f"Run {wandb_run.id} has a non-positive or non-integer n: {n!r}")
                metadata["n"] = int(n)
            rows.append({**summarize_run(wandb_run, run_dir, diversity), **metadata})
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"Skipping run {wandb_run.id} ({wandb_run.name}): {exc}")
    if not rows:
        raise ValueError(f"No usable evaluations found in {project}")
    identifiers = [column for column in IDENTIFIER_COLUMNS if column != "n" or project == "ablation_n_topology"]
    columns = [*identifiers, "epoch", "jack_mean_10", "jack_mean_100", "pareto_frac", "valid_frac", f"auc_coverage_{diversity}", f"vendi_{diversity}"]
    frame = pd.DataFrame(rows, columns=columns)
    if project == "ablation_n_topology":
        return frame.sort_values(["n", "seed", "run"], kind="stable").reset_index(drop=True)
    scalarization_order = {name: index for index, name in enumerate(SCALARIZATIONS)}
    frame["_scalarization_order"] = frame["scalarization"].map(scalarization_order)
    return frame.sort_values(["_scalarization_order", "seed", "run"], kind="stable").drop(columns="_scalarization_order").reset_index(drop=True)


def build_aggregates(individual_metrics: pd.DataFrame, group_by: str = "scalarization") -> pd.DataFrame:
    rows = []
    group_values = SCALARIZATIONS if group_by == "scalarization" else sorted(individual_metrics[group_by].unique())
    for group_value in group_values:
        group = individual_metrics[individual_metrics[group_by] == group_value]
        n_runs = len(group)
        if not n_runs:
            continue

        row: dict[str, object] = {group_by: group_value, "n_runs": n_runs}
        for metric in individual_metrics.columns.difference(IDENTIFIER_COLUMNS, sort=False):
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_se"] = values.std(ddof=1) / math.sqrt(n_runs)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    try:
        api = wandb.Api()
    except Exception as exc:
        raise RuntimeError("Could not initialize the W&B API. Confirm network access and run 'wandb login'.") from exc

    for project, diversity in PROJECTS.items():
        individual_metrics = build_individual_metrics(api, project, diversity)
        group_by = "n" if project == "ablation_n_topology" else "scalarization"
        aggregates = build_aggregates(individual_metrics, group_by=group_by)
        output_dir = OUTPUT_ROOT / project
        output_dir.mkdir(parents=True, exist_ok=True)
        individual_path = output_dir / "filtered_individual_metrics.csv"
        summary_path = output_dir / "summary.csv"
        individual_metrics.to_csv(individual_path, index=False)
        aggregates.to_csv(summary_path, index=False)
        print(f"Wrote {len(individual_metrics)} rows to {individual_path}")
        print(f"Wrote {len(aggregates)} rows to {summary_path}")


if __name__ == "__main__":
    main()
