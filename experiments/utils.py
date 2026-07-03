

from omegaconf import OmegaConf
import itertools
import copy
import os
import random
import numpy as np
import torch


def collect_grid_axes(node, path=()):
    axes = []
    if isinstance(node, dict):
        for key, value in node.items():
            axes.extend(collect_grid_axes(value, path + (key,)))
    elif isinstance(node, list):
        axes.append((path, node))
    return axes


def set_nested_value(node, path, value):
    current = node
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def resolve_config(args):
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    grid_axes = collect_grid_axes(config_dict)
    grid_values = [values for _, values in grid_axes]
    grid_combinations = list(itertools.product(*grid_values)) if grid_values else [()]

    if args.list_configs:
        print(len(grid_combinations))
        raise SystemExit(0)

    config_idx = args.config_idx
    if config_idx is None:
        config_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if config_idx < 0 or config_idx >= len(grid_combinations):
        raise ValueError(f"config_idx={config_idx} is out of range for {len(grid_combinations)} configs")

    resolved_dict = copy.deepcopy(config_dict)
    for (path, _), value in zip(grid_axes, grid_combinations[config_idx]):
        set_nested_value(resolved_dict, path, value)

    return OmegaConf.create(resolved_dict), config_idx


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
