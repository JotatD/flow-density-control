"""Multi-objective rewards for PeptidesGym peptide samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from diffusiongym.types import D
from genexp.mo.base import MOReward
from peptidesgym.rewards import (
    BindingAffinityReward,
    HemolysisReward,
    NonfoulingReward,
    PermeabilityReward,
    SolubilityReward,
)
from peptidesgym.types import String
from peptune.src.utils.app import PeptideAnalyzer


_REWARD_CLASSES = {
    "solubility": SolubilityReward,
    "hemolysis": HemolysisReward,
    "nonfouling": NonfoulingReward,
    "permeability": PermeabilityReward,
    "binding_affinity": BindingAffinityReward,
}

_REFERENCE_POINTS = {
    "solubility": 0.0,
    "hemolysis": 0.0,
    "nonfouling": 0.0,
    "permeability": -10.0,
    "binding_affinity": 4.5,
    "validity": 0.0,
}


class PeptideMOReward(MOReward[D]):
    """Combine selected PeptidesGym rewards into one ordered reward tensor.

    ``validity`` is an optional objective. Independently of whether it is
    selected, validity is used to zero property rewards for invalid peptides
    when ``zero_invalid`` is enabled.
    """

    def __init__(
        self,
        reward_names: str | Sequence[str],
        protein_sequence: str | None = None,
        zero_invalid: bool = True,
        base_path: str | Path | None = None,
        device: torch.device | str | None = None,
        emb_model: Any | None = None,
    ) -> None:
        if isinstance(reward_names, str):
            reward_names = (reward_names,)
        else:
            reward_names = tuple(reward_names)

        if not reward_names:
            raise ValueError("reward_names must contain at least one reward")

        unknown = [name for name in reward_names if name not in _REFERENCE_POINTS]
        if unknown:
            supported = ", ".join(_REFERENCE_POINTS)
            raise ValueError(f"Unknown peptide reward name(s): {unknown}. Supported names: {supported}")

        if len(set(reward_names)) != len(reward_names):
            raise ValueError("reward_names must not contain duplicates")

        if "binding_affinity" in reward_names and not protein_sequence:
            raise ValueError("protein_sequence is required when binding_affinity is selected")

        self.reward_names = reward_names
        self.zero_invalid = zero_invalid
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.analyzer = PeptideAnalyzer()
        self.rewards: dict[str, Any] = {}

        shared_emb_model = emb_model
        for name in reward_names:
            if name == "validity":
                continue

            reward_class = _REWARD_CLASSES[name]
            reward_kwargs: dict[str, Any] = {
                "base_path": base_path,
                "device": self.device,
                "emb_model": shared_emb_model,
            }
            if name == "binding_affinity":
                reward_kwargs["prot_seq"] = protein_sequence

            reward = reward_class(**reward_kwargs)
            self.rewards[name] = reward

            if shared_emb_model is None:
                shared_emb_model = getattr(reward, "emb_model", None)
                if shared_emb_model is None:
                    shared_emb_model = getattr(reward, "pep_model", None)

        ref_point = torch.tensor(
            [_REFERENCE_POINTS[name] for name in reward_names],
            dtype=torch.float32,
        )
        super().__init__(ref_point=ref_point, num_rew=len(reward_names))

    def _get_output_device(self, sample: D, latent: D) -> torch.device:
        for value in (sample, latent):
            value_device = getattr(value, "device", None)
            if value_device is not None:
                return torch.device(value_device)
        return self.device

    def _validity(self, sequences: Sequence[str], device: torch.device) -> torch.Tensor:
        values = []
        for sequence in sequences:
            try:
                values.append(bool(self.analyzer.is_peptide(sequence)))
            except Exception:
                values.append(False)
        return torch.tensor(values, dtype=torch.bool, device=device)

    @staticmethod
    def _as_reward_vector(values: Any, expected_size: int, device: torch.device, name: str) -> torch.Tensor:
        result = torch.as_tensor(values, dtype=torch.float32, device=device)
        if result.numel() != expected_size:
            raise ValueError(
                f"PeptidesGym reward {name!r} returned {result.numel()} values; expected {expected_size}"
            )
        return result.reshape(expected_size)

    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        sequences = tuple(sample.data)
        output_device = self._get_output_device(sample, latent)
        valids = self._validity(sequences, output_device)
        values = torch.zeros(
            (len(sequences), self.num_rew),
            dtype=torch.float32,
            device=output_device,
        )

        if self.zero_invalid:
            selected_indices = valids.nonzero(as_tuple=False).flatten()
            selected_sequences = [sequences[index] for index in selected_indices.cpu().tolist()]
            selected_sample = String(selected_sequences)
            selected_latent = latent[selected_indices.to(getattr(latent, "device", output_device))]
        else:
            selected_indices = torch.arange(len(sequences), device=output_device)
            selected_sample = sample
            selected_latent = latent

        for column, name in enumerate(self.reward_names):
            if name == "validity":
                values[:, column] = valids.to(dtype=torch.float32)
                continue

            if selected_indices.numel() == 0:
                continue

            reward_values, _ = self.rewards[name](selected_sample, selected_latent, **kwargs)
            reward_vector = self._as_reward_vector(
                reward_values,
                expected_size=selected_indices.numel(),
                device=output_device,
                name=name,
            )
            values[selected_indices, column] = reward_vector

        return values, {"valids": valids}


__all__ = ["PeptideMOReward"]
