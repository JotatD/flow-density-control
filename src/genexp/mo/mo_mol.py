from typing import Any

import torch
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.rewards.utils import graph_to_mols, is_not_fragmented, is_valid
from diffusiongym.molecules.rewards.xtb import parallel_xtb
from posebusters import PoseBusters
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

# from src.utils import write_sdf_file
from genexp.mo.base import MOReward
from genexp.mo.sa_score import calculateScore


def _remove_hydrogens(rdmol: Chem.Mol) -> Chem.Mol:
    """Return the conventional heavy-atom representation when possible."""
    try:
        return Chem.RemoveHs(rdmol)
    except Exception:
        # Generated invalid molecules should be rejected by the existing
        # validity filters rather than crashing the whole reward batch here.
        return rdmol


class RDkitReward(MOReward[DDGraph]):
    supported_rewards = {"qed", "sa", "dipole"}

    def __init__(self, rewards: list[str], valid_2d: str = "none", valid_3d: str = "none", invalid_val: float = 0.0) -> None:
        RDLogger.DisableLog("rdApp.*")
        
        if valid_3d != "none":
            full_bust = "mol" if (valid_3d == "full") else "mol_fast"
            self.buster = PoseBusters(config=full_bust, max_workers=None)
        else:
            self.buster = None
            
        self.validate_2d: bool = valid_2d == "full"
        if invalid_val > 0:
            raise ValueError(f"invalid_val must be <= 0 for {type(self).__name__}")

        self.reward_names = list(rewards)
        unknown_rewards = [reward for reward in self.reward_names if reward not in self.supported_rewards]
        if unknown_rewards:
            raise ValueError(f"Unknown rewards: {unknown_rewards}")

        self.rewards = self.reward_names
        self.num_rew = len(self.reward_names)
        self.invalid_val = invalid_val
        self.ref_point = torch.full((self.num_rew,), fill_value=self.invalid_val, dtype=torch.float32)

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        rdmol = _remove_hydrogens(rdmol)
        sa = calculateScore(rdmol)
        return (10 - sa) / 9
    
    def evaluate_reward(self, mol: Chem.Mol, xtb_result: Any = None) -> torch.Tensor:
        reward_values = []
        for reward_name in self.reward_names:
            if reward_name == "qed":
                reward_values.append(self.calculate_qed(mol))
            elif reward_name == "sa":
                reward_values.append(self.calculate_sa(mol))
            elif reward_name == "dipole":
                if xtb_result is None:
                    raise ValueError("dipole reward requires a successful xTB result")
                reward_values.append(float(xtb_result.dipole_moment) / 20.0)
        return torch.tensor(reward_values, dtype=torch.float32)

    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        mols = graph_to_mols(sample)
        return self.compute(mols, sample.device)

    def compute(self, mols: list[Chem.Mol], device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
        valid_mols: list[Chem.Mol] = []
        valid_indices: list[int] = []
        
        for i, mol in enumerate(mols):
            if not self.validate_2d or (is_valid(mol) and is_not_fragmented(mol)):
                valid_mols.append(mol)
                valid_indices.append(i)

        rewards = torch.full((len(mols), self.num_rew), fill_value=self.invalid_val, device=device, dtype=torch.float32)
        valids = torch.zeros(len(mols), device=device, dtype=torch.bool)

        if not valid_mols:
            return rewards, {"valids": valids}

        if self.buster is not None:
            posebuster_valids = self.buster.bust(valid_mols).all(axis=1).values  # ty: ignore[unresolved-attribute]
            valid_mols = [mol for mol, is_valid in zip(valid_mols, posebuster_valids) if is_valid]
            valid_indices = [idx for idx, is_valid in zip(valid_indices, posebuster_valids) if is_valid]

        uses_dipole = "dipole" in self.reward_names
        xtb_results = parallel_xtb(valid_mols) if uses_dipole and valid_mols else [None] * len(valid_mols)

        for idx, xtb_result in zip(valid_indices, xtb_results):
            if uses_dipole and xtb_result is None:
                continue
            rewards[idx] = self.evaluate_reward(mols[idx], xtb_result).to(device)
            valids[idx] = True

        return rewards, {"valids": valids}


class TopologyMetrics(RDkitReward):
    def __init__(self, valid_2d: str = "none", valid_3d: str = "none", invalid_val: float = 0.0) -> None:
        super().__init__(["qed", "sa"], valid_2d=valid_2d, valid_3d=valid_3d, invalid_val=invalid_val)


class MolecularMetrics(RDkitReward):
    def __init__(self, valid_2d: str = "none", valid_3d: str = "none", invalid_val: float = 0.0) -> None:
        super().__init__(["qed", "sa", "dipole"], valid_2d=valid_2d, valid_3d=valid_3d, invalid_val=invalid_val)
