from typing import Any

import torch
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.rewards.utils import graph_to_mols, is_not_fragmented, is_valid, safe_mmff_relax
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


class MolecularMetrics(MOReward[DDGraph]):
    def __init__(self, do_relax: bool = True, valid_2d: str = "none", valid_3d: str = "none", invalid_val: float = 0.0) -> None:
        RDLogger.DisableLog("rdApp.*")
        self.relax = safe_mmff_relax if do_relax else lambda mol: mol

        if valid_3d != "none":
            full_bust = "mol" if valid_3d == "full" else "mol_fast"
            self.buster = PoseBusters(config=full_bust, max_workers=None)
        else:
            self.buster = None

        self.validate_2d: bool = valid_2d == "full"
        self.invalid_val = invalid_val

        if self.invalid_val > 0:
            raise ValueError("invalid_val must be <= 0 for MolecularMetrics")

        self.num_rew = 3
        self.ref_point = torch.full((self.num_rew,), fill_value=self.invalid_val, dtype=torch.float32)

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        return (10 - sa) / 9

    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        mols = graph_to_mols(sample)
        valid_mols: list[Chem.Mol] = []
        valid_indices: list[int] = []

        for i, mol in enumerate(mols):
            if not self.validate_2d or (is_valid(mol) and is_not_fragmented(mol)):
                try:
                    relaxed_mol = self.relax(mol)
                except Exception:  # noqa: BLE001
                    relaxed_mol = None
                if relaxed_mol is not None:
                    valid_mols.append(relaxed_mol)
                    valid_indices.append(i)

        rewards = torch.full((len(sample), self.num_rew), fill_value=self.invalid_val, device=sample.device, dtype=torch.float32)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)

        if not valid_mols:
            return rewards, {"valids": valids}

        if self.buster is not None:
            posebuster_valids = self.buster.bust(valid_mols).all(axis=1).values  # ty: ignore[unresolved-attribute]
            valid_mols = [mol for mol, is_valid in zip(valid_mols, posebuster_valids) if is_valid]
            valid_indices = [idx for idx, is_valid in zip(valid_indices, posebuster_valids) if is_valid]

        xtb_results = parallel_xtb(valid_mols) if valid_mols else []

        for idx, result in zip(valid_indices, xtb_results):
            if result is None:
                continue
            mol = mols[idx]
            rewards[idx, 0] = float(self.calculate_qed(mol))
            rewards[idx, 1] = float(self.calculate_sa(mol))
            rewards[idx, 2] = float(result.dipole_moment) / 20.0
            valids[idx] = True

        return rewards, {"valids": valids}


class TopologyMetrics(MOReward[DDGraph]):
    def __init__(self, valid_2d: str = "none", valid_3d: str = "none", invalid_val: float = 0.0) -> None:
        RDLogger.DisableLog("rdApp.*")

        if valid_3d != "none":
            full_bust = "mol" if (valid_3d == "full") else "mol_fast"
            self.buster = PoseBusters(config=full_bust, max_workers=None)
        else:
            self.buster = None

        self.validate_2d: bool = valid_2d == "full"
        self.invalid_val = invalid_val
        
        if self.invalid_val > 0:
            raise ValueError("invalid_val must be <= 0 for TopologyMetrics")

        self.num_rew = 2
        self.ref_point = torch.full((self.num_rew,), fill_value=self.invalid_val, dtype=torch.float32)

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(_remove_hydrogens(rdmol))

    @staticmethod
    def calculate_sa(rdmol):
        rdmol = _remove_hydrogens(rdmol)
        sa = calculateScore(rdmol)
        sa_n = (10 - sa) / 9
        return sa_n


    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        mols = graph_to_mols(sample)
        # is_valid mutates its input, including its implicit-H settings. Keep the
        # explicit-H molecules for validation and separate heavy-atom copies for
        # QED/SA scoring.
        score_mols = [_remove_hydrogens(Chem.Mol(mol)) for mol in mols]
        valid_mols: list[Chem.Mol] = []
        valid_indices: list[int] = []

        for i, mol in enumerate(mols):
            if not self.validate_2d or (is_valid(mol) and is_not_fragmented(mol)):
                valid_mols.append(mol)
                valid_indices.append(i)

        rewards = torch.full((len(sample), self.num_rew), fill_value=self.invalid_val, device=sample.device, dtype=torch.float32)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)

        if not valid_mols:
            return rewards, {"valids": valids}

        if self.buster is not None:
            posebuster_valids = self.buster.bust(valid_mols).all(axis=1).values  # ty: ignore[unresolved-attribute]
        else:
            posebuster_valids = [True] * len(valid_mols)
            
        valid_indices = [idx for idx, is_valid in zip(valid_indices, posebuster_valids) if is_valid]

        for idx in valid_indices:
            mol = score_mols[idx]
            rewards[idx, 0] = float(self.calculate_qed(mol))
            rewards[idx, 1] = float(self.calculate_sa(mol))
            valids[idx] = True

        return rewards, {"valids": valids}


class RDkitReward(MOReward[DDGraph]):
    def __init__(self, rewards: list[str], valid_2d: str = "none", valid_3d: str = "none", invalid_val: float = 0.0) -> None:
        RDLogger.DisableLog("rdApp.*")
        
        if valid_3d != "none":
            full_bust = "mol" if (valid_3d == "full") else "mol_fast"
            self.buster = PoseBusters(config=full_bust, max_workers=None)
        else:
            self.buster = None
            
        self.validate_2d: bool = (valid_2d == "full")
        if invalid_val > 0:
            raise ValueError("invalid_val must be <= 0 for RDkitReward")

        self.rewards = rewards
        self.num_rew = len(rewards)
        self.invalid_val = invalid_val
        self.ref_point = torch.full((self.num_rew,), fill_value=self.invalid_val, dtype=torch.float32)

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        sa_n = (10 - sa) / 9
        return sa_n
    
    def evaluate_reward(self, mol: Chem.Mol) -> torch.Tensor:
        reward_values = []
        for reward_name in self.rewards:
            if reward_name == "qed":
                reward_values.append(self.calculate_qed(mol))
            elif reward_name == "sa":
                reward_values.append(self.calculate_sa(mol))
            else:
                raise ValueError(f"Unknown reward: {reward_name}")
        return torch.tensor(reward_values, dtype=torch.float32)

    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        mols = graph_to_mols(sample)

        valid_mols: list[Chem.Mol] = []
        valid_indices: list[int] = []
        
        for i, mol in enumerate(mols):
            if not self.validate_2d or (is_valid(mol) and is_not_fragmented(mol)):
                valid_mols.append(mol)
                valid_indices.append(i)

        rewards = torch.full((len(sample), self.num_rew), fill_value=self.invalid_val, device=sample.device, dtype=torch.float32)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)

        if not valid_mols:
            return rewards, {"valids": valids}

        if self.buster is not None:
            posebuster_valids = self.buster.bust(valid_mols).all(axis=1).values  # ty: ignore[unresolved-attribute]
        else:
            posebuster_valids = [True] * len(valid_mols)
            
        valid_indices = [idx for idx, is_valid in zip(valid_indices, posebuster_valids) if is_valid]

        for idx in valid_indices:
            mol = mols[idx]
            rewards[idx] = self.evaluate_reward(mol)
            valids[idx] = True

        return rewards, {"valids": valids}
