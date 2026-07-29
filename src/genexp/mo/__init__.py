"""Multi-objective rewards."""

from genexp.mo.base import CombinedRewards, MOReward
dxtb = []
try:
    from genexp.mo.dxtb import DXTBDipoleL2, DXTBEnergy, DXTBTask
    dxtb = ["DXTBDipoleL2", "DXTBEnergy", "DXTBTask"]
except ImportError:
    dxtb = []
    pass

from genexp.mo.zdt import ZDT1Torch, ZDT2Torch, ZDT3Torch, ZDT4Torch, ZDT6Torch

base = [
    "CombinedRewards",
    "MOReward",
    "ZDT1Torch",
    "ZDT2Torch",
    "ZDT3Torch",
    "ZDT4Torch",
    "ZDT6Torch",
]

base.extend(dxtb)
__all__ = base