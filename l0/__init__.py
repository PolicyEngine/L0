"""
L0 Regularization Package

A PyTorch implementation of L0 regularization for neural network sparsification
and intelligent sampling, based on Louizos, Welling, & Kingma (2017).
"""

__version__ = "0.5.0"

from .distributions import HardConcrete
from .gates import FeatureGate, HybridGate, L0Gate, SampleGate
from .layers import (
    L0Conv2d,
    L0DepthwiseConv2d,
    L0Linear,
    SparseMLP,
    prune_model,
)
from .penalties import (
    PenaltyTracker,
    TemperatureScheduler,
    compute_l0_penalty,
    compute_l0l2_penalty,
    compute_l2_penalty,
    get_active_parameter_count,
    get_sparsity_stats,
    update_temperatures,
)
from .sparse import SparseL0Linear

__all__ = [
    # Distributions
    "HardConcrete",
    # Layers
    "L0Linear",
    "L0Conv2d",
    "L0DepthwiseConv2d",
    "SparseMLP",
    "prune_model",
    # Sparse
    "SparseL0Linear",
    # Gates
    "L0Gate",
    "SampleGate",
    "FeatureGate",
    "HybridGate",
    # Penalties
    "compute_l0_penalty",
    "compute_l2_penalty",
    "compute_l0l2_penalty",
    "get_sparsity_stats",
    "get_active_parameter_count",
    "TemperatureScheduler",
    "update_temperatures",
    "PenaltyTracker",
]
