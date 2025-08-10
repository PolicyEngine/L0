"""
L0 Regularization Package

A PyTorch implementation of L0 regularization for neural network sparsification
and intelligent sampling, based on Louizos, Welling, & Kingma (2017).
"""

__version__ = "0.1.0"

from .distributions import HardConcrete
from .layers import L0Linear, L0Conv2d, L0DepthwiseConv2d, SparseMLP, prune_model
from .gates import L0Gate, SampleGate, FeatureGate, HybridGate
from .penalties import (
    compute_l0_penalty,
    compute_l2_penalty,
    compute_l0l2_penalty,
    get_sparsity_stats,
    get_active_parameter_count,
    TemperatureScheduler,
    update_temperatures,
    PenaltyTracker,
)

__all__ = [
    # Distributions
    "HardConcrete",
    # Layers
    "L0Linear",
    "L0Conv2d",
    "L0DepthwiseConv2d",
    "SparseMLP",
    "prune_model",
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