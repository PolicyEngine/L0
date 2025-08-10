"""
Standalone gate modules for sample and feature selection.

These gates can be used independently of neural network layers
for tasks like survey calibration and feature selection.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Union
import numpy as np
from .distributions import HardConcrete


class L0Gate(nn.Module):
    """
    Generic L0 gate for selecting items (samples, features, etc.).

    Parameters
    ----------
    size : int
        Number of items to gate
    temperature : float
        Temperature for HardConcrete distribution
    init_mean : float
        Initial mean activation probability
    """

    def __init__(
        self,
        size: int,
        temperature: float = 0.5,
        init_mean: float = 0.5,
    ):
        super().__init__()
        self.size = size
        self.temperature = temperature
        self.init_mean = init_mean

        # Create HardConcrete gates
        self.gates = HardConcrete(
            size,
            temperature=temperature,
            init_mean=init_mean,
        )

    def get_gates(self) -> torch.Tensor:
        """Get current gate values."""
        return self.gates()

    def apply(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Apply gates to a tensor along specified dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to gate
        dim : int
            Dimension along which to apply gates

        Returns
        -------
        torch.Tensor
            Gated tensor
        """
        gates = self.get_gates()

        # Reshape gates to match tensor dimension
        shape = [1] * len(x.shape)
        shape[dim] = -1
        gates = gates.view(shape)

        return x * gates

    def get_penalty(self) -> torch.Tensor:
        """Get L0 penalty (expected number of active gates)."""
        return self.gates.get_penalty()

    def get_active_indices(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get indices of active items.

        Parameters
        ----------
        threshold : float
            Threshold for considering a gate active

        Returns
        -------
        torch.Tensor
            Indices of active items
        """
        with torch.no_grad():
            gates = self.get_gates()
            active_mask = gates > threshold
            return torch.where(active_mask)[0]


class SampleGate(L0Gate):
    """
    L0 gate specifically for sample/observation selection.

    Useful for survey calibration and intelligent subsampling.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    target_samples : int
        Target number of samples to select
    temperature : float
        Temperature for HardConcrete distribution
    """

    def __init__(
        self,
        n_samples: int,
        target_samples: int,
        temperature: float = 0.25,
    ):
        # Initialize with mean set to achieve target
        init_mean = target_samples / n_samples
        super().__init__(
            size=n_samples,
            temperature=temperature,
            init_mean=init_mean,
        )
        self.n_samples = n_samples
        self.target_samples = target_samples

    def select_samples(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select samples from data.

        Parameters
        ----------
        data : torch.Tensor
            Data tensor of shape (n_samples, ...)

        Returns
        -------
        selected_data : torch.Tensor
            Selected samples
        indices : torch.Tensor
            Indices of selected samples
        """
        indices = self.get_active_indices()
        selected_data = data[indices]
        return selected_data, indices

    def select_weighted_samples(
        self, data: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select samples with associated weights.

        Parameters
        ----------
        data : torch.Tensor
            Data tensor of shape (n_samples, ...)
        weights : torch.Tensor
            Weight vector of shape (n_samples,)

        Returns
        -------
        selected_data : torch.Tensor
            Selected samples
        selected_weights : torch.Tensor
            Weights of selected samples (scaled by gates)
        indices : torch.Tensor
            Indices of selected samples
        """
        gates = self.get_gates()

        # Apply gates to weights
        gated_weights = weights * gates

        # Select non-zero weights
        active_mask = gated_weights > 1e-8
        indices = torch.where(active_mask)[0]

        selected_data = data[indices]
        selected_weights = gated_weights[indices]

        return selected_data, selected_weights, indices


class FeatureGate(L0Gate):
    """
    L0 gate for feature selection.

    Parameters
    ----------
    n_features : int
        Total number of features
    max_features : int
        Maximum number of features to select
    temperature : float
        Temperature for HardConcrete distribution
    """

    def __init__(
        self,
        n_features: int,
        max_features: int,
        temperature: float = 0.1,
    ):
        # Initialize to select approximately max_features
        init_mean = min(max_features / n_features, 0.99)
        super().__init__(
            size=n_features,
            temperature=temperature,
            init_mean=init_mean,
        )
        self.n_features = n_features
        self.max_features = max_features

    def select_features(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select features from data.

        Parameters
        ----------
        data : torch.Tensor
            Data tensor of shape (..., n_features)

        Returns
        -------
        selected_data : torch.Tensor
            Data with selected features
        feature_indices : torch.Tensor
            Indices of selected features
        """
        feature_indices = self.get_active_indices()
        selected_data = data[..., feature_indices]
        return selected_data, feature_indices

    def select_features_with_names(
        self, data: torch.Tensor, feature_names: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Select features and return their names.

        Parameters
        ----------
        data : torch.Tensor
            Data tensor of shape (..., n_features)
        feature_names : List[str]
            Names of all features

        Returns
        -------
        selected_data : torch.Tensor
            Data with selected features
        selected_names : List[str]
            Names of selected features
        """
        selected_data, indices = self.select_features(data)
        selected_names = [feature_names[i] for i in indices]
        return selected_data, selected_names

    def get_feature_importance(self) -> torch.Tensor:
        """
        Get feature importance scores based on gate probabilities.

        Returns
        -------
        torch.Tensor
            Importance score for each feature
        """
        with torch.no_grad():
            return self.gates.get_active_prob()


class HybridGate(nn.Module):
    """
    Hybrid selection combining L0 and random sampling.

    Useful for combining intelligent selection with random sampling
    for better coverage and reduced bias.

    Parameters
    ----------
    n_items : int
        Total number of items
    l0_fraction : float
        Fraction to select via L0 gates
    random_fraction : float
        Fraction to select randomly
    target_items : int
        Total target number of items
    temperature : float
        Temperature for L0 gates
    """

    def __init__(
        self,
        n_items: int,
        l0_fraction: float = 0.25,
        random_fraction: float = 0.75,
        target_items: Optional[int] = None,
        temperature: float = 0.25,
    ):
        super().__init__()

        assert l0_fraction + random_fraction <= 1.0

        self.n_items = n_items
        self.l0_fraction = l0_fraction
        self.random_fraction = random_fraction
        self.target_items = target_items or int(n_items * 0.2)

        # Number of items for each selection method
        self.n_l0 = int(self.target_items * l0_fraction)
        self.n_random = int(self.target_items * random_fraction)

        # L0 gates for intelligent selection
        self.l0_gate = SampleGate(
            n_samples=n_items,
            target_samples=self.n_l0,
            temperature=temperature,
        )

    def select(
        self,
        data: torch.Tensor,
        random_seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Select items using hybrid approach.

        Parameters
        ----------
        data : torch.Tensor
            Data tensor of shape (n_items, ...)
        random_seed : Optional[int]
            Random seed for reproducibility

        Returns
        -------
        selected_data : torch.Tensor
            Selected items
        indices : torch.Tensor
            Indices of selected items
        selection_type : np.ndarray
            Type of selection for each item ('l0' or 'random')
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # L0 selection
        l0_indices = self.l0_gate.get_active_indices()

        # Random selection (excluding L0 selected indices)
        all_indices = torch.arange(self.n_items)
        mask = torch.ones(self.n_items, dtype=torch.bool)
        mask[l0_indices] = False
        available_indices = all_indices[mask]

        # Randomly select from available indices
        n_random_actual = min(self.n_random, len(available_indices))
        if n_random_actual > 0:
            perm = torch.randperm(len(available_indices))[:n_random_actual]
            random_indices = available_indices[perm]
        else:
            random_indices = torch.tensor([], dtype=torch.long)

        # Combine indices
        all_selected = torch.cat([l0_indices, random_indices])
        all_selected = torch.sort(all_selected)[0]

        # Track selection type
        selection_type = np.array(
            ["l0"] * len(l0_indices) + ["random"] * len(random_indices)
        )

        # Reorder selection_type to match sorted indices
        sort_order = torch.cat(
            [
                torch.arange(len(l0_indices)),
                torch.arange(len(random_indices)) + len(l0_indices),
            ]
        )
        idx_map = {idx.item(): i for i, idx in enumerate(all_selected)}
        l0_positions = [idx_map[idx.item()] for idx in l0_indices]
        random_positions = [idx_map[idx.item()] for idx in random_indices]

        selection_type_sorted = np.empty(len(all_selected), dtype="<U6")
        for pos in l0_positions:
            selection_type_sorted[pos] = "l0"
        for pos in random_positions:
            selection_type_sorted[pos] = "random"

        selected_data = data[all_selected]

        return selected_data, all_selected, selection_type_sorted

    def get_penalty(self) -> torch.Tensor:
        """Get L0 penalty from the L0 component."""
        return self.l0_gate.get_penalty()
