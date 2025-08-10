"""
Penalty computation and management utilities for L0 regularization.

Provides functions for computing penalties, tracking sparsity,
and managing temperature schedules.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
import numpy as np
from collections import defaultdict


def compute_l0_penalty(model: nn.Module) -> torch.Tensor:
    """
    Compute total L0 penalty across all L0 layers in a model.

    Parameters
    ----------
    model : nn.Module
        Model containing L0 layers

    Returns
    -------
    torch.Tensor
        Total L0 penalty (expected number of active parameters)
    """
    penalty = torch.tensor(0.0)

    for module in model.modules():
        if hasattr(module, "get_l0_penalty"):
            penalty = penalty + module.get_l0_penalty()

    return penalty


def compute_l2_penalty(model: nn.Module) -> torch.Tensor:
    """
    Compute L2 penalty (weight decay) across all layers.

    Parameters
    ----------
    model : nn.Module
        Model to compute L2 penalty for

    Returns
    -------
    torch.Tensor
        Total L2 penalty (sum of squared weights)
    """
    penalty = torch.tensor(0.0)

    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            penalty = penalty + (param**2).sum()

    return penalty


def compute_l0l2_penalty(
    model: nn.Module,
    l0_lambda: float = 1e-3,
    l2_lambda: float = 1e-4,
) -> torch.Tensor:
    """
    Compute combined L0L2 penalty.

    This combination is recommended in practice to prevent overfitting
    while maintaining sparsity.

    Parameters
    ----------
    model : nn.Module
        Model to compute penalties for
    l0_lambda : float
        Weight for L0 penalty
    l2_lambda : float
        Weight for L2 penalty

    Returns
    -------
    torch.Tensor
        Combined penalty value
    """
    l0_penalty = compute_l0_penalty(model)
    l2_penalty = compute_l2_penalty(model)

    return l0_lambda * l0_penalty + l2_lambda * l2_penalty


def get_sparsity_stats(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Get sparsity statistics for all L0 layers in a model.

    Parameters
    ----------
    model : nn.Module
        Model containing L0 layers

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping layer names to their statistics
    """
    stats = {}

    for name, module in model.named_modules():
        if hasattr(module, "get_sparsity") and hasattr(
            module, "get_l0_penalty"
        ):
            active_params = module.get_l0_penalty().item()

            # Calculate total parameters
            if hasattr(module, "weight"):
                total_params = module.weight.numel()
            elif hasattr(module, "gates"):
                total_params = module.gates.gate_size
                if isinstance(total_params, tuple):
                    total_params = np.prod(total_params)
            else:
                total_params = active_params / (1 - module.get_sparsity())

            stats[name] = {
                "sparsity": module.get_sparsity(),
                "active_params": active_params,
                "total_params": total_params,
            }

    return stats


def get_active_parameter_count(model: nn.Module) -> int:
    """
    Get the total number of active parameters in the model.

    Parameters
    ----------
    model : nn.Module
        Model containing L0 layers

    Returns
    -------
    int
        Total number of active parameters
    """
    total = 0

    for module in model.modules():
        if hasattr(module, "get_l0_penalty"):
            total += int(module.get_l0_penalty().item())

    return total


class TemperatureScheduler:
    """
    Temperature scheduler for annealing during training.

    Gradually decreases temperature to make gates more discrete.

    Parameters
    ----------
    initial_temp : float
        Starting temperature
    final_temp : float
        Final temperature
    anneal_epochs : int
        Number of epochs over which to anneal
    schedule : str
        Type of schedule ('linear', 'exponential', 'cosine')
    """

    def __init__(
        self,
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        anneal_epochs: int = 100,
        schedule: str = "exponential",
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_epochs = anneal_epochs
        self.schedule = schedule

    def get_temperature(self, epoch: int) -> float:
        """
        Get temperature for a given epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number

        Returns
        -------
        float
            Temperature value
        """
        if epoch >= self.anneal_epochs:
            return self.final_temp

        progress = epoch / self.anneal_epochs

        if self.schedule == "linear":
            temp = (
                self.initial_temp
                - (self.initial_temp - self.final_temp) * progress
            )

        elif self.schedule == "exponential":
            log_temp = (
                np.log(self.initial_temp)
                - (np.log(self.initial_temp) - np.log(self.final_temp))
                * progress
            )
            temp = np.exp(log_temp)

        elif self.schedule == "cosine":
            temp = (
                self.final_temp
                + (self.initial_temp - self.final_temp)
                * (1 + np.cos(np.pi * progress))
                / 2
            )

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule}")

        return float(temp)


def update_temperatures(model: nn.Module, temperature: float) -> None:
    """
    Update temperature for all HardConcrete distributions in model.

    Parameters
    ----------
    model : nn.Module
        Model containing HardConcrete gates
    temperature : float
        New temperature value
    """
    for module in model.modules():
        # Check for HardConcrete gates in layers
        if hasattr(module, "weight_gates"):
            module.weight_gates.temperature = temperature
        if hasattr(module, "channel_gates"):
            module.channel_gates.temperature = temperature
        if hasattr(module, "gates"):
            if hasattr(module.gates, "temperature"):
                module.gates.temperature = temperature

        # Direct HardConcrete modules
        from .distributions import HardConcrete

        if isinstance(module, HardConcrete):
            module.temperature = temperature


class PenaltyTracker:
    """
    Track penalties and sparsity metrics during training.

    Useful for monitoring and visualization.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = defaultdict(list)

    def log(self, name: str, value: float) -> None:
        """
        Log a metric value.

        Parameters
        ----------
        name : str
            Metric name
        value : float
            Metric value
        """
        self.history[name].append(value)

    def get_history(self, name: str) -> List[float]:
        """
        Get history for a metric.

        Parameters
        ----------
        name : str
            Metric name

        Returns
        -------
        List[float]
            History of values
        """
        return self.history[name]

    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Parameters
        ----------
        name : str
            Metric name

        Returns
        -------
        Dict[str, float]
            Statistics (mean, min, max, last)
        """
        values = self.history[name]
        if not values:
            return {}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "last": values[-1],
        }

    def clear(self) -> None:
        """Clear all history."""
        self.history.clear()

    def save_to_csv(self, filepath: str) -> None:
        """
        Save history to CSV file.

        Parameters
        ----------
        filepath : str
            Path to save CSV
        """
        import pandas as pd

        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)

    def plot_history(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot metric history.

        Parameters
        ----------
        metrics : Optional[List[str]]
            Metrics to plot (all if None)
        save_path : Optional[str]
            Path to save figure
        """
        import matplotlib.pyplot as plt

        metrics = metrics or list(self.history.keys())

        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(10, 4 * len(metrics)), squeeze=False
        )

        for i, metric in enumerate(metrics):
            values = self.history[metric]
            axes[i, 0].plot(values)
            axes[i, 0].set_title(metric)
            axes[i, 0].set_xlabel("Step")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
