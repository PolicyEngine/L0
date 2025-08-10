"""
Hard Concrete distribution for L0 regularization.

Based on Louizos, Welling, & Kingma (2017): "Learning Sparse Neural Networks
through L0 Regularization" (https://arxiv.org/abs/1712.01312)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union


class HardConcrete(nn.Module):
    """
    Hard Concrete distribution for L0 regularization.
    
    This distribution enables differentiable approximation of the L0 norm
    by using stochastic gates that can be optimized via gradient descent.
    
    Parameters
    ----------
    input_dim : int
        Size of the first dimension (e.g., number of samples or features)
    output_dim : Optional[int]
        Size of the second dimension for 2D gates (e.g., for weight matrices)
    temperature : float
        Temperature parameter controlling the hardness of the distribution.
        Lower values make the distribution more discrete.
    stretch : float
        Stretch parameter for the hard concrete transformation
    init_mean : float
        Initial mean activation probability for the gates (0 to 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        temperature: float = 0.5,
        stretch: float = 0.1,
        init_mean: float = 0.5,
    ):
        super().__init__()
        
        # Set gate dimensions
        if output_dim is None:
            self.gate_size = (input_dim,)
        else:
            self.gate_size = (input_dim, output_dim)
        
        # Learnable parameters - logits for each gate
        self.qz_logits = nn.Parameter(torch.zeros(self.gate_size))
        
        # Distribution parameters
        self.temperature = temperature
        self.stretch = stretch
        self.gamma = -stretch  # Lower bound after stretching
        self.zeta = 1.0 + stretch  # Upper bound after stretching
        self.init_mean = init_mean
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize gate logits based on desired initial mean."""
        if self.init_mean is not None:
            # Convert mean probability to logit space
            init_val = math.log(self.init_mean / (1 - self.init_mean))
            self.qz_logits.data.fill_(init_val)
    
    def forward(
        self, input_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Sample or compute gates.
        
        Parameters
        ----------
        input_shape : Optional[Tuple[int, ...]]
            Shape of the input tensor for broadcasting gates if needed
        
        Returns
        -------
        torch.Tensor
            Gate values in [0, 1], either sampled (training) or deterministic (eval)
        """
        if self.training:
            gates = self._sample_gates()
        else:
            gates = self._deterministic_gates()
        
        # Broadcast gates if needed for convolution layers
        if input_shape is not None and len(input_shape) > len(gates.shape):
            # Add dimensions for spatial dims (e.g., height, width in Conv2d)
            for _ in range(len(input_shape) - len(gates.shape)):
                gates = gates.unsqueeze(-1)
        
        return gates
    
    def _sample_gates(self) -> torch.Tensor:
        """
        Sample gates using the reparameterization trick.
        
        Returns
        -------
        torch.Tensor
            Sampled gate values in [0, 1]
        """
        # Sample uniform noise (avoiding exact 0 and 1 for numerical stability)
        u = torch.zeros_like(self.qz_logits).uniform_(1e-8, 1.0 - 1e-8)
        
        # Apply the concrete distribution transformation
        # s = sigmoid((log(u) - log(1-u) + logits) / temperature)
        s = torch.log(u) - torch.log(1 - u) + self.qz_logits
        s = torch.sigmoid(s / self.temperature)
        
        # Stretch and clamp to create hard concrete
        s = s * (self.zeta - self.gamma) + self.gamma
        gates = torch.clamp(s, 0, 1)
        
        return gates
    
    def _deterministic_gates(self) -> torch.Tensor:
        """
        Compute deterministic gates for evaluation.
        
        Returns
        -------
        torch.Tensor
            Deterministic gate values in [0, 1]
        """
        # Use mean of the distribution
        probs = torch.sigmoid(self.qz_logits)
        
        # Apply stretching transformation
        gates = probs * (self.zeta - self.gamma) + self.gamma
        
        return torch.clamp(gates, 0, 1)
    
    def get_penalty(self) -> torch.Tensor:
        """
        Compute the expected L0 norm (number of active gates).
        
        Returns
        -------
        torch.Tensor
            Expected number of non-zero gates
        """
        # Shift logits to account for hard concrete bounds
        logits_shifted = self.qz_logits - self.temperature * math.log(
            -self.gamma / self.zeta
        )
        
        # Probability that gate is active (non-zero)
        prob_active = torch.sigmoid(logits_shifted)
        
        return prob_active.sum()
    
    def get_active_prob(self) -> torch.Tensor:
        """
        Get the probability that each gate is active.
        
        Returns
        -------
        torch.Tensor
            Probability of each gate being non-zero
        """
        logits_shifted = self.qz_logits - self.temperature * math.log(
            -self.gamma / self.zeta
        )
        return torch.sigmoid(logits_shifted)
    
    def get_sparsity(self) -> float:
        """
        Get the overall sparsity level (fraction of inactive gates).
        
        Returns
        -------
        float
            Sparsity level in [0, 1] where 1 means all gates are inactive
        """
        with torch.no_grad():
            prob_active = self.get_active_prob()
            return 1.0 - prob_active.mean().item()
    
    @torch.no_grad()
    def get_num_active(self) -> int:
        """
        Get the number of active gates (for monitoring).
        
        Returns
        -------
        int
            Number of gates that are likely active
        """
        prob_active = self.get_active_prob()
        return int((prob_active > 0.5).sum().item())
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"gate_size={self.gate_size}, temperature={self.temperature}, "
            f"stretch={self.stretch}, init_mean={self.init_mean}"
        )