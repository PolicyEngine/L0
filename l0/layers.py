"""
L0 regularized neural network layers.

Implements sparse layers using the Hard Concrete distribution
for various layer types (Linear, Conv2d, etc.).
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import HardConcrete


class L0Linear(nn.Module):
    """
    Linear layer with L0 regularization using HardConcrete gates.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool
        Whether to include bias term
    temperature : float
        Temperature for HardConcrete distribution
    init_sparsity : float
        Initial sparsity level (1 - init_mean for gates)
    use_l2 : bool
        Whether to support L2 penalty computation (L0L2 regularization)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 0.5,
        init_sparsity: float = 0.5,
        use_l2: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_l2 = use_l2

        # Weight parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        # L0 gates for weights
        self.weight_gates = HardConcrete(
            out_features,
            in_features,
            temperature=temperature,
            init_mean=1.0 - init_sparsity,  # Convert sparsity to activation
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight and bias parameters."""
        nn.init.kaiming_normal_(self.weight, mode="fan_out")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, in_features)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features)
        """
        gates = self.weight_gates()
        masked_weight = self.weight * gates
        return F.linear(input, masked_weight, self.bias)

    def get_l0_penalty(self) -> torch.Tensor:
        """Get L0 penalty (expected number of active weights)."""
        return self.weight_gates.get_penalty()

    def get_l2_penalty(self) -> torch.Tensor:
        """Get L2 penalty (weight norm squared)."""
        if not self.use_l2:
            return torch.tensor(0.0, device=self.weight.device)
        return (self.weight**2).sum()

    def get_sparsity(self) -> float:
        """Get current sparsity level."""
        return self.weight_gates.get_sparsity()

    def extra_repr(self) -> str:
        """String representation."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class L0Conv2d(nn.Module):
    """
    2D Convolutional layer with L0 regularization.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple
        Size of the convolutional kernel
    stride : int or tuple
        Stride of the convolution
    padding : int or tuple
        Padding added to input
    temperature : float
        Temperature for HardConcrete distribution
    init_sparsity : float
        Initial sparsity level
    structured : bool
        If True, use channel-wise structured sparsity
    use_l2 : bool
        Whether to support L2 penalty computation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        temperature: float = 0.5,
        init_sparsity: float = 0.5,
        structured: bool = False,
        use_l2: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.structured = structured
        self.use_l2 = use_l2

        # Weight parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # L0 gates
        if structured:
            # Channel-wise gates for structured sparsity
            self.channel_gates = HardConcrete(
                out_channels,
                temperature=temperature,
                init_mean=1.0 - init_sparsity,
            )
        else:
            # Per-weight gates
            self.weight_gates = HardConcrete(
                out_channels,
                in_channels * self.kernel_size[0] * self.kernel_size[1],
                temperature=temperature,
                init_mean=1.0 - init_sparsity,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight and bias parameters."""
        nn.init.kaiming_normal_(self.weight, mode="fan_out")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width)

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if self.structured:
            # Apply channel-wise gates
            gates = self.channel_gates()
            gates = gates.view(-1, 1, 1, 1)  # Reshape for broadcasting
            masked_weight = self.weight * gates
        else:
            # Apply per-weight gates
            gates = self.weight_gates()
            gates = gates.view(self.weight.shape)
            masked_weight = self.weight * gates

        return F.conv2d(
            input, masked_weight, self.bias, self.stride, self.padding
        )

    def get_l0_penalty(self) -> torch.Tensor:
        """Get L0 penalty."""
        if self.structured:
            return self.channel_gates.get_penalty()
        else:
            return self.weight_gates.get_penalty()

    def get_l2_penalty(self) -> torch.Tensor:
        """Get L2 penalty."""
        if not self.use_l2:
            return torch.tensor(0.0, device=self.weight.device)
        return (self.weight**2).sum()

    def get_sparsity(self) -> float:
        """Get current sparsity level."""
        if self.structured:
            return self.channel_gates.get_sparsity()
        else:
            return self.weight_gates.get_sparsity()


class L0DepthwiseConv2d(nn.Module):
    """
    Depthwise 2D convolution with L0 regularization.

    Each input channel is convolved with its own set of filters.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        temperature: float = 0.5,
        init_sparsity: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        # Depthwise convolution: groups = in_channels
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        # Channel-wise gates for depthwise conv
        self.channel_gates = HardConcrete(
            in_channels,
            temperature=temperature,
            init_mean=1.0 - init_sparsity,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated channels."""
        gates = self.channel_gates()
        gates = gates.view(1, -1, 1, 1)  # Reshape for broadcasting

        # Apply gates to input channels
        gated_input = input * gates
        return self.depthwise_conv(gated_input)

    def get_l0_penalty(self) -> torch.Tensor:
        """Get L0 penalty."""
        return self.channel_gates.get_penalty()

    def get_sparsity(self) -> float:
        """Get current sparsity level."""
        return self.channel_gates.get_sparsity()


class SparseMLP(nn.Module):
    """
    Example MLP with L0 regularization on all layers.

    Parameters
    ----------
    input_dim : int
        Input dimension (e.g., 784 for MNIST)
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output dimension (e.g., 10 for MNIST)
    init_sparsity : float
        Initial sparsity level for all layers
    temperature : float
        Temperature for HardConcrete distribution
    use_l2 : bool
        Whether to use L0L2 combined regularization
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        init_sparsity: float = 0.5,
        temperature: float = 0.5,
        use_l2: bool = False,
    ):
        super().__init__()

        self.fc1 = L0Linear(
            input_dim,
            hidden_dim,
            init_sparsity=init_sparsity,
            temperature=temperature,
            use_l2=use_l2,
        )
        self.fc2 = L0Linear(
            hidden_dim,
            hidden_dim,
            init_sparsity=init_sparsity,
            temperature=temperature,
            use_l2=use_l2,
        )
        self.fc3 = L0Linear(
            hidden_dim,
            output_dim,
            init_sparsity=init_sparsity,
            temperature=temperature,
            use_l2=use_l2,
        )
        self.use_l2 = use_l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_l0_loss(self) -> torch.Tensor:
        """Get total L0 penalty across all layers."""
        l0_loss = torch.tensor(0.0)
        for module in self.modules():
            if isinstance(module, (L0Linear, L0Conv2d, L0DepthwiseConv2d)):
                l0_loss = l0_loss + module.get_l0_penalty()
        return l0_loss

    def get_l2_loss(self) -> torch.Tensor:
        """Get total L2 penalty across all layers."""
        if not self.use_l2:
            return torch.tensor(0.0)

        l2_loss = torch.tensor(0.0)
        for module in self.modules():
            if hasattr(module, "get_l2_penalty"):
                l2_loss = l2_loss + module.get_l2_penalty()
        return l2_loss

    def get_sparsity_stats(self) -> dict[str, dict[str, Any]]:
        """Get sparsity statistics for all layers."""
        stats = {}
        for name, module in self.named_modules():
            if hasattr(module, "get_sparsity") and hasattr(
                module, "get_l0_penalty"
            ):
                stats[name] = {
                    "sparsity": module.get_sparsity(),
                    "active_params": module.get_l0_penalty().item(),
                }
        return stats


def prune_model(model: nn.Module, threshold: float = 0.05) -> nn.Module:
    """
    Prune a model by zeroing weights with low activation probability.

    Parameters
    ----------
    model : nn.Module
        Model to prune
    threshold : float
        Probability threshold below which to prune weights

    Returns
    -------
    nn.Module
        Pruned model (modified in-place)
    """
    model.eval()  # Ensure deterministic gates

    for module in model.modules():
        if hasattr(module, "weight_gates"):
            with torch.no_grad():
                prob_active = module.weight_gates.get_active_prob()
                mask = (prob_active > threshold).float()

                # Reshape mask to match weight dimensions
                if len(module.weight.shape) > len(mask.shape):
                    mask = mask.view(module.weight.shape)

                # Zero out pruned weights
                module.weight.data *= mask

        elif hasattr(module, "channel_gates"):
            with torch.no_grad():
                prob_active = module.channel_gates.get_active_prob()
                mask = (prob_active > threshold).float()

                # Apply channel-wise masking
                if isinstance(module, L0Conv2d):
                    mask = mask.view(-1, 1, 1, 1)
                elif isinstance(module, L0DepthwiseConv2d):
                    mask = mask.view(1, -1, 1, 1)

                module.weight.data *= mask

    return model
