# API Reference

```{note}
For complete auto-generated API documentation from docstrings, MyST Parser with Sphinx is recommended.
The current MyST command-line tool (mystmd) focuses on simplicity and doesn't yet support autodoc.
```

## Core Modules

### l0.distributions

#### HardConcrete

The core distribution for L0 regularization, providing a differentiable approximation of discrete gates.

```python
class HardConcrete(nn.Module):
    """
    Hard Concrete distribution for L0 regularization.
    
    Parameters
    ----------
    *gate_size : int
        Dimensions of the gate tensor
    temperature : float, default=0.5
        Temperature parameter controlling hardness
    init_mean : float, default=0.5
        Initial mean activation probability
    stretch : float, default=0.1
        Stretch parameter for exact zeros
    """
```

**Methods:**

- `forward(input_shape=None)`: Sample gates (stochastic in training, deterministic in eval)
- `get_penalty()`: Compute L0 penalty (expected number of active parameters)
- `get_sparsity()`: Get current sparsity level (fraction of zeros)
- `get_active_prob()`: Get probability of each gate being active
- `get_num_active()`: Count currently active gates

**Example:**

```python
from l0.distributions import HardConcrete

# Create gates for a weight matrix
gates = HardConcrete(256, 128, temperature=0.3, init_mean=0.8)

# Sample gates
z = gates()  # Shape: (256, 128)

# Get L0 penalty for loss
penalty = gates.get_penalty()  # Scalar tensor

# Check sparsity
sparsity = gates.get_sparsity()  # Float between 0 and 1
```

### l0.layers

#### L0Linear

Linear layer with L0 regularization on weights.

```python
class L0Linear(nn.Module):
    """
    Linear layer with L0 regularization.
    
    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool, default=True
        Include bias term
    temperature : float, default=0.5
        Temperature for Hard Concrete
    init_sparsity : float, default=0.5
        Initial sparsity level
    use_l2 : bool, default=False
        Enable L2 penalty computation
    """
```

**Example:**

```python
from l0.layers import L0Linear

# Create sparse linear layer
layer = L0Linear(784, 256, init_sparsity=0.9)

# Forward pass
x = torch.randn(32, 784)
output = layer(x)  # Shape: (32, 256)

# Get penalties
l0_penalty = layer.get_l0_penalty()
l2_penalty = layer.get_l2_penalty()  # If use_l2=True
```

#### L0Conv2d

2D Convolutional layer with L0 regularization.

```python
class L0Conv2d(nn.Module):
    """
    2D Convolution with L0 regularization.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple
        Size of convolution kernel
    stride : int or tuple, default=1
        Stride of convolution
    padding : int or tuple, default=0
        Padding added to input
    bias : bool, default=True
        Include bias term
    temperature : float, default=0.5
        Temperature parameter
    init_sparsity : float, default=0.5
        Initial sparsity
    structured : bool, default=False
        Use channel-wise structured sparsity
    use_l2 : bool, default=False
        Enable L2 penalty
    """
```

**Example with Structured Sparsity:**

```python
from l0.layers import L0Conv2d

# Channel-wise structured sparsity
conv = L0Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    structured=True,  # Prune entire channels
    init_sparsity=0.7
)

x = torch.randn(8, 64, 32, 32)
output = conv(x)  # Shape: (8, 128, 30, 30)
```

#### L0DepthwiseConv2d

Depthwise separable convolution with L0 regularization.

```python
class L0DepthwiseConv2d(nn.Module):
    """
    Depthwise 2D convolution with L0 regularization.
    
    Parameters
    ----------
    in_channels : int
        Number of input/output channels
    kernel_size : int or tuple
        Size of convolution kernel
    stride : int or tuple, default=1
        Stride of convolution
    padding : int or tuple, default=0
        Padding added to input
    bias : bool, default=True
        Include bias term
    temperature : float, default=0.5
        Temperature parameter
    init_sparsity : float, default=0.5
        Initial sparsity level
    """
```

#### SparseMLP

Example multi-layer perceptron with L0 regularization.

```python
class SparseMLP(nn.Module):
    """
    Example MLP with L0 regularization.
    
    Parameters
    ----------
    input_dim : int, default=784
        Input dimension
    hidden_dim : int, default=256
        Hidden layer dimension
    output_dim : int, default=10
        Output dimension
    init_sparsity : float, default=0.5
        Initial sparsity for all layers
    temperature : float, default=0.5
        Temperature for all layers
    use_l2 : bool, default=False
        Enable L0L2 combined regularization
    """
```

**Training Example:**

```python
from l0.layers import SparseMLP
import torch.nn.functional as F

model = SparseMLP(784, 256, 10, init_sparsity=0.8)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for x, y in dataloader:
    logits = model(x)
    
    # Task loss + L0 regularization
    task_loss = F.cross_entropy(logits, y)
    l0_loss = model.get_l0_loss()
    total_loss = task_loss + 1e-3 * l0_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Monitor sparsity
    stats = model.get_sparsity_stats()
    print(f"Layer sparsities: {stats}")
```

### l0.gates

Standalone gate modules for feature and sample selection.

#### L0Gate

Generic L0 gate for any tensor dimension.

```python
class L0Gate(nn.Module):
    """
    Generic L0 gate module.
    
    Parameters
    ----------
    size : int
        Number of gates
    temperature : float, default=0.5
        Temperature parameter
    init_mean : float, default=0.5
        Initial activation probability
    """
```

#### SampleGate

Gate for selecting samples from a dataset.

```python
class SampleGate(L0Gate):
    """
    L0 gate for sample selection.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    target_samples : int
        Target number of samples to select
    temperature : float, default=0.5
        Temperature parameter
    """
```

**Example:**

```python
from l0.gates import SampleGate

# Select ~100 samples from 1000
gate = SampleGate(n_samples=1000, target_samples=100)

# Select samples
data = torch.randn(1000, 50)
selected_data, indices = gate.select_samples(data)
print(f"Selected {len(indices)} samples")
```

#### FeatureGate

Gate for feature selection.

```python
class FeatureGate(L0Gate):
    """
    L0 gate for feature selection.
    
    Parameters
    ----------
    n_features : int
        Total number of features
    max_features : int
        Maximum features to select
    temperature : float, default=0.5
        Temperature parameter
    """
```

**Example:**

```python
from l0.gates import FeatureGate

# Select up to 10 features from 100
gate = FeatureGate(n_features=100, max_features=10)

# Feature selection with names
data = torch.randn(50, 100)
feature_names = [f"feat_{i}" for i in range(100)]

selected_data, selected_names = gate.select_features_with_names(
    data, feature_names
)
print(f"Selected features: {selected_names}")

# Get feature importance
importance = gate.get_feature_importance()
top_10 = torch.topk(importance, 10)
```

#### HybridGate

Combines L0 selection with random sampling.

```python
class HybridGate(nn.Module):
    """
    Hybrid selection combining L0 and random sampling.
    
    Parameters
    ----------
    n_items : int
        Total number of items
    l0_fraction : float
        Fraction selected via L0
    random_fraction : float
        Fraction selected randomly
    target_items : int
        Target total items to select
    temperature : float, default=0.5
        Temperature for L0 gates
    """
```

### l0.penalties

Utility functions for penalty computation and management.

#### Functions

```python
def compute_l0_penalty(model: nn.Module) -> torch.Tensor:
    """Compute total L0 penalty across all L0 layers."""

def compute_l2_penalty(model: nn.Module) -> torch.Tensor:
    """Compute total L2 penalty across all layers with weights."""

def compute_l0l2_penalty(
    model: nn.Module,
    l0_lambda: float = 1e-3,
    l2_lambda: float = 1e-4
) -> torch.Tensor:
    """Compute combined L0L2 penalty."""

def get_sparsity_stats(model: nn.Module) -> dict:
    """Get sparsity statistics for all L0 layers."""

def get_active_parameter_count(model: nn.Module) -> int:
    """Count total active parameters in model."""

def update_temperatures(model: nn.Module, temperature: float):
    """Update temperature for all L0 gates in model."""
```

#### TemperatureScheduler

Anneal temperature during training for better convergence.

```python
class TemperatureScheduler:
    """
    Temperature annealing scheduler.
    
    Parameters
    ----------
    initial_temp : float
        Starting temperature
    final_temp : float
        Final temperature
    anneal_epochs : int
        Number of epochs for annealing
    """
```

**Example:**

```python
from l0.penalties import TemperatureScheduler, update_temperatures

scheduler = TemperatureScheduler(
    initial_temp=1.0,
    final_temp=0.1,
    anneal_epochs=50
)

for epoch in range(100):
    temp = scheduler.get_temperature(epoch)
    update_temperatures(model, temp)
    # ... training loop ...
```

#### PenaltyTracker

Track penalties and sparsity during training.

```python
class PenaltyTracker:
    """Track penalties and statistics during training."""
    
    def log(self, name: str, value: float):
        """Log a metric value."""
    
    def get_stats(self, name: str) -> dict:
        """Get statistics for a metric."""
    
    def get_history(self, name: str) -> list:
        """Get full history of a metric."""
```

### Model Utilities

#### Pruning

```python
from l0.layers import prune_model

# Prune weights with activation probability < 0.05
pruned_model = prune_model(model, threshold=0.05)

# Pruned weights are set to exactly 0
# Model can be saved with sparse storage
```

## Complete Training Example

```python
import torch
import torch.nn.functional as F
from l0.layers import SparseMLP
from l0.penalties import (
    TemperatureScheduler,
    update_temperatures,
    PenaltyTracker
)

# Initialize model and training
model = SparseMLP(784, 256, 10, init_sparsity=0.9, use_l2=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
temp_scheduler = TemperatureScheduler(1.0, 0.1, 50)
tracker = PenaltyTracker()

# Training parameters
l0_lambda = 1e-3
l2_lambda = 1e-4

for epoch in range(100):
    # Update temperature
    temp = temp_scheduler.get_temperature(epoch)
    update_temperatures(model, temp)
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # Forward pass
        logits = model(x)
        
        # Compute losses
        task_loss = F.cross_entropy(logits, y)
        l0_loss = model.get_l0_loss()
        l2_loss = model.get_l2_loss()
        
        total_loss = task_loss + l0_lambda * l0_loss + l2_lambda * l2_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        tracker.log("task_loss", task_loss.item())
        tracker.log("l0_penalty", l0_loss.item())
        tracker.log("sparsity", list(model.get_sparsity_stats().values())[0]["sparsity"])
    
    # Print epoch statistics
    stats = model.get_sparsity_stats()
    print(f"Epoch {epoch}: Temp={temp:.3f}")
    for name, layer_stats in stats.items():
        print(f"  {name}: {layer_stats['sparsity']:.1%} sparse, "
              f"{layer_stats['active_params']:.0f} active")

# After training, prune the model
from l0.layers import prune_model
pruned = prune_model(model, threshold=0.05)
print(f"Final model has {get_active_parameter_count(pruned)} active parameters")
```