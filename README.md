# L0 Regularization

A PyTorch implementation of L0 regularization for neural network sparsification and intelligent sampling, based on [Louizos, Welling, & Kingma (2017)](https://arxiv.org/abs/1712.01312).

## Features

- **Hard Concrete Distribution**: Differentiable approximation of L0 norm
- **Sparse Neural Network Layers**: L0Linear, L0Conv2d with automatic pruning
- **Intelligent Sampling**: Sample/feature selection gates for calibration
- **L0L2 Combined Penalty**: Recommended approach to prevent overfitting
- **Temperature Scheduling**: Annealing for improved convergence
- **TDD Development**: Comprehensive test coverage

## Installation

```bash
pip install l0
```

For development:
```bash
git clone https://github.com/PolicyEngine/L0.git
cd L0
pip install -e .[dev]
```

## Quick Start

### Neural Network Sparsification

```python
import torch
from l0 import L0Linear, compute_l0l2_penalty, TemperatureScheduler, update_temperatures

# Create a sparse model
class SparseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = L0Linear(784, 256, init_sparsity=0.5)
        self.fc2 = L0Linear(256, 10, init_sparsity=0.7)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SparseModel()
optimizer = torch.optim.Adam(model.parameters())
scheduler = TemperatureScheduler(initial_temp=1.0, final_temp=0.1)

# Training loop
for epoch in range(100):
    # Update temperature
    temp = scheduler.get_temperature(epoch)
    update_temperatures(model, temp)
    
    # Forward pass
    output = model(input_data)
    ce_loss = criterion(output, target)
    
    # Add L0L2 penalty
    penalty = compute_l0l2_penalty(model, l0_lambda=1e-3, l2_lambda=1e-4)
    loss = ce_loss + penalty
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Intelligent Sample Selection

```python
from l0 import SampleGate, HybridGate

# Pure L0 selection
gate = SampleGate(n_samples=10000, target_samples=1000)
selected_data, indices = gate.select_samples(data)

# Hybrid selection (25% L0, 75% random)
hybrid = HybridGate(
    n_items=10000,
    l0_fraction=0.25,
    random_fraction=0.75,
    target_items=1000
)
selected, indices, types = hybrid.select(data)
```

### Feature Selection

```python
from l0 import FeatureGate

# Select top features
gate = FeatureGate(n_features=1000, max_features=50)
selected_data, feature_indices = gate.select_features(data)

# Get feature importance
importance = gate.get_feature_importance()
```

## Integration with PolicyEngine

This package is designed to work with PolicyEngine's calibration system:

```python
# In policyengine-us-data or similar
from l0 import HardConcrete

# Use for household selection in CPS calibration
gates = HardConcrete(
    len(household_weights),
    temperature=0.25,
    init_mean=0.999  # Start with most households
)

# Apply gates during reweighting
masked_weights = weights * gates()
```

## Documentation

Full documentation available at: https://policyengine.github.io/L0/

## Testing

Run tests with:
```bash
pytest tests/ -v --cov=l0
```

## Acknowledgments

This implementation is inspired by and builds upon the [original L0 regularization code](https://github.com/AMLab-Amsterdam/L0_regularization) by AMLab Amsterdam, which accompanied the paper by Louizos et al. (2018).

## Citation

If you use this package, please cite:

```bibtex
@article{louizos2017learning,
  title={Learning Sparse Neural Networks through L0 Regularization},
  author={Louizos, Christos and Welling, Max and Kingma, Diederik P},
  journal={arXiv preprint arXiv:1712.01312},
  year={2017}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
