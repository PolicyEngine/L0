# L0 Regularization

[![PyPI version](https://badge.fury.io/py/l0-python.svg)](https://pypi.org/project/l0-python/)
[![CI](https://github.com/PolicyEngine/L0/actions/workflows/push.yml/badge.svg)](https://github.com/PolicyEngine/L0/actions)

A PyTorch implementation of L0 regularization based on [Louizos, Welling, & Kingma (2017)](https://arxiv.org/abs/1712.01312), with a **critical bug fix** to the original authors' implementation.

## Why This Package?

The original L0 implementation from AMLab-Amsterdam contains a bug in test-time gate computation where the temperature parameter is incorrectly omitted. This prevents proper sparsity and degrades performance. Our implementation corrects this error:

```python
# Original (buggy): gates never fully close
pi = sigmoid(log_alpha)

# Corrected: temperature division required for proper sparsity
pi = sigmoid(log_alpha / temperature)
```

This fix enables gates to achieve true 0/1 values, producing exact sparsity as intended by the L0 formulation.

## Installation

```bash
pip install l0-python
```

For development:
```bash
git clone https://github.com/PolicyEngine/L0.git
cd L0
pip install -e .[dev]
```

## Primary Use Case: Survey Calibration

This package was developed for PolicyEngine's survey calibration system, where we select a sparse subset of survey households while matching population targets.

```python
import numpy as np
from scipy import sparse as sp
from l0.calibration import SparseCalibrationWeights

# Setup: Q targets, N households
Q, N = 200, 10000
M = sp.random(Q, N, density=0.3, format="csr")  # Household characteristics
y = np.random.uniform(1e6, 1e8, size=Q)          # Population targets

# Initialize model
model = SparseCalibrationWeights(
    n_features=N,
    beta=0.35,
    gamma=-0.1,
    zeta=1.1,
    init_keep_prob=0.5,      # Start with 50% active probability
    init_weights=1.0,        # Or pass array of initial weights
    log_weight_jitter_sd=0.05,
    device="cuda",           # GPU acceleration
)

# Train with L0+L2 regularization
model.fit(
    M=M,
    y=y,
    lambda_l0=1e-6,          # Controls sparsity level
    lambda_l2=1e-8,          # Prevents weight explosion
    lr=0.15,
    epochs=2000,
    loss_type="relative",    # Scale-invariant loss
    verbose=True,
    verbose_freq=100,
)

# Get results
active = model.get_active_weights()
print(f"Selected {active['count']} of {N} households ({100*active['count']/N:.1f}%)")
print(f"Sparsity: {model.get_sparsity():.1%}")

# Predict calibrated totals
y_pred = model.predict(M)
```

### Key Features for Calibration

- **Non-negative weights**: All weights constrained to be positive via log-space parameterization
- **Sparse solutions**: L0 penalty directly minimizes the count of active weights
- **Relative loss**: Scale-invariant loss for targets spanning orders of magnitude
- **Group-wise averaging**: Balance loss contributions across target groups with different cardinalities
- **GPU support**: CUDA acceleration for large-scale problems

## Neural Network Sparsification

The package also supports traditional neural network pruning:

```python
import torch
from l0 import L0Linear, compute_l0l2_penalty, TemperatureScheduler, update_temperatures

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

for epoch in range(100):
    temp = scheduler.get_temperature(epoch)
    update_temperatures(model, temp)

    output = model(input_data)
    ce_loss = criterion(output, target)
    penalty = compute_l0l2_penalty(model, l0_lambda=1e-3, l2_lambda=1e-4)
    loss = ce_loss + penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Available Layers

- `L0Linear`: Fully connected layer with L0 gates on weights
- `L0Conv2d`: 2D convolution with channel-wise L0 gates
- `L0DepthwiseConv2d`: Depthwise convolution with L0 gates
- `SparseMLP`: Multi-layer perceptron with built-in L0 regularization

## Intelligent Sampling Gates

Standalone gates for sample/feature selection:

```python
from l0 import SampleGate, FeatureGate, HybridGate

# Select samples via learned gates
gate = SampleGate(n_samples=10000, target_samples=1000)
selected_data, indices = gate.select_samples(data)

# Select features
gate = FeatureGate(n_features=1000, max_features=50)
selected_data, indices = gate.select_features(data)

# Hybrid: combine L0 selection with random sampling
hybrid = HybridGate(
    n_items=10000,
    l0_fraction=0.25,    # 25% via learned L0 gates
    random_fraction=0.75 # 75% random for coverage
)
selected, indices, types = hybrid.select(data)
```

## How L0 Regularization Works

Unlike post-hoc pruning (setting small weights to zero), L0 regularization integrates sparsity into the training objective:

1. **Stochastic gates**: Each weight has a learned gate parameter controlling activation probability
2. **Hard Concrete distribution**: Enables differentiable 0/1 gate values during training
3. **Expected L0 penalty**: Minimizes the expected number of active gates
4. **Temperature annealing**: Gradually sharpens gates from soft to hard decisions

The result: the network learns *which* weights should be zero as part of optimization, not as a post-processing step.

## Testing

```bash
pytest tests/ -v --cov=l0
```

## Citation

If you use this package, please cite the original paper:

```bibtex
@article{louizos2017learning,
  title={Learning Sparse Neural Networks through L0 Regularization},
  author={Louizos, Christos and Welling, Max and Kingma, Diederik P},
  journal={arXiv preprint arXiv:1712.01312},
  year={2017}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
