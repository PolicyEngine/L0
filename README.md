# L0 Regularization

[![PyPI version](https://badge.fury.io/py/l0-python.svg)](https://pypi.org/project/l0-python/)
[![CI](https://github.com/PolicyEngine/L0/actions/workflows/push.yml/badge.svg)](https://github.com/PolicyEngine/L0/actions)

A PyTorch implementation of L0 regularization based on [Louizos, Welling, & Kingma (2017)](https://arxiv.org/abs/1712.01312), designed for survey calibration and sparse regression.

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

## Our Approach to Test-Time Gates

The original Hard Concrete formulation uses temperature (β) during training to control the sharpness of stochastic gates. At test time, there's a design choice: whether to include temperature in the deterministic gate computation.

We include temperature at test time:
```python
# Our approach: include temperature
z = sigmoid(log_alpha / beta) * (zeta - gamma) + gamma

# Alternative: omit temperature
z = sigmoid(log_alpha) * (zeta - gamma) + gamma
```

Including temperature produces sharper 0/1 decisions, which we find beneficial for achieving clean sparsity in our applications. See `examples/sparse_regression_demo.py` for a demonstration on a 4-variable regression problem.

## Primary Use Case: Survey Calibration

This package was developed for PolicyEngine's survey calibration, where we select a sparse subset of survey households while matching population targets.

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
    init_keep_prob=0.5,
    init_weights=1.0,
    log_weight_jitter_sd=0.05,
    device="cuda",
)

# Train with L0+L2 regularization
model.fit(
    M=M,
    y=y,
    lambda_l0=1e-6,
    lambda_l2=1e-8,
    lr=0.15,
    epochs=2000,
    loss_type="relative",
    verbose=True,
)

# Get results
active = model.get_active_weights()
print(f"Selected {active['count']} of {N} households")
print(f"Sparsity: {model.get_sparsity():.1%}")
```

### Key Features

- **Non-negative weights**: Constrained via log-space parameterization
- **L0 sparsity**: Directly minimizes the count of active weights
- **Relative loss**: Scale-invariant for targets spanning orders of magnitude
- **Group-wise averaging**: Balance loss across target groups with different sizes
- **GPU support**: CUDA acceleration for large problems

## Sparse Regression

For sparse linear regression with scipy sparse matrices:

```python
from scipy import sparse as sp
from l0.sparse import SparseL0Linear

# Sparse design matrix
X = sp.random(1000, 500, density=0.1, format="csr")
y = np.random.randn(1000)

model = SparseL0Linear(n_features=500)
model.fit(X, y, lambda_l0=0.001, epochs=1000)

# Get sparse coefficients
coef = model.get_coefficients(threshold=0.01)
```

## Example: Variable Selection

The `examples/sparse_regression_demo.py` script demonstrates L0 regularization on a simple problem where the true coefficients are `[1, 0, -2, 0]`:

```bash
python examples/sparse_regression_demo.py
```

Output:
```
True coefficients:        [ 1.  0. -2.  0.]
Recovered coefficients:   [ 1.039  0.    -2.069 -0.   ]
Gates:                    [1. 0. 1. 0.]
```

The model correctly identifies that only variables 1 and 3 contribute to the outcome.

## Testing

```bash
pytest tests/ -v --cov=l0
```

## Citation

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
