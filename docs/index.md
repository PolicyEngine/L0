# L0 Regularization Documentation

```{toctree}
:hidden:
:maxdepth: 2

quickstart
theory
api/index
examples/index
contributing
```

## Welcome

L0 Regularization is a PyTorch package implementing the L0 penalty method from [Louizos, Welling, & Kingma (2017)](https://arxiv.org/abs/1712.01312). It enables:

- **Neural Network Sparsification**: Automatically prune weights during training
- **Intelligent Sampling**: Select optimal subsets from large datasets
- **Feature Selection**: Identify important features with differentiable gates

## Key Features

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🧠 Hard Concrete Distribution
:link: theory
:link-type: doc

Differentiable approximation of the L0 norm using stochastic gates
:::

:::{grid-item-card} 🔧 Sparse Layers
:link: api/layers
:link-type: doc

Drop-in replacements for PyTorch layers with built-in sparsity
:::

:::{grid-item-card} 📊 Smart Selection
:link: api/gates
:link-type: doc

Intelligent sample and feature selection for calibration tasks
:::

::::

## Quick Installation

```bash
pip install l0
```

## Simple Example

```python
from l0 import L0Linear, compute_l0l2_penalty

# Create a sparse layer
layer = L0Linear(100, 50, init_sparsity=0.7)

# Use in training
output = layer(input_data)
penalty = compute_l0l2_penalty(model)
loss = task_loss + penalty
```

## Why L0 Regularization?

Traditional regularization methods (L1, L2) approximate sparsity. L0 directly optimizes for the number of non-zero parameters:

- **True Sparsity**: Actually zeros out parameters, not just shrinks them
- **Differentiable**: Uses Hard Concrete distribution for gradient-based optimization
- **Hardware Efficient**: Sparse models run faster and use less memory

## PolicyEngine Integration

This package is designed to work seamlessly with PolicyEngine's calibration systems, particularly for intelligent household sampling in survey data.

## Getting Started

:::::{grid} 2
:gutter: 3

::::{grid-item}
**New to L0?**

Start with our [quickstart guide](quickstart) to understand the basics.
::::

::::{grid-item}
**Coming from PolicyEngine?**

See our [integration guide](examples/policyengine_integration) for migration instructions.
::::

:::::

## Citation

```bibtex
@article{louizos2017learning,
  title={Learning Sparse Neural Networks through L0 Regularization},
  author={Louizos, Christos and Welling, Max and Kingma, Diederik P},
  journal={arXiv preprint arXiv:1712.01312},
  year={2017}
}
```