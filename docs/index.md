# L0 Regularization for Neural Networks

This library provides a PyTorch implementation of L0 regularization using the Hard Concrete distribution, as introduced in {cite}`louizos2018learning`. L0 regularization enables learning truly sparse neural networks by penalizing the number of non-zero parameters rather than their magnitude.

## Key Features

- **True Sparsity**: Learn networks with actual zero weights, not just small values
- **Hard Concrete Distribution**: Differentiable approximation of discrete gate variables
- **Flexible Architecture Support**: Works with Linear, Conv2d, and custom layers
- **L0L2 Combined Regularization**: Option to combine L0 penalty with L2 weight decay
- **Structured Sparsity**: Support for channel-wise and filter-wise pruning
- **Standalone Gates**: Use L0 gates independently for feature/sample selection

## Why L0 Regularization?

Traditional regularization methods like L1 and L2 penalties encourage small weights but rarely produce exact zeros. L0 regularization directly penalizes the count of non-zero parameters:

```{math}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \|\theta\|_0
```

Where $\|\theta\|_0$ counts the number of non-zero parameters. Since this is non-differentiable, we use the Hard Concrete distribution to create a differentiable approximation through stochastic gates.

## The Hard Concrete Distribution

The Hard Concrete distribution {cite}`maddison2017concrete,jang2016categorical` provides a continuous relaxation of discrete Bernoulli variables. For each parameter, we learn a gate $z_i \in [0, 1]$ that determines whether the parameter is active:

1. **During Training**: Gates are sampled stochastically from the Hard Concrete distribution
2. **During Inference**: Gates become deterministic based on learned probabilities
3. **After Training**: Parameters with gate probabilities below a threshold can be pruned

The distribution is parameterized by:
- **Temperature** ($\tau$): Controls the "hardness" of the gates (lower = more binary-like)
- **Stretch** ($\gamma, \zeta$): Extends the support to $[\gamma, \zeta]$ to encourage exact zeros

## Quick Example

```python
import torch
from l0.layers import L0Linear

# Create a sparse linear layer
layer = L0Linear(
    in_features=784,
    out_features=256,
    temperature=0.5,
    init_sparsity=0.9  # Start with 90% sparsity
)

# Use in training
x = torch.randn(32, 784)
output = layer(x)

# Get L0 penalty for loss
l0_penalty = layer.get_l0_penalty()
loss = task_loss + lambda_l0 * l0_penalty

# Check current sparsity
print(f"Sparsity: {layer.get_sparsity():.2%}")
```

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/PolicyEngine/L0.git
```

## Documentation Structure

- [Installation Guide](installation.md): Detailed installation instructions and requirements
- [API Reference](api_reference.md): Complete API documentation for all modules
- [Examples](examples.md): Comprehensive examples and use cases

## Citation

If you use this library in your research, please cite the original L0 regularization paper:

```bibtex
@article{louizos2018learning,
  title={Learning sparse neural networks through {$L_0$} regularization},
  author={Louizos, Christos and Welling, Max and Kingma, Diederik P},
  journal={arXiv preprint arXiv:1712.01312},
  year={2018}
}
```

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/PolicyEngine/L0) for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.