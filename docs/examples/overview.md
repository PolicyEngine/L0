# Examples Overview

L0 regularization provides differentiable sparsity for any model trained with gradient-based optimization. While commonly used with neural networks, L0 works with any differentiable model including:

- Linear/logistic regression
- Gradient boosting (with differentiable trees)
- Matrix factorization
- Any custom PyTorch model

## Key Concepts

L0 regularization differs from L1/L2 by directly controlling the **number** of non-zero parameters rather than their magnitude. This makes it ideal for:

- **Feature selection**: Identifying the most important features
- **Model compression**: Reducing model size while maintaining accuracy
- **Interpretability**: Creating sparse, interpretable models
- **Calibration**: Selecting representative samples from large datasets

## How L0 Works

The L0 penalty counts non-zero parameters, but this isn't differentiable. We solve this using:

1. **Stochastic gates**: Each parameter has a learnable gate that controls if it's active
2. **Hard Concrete distribution**: Makes gates differentiable while producing hard 0/1 decisions
3. **Temperature annealing**: Gradually makes decisions more discrete during training

## Examples Structure

Our examples progress from simple to advanced:

1. **[Basic L0](basic_l0.ipynb)**: Simple linear regression with L0 regularization
2. **[Neural Networks](neural_networks.ipynb)**: CNNs and MLPs with structured sparsity
3. **[Feature Selection](feature_selection.ipynb)**: Using gates for feature importance
4. **[Comparisons](comparison.ipynb)**: L0 vs L1/L2 regularization
5. **[Advanced Techniques](advanced.ipynb)**: Temperature scheduling, hybrid selection

## Not Just for Neural Networks

While neural networks are a common use case, L0 regularization works with **any gradient-based optimization**:

```python
# Works with simple gradient descent
import torch
import torch.nn as nn
from l0.layers import L0Linear

# Simple linear model (no hidden layers)
model = L0Linear(n_features, 1, bias=False)

# Standard gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X)
    loss = mse_loss(predictions, y) + lambda_l0 * model.get_l0_penalty()
    loss.backward()  # Gradients flow through Hard Concrete gates
    optimizer.step()
```

The key requirement is that your optimization uses gradients - L0 provides a differentiable approximation to the non-differentiable sparsity constraint.