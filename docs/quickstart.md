# Quick Start Guide

This guide will help you get started with L0 regularization in 5 minutes.

## Installation

```bash
pip install l0
```

Or for development:

```bash
git clone https://github.com/PolicyEngine/L0.git
cd L0
pip install -e .[dev]
```

## Basic Concepts

L0 regularization uses **stochastic gates** to learn which parameters to keep:

1. Each weight has an associated gate (0 or 1)
2. Gates are sampled from a Hard Concrete distribution
3. The distribution parameters are learned via backpropagation
4. Temperature controls how "hard" (binary) the gates are

## Your First Sparse Model

```python
import torch
import torch.nn as nn
from l0 import L0Linear, compute_l0l2_penalty

class SparseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Replace nn.Linear with L0Linear
        self.fc1 = L0Linear(784, 256, init_sparsity=0.5)
        self.fc2 = L0Linear(256, 128, init_sparsity=0.7)
        self.fc3 = L0Linear(128, 10, init_sparsity=0.9)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Create model
model = SparseNet()

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    # Forward pass
    output = model(input_batch)
    
    # Task loss (e.g., cross-entropy)
    task_loss = criterion(output, targets)
    
    # L0L2 regularization (recommended)
    reg_loss = compute_l0l2_penalty(model, l0_lambda=1e-3, l2_lambda=1e-4)
    
    # Total loss
    loss = task_loss + reg_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Temperature Scheduling

Lower temperature makes gates more binary. Schedule it during training:

```python
from l0 import TemperatureScheduler, update_temperatures

scheduler = TemperatureScheduler(
    initial_temp=1.0,
    final_temp=0.1,
    anneal_epochs=50
)

for epoch in range(100):
    # Update temperature
    temp = scheduler.get_temperature(epoch)
    update_temperatures(model, temp)
    
    # Rest of training loop...
```

## Monitoring Sparsity

Track how sparse your model becomes:

```python
from l0 import get_sparsity_stats

stats = get_sparsity_stats(model)
for layer_name, info in stats.items():
    print(f"{layer_name}: {info['sparsity']:.1%} sparse, "
          f"{info['active_params']:.0f}/{info['total_params']} active")
```

## Sample Selection

Use L0 for intelligent data sampling:

```python
from l0 import SampleGate

# Select 1000 from 10000 samples
gate = SampleGate(n_samples=10000, target_samples=1000)

# Optimize gates based on some objective
optimizer = torch.optim.Adam(gate.parameters())

for _ in range(100):
    selected_data, indices = gate.select_samples(full_data)
    loss = compute_selection_loss(selected_data, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Next Steps

- Read about the [theory](theory) behind L0 regularization
- Explore the full [API reference](api/index)
- See complete [examples](examples/index)
- Learn about [PolicyEngine integration](examples/policyengine_integration)