# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

L0 is a PyTorch package implementing L0 regularization from Louizos, Welling, & Kingma (2017) [https://arxiv.org/abs/1712.01312]. It provides differentiable sparsity for neural networks and intelligent sampling through the Hard Concrete distribution.

## Current Implementation Status

### Completed Modules
- **l0/distributions.py**: HardConcrete distribution (core L0 mechanism)
- **l0/layers.py**: L0Linear, L0Conv2d, L0DepthwiseConv2d, SparseMLP
- **l0/gates.py**: L0Gate, SampleGate, FeatureGate, HybridGate
- **l0/penalties.py**: L0/L2/L0L2 penalties, TemperatureScheduler, PenaltyTracker
- **tests/**: Comprehensive test coverage using TDD approach
- **CI/CD**: GitHub Actions workflow for Python 3.13

### Package Structure
```
l0/
├── l0/
│   ├── __init__.py         # Main exports
│   ├── distributions.py    # HardConcrete distribution
│   ├── layers.py           # Neural network layers with L0
│   ├── gates.py            # Standalone gates for selection
│   └── penalties.py        # Penalty computation and utilities
├── tests/
│   ├── test_distributions.py
│   ├── test_layers.py
│   ├── test_gates.py
│   └── test_penalties.py
├── docs/                   # Jupyter Book documentation (pending)
├── examples/               # Example notebooks (pending)
├── .github/workflows/ci.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

## Development Commands

```bash
# Install for development (Python 3.13 required)
pip install -e .[dev]

# Run all tests
pytest tests/ -v --cov=l0

# Run specific test
pytest tests/test_layers.py::TestL0Linear -v

# Format code (79 char line length)
black . -l 79

# Check formatting
black . -l 79 --check

# Lint with ruff
ruff check .

# Type checking
mypy l0/

# Build documentation (when ready)
jupyter-book build docs
```

## Key Design Decisions

### 1. L0L2 Combined Penalty
Based on research, pure L0 can overfit. The package defaults to supporting L0L2:
```python
layer = L0Linear(100, 50, use_l2=True)
penalty = compute_l0l2_penalty(model, l0_lambda=1e-3, l2_lambda=1e-4)
```

### 2. Temperature Scheduling
Temperature annealing is critical for convergence:
```python
scheduler = TemperatureScheduler(
    initial_temp=2.0,
    final_temp=0.1,
    anneal_epochs=100,
    schedule='exponential'  # or 'linear', 'cosine'
)
```

### 3. Hybrid Selection
For calibration tasks, combine L0 with random sampling:
```python
gate = HybridGate(
    n_items=10000,
    l0_fraction=0.25,    # 25% intelligent selection
    random_fraction=0.75  # 75% random for coverage
)
```

## Integration with PolicyEngine

This package is designed to replace the L0 implementation in policyengine-us-data:

```python
# Current usage in policyengine-us-data
from l0 import HardConcrete

# For CPS household selection
gates = HardConcrete(
    len(household_weights),
    temperature=0.25,      # Low temp for hard decisions
    init_mean=0.999        # Start with most households active
)

# L0 penalty targeting 20k-25k households
l0_lambda = 5.0e-07  # Tuned value from PolicyEngine
```

## Testing Philosophy

- **TDD**: Tests written before implementation
- **Comprehensive Coverage**: All public APIs tested
- **Property Testing**: Verify mathematical properties (bounds, gradients)
- **Integration Tests**: Full training loops in test suite

## Code Standards

- **Python 3.13**: Required for latest features
- **Black Formatter**: 79-character line length (PolicyEngine standard)
- **Type Hints**: All public functions fully typed
- **Docstrings**: NumPy style with examples
- **Imports**: Grouped (stdlib, third-party, local) and alphabetized

## Common Tasks

### Adding a New Layer Type
1. Write tests in `tests/test_layers.py`
2. Implement in `l0/layers.py` inheriting from `nn.Module`
3. Add `get_l0_penalty()` and `get_sparsity()` methods
4. Export in `l0/__init__.py`
5. Update README with usage example

### Debugging Sparsity Issues
```python
# Monitor sparsity during training
tracker = PenaltyTracker()
stats = get_sparsity_stats(model)
print(f"Layer sparsity: {stats}")
print(f"Active params: {get_active_parameter_count(model)}")
```

### Preparing for PolicyEngine Integration
1. Ensure temperature=0.25 works well (PolicyEngine default)
2. Test with init_mean=0.999 (PolicyEngine's high initial activation)
3. Verify gradient stability with l0_lambda=5e-07 range

## Documentation Strategy

Using Jupyter Book (next.jupyterbook.org) for interactive docs:
- Examples as executable notebooks
- API reference from docstrings
- Integration guides for PolicyEngine
- Visualization of sparsity patterns

## Future Enhancements

- JAX backend support for better PolicyEngine integration
- Automatic hyperparameter tuning for target sparsity
- Structured pruning patterns (channel, block, attention heads)
- Integration with quantization for further compression