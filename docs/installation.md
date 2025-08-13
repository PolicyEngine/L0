# Installation Guide

## Requirements

- Python 3.10 or higher (tested up to 3.13)
- PyTorch 2.0 or higher
- NumPy

## Installing from GitHub

Install the latest version directly from GitHub:

```bash
pip install git+https://github.com/PolicyEngine/L0.git
```

## Installing from Source

For the latest development version or to contribute:

```bash
# Clone the repository
git clone https://github.com/PolicyEngine/L0.git
cd L0

# Install in development mode (recommended: use uv)
uv pip install -e .

# Or with make
make install
```

## Installing with Development Dependencies

If you want to contribute or run tests:

```bash
# Install with all development dependencies (using uv)
uv pip install -e ".[dev]"

# Or using make
make install-dev
```

This installs additional tools for:
- Testing: pytest, pytest-cov
- Code quality: black, ruff, mypy
- Documentation: jupyter-book, mystmd

## Verifying Installation

You can verify your installation by running:

```python
import l0
print(l0.__version__)

# Test basic functionality
from l0.layers import L0Linear
layer = L0Linear(10, 5)
print(f"Created L0Linear layer: {layer}")
```

## Running Tests

After installation, you can run the test suite:

```bash
# Run all tests
make test

# Or directly with pytest
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=l0 --cov-report=html
```

## Building Documentation

To build the documentation locally:

```bash
# Using make (recommended)
make docs-serve  # Start development server at http://localhost:3000
make docs        # Build static HTML

# Or using MyST directly
cd docs && myst start      # Start development server
cd docs && myst build --html  # Build static site
```

The documentation will be available at `http://localhost:3000` (for development server) or in `docs/_build/` (for static build).

## Troubleshooting

### CUDA Support

L0 works with both CPU and CUDA tensors. If you have CUDA installed:

```python
import torch
from l0.layers import L0Linear

# Check CUDA availability
if torch.cuda.is_available():
    layer = L0Linear(10, 5).cuda()
    x = torch.randn(32, 10).cuda()
    output = layer(x)
    print("CUDA acceleration enabled")
```

### Common Issues

1. **ImportError: No module named 'l0'**
   - Ensure you're in the correct environment
   - If installed from source, make sure you used `-e` flag

2. **PyTorch version compatibility**
   - L0 requires PyTorch 2.0+
   - Update PyTorch: `pip install torch>=2.0`

3. **Type hints not working**
   - L0 uses Python 3.10+ union syntax (`|`)
   - Ensure you're using Python 3.10 or higher

## Platform Support

L0 is tested on:
- Linux (Ubuntu 20.04+)
- macOS (11.0+)
- Windows (10+)

All platforms support both CPU and GPU computation.

## Development Setup

For active development:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/L0.git
cd L0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (optional)
pre-commit install

# Run formatters
make format

# Run linters
make lint

# Run tests
make test
```

## Continuous Integration

The project uses GitHub Actions for CI/CD. Every push and pull request triggers:
- Linting with Ruff
- Type checking with MyPy  
- Code formatting with Black
- Full test suite with pytest
- Coverage reporting

Make sure your code passes all checks before submitting a PR:

```bash
make format  # Auto-fix formatting
make test    # Run tests
```