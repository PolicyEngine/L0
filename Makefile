# L0 Regularization Package Makefile

.PHONY: help install install-dev test format type-check clean docs docs-serve changelog build publish

help:
	@echo "Available commands:"
	@echo "  make install      Install package in editable mode"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make test         Run tests with coverage"
	@echo "  make format       Format code with black"
	@echo "  make type-check   Run mypy type checker"
	@echo "  make changelog    Update changelog and version"
	@echo "  make clean        Remove build artifacts"
	@echo "  make build        Build Python package"
	@echo "  make publish      Publish to PyPI"
	@echo "  make docs         Build documentation"
	@echo "  make docs-serve   Serve documentation locally"

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=l0 --cov-report=term-missing

format:
	black . -l 79

type-check:
	mypy l0 --ignore-missing-imports

changelog:
	python .github/bump_version.py
	towncrier build --yes --version $$(python -c "import re; print(re.search(r'version = \"(.+?)\"', open('pyproject.toml').read()).group(1))")

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:
	python -m build

publish:
	twine upload dist/*

docs:
	cd docs && myst build --html

docs-serve:
	cd docs && myst start

# Convenience targets
all: format type-check test

ci: format type-check test