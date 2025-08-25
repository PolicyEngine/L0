"""
Tests for sparse L0 linear models.
"""
# ruff: noqa: N806

import numpy as np
import pytest
import torch
from scipy import sparse as sp
from sklearn.linear_model import LinearRegression

from l0.sparse import SparseL0Linear


class TestSparseL0Linear:
    """Test cases for SparseL0Linear model."""

    def test_sparse_data_recovery(self):
        """
        Test that the model can recover sparse coefficients from sparse data.
        Uses the same data generating process as the demonstration.
        """
        # Set seeds for reproducibility
        torch.manual_seed(12543)
        np.random.seed(12543)

        # Data parameters
        n = 1000
        p = 100
        X_sparsity = 0.95  # 95% of X entries are zero
        beta_sparsity = 0.95  # 95% of true coefficients are zero

        # Create sparse true coefficients
        num_nonzero = int(p * (1 - beta_sparsity))
        nonzero_indices = np.random.choice(p, num_nonzero, replace=False)
        b = np.zeros(p)
        b[nonzero_indices] = np.random.choice(
            [-3, -2, -1, 1, 2, 3], num_nonzero
        )

        # Generate sparse X matrix
        X_dense = np.random.randn(n, p) * 0.5
        mask = np.random.random((n, p)) < X_sparsity
        X_dense[mask] = 0

        # Add correlation structure
        for i in range(0, p - 1, 2):
            if i + 1 < p:
                corr_mask = ~(mask[:, i] | mask[:, i + 1])
                X_dense[corr_mask, i + 1] += 0.3 * X_dense[corr_mask, i]

        # Convert to sparse
        X_sparse = sp.csr_matrix(X_dense)

        # Generate y with intercept
        b0 = 30
        sigma_e = 1.5
        y_true = b0 + X_sparse @ b
        y = y_true + sigma_e * np.random.randn(n)

        # Fit baseline linear regression for lambda selection
        lr = LinearRegression(fit_intercept=True).fit(X_sparse, y)
        residuals = y - lr.predict(X_sparse)
        sigma2_hat = np.var(residuals)
        lambda_reg = 0.01 * sigma2_hat

        # Fit our sparse L0 model
        model = SparseL0Linear(
            n_features=p, fit_intercept=True, init_keep_prob=0.5
        )

        model.fit(
            X_sparse,
            y,
            lambda_reg=lambda_reg,
            lr=0.01,
            epochs=2000,
            verbose=False,
        )

        # Check feature selection performance
        selected = model.get_selected_features().cpu().numpy()
        true_positives = len(set(selected) & set(nonzero_indices))

        # We expect to recover at least 60% of true features
        recall = (
            true_positives / len(nonzero_indices)
            if len(nonzero_indices) > 0
            else 0
        )
        assert recall >= 0.6, f"Recall {recall:.2%} is too low"

        # We expect reasonable precision (at least 50%)
        if len(selected) > 0:
            precision = true_positives / len(selected)
            assert precision >= 0.5, f"Precision {precision:.2%} is too low"

        # Check that sparsity is achieved
        sparsity = model.get_sparsity()
        assert sparsity >= 0.7, f"Sparsity {sparsity:.2%} is too low"

        # Check prediction quality on training data
        y_pred = model.predict(X_sparse).cpu().numpy()
        mse = np.mean((y - y_pred) ** 2)
        baseline_mse = np.mean((y - lr.predict(X_sparse)) ** 2)

        # Should be comparable to baseline
        assert mse < 1.5 * baseline_mse, "MSE is much worse than baseline"

    def test_no_intercept(self):
        """Test that the model works without intercept."""
        np.random.seed(42)
        torch.manual_seed(42)

        n, p = 100, 20
        X_dense = np.random.randn(n, p)
        mask = np.random.random((n, p)) < 0.8
        X_dense[mask] = 0
        X_sparse = sp.csr_matrix(X_dense)

        # Generate y without intercept
        b = np.zeros(p)
        b[:3] = [2, -3, 1]
        y = X_sparse @ b + 0.1 * np.random.randn(n)

        # Fit model without intercept
        model = SparseL0Linear(
            n_features=p, fit_intercept=False, init_keep_prob=0.5
        )

        assert (
            model.bias is None
        ), "Bias should be None when fit_intercept=False"

        model.fit(X_sparse, y, epochs=500, verbose=False)

        # Check that we find the right features
        selected = model.get_selected_features().cpu().numpy()
        assert (
            0 in selected or 1 in selected or 2 in selected
        ), "Should select at least one true feature"

        # Make predictions
        y_pred = model.predict(X_sparse)
        assert y_pred.shape == (n,), "Predictions should have correct shape"

    def test_sparse_efficiency(self):
        """Test that sparse computations remain sparse."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Create very sparse data
        n, p = 500, 1000
        X_dense = np.random.randn(n, p)
        mask = np.random.random((n, p)) < 0.99  # 99% sparse
        X_dense[mask] = 0
        X_sparse = sp.csr_matrix(X_dense)

        # Check sparsity
        sparsity = 1 - X_sparse.nnz / (n * p)
        assert sparsity > 0.98, "Data should be very sparse"

        y = np.random.randn(n)

        model = SparseL0Linear(n_features=p, fit_intercept=True)

        # The model should handle this efficiently
        model.fit(X_sparse, y, epochs=100, verbose=False)

        # Check that cached tensor is sparse
        assert (
            model._cached_X_torch is not None
        ), "Should cache the sparse tensor"
        assert (
            model._cached_X_torch.is_sparse
        ), "Cached tensor should be sparse"

        # Multiple predictions should reuse cached tensor
        y_pred1 = model.predict(X_sparse)
        y_pred2 = model.predict(X_sparse)

        # Should get same predictions
        torch.testing.assert_close(y_pred1, y_pred2)

    def test_device_support(self):
        """Test that model works on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        np.random.seed(42)
        torch.manual_seed(42)

        n, p = 50, 10
        X_dense = np.random.randn(n, p)
        X_sparse = sp.csr_matrix(X_dense)
        y = np.random.randn(n)

        # Test on CUDA
        model = SparseL0Linear(n_features=p, fit_intercept=True, device="cuda")

        assert model.weight.device.type == "cuda"
        assert model.log_alpha.device.type == "cuda"

        model.fit(X_sparse, y, epochs=100, verbose=False)
        y_pred = model.predict(X_sparse)

        assert y_pred.device.type == "cuda"

    def test_deterministic_vs_stochastic(self):
        """Test that deterministic mode gives consistent predictions."""
        np.random.seed(42)
        torch.manual_seed(42)

        n, p = 100, 20
        X_dense = np.random.randn(n, p)
        X_sparse = sp.csr_matrix(X_dense)
        y = np.random.randn(n)

        model = SparseL0Linear(n_features=p, fit_intercept=True)
        model.fit(X_sparse, y, epochs=200, verbose=False)

        # Convert X for forward pass
        X_torch = model._convert_sparse_to_torch(X_sparse)

        # Deterministic predictions should be the same
        with torch.no_grad():
            y_det1 = model.forward(X_torch, deterministic=True)
            y_det2 = model.forward(X_torch, deterministic=True)

        torch.testing.assert_close(y_det1, y_det2)

        # Stochastic predictions should differ
        with torch.no_grad():
            y_stoch1 = model.forward(X_torch, deterministic=False)
            y_stoch2 = model.forward(X_torch, deterministic=False)

        assert not torch.allclose(
            y_stoch1, y_stoch2
        ), "Stochastic predictions should differ"
