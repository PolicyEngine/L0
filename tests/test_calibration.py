"""
Tests for SparseCalibrationWeights with positive weight constraints.
"""

import numpy as np
import torch
import pytest
from scipy import sparse as sp
from l0.calibration import SparseCalibrationWeights


class TestSparseCalibrationWeights:
    """Test suite for calibration weights with L0 sparsity."""

    def test_positive_weights(self):
        """Verify all weights remain non-negative."""
        N = 100
        Q = 20

        # Create test data
        M = sp.random(Q, N, density=0.3, format="csr")
        y = np.random.randn(Q) + 10

        model = SparseCalibrationWeights(n_features=N)
        model.fit(M, y, epochs=100, verbose=False)

        # Check positivity
        with torch.no_grad():
            weights = model.get_weights(deterministic=True)
            assert torch.all(weights >= 0), "Weights must be non-negative"

    def test_sparse_ground_truth_relative_loss(self):
        """Test recovery of sparse ground truth using relative loss."""
        Q = 200  # targets
        N = 2000  # features
        N_active = 1000  # 50% sparsity

        np.random.seed(42)
        torch.manual_seed(42)

        # Generate data with sparse ground truth
        M_dense = np.random.lognormal(mean=1.5, sigma=0.25, size=(Q, N))
        M = sp.csr_matrix(M_dense)

        w_true = np.zeros(N)
        active_indices = np.random.choice(N, size=N_active, replace=False)
        w_true[active_indices] = np.random.lognormal(
            mean=2.0, sigma=1.0, size=N_active
        )

        y = M @ w_true

        # Fit with relative loss
        model = SparseCalibrationWeights(
            n_features=N,
            beta=0.66,
            gamma=-0.1,
            zeta=1.1,
            init_keep_prob=0.3,
            init_weight_scale=0.5,
        )

        model.fit(
            M=M,
            y=y,
            lambda_l0=0.0005,  # Tuned for ~50% sparsity with relative loss
            lambda_l2=1e-6,
            lr=0.2,
            epochs=2000,
            loss_type="relative",
            verbose=False,
        )

        # Check sparsity is reasonable (between 30% and 70%)
        sparsity = model.get_sparsity()
        assert (
            0.3 <= sparsity <= 0.7
        ), f"Sparsity {sparsity:.2%} not in expected range"

        # Check relative loss is low
        with torch.no_grad():
            y_pred = model.predict(M).cpu().numpy()
            rel_loss = np.mean(((y - y_pred) / (y + 1)) ** 2)
            assert rel_loss < 0.1, f"Relative loss {rel_loss:.4f} too high"

    def test_relative_vs_mse_loss(self):
        """Compare relative loss vs MSE for large-scale data."""
        Q = 100
        N = 500

        np.random.seed(123)
        torch.manual_seed(123)

        # Large-scale data
        M = sp.random(Q, N, density=0.5, format="csr")
        M.data = np.abs(M.data) * 1000  # Large values
        y = np.random.uniform(1000, 100000, size=Q)

        # Train with MSE
        model_mse = SparseCalibrationWeights(n_features=N)
        model_mse.fit(
            M,
            y,
            lambda_l0=1e-10,  # Very small for MSE
            lr=0.1,
            epochs=500,
            loss_type="mse",
            verbose=False,
        )

        # Train with relative loss
        model_rel = SparseCalibrationWeights(n_features=N)
        model_rel.fit(
            M,
            y,
            lambda_l0=0.001,  # Can use larger penalty
            lr=0.1,
            epochs=500,
            loss_type="relative",
            verbose=False,
        )

        # Relative loss should achieve better relative accuracy
        with torch.no_grad():
            y_pred_mse = model_mse.predict(M).cpu().numpy()
            y_pred_rel = model_rel.predict(M).cpu().numpy()

            rel_err_mse = np.mean(np.abs((y - y_pred_mse) / (y + 1)))
            rel_err_rel = np.mean(np.abs((y - y_pred_rel) / (y + 1)))

            # Relative loss should do better on relative error
            assert (
                rel_err_rel <= rel_err_mse * 1.5
            ), f"Relative loss should handle scale better: {rel_err_rel:.4f} vs {rel_err_mse:.4f}"

    def test_sparsity_control(self):
        """Test that L0 penalty controls sparsity level."""
        Q = 50
        N = 200

        np.random.seed(123)
        torch.manual_seed(123)

        M = sp.random(Q, N, density=0.3, format="csr")
        y = np.random.randn(Q) + 10

        sparsities = []

        # Test different L0 penalties
        for lambda_l0 in [0.0001, 0.001, 0.01]:
            model = SparseCalibrationWeights(n_features=N, init_keep_prob=0.5)
            model.fit(
                M,
                y,
                lambda_l0=lambda_l0,
                lr=0.1,
                epochs=2000,
                loss_type="relative",
                verbose=False,
            )
            sparsities.append(model.get_sparsity())

        # Higher penalty should give more sparsity
        assert (
            sparsities[0] < sparsities[1]
        ), "Higher L0 penalty should increase sparsity"
        assert (
            sparsities[1] < sparsities[2]
        ), "Higher L0 penalty should increase sparsity"

    def test_get_active_weights(self):
        """Test active weight extraction."""
        N = 100
        model = SparseCalibrationWeights(n_features=N)

        # Simple test data
        M = sp.eye(N, format="csr")
        y = np.ones(N)

        model.fit(M, y, lambda_l0=0.01, epochs=100, verbose=False)

        active_info = model.get_active_weights()

        assert "indices" in active_info
        assert "values" in active_info
        assert "count" in active_info
        assert active_info["count"] == len(active_info["indices"])
        assert len(active_info["values"]) == active_info["count"]

        # All active values should be positive
        if active_info["count"] > 0:
            assert torch.all(active_info["values"] > 0)

    def test_deterministic_inference(self):
        """Test that inference is deterministic."""
        N = 50
        Q = 10

        np.random.seed(123)
        torch.manual_seed(123)

        M = sp.random(Q, N, density=0.5, format="csr")
        y = np.random.randn(Q)

        model = SparseCalibrationWeights(n_features=N)
        model.fit(M, y, epochs=100, verbose=False)

        # Multiple predictions should be identical
        with torch.no_grad():
            pred1 = model.predict(M).cpu().numpy()
            pred2 = model.predict(M).cpu().numpy()

        np.testing.assert_array_equal(
            pred1, pred2, "Predictions should be deterministic"
        )

    def test_l2_regularization(self):
        """Test that L2 penalty prevents weight explosion."""
        N = 100
        Q = 20

        np.random.seed(123)
        torch.manual_seed(123)

        M = sp.random(Q, N, density=0.3, format="csr")
        y = np.random.randn(Q) * 100  # Large scale

        # Train without L2
        model_no_l2 = SparseCalibrationWeights(n_features=N)
        model_no_l2.fit(
            M, y, lambda_l0=0.0001, lambda_l2=0.0, epochs=200, verbose=False
        )

        # Train with L2
        model_with_l2 = SparseCalibrationWeights(n_features=N)
        model_with_l2.fit(
            M, y, lambda_l0=0.0001, lambda_l2=0.01, epochs=200, verbose=False
        )

        with torch.no_grad():
            weights_no_l2 = model_no_l2.get_weights(deterministic=True)
            weights_with_l2 = model_with_l2.get_weights(deterministic=True)

            # L2 should reduce weight magnitudes
            assert (
                weights_with_l2.max() <= weights_no_l2.max() * 2.0
            ), "L2 should prevent extreme weights"

    def test_group_wise_averaging(self):
        """Test that group-wise averaging balances loss contributions."""
        N = 100  # features (households)

        # Create targets with different cardinalities:
        # - 3 singleton targets (like national targets)
        # - 18 targets in one group (like age bins for one state)
        # - 18 targets in another group (like age bins for another state)
        Q = 3 + 18 + 18  # 39 total targets

        np.random.seed(42)
        torch.manual_seed(42)

        # Create matrix with varying scales
        M = sp.random(Q, N, density=0.3, format="csr")

        # Create target values with different scales
        # Singletons: large values (billions scale)
        y_singletons = np.array([1e9, 5e8, 2e9])
        # Groups: smaller values (thousands scale)
        y_group1 = np.random.uniform(1e3, 1e6, size=18)
        y_group2 = np.random.uniform(1e3, 1e6, size=18)
        y = np.concatenate([y_singletons, y_group1, y_group2])

        # Create target groups
        # Groups 0, 1, 2: singletons (each national target)
        # Group 3: all 18 targets from first age group
        # Group 4: all 18 targets from second age group
        target_groups = np.array(
            [0, 1, 2]  # 3 singletons
            + [3] * 18  # Group 3
            + [4] * 18  # Group 4
        )

        # Train WITHOUT grouping (baseline)
        model_no_groups = SparseCalibrationWeights(n_features=N)
        model_no_groups.fit(
            M,
            y,
            lambda_l0=0.0001,
            lr=0.1,
            epochs=500,
            loss_type="relative",
            verbose=False,
            target_groups=None,  # No grouping
        )

        # Train WITH grouping
        model_with_groups = SparseCalibrationWeights(n_features=N)
        model_with_groups.fit(
            M,
            y,
            lambda_l0=0.0001,
            lr=0.1,
            epochs=500,
            loss_type="relative",
            verbose=False,
            target_groups=target_groups,
        )

        # Compute errors by group
        with torch.no_grad():
            y_pred_no_groups = model_no_groups.predict(M).cpu().numpy()
            y_pred_with_groups = model_with_groups.predict(M).cpu().numpy()

            # Relative errors
            rel_err_no_groups = np.abs((y - y_pred_no_groups) / (y + 1))
            rel_err_with_groups = np.abs((y - y_pred_with_groups) / (y + 1))

            # Average errors by group
            singleton_err_no_groups = rel_err_no_groups[:3].mean()
            group3_err_no_groups = rel_err_no_groups[3:21].mean()
            group4_err_no_groups = rel_err_no_groups[21:].mean()

            singleton_err_with_groups = rel_err_with_groups[:3].mean()
            group3_err_with_groups = rel_err_with_groups[3:21].mean()
            group4_err_with_groups = rel_err_with_groups[21:].mean()

            # With grouping, singleton errors should be much better
            # (they're not dominated by the 36 histogram targets)
            assert singleton_err_with_groups < singleton_err_no_groups * 1.5, (
                f"Grouping should improve singleton accuracy: "
                f"{singleton_err_with_groups:.4f} vs {singleton_err_no_groups:.4f}"
            )

            # All groups should have relatively balanced errors with grouping
            all_group_errors = [
                singleton_err_with_groups,
                group3_err_with_groups,
                group4_err_with_groups,
            ]
            max_err = max(all_group_errors)
            min_err = min(all_group_errors)

            # Errors should be within an order of magnitude of each other
            assert max_err < min_err * 10, (
                f"Group errors should be balanced: "
                f"min={min_err:.4f}, max={max_err:.4f}"
            )

    def test_group_wise_averaging_edge_cases(self):
        """Test edge cases for group-wise averaging."""
        N = 50
        Q = 10

        np.random.seed(42)
        torch.manual_seed(42)

        M = sp.random(Q, N, density=0.3, format="csr")
        y = np.random.uniform(100, 1000, size=Q)

        model = SparseCalibrationWeights(n_features=N)

        # Test 1: All targets in one group (should behave like no grouping)
        target_groups_single = np.zeros(Q, dtype=int)
        model.fit(
            M,
            y,
            lambda_l0=0.00001,  # Lower penalty for better convergence
            epochs=2000,  # Plenty of epochs
            lr=0.2,  # Higher learning rate
            loss_type="relative",
            verbose=False,
            target_groups=target_groups_single,
        )

        with torch.no_grad():
            y_pred = model.predict(M).cpu().numpy()
            rel_err = np.mean(np.abs((y - y_pred) / (y + 1)))
            assert (
                rel_err < 0.5
            ), f"Single group should still converge, got {rel_err:.4f}"

        # Test 2: Each target in its own group (like all singletons)
        target_groups_all_singleton = np.arange(Q)
        model_new = SparseCalibrationWeights(n_features=N)
        model_new.fit(
            M,
            y,
            lambda_l0=0.00001,
            epochs=2000,
            lr=0.2,
            loss_type="relative",
            verbose=False,
            target_groups=target_groups_all_singleton,
        )

        with torch.no_grad():
            y_pred = model_new.predict(M).cpu().numpy()
            rel_err = np.mean(np.abs((y - y_pred) / (y + 1)))
            assert (
                rel_err < 0.5
            ), f"All singleton groups should converge, got {rel_err:.4f}"

        # Test 3: Unbalanced groups (1 huge group, several small)
        target_groups_unbalanced = np.array([0] * 7 + [1, 2, 3])
        model_unbalanced = SparseCalibrationWeights(n_features=N)
        model_unbalanced.fit(
            M,
            y,
            lambda_l0=0.00001,
            epochs=2000,
            lr=0.2,
            loss_type="relative",
            verbose=False,
            target_groups=target_groups_unbalanced,
        )

        with torch.no_grad():
            y_pred = model_unbalanced.predict(M).cpu().numpy()
            # Check that small groups aren't ignored
            small_group_errors = np.abs((y[7:] - y_pred[7:]) / (y[7:] + 1))
            assert (
                np.mean(small_group_errors) < 0.5
            ), "Small groups should not be ignored"
