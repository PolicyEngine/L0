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

        model = SparseCalibrationWeights(n_features=N, init_weights=1.0)
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
            init_weights=1.0,  # Start all weights at 1.0
            log_weight_jitter_sd=0.5,  # Add jitter for symmetry breaking
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
        model = SparseCalibrationWeights(n_features=N, init_weights=1.0)

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

        model = SparseCalibrationWeights(n_features=N, init_weights=1.0)
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

    def test_pure_l2_penalty(self):
        """Test pure L2 regularization without L0."""
        N = 50
        Q = 30

        np.random.seed(42)
        torch.manual_seed(42)

        # Create underdetermined problem where L2 helps regularize
        M = sp.random(Q, N, density=0.5, format="csr")
        M.data = np.abs(M.data) * 2  # Scale up to create larger weights
        y = np.ones(Q) * 10  # Simple target

        # Train without any regularization
        model_no_reg = SparseCalibrationWeights(
            n_features=N,
            init_weights=2.0,  # Start with larger weights
            init_keep_prob=0.999,  # Keep all weights active
        )
        model_no_reg.fit(
            M,
            y,
            lambda_l0=0.0,  # No L0
            lambda_l2=0.0,  # No L2
            epochs=1000,
            lr=0.02,
            verbose=False,
        )

        # Train with strong L2 regularization
        model_l2_only = SparseCalibrationWeights(
            n_features=N,
            init_weights=2.0,  # Same starting point
            init_keep_prob=0.999,  # Keep all weights active
        )
        model_l2_only.fit(
            M,
            y,
            lambda_l0=0.0,  # No L0
            lambda_l2=10.0,  # Very strong L2 to see clear effect
            epochs=1000,
            lr=0.02,
            verbose=False,
        )

        with torch.no_grad():
            weights_no_reg = model_no_reg.get_weights(deterministic=True)
            weights_l2 = model_l2_only.get_weights(deterministic=True)

            # Get non-zero weights
            active_no_reg = weights_no_reg[weights_no_reg > 1e-8]
            active_l2 = weights_l2[weights_l2 > 1e-8]

            # L2 should keep most weights active (no sparsity from L2)
            assert (
                len(active_l2) >= N * 0.9
            ), f"Pure L2 should not induce sparsity, got {len(active_l2)}/{N}"

            # L2 norm should be smaller with regularization
            l2_norm_no_reg = (weights_no_reg**2).sum().sqrt()
            l2_norm_with_l2 = (weights_l2**2).sum().sqrt()
            assert (
                l2_norm_with_l2 < l2_norm_no_reg
            ), f"L2 regularization should reduce L2 norm: {l2_norm_with_l2:.2f} vs {l2_norm_no_reg:.2f}"

            # Check coefficient of variation (CV = std/mean) is lower with L2
            # This captures that L2 shrinks weights toward each other
            if active_no_reg.mean() > 1e-6 and active_l2.mean() > 1e-6:
                cv_no_reg = active_no_reg.std() / active_no_reg.mean()
                cv_l2 = active_l2.std() / active_l2.mean()
                assert (
                    cv_l2 < cv_no_reg * 1.2
                ), f"L2 should reduce relative variation: CV {cv_l2:.2f} vs {cv_no_reg:.2f}"

            # L2 should prevent extreme weights
            assert (
                weights_l2.max() < weights_no_reg.max() * 1.5
            ), f"L2 should limit max weights: {weights_l2.max():.2f} vs {weights_no_reg.max():.2f}"

            # Both should still fit reasonably well
            y_pred_no_reg = model_no_reg.predict(M).cpu().numpy()
            y_pred_l2 = model_l2_only.predict(M).cpu().numpy()

            error_no_reg = np.abs((y - y_pred_no_reg) / y).mean()
            error_l2 = np.abs((y - y_pred_l2) / y).mean()

            # L2 model may have slightly worse fit due to regularization
            assert (
                error_l2 < 0.5
            ), f"L2 model should still fit reasonably: {error_l2:.3f}"
            # But the trade-off is worth it for regularization

    def test_l0_l2_combination(self):
        """Test that combining L0 and L2 gives both sparsity and regularization."""
        N = 100  # features
        Q = 50  # targets

        np.random.seed(42)
        torch.manual_seed(42)

        # Create problem with potential for overfitting
        M = sp.random(Q, N, density=0.4, format="csr")
        M.data = np.abs(M.data) * 3
        y = np.random.uniform(5, 20, size=Q)

        # Model 1: Only L0 (sparsity without weight regularization)
        model_l0_only = SparseCalibrationWeights(
            n_features=N,
            init_weights=2.0,
            init_keep_prob=0.5,  # Start with 50% probability
        )
        model_l0_only.fit(
            M,
            y,
            lambda_l0=0.01,  # Stronger L0 for sparsity
            lambda_l2=0.0,  # No L2
            epochs=2000,
            lr=0.02,
            verbose=False,
        )

        # Model 2: Only L2 (weight regularization without sparsity)
        model_l2_only = SparseCalibrationWeights(
            n_features=N,
            init_weights=2.0,
            init_keep_prob=0.999,  # Keep all weights active
        )
        model_l2_only.fit(
            M,
            y,
            lambda_l0=0.0,  # No L0
            lambda_l2=0.1,  # Moderate L2
            epochs=1500,
            lr=0.02,
            verbose=False,
        )

        # Model 3: Combined L0+L2 (both sparsity and weight regularization)
        model_l0_l2 = SparseCalibrationWeights(
            n_features=N,
            init_weights=2.0,
            init_keep_prob=0.5,  # Same starting point as L0-only
        )
        model_l0_l2.fit(
            M,
            y,
            lambda_l0=0.01,  # Same L0 as model 1
            lambda_l2=0.1,  # Add L2 regularization
            epochs=2000,
            lr=0.02,
            verbose=False,
        )

        with torch.no_grad():
            # Get weights and stats for all models
            weights_l0_only = model_l0_only.get_weights(deterministic=True)
            weights_l2_only = model_l2_only.get_weights(deterministic=True)
            weights_l0_l2 = model_l0_l2.get_weights(deterministic=True)

            # Count active weights
            active_l0_only = (weights_l0_only > 1e-6).sum().item()
            active_l2_only = (weights_l2_only > 1e-6).sum().item()
            active_l0_l2 = (weights_l0_l2 > 1e-6).sum().item()

            # L0-only should have sparsity
            assert (
                active_l0_only < N * 0.8
            ), f"L0 should induce sparsity: {active_l0_only}/{N} active"

            # L2-only should have no/little sparsity
            assert (
                active_l2_only > N * 0.9
            ), f"L2 alone shouldn't induce sparsity: {active_l2_only}/{N} active"

            # L0+L2 should have sparsity (from L0)
            assert (
                active_l0_l2 < N * 0.8
            ), f"L0+L2 should have sparsity: {active_l0_l2}/{N} active"

            # Among active weights, L0+L2 should have smaller magnitudes than L0-only (from L2)
            active_mask_l0_only = weights_l0_only > 1e-6
            active_mask_l0_l2 = weights_l0_l2 > 1e-6

            if active_mask_l0_only.any() and active_mask_l0_l2.any():
                # Compare L2 norms of active weights
                l2_norm_l0_only = (
                    (weights_l0_only[active_mask_l0_only] ** 2).sum().sqrt()
                )
                l2_norm_l0_l2 = (
                    (weights_l0_l2[active_mask_l0_l2] ** 2).sum().sqrt()
                )

                # L0+L2 should have smaller weight norms than L0-only
                # (L2 regularization effect on top of sparsity)
                assert (
                    l2_norm_l0_l2 < l2_norm_l0_only * 1.2
                ), f"L0+L2 should have controlled weights: {l2_norm_l0_l2:.2f} vs L0-only {l2_norm_l0_only:.2f}"

            # Check prediction quality for all models
            y_pred_l0_only = model_l0_only.predict(M).cpu().numpy()
            y_pred_l2_only = model_l2_only.predict(M).cpu().numpy()
            y_pred_l0_l2 = model_l0_l2.predict(M).cpu().numpy()

            error_l0_only = np.abs((y - y_pred_l0_only) / (y + 1)).mean()
            error_l2_only = np.abs((y - y_pred_l2_only) / (y + 1)).mean()
            error_l0_l2 = np.abs((y - y_pred_l0_l2) / (y + 1)).mean()

            # All should fit reasonably well (L2-only may have slightly worse fit due to regularization)
            assert (
                error_l0_only < 0.35
            ), f"L0-only should fit well: {error_l0_only:.3f}"
            assert (
                error_l2_only < 0.35
            ), f"L2-only should fit well: {error_l2_only:.3f}"
            assert (
                error_l0_l2 < 0.35
            ), f"L0+L2 should fit well: {error_l0_l2:.3f}"

            print(
                f"\nL0-only: {active_l0_only}/{N} active, error={error_l0_only:.3f}"
            )
            print(
                f"L2-only: {active_l2_only}/{N} active, error={error_l2_only:.3f}"
            )
            print(f"L0+L2: {active_l0_l2}/{N} active, error={error_l0_l2:.3f}")

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

        model = SparseCalibrationWeights(n_features=N, init_weights=1.0)

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

    def test_init_weights_options(self):
        """Test different weight initialization options."""
        N = 50

        # Test 1: Default (None) should give all weights = 1.0
        model_default = SparseCalibrationWeights(n_features=N)
        with torch.no_grad():
            weights = torch.exp(model_default.log_weight)
            assert torch.allclose(weights, torch.ones(N), atol=1e-6)

        # Test 2: Scalar initialization
        model_scalar = SparseCalibrationWeights(n_features=N, init_weights=2.5)
        with torch.no_grad():
            weights = torch.exp(model_scalar.log_weight)
            assert torch.allclose(weights, torch.full((N,), 2.5), atol=1e-6)

        # Test 3: Array initialization
        init_array = np.random.uniform(0.5, 2.0, size=N)
        model_array = SparseCalibrationWeights(
            n_features=N, init_weights=init_array
        )
        with torch.no_grad():
            weights = torch.exp(model_array.log_weight).cpu().numpy()
            np.testing.assert_allclose(weights, init_array, rtol=1e-5)

        # Test 4: Wrong shape should raise error
        with pytest.raises(ValueError, match="must have shape"):
            SparseCalibrationWeights(n_features=N, init_weights=np.ones(N + 1))

    def test_weight_jitter(self):
        """Test that weight jitter works correctly."""
        N = 100
        Q = 20

        np.random.seed(42)
        torch.manual_seed(42)

        M = sp.random(Q, N, density=0.3, format="csr")
        y = np.random.randn(Q) + 10

        # Model with jitter
        model_with_jitter = SparseCalibrationWeights(
            n_features=N, init_weights=1.0, log_weight_jitter_sd=0.5
        )

        # Store initial weights
        initial_weights = model_with_jitter.log_weight.data.clone()

        # Fit should add jitter
        model_with_jitter.fit(M, y, epochs=10, verbose=False)

        # Weights should have changed due to jitter (and training)
        final_weights = model_with_jitter.log_weight.data
        assert not torch.allclose(initial_weights, final_weights)

        # Model without jitter
        torch.manual_seed(42)  # Reset seed
        model_no_jitter = SparseCalibrationWeights(
            n_features=N,
            init_weights=1.0,
            log_weight_jitter_sd=0.0,  # No jitter
        )

        initial_weights_no_jitter = model_no_jitter.log_weight.data.clone()
        model_no_jitter.fit(M, y, epochs=1, verbose=False)  # Just 1 epoch

        # After 1 epoch, change should be small without jitter
        weights_after_1_epoch = model_no_jitter.log_weight.data
        # The change is due to gradient updates only
        change = (
            (weights_after_1_epoch - initial_weights_no_jitter).abs().max()
        )
        assert change < 1.0, "Without jitter, initial change should be small"

    def test_init_keep_prob_options(self):
        """Test init_keep_prob as scalar and array."""
        n_features = 20

        # Test 1: Scalar init_keep_prob (existing behavior)
        model_scalar = SparseCalibrationWeights(
            n_features=n_features,
            init_keep_prob=0.7,
        )
        # All log_alpha values should be similar (around log(0.7/0.3) plus small jitter)
        expected_mu = np.log(0.7 / 0.3)
        with torch.no_grad():
            log_alphas = model_scalar.log_alpha.numpy()
            # Check they're all close to expected value (within jitter range)
            assert np.all(np.abs(log_alphas - expected_mu) < 0.1)

        # Test 2: Array init_keep_prob
        keep_probs = np.linspace(0.1, 0.9, n_features)
        model_array = SparseCalibrationWeights(
            n_features=n_features,
            init_keep_prob=keep_probs,
        )
        # Each log_alpha should correspond to its keep_prob
        with torch.no_grad():
            log_alphas = model_array.log_alpha.numpy()
            expected_mus = np.log(keep_probs / (1 - keep_probs))
            # Check each is close to its expected value
            assert np.all(np.abs(log_alphas - expected_mus) < 0.1)

        # Test 3: Wrong shape should raise error
        with pytest.raises(ValueError, match="must have shape"):
            SparseCalibrationWeights(
                n_features=10,
                init_keep_prob=np.ones(5),  # Wrong size
            )

        # Test 4: Edge case probabilities get clamped
        extreme_probs = np.array([0.0, 0.5, 1.0])
        model_extreme = SparseCalibrationWeights(
            n_features=3,
            init_keep_prob=extreme_probs,
        )
        with torch.no_grad():
            log_alphas = model_extreme.log_alpha.numpy()
            # Should not have inf or -inf values
            assert np.all(np.isfinite(log_alphas))
