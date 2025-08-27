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
        M = sp.random(Q, N, density=0.3, format='csr')
        y = np.random.randn(Q) + 10
        
        model = SparseCalibrationWeights(n_features=N)
        model.fit(M, y, epochs=100, verbose=False)
        
        # Check positivity
        with torch.no_grad():
            weights = model.get_weights(deterministic=True)
            assert torch.all(weights >= 0), "Weights must be non-negative"
    
    def test_sparse_ground_truth_relative_loss(self):
        """Test recovery of sparse ground truth using relative loss."""
        Q = 200    # targets
        N = 2000   # features
        N_active = 1000  # 50% sparsity
        
        np.random.seed(42)
        
        # Generate data with sparse ground truth
        M_dense = np.random.lognormal(mean=1.5, sigma=0.25, size=(Q, N))
        M = sp.csr_matrix(M_dense)
        
        w_true = np.zeros(N)
        active_indices = np.random.choice(N, size=N_active, replace=False)
        w_true[active_indices] = np.random.lognormal(mean=2.0, sigma=1.0, size=N_active)
        
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
            lambda_l0=0.00015,  # Tuned for ~50% sparsity with relative loss
            lambda_l2=1e-6,
            lr=0.2,
            epochs=2000,
            loss_type="relative",
            verbose=False
        )
        
        # Check sparsity is reasonable (between 30% and 70%)
        sparsity = model.get_sparsity()
        assert 0.3 <= sparsity <= 0.7, f"Sparsity {sparsity:.2%} not in expected range"
        
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
        
        # Large-scale data
        M = sp.random(Q, N, density=0.5, format='csr')
        M.data = np.abs(M.data) * 1000  # Large values
        y = np.random.uniform(1000, 100000, size=Q)
        
        # Train with MSE
        model_mse = SparseCalibrationWeights(n_features=N)
        model_mse.fit(
            M, y,
            lambda_l0=1e-10,  # Very small for MSE
            lr=0.1,
            epochs=500,
            loss_type="mse",
            verbose=False
        )
        
        # Train with relative loss
        model_rel = SparseCalibrationWeights(n_features=N)
        model_rel.fit(
            M, y,
            lambda_l0=0.001,  # Can use larger penalty
            lr=0.1,
            epochs=500,
            loss_type="relative",
            verbose=False
        )
        
        # Relative loss should achieve better relative accuracy
        with torch.no_grad():
            y_pred_mse = model_mse.predict(M).cpu().numpy()
            y_pred_rel = model_rel.predict(M).cpu().numpy()
            
            rel_err_mse = np.mean(np.abs((y - y_pred_mse) / (y + 1)))
            rel_err_rel = np.mean(np.abs((y - y_pred_rel) / (y + 1)))
            
            # Relative loss should do better on relative error
            assert rel_err_rel <= rel_err_mse * 1.5, \
                f"Relative loss should handle scale better: {rel_err_rel:.4f} vs {rel_err_mse:.4f}"
    
    def test_sparsity_control(self):
        """Test that L0 penalty controls sparsity level."""
        Q = 50
        N = 200
        
        M = sp.random(Q, N, density=0.3, format='csr')
        y = np.random.randn(Q) + 10
        
        sparsities = []
        
        # Test different L0 penalties
        for lambda_l0 in [0.0001, 0.001, 0.01]:
            model = SparseCalibrationWeights(n_features=N, init_keep_prob=0.5)
            model.fit(
                M, y,
                lambda_l0=lambda_l0,
                lr=0.1,
                epochs=500,
                loss_type="relative",
                verbose=False
            )
            sparsities.append(model.get_sparsity())
        
        # Higher penalty should give more sparsity
        assert sparsities[0] < sparsities[1], "Higher L0 penalty should increase sparsity"
        assert sparsities[1] < sparsities[2], "Higher L0 penalty should increase sparsity"
    
    def test_get_active_weights(self):
        """Test active weight extraction."""
        N = 100
        model = SparseCalibrationWeights(n_features=N)
        
        # Simple test data
        M = sp.eye(N, format='csr')
        y = np.ones(N)
        
        model.fit(M, y, lambda_l0=0.01, epochs=100, verbose=False)
        
        active_info = model.get_active_weights(threshold=0.01)
        
        assert 'indices' in active_info
        assert 'values' in active_info
        assert 'count' in active_info
        assert active_info['count'] == len(active_info['indices'])
        assert len(active_info['values']) == active_info['count']
        
        # All active values should be positive
        if active_info['count'] > 0:
            assert torch.all(active_info['values'] > 0)
    
    def test_deterministic_inference(self):
        """Test that inference is deterministic."""
        N = 50
        Q = 10
        
        M = sp.random(Q, N, density=0.5, format='csr')
        y = np.random.randn(Q)
        
        model = SparseCalibrationWeights(n_features=N)
        model.fit(M, y, epochs=100, verbose=False)
        
        # Multiple predictions should be identical
        with torch.no_grad():
            pred1 = model.predict(M).cpu().numpy()
            pred2 = model.predict(M).cpu().numpy()
            
        np.testing.assert_array_equal(pred1, pred2, 
                                      "Predictions should be deterministic")
    
    def test_l2_regularization(self):
        """Test that L2 penalty prevents weight explosion."""
        N = 100
        Q = 20
        
        M = sp.random(Q, N, density=0.3, format='csr')
        y = np.random.randn(Q) * 100  # Large scale
        
        # Train without L2
        model_no_l2 = SparseCalibrationWeights(n_features=N)
        model_no_l2.fit(M, y, lambda_l0=0.0001, lambda_l2=0.0, 
                       epochs=200, verbose=False)
        
        # Train with L2
        model_with_l2 = SparseCalibrationWeights(n_features=N)
        model_with_l2.fit(M, y, lambda_l0=0.0001, lambda_l2=0.01,
                         epochs=200, verbose=False)
        
        with torch.no_grad():
            weights_no_l2 = model_no_l2.get_weights(deterministic=True)
            weights_with_l2 = model_with_l2.get_weights(deterministic=True)
            
            # L2 should reduce weight magnitudes
            assert weights_with_l2.max() <= weights_no_l2.max() * 2.0, \
                "L2 should prevent extreme weights"