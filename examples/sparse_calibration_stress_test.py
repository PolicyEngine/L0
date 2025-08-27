"""
Test script for SparseCalibrationWeights with truly sparse ground truth.
"""

import numpy as np
import torch
from scipy import sparse as sp
from l0.calibration import SparseCalibrationWeights

# Data generating process with sparse ground truth
Q = 2000  # number of targets/samples
N = 20000  # number of households/features

WEIGHT_SPARSITY = 0.50
METRIC_SPARSITY = 0.90

N_active = int(N * (1 - WEIGHT_SPARSITY))
np.random.seed(34543)
torch.manual_seed(34543)

# Metric matrix M (shape Q x N) - underdetermined system
M_dense = np.random.lognormal(mean=1.5, sigma=0.25, size=(Q, N))

# Add metric sparsity
mask = np.random.random((Q, N)) < (1 - METRIC_SPARSITY)
M_dense = M_dense * mask
M = sp.csr_matrix(M_dense)

# Create sparse true weights
w_true = np.zeros(N)
active_indices = np.random.choice(N, size=N_active, replace=False)
w_true[active_indices] = np.random.lognormal(
    mean=2.0, sigma=1.0, size=N_active
)

# Target vector y
y = M @ w_true

# Checks to make sure we really have this level of sparsity:
print(f"non-zeros elements in M: {M.nnz:,} ({M.nnz / (Q * N) * 100:.1f}%)")
print(
    f"non-zeros elements in w_true: {np.count_nonzero(w_true)} ({ np.count_nonzero(w_true) / N * 100:.1f}%)"
)

model = SparseCalibrationWeights(
    n_features=N,
    beta=0.66,
    gamma=-0.1,
    zeta=1.1,
    init_keep_prob=0.5,
    init_weight_scale=0.5,
    device="cpu",
)

model.fit(
    M=M,
    y=y,
    lambda_l0=0.00005,
    lambda_l2=0.0,
    lr=0.2,
    epochs=5000,
    loss_type="relative",
    verbose=True,
    verbose_freq=500,
)

with torch.no_grad():
    final_weights = model.get_weights(deterministic=True).cpu().numpy()
    y_pred = model.predict(M).cpu().numpy()

    relative_loss = np.mean(((y - y_pred) / (y + 1)) ** 2)
    print(f"Relative Loss: {relative_loss:.6f}")

    assert np.all(final_weights >= 0), "Weights should be non-negative!"

    active_weights = final_weights[final_weights > 1e-6]
    print(
        f"Final Sparsity (% zero): {100 - len(active_weights) / N * 100:.1f}%"
    )
    if len(active_weights) > 0:
        print(f"\nActive weight statistics:")
        print(f"  Min: {active_weights.min():.4f}")
        print(f"  Max: {active_weights.max():.4f}")
        print(f"  Mean: {active_weights.mean():.4f}")
        print(f"  Median: {np.median(active_weights):.4f}")

    # Check overlap with true active set
    active_info = model.get_active_weights()
    model_active = active_info["indices"].cpu().numpy()
    overlap = len(np.intersect1d(model_active, active_indices))
    precision = overlap / len(model_active) if len(model_active) > 0 else 0
    recall = overlap / len(active_indices)
    r_squared = np.corrcoef(y_pred, y)[0, 1] ** 2

    print(f"R²: {r_squared:.6f}")
    print(f"\nSelection accuracy:")
    print(f"  Overlap with true active set: {overlap}/{N_active}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
