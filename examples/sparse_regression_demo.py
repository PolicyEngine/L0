"""
Sparse Regression with L0 Regularization

Demonstrates the Hard Concrete distribution for variable selection on a
simple 4-variable regression problem where the true coefficients are
[1, 0, -2, 0] - only x1 and x3 contribute to y.

This example shows why we include temperature (beta) in the test-time
gate computation, which produces sharper 0/1 decisions.
"""

import numpy as np
import torch
from scipy.stats import multivariate_normal

torch.manual_seed(12543)
np.random.seed(12543)


def sample_z(log_alpha, beta, zeta, gamma):
    """Sample from Hard Concrete distribution during training."""
    p = log_alpha.numel()
    eps = 1e-6
    u = torch.rand(p).clamp(eps, 1 - eps)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    return s_bar.clamp(0, 1)


def complexity_loss(log_alpha, beta, zeta, gamma):
    """Expected L0 penalty: probability each gate is active."""
    c = -beta * torch.log(torch.tensor(-gamma / zeta))
    return torch.sigmoid(log_alpha + c).sum()


def final_gate(log_alpha, beta, zeta, gamma):
    """
    Deterministic gate for inference.

    We include temperature (beta) here, unlike some implementations.
    This produces sharper decisions: gates closer to exactly 0 or 1.
    """
    return ((log_alpha / beta).sigmoid() * (zeta - gamma) + gamma).clamp(0, 1)


def final_gate_no_temp(log_alpha, beta, zeta, gamma):
    """Alternative: without temperature division (softer gates)."""
    return (log_alpha.sigmoid() * (zeta - gamma) + gamma).clamp(0, 1)


# Data generating process
n = 500
b_true = np.array([1.0, 0.0, -2.0, 0.0])  # Only x1 and x3 matter
b0_true = 30.0
p = len(b_true)

# Correlated features
rho = 0.5
sigma_X = np.full((p, p), rho)
np.fill_diagonal(sigma_X, 1)
sigma_e = 1.5

X_np = multivariate_normal.rvs(mean=np.zeros(p), cov=sigma_X, size=n)
y_np = b0_true + X_np @ b_true + sigma_e * np.random.randn(n)

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

# Initialize parameters
b0_param = torch.tensor(y_np.mean(), dtype=torch.float32, requires_grad=True)
b_param = torch.tensor(
    0.1 * np.random.rand(p), dtype=torch.float32, requires_grad=True
)

# L0 gate parameters
keep_prob = 0.5
mu = np.log(keep_prob / (1 - keep_prob))
log_alpha = torch.tensor(
    np.random.normal(mu, 0.01, size=p), dtype=torch.float32, requires_grad=True
)

# Hard Concrete hyperparameters
beta = 2 / 3  # Temperature
gamma = -0.1  # Stretch lower bound
zeta = 1.1  # Stretch upper bound

# Regularization strength
lambda_l0 = 0.2

# Training
optimizer = torch.optim.Adam([b0_param, b_param, log_alpha], lr=0.01)

print("Training L0-regularized regression...")
print(f"True coefficients: {b_true}")
print()

for epoch in range(1, 5001):
    z = sample_z(log_alpha, beta, zeta, gamma)
    y_hat = b0_param + X @ (b_param * z)

    data_loss = (y - y_hat).pow(2).mean()
    l0_loss = complexity_loss(log_alpha, beta, zeta, gamma)
    loss = data_loss + lambda_l0 * l0_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        with torch.no_grad():
            z_test = final_gate(log_alpha, beta, zeta, gamma)
            b_final = b_param * z_test
            print(
                f"Epoch {epoch}: loss={loss.item():.4f}, coeffs={b_final.numpy().round(3)}"
            )

# Final comparison
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

with torch.no_grad():
    z_with_temp = final_gate(log_alpha, beta, zeta, gamma)
    z_without_temp = final_gate_no_temp(log_alpha, beta, zeta, gamma)

    b_with_temp = b_param * z_with_temp
    b_without_temp = b_param * z_without_temp

    print(f"\nTrue coefficients:        {b_true}")
    print(f"\nWith temperature (ours):  {b_with_temp.numpy().round(4)}")
    print(f"  Gates:                  {z_with_temp.numpy().round(4)}")
    print(f"\nWithout temperature:      {b_without_temp.numpy().round(4)}")
    print(f"  Gates:                  {z_without_temp.numpy().round(4)}")

    # Show the mathematical difference
    print("\n" + "-" * 60)
    print("Gate computation comparison (for learned log_alpha values):")
    print("-" * 60)
    print(f"log_alpha:                {log_alpha.numpy().round(3)}")
    print(
        f"sigmoid(log_alpha/beta):  {(log_alpha/beta).sigmoid().numpy().round(3)}"
    )
    print(f"sigmoid(log_alpha):       {log_alpha.sigmoid().numpy().round(3)}")
    print("\nDividing by temperature (beta=2/3) amplifies the logit,")
    print("pushing sigmoid outputs closer to 0 or 1.")
