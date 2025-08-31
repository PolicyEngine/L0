"""
L0-regularized calibration weights for survey data.

Implements sparse calibration weights that are constrained to be non-negative,
suitable for survey weighting and importance sampling applications.
"""

# ruff: noqa: N803 N806

import torch
import torch.nn as nn
import numpy as np
from scipy import sparse as sp


class SparseCalibrationWeights(nn.Module):
    """
    L0-regularized calibration weights with positivity constraint.

    Designed for survey calibration where weights must be non-negative
    and sparsity is desired to select a subset of households/units.

    Parameters
    ----------
    n_features : int
        Number of calibration weights (e.g., number of households)
    beta : float
        Temperature parameter for Hard Concrete distribution
    gamma : float
        Lower bound of stretched concrete distribution
    zeta : float
        Upper bound of stretched concrete distribution
    init_keep_prob : float
        Initial probability of keeping each weight active
    init_weight_scale : float
        Initial scale for log weights (controls initial weight magnitude)
    device : str or torch.device
        Device to run computations on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        n_features: int,
        beta: float = 2 / 3,
        gamma: float = -0.1,
        zeta: float = 1.1,
        init_keep_prob: float = 0.5,
        init_weight_scale: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.n_features = n_features
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.device = torch.device(device)

        # Log weights to ensure positivity via exp transformation
        self.log_weight = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=init_weight_scale,
                size=(n_features,),
                device=self.device,
            )
        )

        # L0 gate parameters
        mu = torch.log(torch.tensor(init_keep_prob / (1 - init_keep_prob)))
        self.log_alpha = nn.Parameter(
            torch.normal(
                mu.item(), 0.01, size=(n_features,), device=self.device
            )
        )

        # Cache for sparse tensor conversion
        self._cached_M_torch: torch.sparse.Tensor | None = None
        self._cached_M_shape: tuple[int, int] | None = None

    def _convert_sparse_to_torch(
        self, M_sparse: sp.spmatrix
    ) -> torch.sparse.Tensor:
        """
        Convert scipy sparse matrix to torch sparse tensor.

        Caches the result if the shape matches to avoid redundant conversions.
        """
        if (
            self._cached_M_torch is not None
            and self._cached_M_shape == M_sparse.shape
            and M_sparse.nnz == self._cached_M_torch._nnz()
        ):
            return self._cached_M_torch

        M_coo = M_sparse.tocoo()
        indices = torch.LongTensor(np.vstack([M_coo.row, M_coo.col])).to(
            self.device
        )
        values = torch.FloatTensor(M_coo.data).to(self.device)
        M_torch = torch.sparse_coo_tensor(
            indices,
            values,
            M_sparse.shape,
            dtype=torch.float32,
            device=self.device,
        )

        # Cache for future use
        self._cached_M_torch = M_torch
        self._cached_M_shape = M_sparse.shape

        return M_torch

    def _sample_gates(self) -> torch.Tensor:
        """Sample gates using Hard Concrete distribution."""
        eps = 1e-6
        u = torch.rand_like(self.log_alpha).clamp(eps, 1 - eps)
        s = (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta
        s = torch.sigmoid(s)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0, 1)

    def get_deterministic_gates(self) -> torch.Tensor:
        """Get deterministic gate values (for inference)."""
        s = torch.sigmoid(self.log_alpha / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0, 1)

    def get_weights(self, deterministic: bool = False) -> torch.Tensor:
        """
        Get effective calibration weights.

        Parameters
        ----------
        deterministic : bool
            Whether to use deterministic gates (for inference)

        Returns
        -------
        torch.Tensor
            Positive calibration weights with L0 sparsity applied
        """
        # Sample or get deterministic gates
        if deterministic:
            gates = self.get_deterministic_gates()
        else:
            gates = self._sample_gates()

        # Apply exp transformation and gates together
        # This ensures gradient flow matches successful implementations
        return torch.exp(self.log_weight) * gates

    def forward(
        self,
        M: sp.spmatrix,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Apply calibration weights to metric matrix.

        Parameters
        ----------
        M : scipy.sparse matrix
            Metric matrix of shape (n_targets, n_features)
        deterministic : bool
            Whether to use deterministic gates (for inference)

        Returns
        -------
        torch.Tensor
            Weighted sums of shape (n_targets,)
        """
        # Convert sparse matrix to torch
        M_torch = self._convert_sparse_to_torch(M)

        # Get effective weights
        weights = self.get_weights(deterministic=deterministic)

        # Sparse matrix multiplication
        y = torch.sparse.mm(M_torch, weights.unsqueeze(1)).squeeze(1)

        return y

    def get_l0_penalty(self) -> torch.Tensor:
        """
        Compute L0 complexity penalty.

        Returns expected number of active weights.
        """
        c = -self.beta * torch.log(
            torch.tensor(-self.gamma / self.zeta, device=self.device)
        )
        pi = torch.sigmoid(self.log_alpha + c)
        return pi.sum()

    def get_l2_penalty(self) -> torch.Tensor:
        """
        Compute L2 penalty on positive weights.

        Helps prevent weight explosion when gates are partially open.

        Returns
        -------
        torch.Tensor
            L2 norm of positive weights
        """
        positive_weights = torch.exp(self.log_weight)
        return (positive_weights**2).sum()

    def get_sparsity(self) -> float:
        """
        Get current sparsity level.

        Returns
        -------
        float
            Fraction of weights that are effectively zero
        """
        with torch.no_grad():
            gates = self.get_deterministic_gates()
            return (gates < 0.01).float().mean().item()

    def get_active_weights(self, threshold: float = 0.01) -> dict:
        """
        Get indices and values of active (non-zero) weights.

        Parameters
        ----------
        threshold : float
            Gate values below this are considered zero

        Returns
        -------
        dict
            Dictionary with 'indices' and 'values' of active weights
        """
        with torch.no_grad():
            weights = self.get_weights(deterministic=True)
            gates = self.get_deterministic_gates()
            active_mask = gates > threshold

            return {
                "indices": torch.where(active_mask)[0],
                "values": weights[active_mask],
                "count": active_mask.sum().item(),
            }

    def fit(
        self,
        M: sp.spmatrix,
        y: np.ndarray,
        lambda_l0: float = 0.01,
        lambda_l2: float = 0.0,
        lr: float = 0.01,
        epochs: int = 1000,
        loss_type: str = "mse",
        verbose: bool = False,
        verbose_freq: int = 100,
    ) -> "SparseCalibrationWeights":
        """
        Fit calibration weights using gradient descent.

        Parameters
        ----------
        M : scipy.sparse matrix
            Metric matrix of shape (n_targets, n_features)
        y : numpy.ndarray
            Target values to match
        lambda_l0 : float
            L0 regularization strength
        lambda_l2 : float
            L2 regularization strength on positive weights
        lr : float
            Learning rate
        epochs : int
            Number of training epochs
        loss_type : str
            Type of loss function: 'mse' or 'relative'
        verbose : bool
            Whether to print progress
        verbose_freq : int
            How often to print progress

        Returns
        -------
        self
            Fitted model
        """
        # Convert y to tensor
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        # Convert M to torch sparse (will be cached)
        M_torch = self._convert_sparse_to_torch(M)

        # Initialize weights
        nn.init.normal_(self.log_weight, 0, 0.5)

        # Setup optimizer
        optimizer = torch.optim.Adam([self.log_weight, self.log_alpha], lr=lr)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(M, deterministic=False)

            # Compute loss
            if loss_type == "relative":
                # Relative error: (y - y_pred)^2 / (y + 1)^2
                # Adding 1 to avoid division by zero
                relative_errors = (y - y_pred) / (y + 1)
                data_loss = relative_errors.pow(2).mean()
            else:
                # Standard MSE
                data_loss = (y - y_pred).pow(2).mean()

            l0_loss = self.get_l0_penalty()
            loss = data_loss + lambda_l0 * l0_loss

            # Add L2 penalty if specified
            if lambda_l2 > 0:
                l2_loss = self.get_l2_penalty()
                loss = loss + lambda_l2 * l2_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if verbose and (epoch + 1) % verbose_freq == 0:
                with torch.no_grad():
                    active_info = self.get_active_weights()
                    weights = self.get_weights(deterministic=True)
                    # Compute MSE for monitoring even if using relative loss
                    mse = (y - y_pred).pow(2).mean().item()
                    print(
                        f"Epoch {epoch+1:4d}: "
                        f"loss={loss.item():.4f}, "
                        f"data_loss={data_loss.item():.4f}, "
                        f"mse={mse:.4f}, "
                        f"l0={l0_loss.item():.2f}, "
                        f"active={active_info['count']}, "
                        f"mean_weight={weights[weights > 0.01].mean().item() if (weights > 0.01).any() else 0:.3f}"
                    )

        return self

    def predict(self, M: sp.spmatrix) -> torch.Tensor:
        """
        Apply calibration weights to new data.

        Parameters
        ----------
        M : scipy.sparse matrix
            Metric matrix of shape (n_targets, n_features)

        Returns
        -------
        torch.Tensor
            Calibrated predictions
        """
        with torch.no_grad():
            return self.forward(M, deterministic=True)
