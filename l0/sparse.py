"""
L0 regularized linear models for sparse data.

Implements L0-regularized linear regression that efficiently handles
sparse input matrices without converting them to dense format.
"""
# ruff: noqa: N803 N806


import numpy as np
import torch
import torch.nn as nn
from scipy import sparse as sp


class SparseL0Linear(nn.Module):
    """
    L0-regularized linear model for sparse input data.

    Efficiently handles sparse matrices (scipy.sparse) without
    converting to dense format during computation.

    Parameters
    ----------
    n_features : int
        Number of input features
    fit_intercept : bool
        Whether to fit an intercept term
    beta : float
        Temperature parameter for Hard Concrete distribution
    gamma : float
        Lower bound of stretched concrete distribution
    zeta : float
        Upper bound of stretched concrete distribution
    init_keep_prob : float
        Initial probability of keeping each feature
    device : str or torch.device
        Device to run computations on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        n_features: int,
        fit_intercept: bool = True,
        beta: float = 2 / 3,
        gamma: float = -0.1,
        zeta: float = 1.1,
        init_keep_prob: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.n_features = n_features
        self.fit_intercept = fit_intercept
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.device = torch.device(device)

        # Model parameters
        self.weight = nn.Parameter(torch.zeros(n_features, device=self.device))
        if fit_intercept:
            self.bias = nn.Parameter(torch.zeros(1, device=self.device))
        else:
            self.register_parameter("bias", None)

        # L0 gate parameters
        mu = torch.log(torch.tensor(init_keep_prob / (1 - init_keep_prob)))
        self.log_alpha = nn.Parameter(
            torch.normal(mu, 0.01, size=(n_features,), device=self.device)
        )

        # Cache for sparse tensor conversion
        self._cached_X_torch = None
        self._cached_X_shape = None

    def _convert_sparse_to_torch(
        self, X_sparse: sp.spmatrix
    ) -> torch.sparse.Tensor:
        """
        Convert scipy sparse matrix to torch sparse tensor.

        Caches the result if the shape matches to avoid redundant conversions.
        """
        if (
            self._cached_X_torch is not None
            and self._cached_X_shape == X_sparse.shape
            and X_sparse.nnz == self._cached_X_torch._nnz()
        ):
            return self._cached_X_torch

        X_coo = X_sparse.tocoo()
        indices = torch.LongTensor(np.vstack([X_coo.row, X_coo.col])).to(
            self.device
        )
        values = torch.FloatTensor(X_coo.data).to(self.device)
        X_torch = torch.sparse_coo_tensor(
            indices,
            values,
            X_sparse.shape,
            dtype=torch.float32,
            device=self.device,
        )

        # Cache for future use
        self._cached_X_torch = X_torch
        self._cached_X_shape = X_sparse.shape

        return X_torch

    def _sample_gates(self) -> torch.Tensor:
        """Sample gates using Hard Concrete distribution."""
        eps = 1e-6
        u = torch.rand_like(self.log_alpha).clamp(eps, 1 - eps)
        X = (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta
        s = torch.sigmoid(X)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0, 1)

    def get_deterministic_gates(self) -> torch.Tensor:
        """Get deterministic gate values (for inference)."""
        X = self.log_alpha / self.beta
        s = torch.sigmoid(X)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0, 1)

    def forward(
        self,
        X: torch.sparse.Tensor | sp.spmatrix,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass maintaining sparse computation.

        Parameters
        ----------
        X : torch.sparse.Tensor or scipy.sparse matrix
            Sparse input tensor of shape (n_samples, n_features)
        deterministic : bool
            Whether to use deterministic gates (for inference)

        Returns
        -------
        torch.Tensor
            Predictions of shape (n_samples,)
        """
        # Convert scipy sparse to torch if needed
        if isinstance(X, sp.spmatrix):
            X = self._convert_sparse_to_torch(X)

        # Sample or get deterministic gates
        if deterministic:
            gates = self.get_deterministic_gates()
        else:
            gates = self._sample_gates()

        # Apply gates to weights
        gated_weight = self.weight * gates

        # Sparse matrix multiplication
        # X is (n_samples, n_features), gated_weight is (n_features,)
        y = torch.sparse.mm(X, gated_weight.unsqueeze(1)).squeeze(1)

        # Add bias if needed
        if self.fit_intercept and self.bias is not None:
            y = y + self.bias

        return y

    def get_l0_penalty(self) -> torch.Tensor:
        """
        Compute L0 complexity penalty.

        Returns expected number of active parameters.
        """
        c = -self.beta * torch.log(
            torch.tensor(-self.gamma / self.zeta, device=self.device)
        )
        pi = torch.sigmoid(self.log_alpha + c)
        return pi.sum()

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

    def get_selected_features(self, threshold: float = 0.01) -> torch.Tensor:
        """
        Get indices of selected (non-zero) features.

        Parameters
        ----------
        threshold : float
            Gate values below this are considered zero

        Returns
        -------
        torch.Tensor
            Indices of selected features
        """
        with torch.no_grad():
            gates = self.get_deterministic_gates()
            return torch.where(gates > threshold)[0]

    def fit(
        self,
        X_sparse: sp.spmatrix,
        y: torch.Tensor | np.ndarray,
        lambda_reg: float = 0.01,
        lr: float = 0.01,
        epochs: int = 1000,
        verbose: bool = False,
        verbose_freq: int = 100,
    ) -> "SparseL0Linear":
        """
        Fit the model using gradient descent.

        Parameters
        ----------
        X_sparse : scipy.sparse matrix
            Sparse input features
        y : torch.Tensor or numpy.ndarray
            Target values
        lambda_reg : float
            L0 regularization strength
        lr : float
            Learning rate
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
        verbose_freq : int
            How often to print progress

        Returns
        -------
        self
            Fitted model
        """
        import numpy as np

        # Convert y to tensor if needed
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        elif not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(self.device)

        # Convert sparse matrix to torch sparse tensor (will be cached)
        X_torch = self._convert_sparse_to_torch(X_sparse)

        # Initialize weights with small random values
        nn.init.normal_(self.weight, 0, 0.1)
        if self.fit_intercept and self.bias is not None:
            self.bias.data.fill_(y.mean())

        # Setup optimizer
        params = [self.weight, self.log_alpha]
        if self.fit_intercept and self.bias is not None:
            params.append(self.bias)
        optimizer = torch.optim.Adam(params, lr=lr)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_torch, deterministic=False)

            # Compute loss
            mse_loss = (y - y_pred).pow(2).mean()
            l0_loss = self.get_l0_penalty()
            loss = mse_loss + lambda_reg * l0_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if verbose and (epoch + 1) % verbose_freq == 0:
                with torch.no_grad():
                    n_active = (self.get_deterministic_gates() > 0.01).sum()
                    print(
                        f"Epoch {epoch+1:4d}: "
                        f"loss={loss.item():.4f}, "
                        f"mse={mse_loss.item():.4f}, "
                        f"l0={l0_loss.item():.2f}, "
                        f"active={n_active.item()}"
                    )

        return self

    def predict(
        self, X_sparse: sp.spmatrix | torch.sparse.Tensor
    ) -> torch.Tensor:
        """
        Make predictions on new data.

        Parameters
        ----------
        X_sparse : scipy.sparse matrix or torch.sparse.Tensor
            Sparse input features

        Returns
        -------
        torch.Tensor
            Predictions
        """
        # Make deterministic predictions
        with torch.no_grad():
            return self.forward(X_sparse, deterministic=True)
