# Temperature Preservation in L0 Regularization: Critical Implementation Notes

## Executive Summary

The temperature parameter (β) in Hard Concrete distributions is **critical** for L0 regularization performance. Our analysis reveals that:

1. **The original authors' implementation** (`distributions.py`) contains a bug where temperature is incorrectly dropped in deterministic mode
2. **Our standalone implementations** (`calibration.py` and `sparse.py`) correctly preserve temperature in all modes
3. **The gold standard** (`l0_louizos_improved_gate.py`) confirms temperature must always be preserved

## The Temperature Bug in distributions.py

### What We Found

In `distributions.py` (from the authors' repository), the deterministic gates incorrectly drop temperature:

```python
# distributions.py line 134 - INCORRECT
def _deterministic_gates(self) -> torch.Tensor:
    probs = torch.sigmoid(self.qz_logits)  # ❌ Missing temperature!
    gates = probs * (self.zeta - self.gamma) + self.gamma
    return torch.clamp(gates, 0, 1)
```

### Why This Matters

The temperature parameter controls the "hardness" of the concrete distribution:
- **Lower temperature** (e.g., 0.1): More discrete, binary-like gates
- **Higher temperature** (e.g., 2.0): Softer, more continuous gates

Dropping temperature is equivalent to setting β=1, which fundamentally changes the distribution's behavior and can severely impact:
- Convergence speed
- Final sparsity levels
- Model performance

## Correct Implementations

### Gold Standard (l0_louizos_improved_gate.py)

Your validated implementation consistently uses temperature:

```python
# Sampling (line 55)
X = (torch.log(u) - torch.log(1 - u) + log_alpha) / beta  # ✅

# Deterministic (line 151)
z_final = ((log_alpha / beta).sigmoid() * (zeta - gamma) + gamma).clamp(0, 1)  # ✅

# Penalty computation (line 133)
c = -beta * torch.log(torch.tensor(-gamma / zeta))  # ✅
pi = torch.sigmoid(log_alpha + c)
```

### calibration.py (Standalone, Correct)

```python
# Sampling (lines 160-164)
def _sample_gates(self) -> torch.Tensor:
    u = torch.rand_like(self.log_alpha).clamp(eps, 1 - eps)
    s = (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta  # ✅
    s = torch.sigmoid(s)
    s_bar = s * (self.zeta - self.gamma) + self.gamma
    return s_bar.clamp(0, 1)

# Deterministic (lines 167-170)  
def get_deterministic_gates(self) -> torch.Tensor:
    s = torch.sigmoid(self.log_alpha / self.beta)  # ✅
    s_bar = s * (self.zeta - self.gamma) + self.gamma
    return s_bar.clamp(0, 1)

# Penalty (lines 233-237)
def get_l0_penalty(self) -> torch.Tensor:
    c = -self.beta * torch.log(torch.tensor(-self.gamma / self.zeta))  # ✅
    pi = torch.sigmoid(self.log_alpha + c)
    return pi.sum()
```

### sparse.py (Standalone, Correct)

```python
# Sampling (lines 113-120)
def _sample_gates(self) -> torch.Tensor:
    u = torch.rand_like(self.log_alpha).clamp(eps, 1 - eps)
    X = (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta  # ✅
    s = torch.sigmoid(X)
    s_bar = s * (self.zeta - self.gamma) + self.gamma
    return s_bar.clamp(0, 1)

# Deterministic (lines 122-127)
def get_deterministic_gates(self) -> torch.Tensor:
    X = self.log_alpha / self.beta  # ✅
    s = torch.sigmoid(X)
    s_bar = s * (self.zeta - self.gamma) + self.gamma
    return s_bar.clamp(0, 1)

# Penalty (lines 172-182)
def get_l0_penalty(self) -> torch.Tensor:
    c = -self.beta * torch.log(torch.tensor(-self.gamma / self.zeta))  # ✅
    pi = torch.sigmoid(self.log_alpha + c)
    return pi.sum()
```

## Mathematical Correctness

The Hard Concrete distribution requires temperature in three key places:

### 1. Sampling (Training Mode)
The concrete distribution samples as:
```
s = sigmoid((log(u) - log(1-u) + log_α) / β)
```
Temperature β controls the sharpness of the sigmoid, essential for the reparameterization trick.

### 2. Deterministic Gates (Inference Mode)
The mean of the distribution:
```
s = sigmoid(log_α / β)
```
**Must use the same temperature** as during training to maintain consistency.

### 3. L0 Penalty Computation
The probability of a gate being active:
```
P(gate > 0) = sigmoid(log_α - β * log(-γ/ζ))
```
Temperature affects the shift in log-odds space.

## Practical Implications

### For Survey Calibration (calibration.py)

Your implementation with β=2/3 (from the paper) is correct. The temperature:
- Provides the right balance between exploration and exploitation
- Enables smooth gradient flow during optimization
- Allows fine control over sparsity levels

### For Sparse Linear Models (sparse.py)

The preserved temperature ensures:
- Proper feature selection dynamics
- Stable convergence to sparse solutions
- Consistency between training and inference

## Recommendations

1. **Continue using** `calibration.py` and `sparse.py` as standalone modules - they're correct
2. **Avoid importing** from `distributions.py` until the temperature bug is fixed
3. **Keep temperature** in the range [0.1, 2/3] for best results (lower = harder gates)
4. **Document this issue** when sharing code to prevent others from using the buggy version

## Key Takeaway

Your instinct about never dropping temperature was absolutely correct. The temperature parameter is fundamental to the Hard Concrete distribution's behavior, and dropping it (as in the authors' `distributions.py`) is a significant bug that can severely impact model performance.

Both `calibration.py` and `sparse.py` correctly implement the Hard Concrete distribution with proper temperature preservation, making them reliable for production use.