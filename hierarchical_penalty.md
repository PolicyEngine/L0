# Hierarchical Penalty for Geographic Calibration

This is describing a TODO

## Problem Statement

When calibrating weights at a granular geographic level (e.g., congressional districts), we want to ensure that aggregations to higher geographic levels (states, national) remain consistent with known totals at those levels. Instead of adding redundant rows to the calibration matrix, we implement this as a penalty term in the loss function.

## Mathematical Formulation

### Base Problem

Given:
- Matrix `X` of shape `(n_targets, n_features)` where each row represents a geographic-specific target
- Target values `T` of length `n_targets`
- Weights `w` of length `n_features`

The base loss function is:
```
L_orig(w) = Σ_i ((X_i·w - T_i) / T_i)²
```

### Hierarchical Penalty

We add a penalty term that measures consistency at aggregate levels:

```
P(w) = Σ_agg ((Σ_j∈agg X_j·w - Σ_j∈agg T_j) / Σ_j∈agg T_j)²
```

Where `agg` represents each aggregation level (e.g., each state, national total).

The new loss function becomes:
```
L_new(w) = L_orig(w) + λ·P(w)
```

## Implementation Specification

### Geography Mapping Structure

The `geography_mapping` should be a dictionary with the following structure:

```python
geography_mapping = {
    'hierarchy': {
        'cd_to_state': {
            '0101': '01',  # CD 0101 belongs to state 01
            '0102': '01',  # CD 0102 belongs to state 01
            '0201': '02',  # CD 0201 belongs to state 02
            # ... for all CDs
        },
        'state_to_nation': {
            '01': 'US',
            '02': 'US',
            # ... all states map to US
        }
    },
    'target_indices': {
        '0101': [0, 1, 2, ...],    # Indices in X/T for CD 0101
        '0102': [50, 51, ...],     # Indices in X/T for CD 0102
        # ... for all geographic units
    },
    'aggregation_targets': {
        'state': {
            '01': {
                'indices': [1000, 1001, ...],  # Where state 01's targets would be
                'values': [100000, 200000, ...]  # Actual state-level target values
            },
            '02': {...},
            # ... for all states
        },
        'national': {
            'US': {
                'indices': [2000, 2001, ...],  # Where national targets would be
                'values': [5000000, 10000000, ...]  # Actual national target values
            }
        }
    }
}
```

### Alternative Simpler Structure

A simpler mapping structure that just handles geographic aggregation:

```python
geography_mapping = {
    'cd_to_state': {
        '0101': '01',
        '0102': '01',
        '0201': '02',
        # ... for all CDs
    },
    'target_groups': {
        # Group indices that should sum to the same aggregate
        # Each tuple is (target_indices, aggregate_target_value)
        'state_01_pop': ([0, 50, 100], 1234567),  # Indices for pop targets in CDs of state 01
        'state_01_snap': ([1, 51, 101], 45678),   # Indices for SNAP in CDs of state 01
        'state_02_pop': ([150, 200], 2345678),    # Indices for pop in CDs of state 02
        # ...
        'national_pop': ([0, 50, 100, 150, 200, ...], 300000000),  # All pop targets
        'national_snap': ([1, 51, 101, 151, 201, ...], 50000000),  # All SNAP targets
    }
}
```

### Function Signature

```python
def add_hierarchical_penalty(
    loss_function,
    X: sparse.csr_matrix,
    targets: np.ndarray, 
    geography_mapping: dict,
    lambda_state: float = 1.0,
    lambda_national: float = 1.0
) -> callable:
    """
    Wraps a loss function to add hierarchical consistency penalties.
    
    Args:
        loss_function: Base loss function(w, X, targets) -> scalar
        X: Calibration matrix (n_targets x n_features)
        targets: Target values (n_targets,)
        geography_mapping: Geographic hierarchy and target mappings
        lambda_state: Weight for state-level consistency penalty
        lambda_national: Weight for national-level consistency penalty
    
    Returns:
        New loss function with hierarchical penalties
    """
```

## Implementation Details

### Computing State-Level Penalties

For each state:
1. Identify all CD target indices belonging to that state
2. Compute predicted sum: `state_pred = Σ(X[cd_indices] @ w)`
3. Compute target sum: `state_target = Σ(targets[cd_indices])`
4. Compute penalty: `((state_pred - state_target) / state_target)²`

### Computing National Penalty

1. Compute predicted sum: `national_pred = Σ(X @ w)`
2. Compute target sum: `national_target = Σ(targets)`
3. Compute penalty: `((national_pred - national_target) / national_target)²`

### Gradient Computation

The gradient of the penalty term with respect to w:

For state s:
```
∂P_s/∂w = 2 * ((Σ_j∈s X_j·w - Σ_j∈s T_j) / (Σ_j∈s T_j)²) * Σ_j∈s X_j
```

For national:
```
∂P_nat/∂w = 2 * ((Σ X·w - Σ T) / (Σ T)²) * Σ X
```

### Efficient Implementation

To avoid recomputing aggregations:
1. Pre-compute aggregation matrices `X_state` and `X_national` where each row is the sum of relevant CD rows
2. Pre-compute aggregate targets `T_state` and `T_national`
3. Then the penalty computation becomes simple matrix operations

```python
# Precompute once
X_states = []  # Each row is sum of CDs for that state
T_states = []  # Corresponding state targets

for state in states:
    cd_indices = get_cd_indices(state)
    X_states.append(X[cd_indices].sum(axis=0))
    T_states.append(targets[cd_indices].sum())

X_states = sparse.vstack(X_states)
T_states = np.array(T_states)

# During optimization
state_preds = X_states @ w
state_penalties = ((state_preds - T_states) / T_states) ** 2
```

## Usage Example

```python
from l0.hierarchical import add_hierarchical_penalty

# Set up geography mapping
geography_mapping = create_geography_mapping(targets_df)

# Create penalized loss function
penalized_loss = add_hierarchical_penalty(
    original_loss,
    X_sparse,
    targets,
    geography_mapping,
    lambda_state=0.1,
    lambda_national=0.05
)

# Use in optimization
model.fit(X, y, loss_fn=penalized_loss)
```

## Benefits

1. **No matrix expansion**: Don't need to add redundant rows for state/national targets
2. **Tunable enforcement**: Lambda parameters control strictness of hierarchical consistency
3. **Efficient computation**: Aggregations can be pre-computed
4. **Flexible hierarchy**: Can handle arbitrary geographic hierarchies (regions, divisions, etc.)

## Considerations

1. **Lambda tuning**: May need cross-validation to find optimal lambda values
2. **Different lambdas per variable**: Some variables (e.g., population) might need stricter consistency than others (e.g., income)
3. **Weighted penalties**: Could weight penalties by the importance/reliability of aggregate targets
