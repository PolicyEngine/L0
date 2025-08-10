# Examples

This page provides comprehensive examples of using L0 regularization in various scenarios.

## Basic Usage

### Simple Linear Layer

```python
import torch
from l0.layers import L0Linear

# Create a sparse linear layer
layer = L0Linear(
    in_features=100,
    out_features=50,
    temperature=0.5,
    init_sparsity=0.8  # Start with 80% sparsity
)

# Forward pass
x = torch.randn(32, 100)
output = layer(x)

# Get L0 penalty for regularization
l0_penalty = layer.get_l0_penalty()
print(f"Active parameters: {l0_penalty.item():.0f}")
print(f"Sparsity: {layer.get_sparsity():.2%}")
```

### Training a Sparse Neural Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from l0.layers import L0Linear

class SparseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, sparsity=0.9):
        super().__init__()
        self.fc1 = L0Linear(input_dim, hidden_dim, init_sparsity=sparsity)
        self.fc2 = L0Linear(hidden_dim, hidden_dim, init_sparsity=sparsity)
        self.fc3 = L0Linear(hidden_dim, num_classes, init_sparsity=sparsity)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_l0_loss(self):
        l0_loss = 0
        for module in self.modules():
            if hasattr(module, 'get_l0_penalty'):
                l0_loss += module.get_l0_penalty()
        return l0_loss

# Create model and optimizer
model = SparseClassifier(784, 256, 10, sparsity=0.95)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    for x, y in train_loader:
        # Forward pass
        logits = model(x)
        
        # Compute losses
        task_loss = F.cross_entropy(logits, y)
        l0_loss = model.get_l0_loss()
        total_loss = task_loss + 1e-3 * l0_loss  # λ = 0.001
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

## Convolutional Networks

### Sparse CNN for Image Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from l0.layers import L0Conv2d, L0Linear

class SparseCNN(nn.Module):
    def __init__(self, num_classes=10, init_sparsity=0.8):
        super().__init__()
        # Convolutional layers with L0
        self.conv1 = L0Conv2d(3, 32, kernel_size=3, init_sparsity=init_sparsity)
        self.conv2 = L0Conv2d(32, 64, kernel_size=3, init_sparsity=init_sparsity)
        self.conv3 = L0Conv2d(64, 128, kernel_size=3, init_sparsity=init_sparsity)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with L0
        self.fc1 = L0Linear(128 * 3 * 3, 256, init_sparsity=init_sparsity)
        self.fc2 = L0Linear(256, num_classes, init_sparsity=init_sparsity)
        
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Conv block 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Conv block 3
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_total_sparsity(self):
        total_params = 0
        active_params = 0
        
        for module in self.modules():
            if hasattr(module, 'get_l0_penalty'):
                penalty = module.get_l0_penalty().item()
                active_params += penalty
                
                # Calculate total parameters
                if hasattr(module, 'weight'):
                    total_params += module.weight.numel()
        
        return 1 - (active_params / total_params)

# Example usage
model = SparseCNN(num_classes=10, init_sparsity=0.9)
print(f"Initial model sparsity: {model.get_total_sparsity():.2%}")
```

### Structured Sparsity (Channel Pruning)

```python
from l0.layers import L0Conv2d

class StructuredSparseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Use structured=True for channel-wise sparsity
        self.conv1 = L0Conv2d(
            3, 64, kernel_size=3,
            structured=True,  # Prune entire channels
            init_sparsity=0.5
        )
        self.conv2 = L0Conv2d(
            64, 128, kernel_size=3,
            structured=True,
            init_sparsity=0.6
        )
        self.conv3 = L0Conv2d(
            128, 256, kernel_size=3,
            structured=True,
            init_sparsity=0.7
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
    
    def get_active_channels(self):
        """Count active channels in each layer."""
        active = {}
        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d) and module.structured:
                gates = module.channel_gates.get_active_prob()
                active[name] = (gates > 0.5).sum().item()
        return active

model = StructuredSparseCNN()
print("Active channels per layer:", model.get_active_channels())
```

## Feature Selection

### Using FeatureGate for Tabular Data

```python
from l0.gates import FeatureGate
import pandas as pd
import torch

# Load tabular data
df = pd.read_csv('data.csv')
feature_names = df.columns[:-1].tolist()
X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

# Create feature selection gate
feature_gate = FeatureGate(
    n_features=len(feature_names),
    max_features=10,  # Select at most 10 features
    temperature=0.2
)

# Train feature selection
optimizer = torch.optim.Adam(feature_gate.parameters(), lr=0.01)

for epoch in range(100):
    # Select features
    selected_X, selected_names = feature_gate.select_features_with_names(
        X, feature_names
    )
    
    # Use selected features for downstream task
    # ... (train classifier on selected_X)
    
    # Optimize feature selection
    penalty = feature_gate.get_penalty()
    target_loss = (penalty - 10) ** 2  # Target 10 features
    
    optimizer.zero_grad()
    target_loss.backward()
    optimizer.step()

# Get final selected features
feature_gate.eval()
importance = feature_gate.get_feature_importance()
top_features = [feature_names[i] for i in torch.topk(importance, 10).indices]
print(f"Selected features: {top_features}")
```

## Sample Selection

### Active Learning with SampleGate

```python
from l0.gates import SampleGate
import torch

class ActiveLearner:
    def __init__(self, n_samples, n_select, model):
        self.sample_gate = SampleGate(
            n_samples=n_samples,
            target_samples=n_select,
            temperature=0.3
        )
        self.model = model
        
    def select_samples(self, X, y=None):
        """Select most informative samples."""
        self.sample_gate.eval()
        
        # Get model predictions for uncertainty
        with torch.no_grad():
            outputs = self.model(X)
            probs = F.softmax(outputs, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1)
        
        # Weight samples by uncertainty
        selected_X, selected_entropy, indices = \
            self.sample_gate.select_weighted_samples(X, entropy)
        
        return selected_X, indices
    
    def update_selection(self, X, y, selected_indices):
        """Update gate parameters based on selection performance."""
        optimizer = torch.optim.Adam(self.sample_gate.parameters(), lr=0.01)
        
        # Train on selected samples
        selected_X = X[selected_indices]
        selected_y = y[selected_indices]
        
        # ... train model on selected samples ...
        
        # Update gates based on performance
        penalty = self.sample_gate.get_penalty()
        loss = penalty  # Minimize number of samples
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Example usage
n_samples = 10000
model = SparseClassifier(784, 256, 10)
learner = ActiveLearner(n_samples, n_select=1000, model=model)

X = torch.randn(n_samples, 784)
selected_X, indices = learner.select_samples(X)
print(f"Selected {len(indices)} samples for labeling")
```

## Advanced Techniques

### Temperature Annealing

```python
from l0.penalties import TemperatureScheduler, update_temperatures

# Create temperature scheduler
scheduler = TemperatureScheduler(
    initial_temp=2.0,  # Start with soft gates
    final_temp=0.1,    # End with hard gates
    anneal_epochs=50   # Anneal over 50 epochs
)

# Training with temperature annealing
model = SparseMLP(784, 256, 10, init_sparsity=0.9)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    # Update temperature
    temp = scheduler.get_temperature(epoch)
    update_temperatures(model, temp)
    
    print(f"Epoch {epoch}: Temperature = {temp:.3f}")
    
    # Training loop
    for x, y in train_loader:
        # ... training step ...
        pass
```

### Combined L0L2 Regularization

```python
from l0.layers import L0Linear
from l0.penalties import compute_l0l2_penalty

class L0L2Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Enable L2 penalty computation
        self.fc1 = L0Linear(100, 50, use_l2=True, init_sparsity=0.8)
        self.fc2 = L0Linear(50, 25, use_l2=True, init_sparsity=0.8)
        self.fc3 = L0Linear(25, 10, use_l2=True, init_sparsity=0.8)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = L0L2Model()
optimizer = torch.optim.Adam(model.parameters())

# Training with L0L2
for x, y in train_loader:
    logits = model(x)
    
    # Task loss
    task_loss = F.cross_entropy(logits, y)
    
    # Combined L0L2 penalty
    penalty = compute_l0l2_penalty(
        model,
        l0_lambda=1e-3,  # L0 weight
        l2_lambda=1e-4   # L2 weight
    )
    
    total_loss = task_loss + penalty
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Model Pruning and Deployment

```python
from l0.layers import prune_model
from l0.penalties import get_sparsity_stats
import torch

# Train model with L0
model = SparseCNN(num_classes=10, init_sparsity=0.95)
# ... training loop ...

# Evaluate sparsity
stats = get_sparsity_stats(model)
for name, layer_stats in stats.items():
    print(f"{name}: {layer_stats['sparsity']:.1%} sparse")

# Prune model for deployment
model.eval()
pruned_model = prune_model(model, threshold=0.05)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
active_params = sum((p != 0).sum().item() for p in pruned_model.parameters())

print(f"Compression ratio: {total_params/active_params:.1f}x")
print(f"Model size reduction: {(1 - active_params/total_params):.1%}")

# Save pruned model
torch.save({
    'model_state_dict': pruned_model.state_dict(),
    'sparsity_stats': stats,
}, 'pruned_model.pt')

# Load for inference
checkpoint = torch.load('pruned_model.pt')
inference_model = SparseCNN(num_classes=10)
inference_model.load_state_dict(checkpoint['model_state_dict'])
inference_model.eval()
```

## Integration with PolicyEngine

### Household Calibration with L0

```python
from l0.gates import HybridGate
import torch

class HouseholdCalibrator:
    """
    Use L0 regularization for household weight calibration
    in PolicyEngine's CPS enhancement.
    """
    
    def __init__(self, n_households, target_households=50000):
        # Hybrid selection: some via L0, some random
        self.selector = HybridGate(
            n_items=n_households,
            l0_fraction=0.25,      # 25% selected by importance
            random_fraction=0.75,  # 75% random for diversity
            target_items=target_households,
            temperature=0.25       # PolicyEngine default
        )
        
    def calibrate_weights(self, household_data, population_targets):
        """
        Select and reweight households to match population targets.
        """
        # Select households
        selected_data, indices, selection_type = self.selector.select(
            household_data
        )
        
        # Separate L0 and random selections
        l0_mask = selection_type == "l0"
        random_mask = selection_type == "random"
        
        # Apply different weight adjustments
        weights = torch.ones(len(indices))
        weights[l0_mask] *= 1.2   # Higher weight for L0-selected
        weights[random_mask] *= 0.9  # Lower weight for random
        
        # Normalize to match targets
        weights = weights * (population_targets.sum() / weights.sum())
        
        return selected_data, weights, indices
    
    def optimize_selection(self, loss_fn):
        """
        Optimize household selection based on calibration loss.
        """
        optimizer = torch.optim.Adam(self.selector.parameters(), lr=0.01)
        
        for _ in range(100):
            # Get current selection
            penalty = self.selector.l0_gate.get_penalty()
            
            # Compute calibration loss
            calib_loss = loss_fn()
            
            # Combined objective
            total_loss = calib_loss + 1e-4 * penalty
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

# Usage in PolicyEngine context
n_households = 200000  # Full CPS
target_households = 50000  # Enhanced subset

calibrator = HouseholdCalibrator(n_households, target_households)

# Household data (from CPS)
household_data = torch.randn(n_households, 100)  # Features
population_targets = torch.tensor([1e8])  # US population

# Calibrate
selected, weights, indices = calibrator.calibrate_weights(
    household_data, population_targets
)

print(f"Selected {len(indices)} households")
print(f"Average weight: {weights.mean():.2f}")
print(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
```

## Performance Tips

### GPU Acceleration

```python
# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
model = SparseMLP(784, 256, 10).to(device)

# Training on GPU
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    output = model(x)
    # ... rest of training ...
```

### Batch Processing

```python
# Process large datasets in batches
def batch_feature_selection(X, feature_names, batch_size=10000):
    feature_gate = FeatureGate(X.shape[1], max_features=20)
    feature_gate.eval()
    
    selected_features = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        selected, names = feature_gate.select_features_with_names(
            batch, feature_names
        )
        selected_features.append(selected)
    
    return torch.cat(selected_features, dim=0)
```

### Memory-Efficient Training

```python
# Use gradient accumulation for large models
accumulation_steps = 4
model = SparseCNN(init_sparsity=0.95)
optimizer = torch.optim.Adam(model.parameters())

for i, (x, y) in enumerate(train_loader):
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## References

For more details on the L0 regularization method, see {cite}`louizos2018learning`.