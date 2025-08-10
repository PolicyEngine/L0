"""Tests for penalty computation utilities."""

import pytest
import torch
import torch.nn as nn
from typing import Dict


class TestPenalties:
    """Test suite for penalty functions."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model with L0 layers."""
        from l0.layers import L0Linear

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = L0Linear(10, 20, init_sparsity=0.3, use_l2=True)
                self.fc2 = L0Linear(20, 10, init_sparsity=0.5, use_l2=True)
                self.fc3 = nn.Linear(10, 5)  # Regular layer

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        return SimpleModel()

    def test_compute_l0_penalty(self, simple_model):
        """Test computing total L0 penalty."""
        from l0.penalties import compute_l0_penalty

        penalty = compute_l0_penalty(simple_model)

        assert penalty.shape == ()
        assert penalty.item() > 0

        # Manual computation
        expected = 0
        expected += simple_model.fc1.get_l0_penalty().item()
        expected += simple_model.fc2.get_l0_penalty().item()

        assert abs(penalty.item() - expected) < 1e-5

    def test_compute_l2_penalty(self, simple_model):
        """Test computing L2 penalty."""
        from l0.penalties import compute_l2_penalty

        penalty = compute_l2_penalty(simple_model)

        assert penalty.shape == ()
        assert penalty.item() > 0

        # Should include all layers with weights
        expected = 0
        expected += (simple_model.fc1.weight**2).sum().item()
        expected += (simple_model.fc2.weight**2).sum().item()
        expected += (simple_model.fc3.weight**2).sum().item()

        assert abs(penalty.item() - expected) < 1e-5

    def test_compute_l0l2_penalty(self, simple_model):
        """Test computing combined L0L2 penalty."""
        from l0.penalties import compute_l0l2_penalty

        penalty = compute_l0l2_penalty(
            simple_model, l0_lambda=1e-3, l2_lambda=1e-4
        )

        assert penalty.shape == ()
        assert penalty.item() > 0

        # Should be weighted sum
        l0 = compute_l0_penalty(simple_model)
        l2 = compute_l2_penalty(simple_model)
        expected = 1e-3 * l0 + 1e-4 * l2

        assert torch.allclose(penalty, expected)

    def test_get_sparsity_stats(self, simple_model):
        """Test getting sparsity statistics."""
        from l0.penalties import get_sparsity_stats

        stats = get_sparsity_stats(simple_model)

        assert "fc1" in stats
        assert "fc2" in stats
        assert "fc3" not in stats  # Regular layer

        for name, layer_stats in stats.items():
            assert "sparsity" in layer_stats
            assert "active_params" in layer_stats
            assert "total_params" in layer_stats
            assert 0 <= layer_stats["sparsity"] <= 1

    def test_get_active_parameter_count(self, simple_model):
        """Test counting active parameters."""
        from l0.penalties import get_active_parameter_count

        count = get_active_parameter_count(simple_model)

        assert isinstance(count, int)
        assert count > 0
        assert count <= 10 * 20 + 20 * 10  # Max possible L0 params

    def test_temperature_schedule(self):
        """Test temperature scheduling."""
        from l0.penalties import TemperatureScheduler

        scheduler = TemperatureScheduler(
            initial_temp=2.0, final_temp=0.1, anneal_epochs=100
        )

        # Initial temperature
        assert scheduler.get_temperature(0) == 2.0

        # Final temperature
        assert scheduler.get_temperature(100) == 0.1
        assert scheduler.get_temperature(150) == 0.1

        # Midpoint
        mid_temp = scheduler.get_temperature(50)
        assert 0.1 < mid_temp < 2.0

        # Monotonic decrease
        temps = [scheduler.get_temperature(i) for i in range(0, 101, 10)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]

    def test_update_temperatures(self, simple_model):
        """Test updating temperatures in model."""
        from l0.penalties import update_temperatures

        # Set initial temperatures
        update_temperatures(simple_model, 1.0)

        assert simple_model.fc1.weight_gates.temperature == 1.0
        assert simple_model.fc2.weight_gates.temperature == 1.0

        # Update temperatures
        update_temperatures(simple_model, 0.2)

        assert simple_model.fc1.weight_gates.temperature == 0.2
        assert simple_model.fc2.weight_gates.temperature == 0.2

    def test_penalty_tracker(self):
        """Test tracking penalties during training."""
        from l0.penalties import PenaltyTracker

        tracker = PenaltyTracker()

        # Log some values
        for i in range(10):
            tracker.log("l0_penalty", i * 10.0)
            tracker.log("l2_penalty", i * 5.0)
            tracker.log("sparsity", 0.1 + i * 0.05)

        # Get statistics
        stats = tracker.get_stats("l0_penalty")
        assert stats["mean"] == 45.0
        assert stats["min"] == 0.0
        assert stats["max"] == 90.0

        # Get history
        history = tracker.get_history("sparsity")
        assert len(history) == 10
        assert history[-1] == 0.55
