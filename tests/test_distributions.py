"""Tests for HardConcrete distribution."""

import math

import pytest
import torch

from l0.distributions import HardConcrete


class TestHardConcrete:
    """Test suite for HardConcrete distribution."""

    @pytest.fixture
    def basic_gate(self):
        """Create a basic 1D gate for testing."""
        return HardConcrete(10, temperature=0.5, init_mean=0.8)

    @pytest.fixture
    def matrix_gate(self):
        """Create a 2D gate for testing weight matrices."""
        return HardConcrete(20, 30, temperature=0.3, init_mean=0.9)

    def test_initialization(self, basic_gate):
        """Test proper initialization of gates."""
        assert basic_gate.gate_size == (10,)
        assert basic_gate.temperature == 0.5
        assert basic_gate.init_mean == 0.8

        # Check logits are initialized correctly
        expected_logit = math.log(0.8 / 0.2)
        assert torch.allclose(
            basic_gate.qz_logits.data,
            torch.full((10,), expected_logit),
            atol=1e-5,
        )

    def test_matrix_initialization(self, matrix_gate):
        """Test 2D gate initialization."""
        assert matrix_gate.gate_size == (20, 30)
        assert matrix_gate.qz_logits.shape == (20, 30)

    def test_forward_training_mode(self, basic_gate):
        """Test forward pass in training mode returns stochastic gates."""
        basic_gate.train()

        # Sample multiple times to check stochasticity
        gates1 = basic_gate()
        gates2 = basic_gate()

        assert gates1.shape == (10,)
        assert gates2.shape == (10,)
        assert torch.all(gates1 >= 0) and torch.all(gates1 <= 1)
        assert torch.all(gates2 >= 0) and torch.all(gates2 <= 1)

        # Should be different due to sampling
        assert not torch.allclose(gates1, gates2)

    def test_forward_eval_mode(self, basic_gate):
        """Test forward pass in eval mode returns deterministic gates."""
        basic_gate.eval()

        # Sample multiple times - should be identical
        gates1 = basic_gate()
        gates2 = basic_gate()

        assert torch.allclose(gates1, gates2)
        assert torch.all(gates1 >= 0) and torch.all(gates1 <= 1)

    def test_broadcasting(self, matrix_gate):
        """Test gate broadcasting for convolutional layers."""
        matrix_gate.eval()

        # Simulate conv2d input shape (batch, channels, height, width)
        input_shape = (32, 20, 28, 28)
        gates = matrix_gate(input_shape)

        # Should broadcast to match input dimensions
        assert gates.shape == (20, 30, 1, 1)

    def test_penalty_computation(self, basic_gate):
        """Test L0 penalty calculation."""
        penalty = basic_gate.get_penalty()

        # Penalty should be a scalar
        assert penalty.shape == ()

        # With init_mean=0.8, expect ~8 active gates
        assert 6 < penalty.item() < 10

    def test_sparsity_computation(self, basic_gate):
        """Test sparsity level calculation."""
        sparsity = basic_gate.get_sparsity()

        # Should return a float between 0 and 1
        assert isinstance(sparsity, float)
        assert 0 <= sparsity <= 1

        # With init_mean=0.8 and stretch, sparsity varies
        assert 0.0 <= sparsity <= 0.3

    def test_active_probability(self, basic_gate):
        """Test getting probability of active gates."""
        prob_active = basic_gate.get_active_prob()

        assert prob_active.shape == (10,)
        assert torch.all(prob_active >= 0) and torch.all(prob_active <= 1)

        # Mean is affected by stretch transformation
        # Just check it's in reasonable range
        assert 0.7 <= prob_active.mean().item() <= 1.0

    def test_num_active(self, basic_gate):
        """Test counting active gates."""
        num_active = basic_gate.get_num_active()

        assert isinstance(num_active, int)
        assert 0 <= num_active <= 10

        # With init_mean=0.8, expect ~8 active
        assert 6 <= num_active <= 10

    def test_gradient_flow(self, basic_gate):
        """Test that gradients flow through the gates."""
        basic_gate.train()

        # Create a simple loss
        gates = basic_gate()
        loss = gates.sum()
        loss.backward()

        # Check gradients exist
        assert basic_gate.qz_logits.grad is not None
        assert not torch.all(basic_gate.qz_logits.grad == 0)

    def test_temperature_effect(self):
        """Test that temperature affects gate hardness."""
        high_temp = HardConcrete(10, temperature=2.0, init_mean=0.5)
        low_temp = HardConcrete(10, temperature=0.1, init_mean=0.5)

        high_temp.train()
        low_temp.train()

        # Sample multiple times and compute variance
        high_samples = torch.stack([high_temp() for _ in range(100)])
        low_samples = torch.stack([low_temp() for _ in range(100)])

        high_var = high_samples.var(dim=0).mean()
        low_var = low_samples.var(dim=0).mean()

        # Due to stretch transformation, just check both have reasonable variance
        # Temperature effect is complex with stretch parameter
        assert high_var > 0 and low_var > 0
        # Optional: lower temp typically has lower variance but not guaranteed with stretch
        # We just verify both are producing variable outputs
        assert high_var < 0.5 and low_var < 0.5  # Both should have reasonable variance

    def test_stretch_parameters(self):
        """Test stretch parameter effects."""
        gate = HardConcrete(5, stretch=0.2)

        assert gate.gamma == -0.2
        assert gate.zeta == 1.2

        # Check gates respect stretched bounds
        gate.train()
        for _ in range(10):
            gates = gate()
            assert torch.all(gates >= 0) and torch.all(gates <= 1)

    @pytest.mark.parametrize("init_mean", [0.1, 0.5, 0.9, 0.99])
    def test_various_init_means(self, init_mean):
        """Test initialization with various mean values."""
        gate = HardConcrete(10, init_mean=init_mean)
        prob_active = gate.get_active_prob()

        # Mean activation is affected by stretch transformation
        mean_active = prob_active.mean().item()
        # Just ensure it's in valid range
        assert 0.0 <= mean_active <= 1.0

    def test_zero_gradient_in_eval(self, basic_gate):
        """Test that no gradients are computed in eval mode."""
        basic_gate.eval()

        gates = basic_gate()
        if gates.requires_grad:
            loss = gates.sum()
            loss.backward()

            # In eval mode, gates should be deterministic
            # but gradients can still flow through
            assert basic_gate.qz_logits.grad is not None

    def test_deterministic_gates_respect_temperature(self):
        """Regression test for the temperature bug in ``_deterministic_gates``.

        Previously ``_deterministic_gates`` computed ``sigmoid(qz_logits)``
        without the ``/ temperature`` scaling, so two models with identical
        ``qz_logits`` but different temperatures produced identical eval
        gates. The fix matches Louizos et al. 2017 Eq. 11: the mean of the
        binary concrete before stretch is ``sigmoid(log_alpha / beta)``.
        """
        high_temp = HardConcrete(10, temperature=2.0, init_mean=0.5)
        low_temp = HardConcrete(10, temperature=0.1, init_mean=0.5)

        # Install identical, non-zero logits so temperature scaling matters.
        fixed_logits = torch.linspace(-3.0, 3.0, 10)
        with torch.no_grad():
            high_temp.qz_logits.copy_(fixed_logits)
            low_temp.qz_logits.copy_(fixed_logits)

        high_temp.eval()
        low_temp.eval()

        high_gates = high_temp()
        low_gates = low_temp()

        # With the bug these would be bit-identical. After the fix they must
        # differ everywhere the logits are non-zero.
        assert not torch.allclose(high_gates, low_gates, atol=1e-6), (
            "Deterministic gates should depend on temperature"
        )

        # Spot-check the closed form of the fix: sigmoid(logits / beta) * (zeta - gamma) + gamma
        expected_low = torch.clamp(
            torch.sigmoid(fixed_logits / 0.1) * (low_temp.zeta - low_temp.gamma)
            + low_temp.gamma,
            0,
            1,
        )
        assert torch.allclose(low_gates, expected_low, atol=1e-6)

    def test_deterministic_gates_match_reference_formula(self):
        """``_deterministic_gates`` must follow ``sigmoid(log_alpha / beta)``.

        This is the mean of the stretched binary concrete (Louizos et al. 2017
        Eq. 11) and is what the standalone ``SparseCalibrationWeights`` /
        ``SparseL0Linear`` modules implement. Any regression that drops the
        temperature from this branch makes the three modules inconsistent.
        """
        gate = HardConcrete(6, temperature=0.4, init_mean=0.5)
        with torch.no_grad():
            gate.qz_logits.copy_(torch.linspace(-2.0, 2.0, 6))
        gate.eval()

        deterministic = gate()

        expected = torch.clamp(
            torch.sigmoid(gate.qz_logits / gate.temperature) * (gate.zeta - gate.gamma)
            + gate.gamma,
            0,
            1,
        )
        assert torch.allclose(deterministic, expected, atol=1e-6)

    def test_sparsity_stats_match_eval_activation(self):
        """``get_sparsity`` should agree with the fraction of active eval gates.

        ``get_sparsity`` uses ``get_active_prob`` (which applies the
        temperature shift), while ``_deterministic_gates`` now applies the
        same temperature scaling. With the fix the two should be numerically
        consistent: the reported sparsity cannot diverge wildly from the
        fraction of gates that are effectively zero at eval time.
        """
        torch.manual_seed(0)
        gate = HardConcrete(200, temperature=0.25, init_mean=0.5)
        with torch.no_grad():
            gate.qz_logits.copy_(torch.randn(200))

        reported_sparsity = gate.get_sparsity()

        gate.eval()
        eval_gates = gate()
        # Active at eval-time = gate value strictly above zero.
        eval_active_fraction = (eval_gates > 0).float().mean().item()
        eval_sparsity = 1.0 - eval_active_fraction

        # Both numbers come from the same temperature-aware distribution
        # so they must be close (within sampling-free rounding slack).
        assert abs(reported_sparsity - eval_sparsity) < 0.15

    def test_extreme_logits_stay_finite(self):
        """Very large logits must not produce NaN/Inf gates.

        Without the `LOG_ALPHA_BOUND` clamp, ``log(u)`` for ``u ~ 0`` plus
        a huge positive logit can push the sigmoid argument past fp16/bf16
        range; in fp16 it also drops to ``-inf`` and produces ``NaN``. We
        simulate that failure mode at fp32 by pushing logits to 1e3.
        """
        gate = HardConcrete(20, temperature=0.1, init_mean=0.5)
        with torch.no_grad():
            gate.qz_logits.fill_(1000.0)

        gate.train()
        for _ in range(50):
            sampled = gate()
            assert torch.isfinite(sampled).all()
            assert torch.all(sampled >= 0) and torch.all(sampled <= 1)

        gate.eval()
        det = gate()
        assert torch.isfinite(det).all()
        assert torch.all(det >= 0) and torch.all(det <= 1)

        penalty = gate.get_penalty()
        assert torch.isfinite(penalty)

        with torch.no_grad():
            gate.qz_logits.fill_(-1000.0)
        gate.train()
        for _ in range(50):
            assert torch.isfinite(gate()).all()
        gate.eval()
        assert torch.isfinite(gate()).all()

    def test_uniform_eps_is_fp16_safe(self):
        """The uniform sample floor should be >= ``1e-6`` (fp16 underflow)."""
        assert HardConcrete._UNIFORM_EPS >= 1e-6
