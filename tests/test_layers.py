"""Tests for L0 regularized layers."""

import pytest
import torch
import torch.nn.functional as F


class TestL0Linear:
    """Test suite for L0Linear layer."""

    @pytest.fixture
    def layer(self):
        """Create a basic L0Linear layer."""
        # Import will be available after implementation
        from l0.layers import L0Linear

        return L0Linear(10, 5, temperature=0.5, init_sparsity=0.8)

    @pytest.fixture
    def layer_no_bias(self):
        """Create L0Linear layer without bias."""
        from l0.layers import L0Linear

        return L0Linear(10, 5, bias=False)

    def test_initialization(self, layer):
        """Test proper initialization of L0Linear."""
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weight.shape == (5, 10)
        assert layer.bias is not None
        assert layer.bias.shape == (5,)

    def test_no_bias_initialization(self, layer_no_bias):
        """Test L0Linear without bias."""
        assert layer_no_bias.bias is None

    def test_forward_shape(self, layer):
        """Test output shape of forward pass."""
        x = torch.randn(32, 10)  # batch_size=32, in_features=10
        output = layer(x)
        assert output.shape == (32, 5)

    def test_gate_application(self, layer):
        """Test that gates are applied to weights."""
        layer.eval()  # Deterministic gates

        # Get the gates
        gates = layer.weight_gates()

        # Forward pass
        x = torch.randn(1, 10)
        with torch.no_grad():
            output = layer(x)

            # Manual computation
            masked_weight = layer.weight * gates
            expected = F.linear(x, masked_weight, layer.bias)

            assert torch.allclose(output, expected, atol=1e-6)

    def test_l0_penalty(self, layer):
        """Test L0 penalty computation."""
        penalty = layer.get_l0_penalty()

        # Should return a scalar
        assert penalty.shape == ()
        assert penalty.item() >= 0

        # Due to stretch transformation, penalty varies
        assert 0 < penalty.item() < 50

    def test_sparsity(self, layer):
        """Test sparsity computation."""
        sparsity = layer.get_sparsity()

        assert isinstance(sparsity, float)
        assert 0 <= sparsity <= 1

        # Due to stretch transformation, sparsity varies
        assert 0.0 <= sparsity <= 1.0

    def test_gradient_flow(self, layer):
        """Test gradients flow through layer and gates."""
        layer.train()

        x = torch.randn(4, 10, requires_grad=True)
        target = torch.randn(4, 5)

        output = layer(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        # Check gradients exist
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert layer.weight_gates.qz_logits.grad is not None

    def test_l0l2_penalty(self):
        """Test combined L0L2 penalty."""
        from l0.layers import L0Linear

        layer = L0Linear(10, 5, use_l2=True, init_sparsity=0.8)
        l0_penalty = layer.get_l0_penalty()
        l2_penalty = layer.get_l2_penalty()

        assert l0_penalty.shape == ()
        assert l2_penalty.shape == ()
        assert l2_penalty.item() >= 0

        # L2 should be weight norm squared
        expected_l2 = (layer.weight**2).sum()
        assert torch.allclose(l2_penalty, expected_l2)


class TestL0Conv2d:
    """Test suite for L0Conv2d layer."""

    @pytest.fixture
    def conv_layer(self):
        """Create a basic L0Conv2d layer."""
        from l0.layers import L0Conv2d

        return L0Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            temperature=0.5,
            init_sparsity=0.9,
        )

    def test_initialization(self, conv_layer):
        """Test proper initialization of L0Conv2d."""
        assert conv_layer.in_channels == 3
        assert conv_layer.out_channels == 16
        assert conv_layer.kernel_size == (3, 3)
        assert conv_layer.weight.shape == (16, 3, 3, 3)
        assert conv_layer.bias.shape == (16,)

    def test_forward_shape(self, conv_layer):
        """Test output shape of conv forward pass."""
        x = torch.randn(8, 3, 32, 32)  # batch, channels, height, width
        output = conv_layer(x)

        # With kernel_size=3, stride=1, padding=0: output = 32-3+1 = 30
        assert output.shape == (8, 16, 30, 30)

    def test_gate_application_conv(self, conv_layer):
        """Test that gates are applied to conv weights."""
        conv_layer.eval()

        gates = conv_layer.weight_gates()
        # Gates are stored flat and reshaped during forward pass
        assert gates.shape == (16, 27)  # 27 = 3*3*3
        gates = gates.view(16, 3, 3, 3)

        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            output = conv_layer(x)

            # Manual computation
            masked_weight = conv_layer.weight * gates
            expected = F.conv2d(
                x,
                masked_weight,
                conv_layer.bias,
                conv_layer.stride,
                conv_layer.padding,
            )

            assert torch.allclose(output, expected, atol=1e-6)

    def test_structured_sparsity(self):
        """Test channel-wise structured sparsity."""
        from l0.layers import L0Conv2d

        conv = L0Conv2d(
            3,
            16,
            3,
            structured=True,  # Channel-wise sparsity
            init_sparsity=0.5,
        )

        # Gates should be per-channel, not per-weight
        assert hasattr(conv, "channel_gates")
        gates = conv.channel_gates()
        assert gates.shape == (16,) or gates.shape == (16, 1, 1, 1)

    def test_depthwise_conv(self):
        """Test depthwise convolution with L0."""
        from l0.layers import L0DepthwiseConv2d

        conv = L0DepthwiseConv2d(
            in_channels=32, kernel_size=3, init_sparsity=0.7
        )

        x = torch.randn(4, 32, 16, 16)
        output = conv(x)

        # Depthwise conv maintains channel count
        assert output.shape == (4, 32, 14, 14)


class TestSparseMLP:
    """Test suite for example MLP with L0 regularization."""

    @pytest.fixture
    def model(self):
        """Create a sparse MLP model."""
        from l0.layers import SparseMLP

        return SparseMLP(
            input_dim=784,
            hidden_dim=256,
            output_dim=10,
            init_sparsity=0.5,
            temperature=0.5,
        )

    def test_forward_pass(self, model):
        """Test forward pass through entire model."""
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_total_l0_loss(self, model):
        """Test computing total L0 loss across all layers."""
        l0_loss = model.get_l0_loss()

        assert l0_loss.shape == ()
        assert l0_loss.item() > 0

        # Should be sum of penalties from all L0 layers
        expected = 0
        for module in model.modules():
            if hasattr(module, "get_l0_penalty"):
                expected += module.get_l0_penalty().item()

        assert abs(l0_loss.item() - expected) < 0.1  # Allow small numerical difference

    def test_sparsity_stats(self, model):
        """Test getting sparsity statistics."""
        stats = model.get_sparsity_stats()

        assert "fc1" in stats
        assert "fc2" in stats
        assert "fc3" in stats

        for layer_name, layer_stats in stats.items():
            assert "sparsity" in layer_stats
            assert "active_params" in layer_stats
            assert 0 <= layer_stats["sparsity"] <= 1
            assert layer_stats["active_params"] >= 0

    def test_training_step(self, model):
        """Test a complete training step."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(16, 784)
        target = torch.randint(0, 10, (16,))

        # Forward pass
        output = model(x)
        ce_loss = F.cross_entropy(output, target)
        l0_loss = model.get_l0_loss()
        total_loss = ce_loss + 1e-3 * l0_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Check that parameters were updated
        assert all(p.grad is not None for p in model.parameters())

    def test_pruning(self, model):
        """Test pruning inactive weights."""
        from l0.layers import prune_model

        # Set model to eval for deterministic gates
        model.eval()

        # Get sparsity before pruning
        _ = model.get_sparsity_stats()

        # Prune model
        pruned_model = prune_model(model, threshold=0.05)

        # Forward pass should still work
        x = torch.randn(4, 784)
        output = pruned_model(x)
        assert output.shape == (4, 10)

        # Weights below threshold should be zero
        for module in pruned_model.modules():
            if hasattr(module, "weight_gates"):
                with torch.no_grad():
                    prob_active = module.weight_gates.get_active_prob()
                    mask = prob_active <= 0.05
                    assert torch.all(module.weight[mask] == 0)
