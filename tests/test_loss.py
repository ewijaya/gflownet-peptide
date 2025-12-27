"""Tests for loss functions."""

import pytest
import torch

from gflownet_peptide.training.loss import (
    TrajectoryBalanceLoss,
    SubTrajectoryBalanceLoss,
    DetailedBalanceLoss,
)


class TestTrajectoryBalanceLoss:
    """Test suite for TB loss."""

    @pytest.fixture
    def loss_fn(self):
        """Create TB loss function."""
        return TrajectoryBalanceLoss(init_log_z=0.0)

    def test_loss_computation(self, loss_fn):
        """Test basic loss computation."""
        batch_size = 8

        log_pf_sum = torch.randn(batch_size)
        log_pb_sum = torch.zeros(batch_size)  # Uniform backward
        log_rewards = torch.randn(batch_size)

        loss = loss_fn(log_pf_sum, log_pb_sum, log_rewards)

        assert loss.shape == ()  # Scalar
        assert loss >= 0  # Squared loss is non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_finite(self, loss_fn):
        """Loss should be finite and non-negative."""
        log_pf = torch.randn(4, 5).sum(dim=-1)
        log_pb = torch.zeros(4)
        rewards = torch.rand(4) + 0.1
        log_rewards = torch.log(rewards)

        loss = loss_fn(log_pf, log_pb, log_rewards)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_log_z_gradient(self, loss_fn):
        """Test that log_Z receives gradients."""
        batch_size = 8

        log_pf_sum = torch.randn(batch_size, requires_grad=True)
        log_pb_sum = torch.zeros(batch_size)
        log_rewards = torch.randn(batch_size)

        loss = loss_fn(log_pf_sum, log_pb_sum, log_rewards)
        loss.backward()

        # log_Z should have gradient
        assert loss_fn.log_z.grad is not None

    def test_perfect_balance(self, loss_fn):
        """Test that perfect balance gives zero loss."""
        batch_size = 4

        # Manually construct balanced trajectories
        # log Z + log P_F = log R + log P_B
        log_z = 1.0
        log_pf_sum = torch.tensor([2.0, 3.0, 1.5, 2.5])
        log_pb_sum = torch.zeros(batch_size)
        log_rewards = log_z + log_pf_sum  # Perfect balance

        loss_fn.log_z.data = torch.tensor(log_z)
        loss = loss_fn(log_pf_sum, log_pb_sum, log_rewards)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_return_info(self, loss_fn):
        """Test return_info option returns dict with metrics."""
        batch_size = 4

        log_pf_sum = torch.randn(batch_size)
        log_pb_sum = torch.zeros(batch_size)
        log_rewards = torch.randn(batch_size)

        loss, info = loss_fn(log_pf_sum, log_pb_sum, log_rewards, return_info=True)

        assert isinstance(info, dict)
        assert 'loss' in info
        assert 'log_z' in info
        assert 'mean_log_pf' in info
        assert 'mean_log_reward' in info
        assert 'mean_reward' in info
        assert 'residual_mean' in info
        assert 'residual_std' in info

    def test_get_log_z(self, loss_fn):
        """get_log_z should return current log partition function value."""
        initial_log_z = loss_fn.get_log_z()
        assert isinstance(initial_log_z, float)

        # Update log_z
        loss_fn.log_z.data = torch.tensor(5.0)
        assert loss_fn.get_log_z() == 5.0

    def test_init_log_z(self):
        """log_Z should be initialized to specified value."""
        loss_fn = TrajectoryBalanceLoss(init_log_z=3.0)
        assert loss_fn.get_log_z() == 3.0

    def test_log_z_is_parameter(self, loss_fn):
        """log_Z should be a learnable parameter."""
        assert isinstance(loss_fn.log_z, torch.nn.Parameter)
        assert loss_fn.log_z.requires_grad

    def test_loss_decreases_with_training(self):
        """Loss should decrease when optimizing towards balance."""
        loss_fn = TrajectoryBalanceLoss(init_log_z=0.0)
        optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.1)

        # Fixed trajectory data
        log_pf_sum = torch.tensor([1.0, 2.0, 1.5, 2.5])
        log_pb_sum = torch.zeros(4)
        log_rewards = torch.tensor([2.0, 3.0, 2.5, 3.5])  # Target log_Z = 1.0

        initial_loss = loss_fn(log_pf_sum, log_pb_sum, log_rewards).item()

        # Train for a few steps
        for _ in range(100):
            optimizer.zero_grad()
            loss = loss_fn(log_pf_sum, log_pb_sum, log_rewards)
            loss.backward()
            optimizer.step()

        final_loss = loss_fn(log_pf_sum, log_pb_sum, log_rewards).item()

        assert final_loss < initial_loss
        # log_Z should converge towards 1.0
        assert abs(loss_fn.get_log_z() - 1.0) < 0.1


class TestSubTrajectoryBalanceLoss:
    """Test suite for SubTB loss."""

    @pytest.fixture
    def loss_fn(self):
        """Create SubTB loss function."""
        return SubTrajectoryBalanceLoss(init_log_z=0.0)

    def test_sub_tb_loss(self, loss_fn):
        """Test SubTB loss computation."""
        batch_size = 4
        seq_len = 10

        log_pf_per_step = torch.randn(batch_size, seq_len)
        log_pb_per_step = torch.zeros(batch_size, seq_len)
        log_rewards = torch.randn(batch_size)

        loss = loss_fn(log_pf_per_step, log_pb_per_step, log_rewards)

        # SubTB returns shape [1] due to accumulation
        assert loss.numel() == 1
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sub_tb_loss_finite(self, loss_fn):
        """Loss should be finite."""
        log_pf = torch.randn(4, 10)
        log_pb = torch.zeros(4, 10)
        log_rewards = torch.randn(4)

        loss = loss_fn(log_pf, log_pb, log_rewards)

        assert torch.isfinite(loss)

    def test_sub_tb_return_info(self, loss_fn):
        """Test return_info option."""
        log_pf = torch.randn(4, 10)
        log_pb = torch.zeros(4, 10)
        log_rewards = torch.randn(4)

        loss, info = loss_fn(log_pf, log_pb, log_rewards, return_info=True)

        assert isinstance(info, dict)
        assert 'loss' in info
        assert 'log_z' in info

    def test_sub_tb_gradient_flow(self, loss_fn):
        """Gradients should flow through SubTB loss."""
        log_pf = torch.randn(4, 10, requires_grad=True)
        log_pb = torch.zeros(4, 10)
        log_rewards = torch.randn(4)

        loss = loss_fn(log_pf, log_pb, log_rewards)
        loss.backward()

        assert log_pf.grad is not None
        assert loss_fn.log_z.grad is not None

    def test_lambda_sub_effect(self):
        """Different lambda_sub values should produce different losses."""
        log_pf = torch.randn(4, 10)
        log_pb = torch.zeros(4, 10)
        log_rewards = torch.randn(4)

        loss_fn_09 = SubTrajectoryBalanceLoss(init_log_z=0.0, lambda_sub=0.9)
        loss_fn_05 = SubTrajectoryBalanceLoss(init_log_z=0.0, lambda_sub=0.5)

        loss_09 = loss_fn_09(log_pf, log_pb, log_rewards)
        loss_05 = loss_fn_05(log_pf, log_pb, log_rewards)

        # Different lambda values should give different losses
        assert not torch.allclose(loss_09, loss_05)


class TestDetailedBalanceLoss:
    """Test suite for DB loss."""

    @pytest.fixture
    def flow_estimator(self):
        """Create a simple flow estimator."""
        return torch.nn.Linear(10, 1)

    @pytest.fixture
    def loss_fn(self, flow_estimator):
        """Create DB loss function."""
        return DetailedBalanceLoss(flow_estimator=flow_estimator)

    def test_db_loss_computation(self, loss_fn):
        """Test DB loss computation."""
        batch_size = 4
        state_dim = 10

        states = torch.randn(batch_size, state_dim)
        next_states = torch.randn(batch_size, state_dim)
        log_pf = torch.randn(batch_size)
        log_pb = torch.randn(batch_size)
        log_rewards = torch.randn(batch_size)
        is_terminal = torch.zeros(batch_size, dtype=torch.bool)

        loss = loss_fn(states, next_states, log_pf, log_pb, log_rewards, is_terminal)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_db_terminal_states(self, loss_fn):
        """Terminal states should use rewards instead of flow estimate."""
        batch_size = 4
        state_dim = 10

        states = torch.randn(batch_size, state_dim)
        next_states = torch.randn(batch_size, state_dim)
        log_pf = torch.randn(batch_size)
        log_pb = torch.randn(batch_size)
        log_rewards = torch.randn(batch_size)

        # All terminal
        is_terminal = torch.ones(batch_size, dtype=torch.bool)
        loss_terminal = loss_fn(states, next_states, log_pf, log_pb, log_rewards, is_terminal)

        # None terminal
        is_terminal = torch.zeros(batch_size, dtype=torch.bool)
        loss_non_terminal = loss_fn(states, next_states, log_pf, log_pb, log_rewards, is_terminal)

        # Losses should differ
        assert not torch.allclose(loss_terminal, loss_non_terminal)


class TestEntropyRegularization:
    """Test entropy regularization in TrajectoryBalanceLoss.

    Entropy regularization prevents mode collapse by penalizing
    overconfident policies. See docs/reward-comparison-analysis.md.
    """

    def test_default_entropy_weight_is_zero(self):
        """Default entropy weight should be 0 (backward compatible)."""
        loss_fn = TrajectoryBalanceLoss()
        assert loss_fn.entropy_weight == 0.0

    def test_entropy_weight_configurable(self):
        """Entropy weight should be configurable."""
        loss_fn = TrajectoryBalanceLoss(entropy_weight=0.05)
        assert loss_fn.entropy_weight == 0.05

    def test_zero_entropy_weight_matches_original(self):
        """With entropy_weight=0, should match original TB loss."""
        loss_fn_new = TrajectoryBalanceLoss(init_log_z=0.0, entropy_weight=0.0)

        log_pf = torch.tensor([-20.0, -25.0, -22.0, -18.0])
        log_pb = torch.zeros(4)
        log_rewards = torch.tensor([-1.0, -0.5, -0.8, -1.2])

        loss_new = loss_fn_new(log_pf, log_pb, log_rewards)

        # Compute expected TB loss manually
        residual = 0.0 + log_pf - log_rewards - log_pb
        expected_loss = (residual ** 2).mean()

        assert torch.isclose(loss_new, expected_loss, rtol=1e-5)

    def test_entropy_regularization_penalizes_confidence(self):
        """Confident policies (low log_pf) should have higher loss."""
        loss_fn = TrajectoryBalanceLoss(init_log_z=0.0, entropy_weight=0.1)

        log_pb = torch.zeros(4)
        log_rewards = torch.zeros(4)

        # Confident policy (very negative log_pf)
        log_pf_confident = torch.tensor([-70.0, -72.0, -68.0, -71.0])

        # Uncertain policy (less negative log_pf)
        log_pf_uncertain = torch.tensor([-20.0, -22.0, -18.0, -21.0])

        loss_confident = loss_fn(log_pf_confident, log_pb, log_rewards)
        loss_uncertain = loss_fn(log_pf_uncertain, log_pb, log_rewards)

        # Confident policy should have HIGHER total loss due to entropy penalty
        assert loss_confident > loss_uncertain

    def test_entropy_metrics_in_info(self):
        """Loss info should include entropy-related metrics."""
        loss_fn = TrajectoryBalanceLoss(entropy_weight=0.05)

        log_pf = torch.tensor([-25.0, -30.0, -22.0, -28.0])
        log_pb = torch.zeros(4)
        log_rewards = torch.tensor([-0.5, -0.3, -0.7, -0.4])

        loss, info = loss_fn(log_pf, log_pb, log_rewards, return_info=True)

        assert 'mean_entropy' in info
        assert 'entropy_reg' in info
        assert 'tb_loss' in info

        # Verify mean_entropy calculation: H = -E[log_P_F]
        expected_entropy = -log_pf.mean().item()
        assert abs(info['mean_entropy'] - expected_entropy) < 1e-5

    def test_subtb_entropy_weight(self):
        """SubTrajectoryBalanceLoss should also support entropy_weight."""
        loss_fn = SubTrajectoryBalanceLoss(entropy_weight=0.03)
        assert loss_fn.entropy_weight == 0.03

    def test_subtb_entropy_penalizes_confidence(self):
        """STB should also penalize overconfident policies."""
        loss_fn = SubTrajectoryBalanceLoss(init_log_z=0.0, entropy_weight=0.1)

        # Per-step log probs [batch=4, seq_len=5]
        log_pb = torch.zeros(4, 5)
        log_rewards = torch.zeros(4)

        # Confident policy (very negative per-step log probs)
        log_pf_confident = torch.full((4, 5), -14.0)  # sum = -70

        # Uncertain policy
        log_pf_uncertain = torch.full((4, 5), -4.0)   # sum = -20

        loss_confident = loss_fn(log_pf_confident, log_pb, log_rewards)
        loss_uncertain = loss_fn(log_pf_uncertain, log_pb, log_rewards)

        # Confident policy should have HIGHER total loss
        assert loss_confident > loss_uncertain

    def test_subtb_entropy_metrics_in_info(self):
        """STB loss info should include entropy-related metrics."""
        loss_fn = SubTrajectoryBalanceLoss(entropy_weight=0.05)

        log_pf = torch.randn(4, 10)
        log_pb = torch.zeros(4, 10)
        log_rewards = torch.randn(4)

        loss, info = loss_fn(log_pf, log_pb, log_rewards, return_info=True)

        assert 'mean_entropy' in info
        assert 'entropy_reg' in info
        assert 'stb_loss' in info

        # Verify mean_entropy calculation: H = -E[log_P_F]
        expected_entropy = -log_pf.sum(dim=1).mean().item()
        assert abs(info['mean_entropy'] - expected_entropy) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
