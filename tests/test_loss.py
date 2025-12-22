"""Tests for loss functions."""

import pytest
import torch


class TestTrajectoryBalanceLoss:
    """Test suite for TB loss."""

    @pytest.fixture
    def loss_fn(self):
        """Create TB loss function."""
        from gflownet_peptide.training.loss import TrajectoryBalanceLoss

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


class TestSubTrajectoryBalanceLoss:
    """Test suite for SubTB loss."""

    def test_sub_tb_loss(self):
        """Test SubTB loss computation."""
        from gflownet_peptide.training.loss import SubTrajectoryBalanceLoss

        loss_fn = SubTrajectoryBalanceLoss(init_log_z=0.0)

        batch_size = 4
        seq_len = 10

        log_pf_per_step = torch.randn(batch_size, seq_len)
        log_pb_per_step = torch.zeros(batch_size, seq_len)
        log_rewards = torch.randn(batch_size)

        loss = loss_fn(log_pf_per_step, log_pb_per_step, log_rewards)

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
