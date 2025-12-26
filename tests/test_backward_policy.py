"""Tests for backward policy."""

import pytest
import torch

from gflownet_peptide.models.backward_policy import BackwardPolicy, LearnedBackwardPolicy


class TestBackwardPolicy:
    """Test suite for BackwardPolicy (uniform)."""

    @pytest.fixture
    def policy(self):
        """Create a uniform backward policy for testing."""
        return BackwardPolicy(use_uniform=True)

    def test_uniform_log_prob(self, policy):
        """Uniform backward policy should return log(1) = 0."""
        batch_size = 8
        seq_len = 10

        current_seq = torch.randint(0, 20, (batch_size, seq_len))
        parent_seq = current_seq[:, :-1]

        log_prob = policy.log_prob(current_seq, parent_seq)

        assert log_prob.shape == (batch_size,)
        assert torch.allclose(log_prob, torch.zeros(batch_size))

    def test_forward_alias(self, policy):
        """forward() should be an alias for log_prob()."""
        batch_size = 4
        seq_len = 5

        current_seq = torch.randint(0, 20, (batch_size, seq_len))
        parent_seq = current_seq[:, :-1]

        log_prob_direct = policy.log_prob(current_seq, parent_seq)
        log_prob_forward = policy.forward(current_seq, parent_seq)

        assert torch.allclose(log_prob_direct, log_prob_forward)

    def test_device_consistency(self, policy):
        """Output should be on same device as input."""
        batch_size = 4
        seq_len = 5

        # CPU test
        current_seq = torch.randint(0, 20, (batch_size, seq_len))
        parent_seq = current_seq[:, :-1]
        log_prob = policy.log_prob(current_seq, parent_seq)
        assert log_prob.device == current_seq.device

        # GPU test if available
        if torch.cuda.is_available():
            current_seq_cuda = current_seq.cuda()
            parent_seq_cuda = parent_seq.cuda()
            log_prob_cuda = policy.log_prob(current_seq_cuda, parent_seq_cuda)
            assert log_prob_cuda.device == current_seq_cuda.device

    def test_batch_size_one(self, policy):
        """Should work with batch size of 1."""
        current_seq = torch.randint(0, 20, (1, 5))
        parent_seq = current_seq[:, :-1]

        log_prob = policy.log_prob(current_seq, parent_seq)

        assert log_prob.shape == (1,)
        assert log_prob.item() == 0.0

    def test_varying_batch_sizes(self, policy):
        """Should work with various batch sizes."""
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            current_seq = torch.randint(0, 20, (batch_size, 10))
            parent_seq = current_seq[:, :-1]

            log_prob = policy.log_prob(current_seq, parent_seq)

            assert log_prob.shape == (batch_size,)
            assert torch.allclose(log_prob, torch.zeros(batch_size))

    def test_non_uniform_raises_not_implemented(self):
        """Non-uniform backward policy should raise NotImplementedError."""
        policy = BackwardPolicy(use_uniform=False)

        current_seq = torch.randint(0, 20, (4, 5))
        parent_seq = current_seq[:, :-1]

        with pytest.raises(NotImplementedError):
            policy.log_prob(current_seq, parent_seq)


class TestLearnedBackwardPolicy:
    """Test suite for LearnedBackwardPolicy (placeholder)."""

    @pytest.fixture
    def policy(self):
        """Create a learned backward policy for testing."""
        return LearnedBackwardPolicy(
            vocab_size=23,
            d_model=64,
            n_layers=2,
            n_heads=4,
        )

    def test_log_prob_shape(self, policy):
        """Should return correct shape for log probabilities."""
        batch_size = 4
        seq_len = 10

        current_seq = torch.randint(0, 20, (batch_size, seq_len))
        removed_token = torch.randint(0, 20, (batch_size,))

        log_prob = policy.log_prob(current_seq, removed_token)

        assert log_prob.shape == (batch_size,)

    def test_log_prob_valid_range(self, policy):
        """Log probabilities should be non-positive."""
        batch_size = 4
        seq_len = 10

        current_seq = torch.randint(0, 20, (batch_size, seq_len))
        removed_token = torch.randint(0, 20, (batch_size,))

        log_prob = policy.log_prob(current_seq, removed_token)

        assert torch.all(log_prob <= 0)

    def test_gradient_flow(self, policy):
        """Gradients should flow through the model."""
        batch_size = 4
        seq_len = 10

        current_seq = torch.randint(0, 20, (batch_size, seq_len))
        removed_token = torch.randint(0, 20, (batch_size,))

        log_prob = policy.log_prob(current_seq, removed_token)
        loss = -log_prob.mean()
        loss.backward()

        # Check that parameters received gradients
        for param in policy.parameters():
            assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
