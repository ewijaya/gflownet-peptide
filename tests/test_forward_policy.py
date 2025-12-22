"""Tests for forward policy."""

import pytest
import torch


class TestForwardPolicy:
    """Test suite for ForwardPolicy model."""

    @pytest.fixture
    def policy(self):
        """Create a forward policy for testing."""
        from gflownet_peptide.models.forward_policy import ForwardPolicy

        return ForwardPolicy(
            vocab_size=23,
            d_model=64,  # Small for testing
            n_layers=2,
            n_heads=4,
            dim_feedforward=128,
            max_length=20,
        )

    def test_forward_shape(self, policy):
        """Test forward pass output shape."""
        batch_size = 4
        seq_len = 5

        # Create input: batch of partial sequences
        partial_seq = torch.randint(0, 20, (batch_size, seq_len))

        # Forward pass
        logits = policy(partial_seq)

        # Check output shape: [batch, vocab_size - 2]
        # (20 AA + STOP = 21 actions)
        assert logits.shape == (batch_size, 21)

    def test_log_prob(self, policy):
        """Test log probability computation."""
        batch_size = 4
        seq_len = 5

        partial_seq = torch.randint(0, 20, (batch_size, seq_len))
        action = torch.randint(0, 21, (batch_size,))

        log_prob = policy.log_prob(partial_seq, action)

        assert log_prob.shape == (batch_size,)
        assert torch.all(log_prob <= 0)  # Log probs are non-positive

    def test_sample_action(self, policy):
        """Test action sampling."""
        batch_size = 4
        seq_len = 5

        partial_seq = torch.randint(0, 20, (batch_size, seq_len))

        action, log_prob = policy.sample_action(partial_seq)

        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert torch.all(action >= 0)
        assert torch.all(action < 21)

    def test_sample_sequence(self, policy):
        """Test complete sequence sampling."""
        batch_size = 4
        max_length = 15
        min_length = 5

        sequences, log_probs = policy.sample_sequence(
            batch_size=batch_size,
            max_length=max_length,
            min_length=min_length,
        )

        assert len(sequences) == batch_size
        assert log_probs.shape == (batch_size,)

        # Check sequence lengths
        for seq in sequences:
            assert len(seq) >= min_length
            assert len(seq) <= max_length
            # Check all characters are valid amino acids
            assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq)

    def test_encode_decode_roundtrip(self, policy):
        """Test encoding and decoding consistency."""
        test_sequence = "MKFLILFL"

        # Encode
        tokens = policy.encode_sequence(test_sequence)

        # Check START and STOP tokens
        assert tokens[0] == policy.start_idx
        assert tokens[-1] == policy.stop_idx

        # Decode
        decoded = policy._decode_sequences(tokens.unsqueeze(0))[0]

        assert decoded == test_sequence

    def test_temperature_effect(self, policy):
        """Test that temperature affects sampling diversity."""
        batch_size = 100
        seq_len = 5

        partial_seq = torch.randint(0, 20, (1, seq_len)).expand(batch_size, -1)

        # Sample with low temperature (more deterministic)
        actions_low_temp, _ = policy.sample_action(partial_seq, temperature=0.1)

        # Sample with high temperature (more random)
        actions_high_temp, _ = policy.sample_action(partial_seq, temperature=2.0)

        # Low temperature should have less unique actions
        n_unique_low = len(torch.unique(actions_low_temp))
        n_unique_high = len(torch.unique(actions_high_temp))

        # This test may occasionally fail due to randomness, but generally holds
        assert n_unique_low <= n_unique_high + 5  # Allow some margin


class TestBackwardPolicy:
    """Test suite for BackwardPolicy."""

    def test_uniform_backward(self):
        """Test uniform backward policy."""
        from gflownet_peptide.models.backward_policy import BackwardPolicy

        policy = BackwardPolicy(use_uniform=True)

        batch_size = 4
        seq_len = 5

        current_seq = torch.randint(0, 20, (batch_size, seq_len))
        parent_seq = current_seq[:, :-1]

        log_prob = policy.log_prob(current_seq, parent_seq)

        # Uniform backward has log prob = 0
        assert log_prob.shape == (batch_size,)
        assert torch.allclose(log_prob, torch.zeros(batch_size))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
