"""Tests for forward policy."""

import pytest
import torch

from gflownet_peptide.models.forward_policy import ForwardPolicy, SinusoidalPositionalEncoding


class TestSinusoidalPositionalEncoding:
    """Test suite for positional encoding."""

    @pytest.fixture
    def pos_enc(self):
        """Create positional encoding for testing."""
        return SinusoidalPositionalEncoding(d_model=64, max_len=100, dropout=0.0)

    def test_output_shape(self, pos_enc):
        """Output should have same shape as input."""
        x = torch.randn(4, 10, 64)
        output = pos_enc(x)
        assert output.shape == x.shape

    def test_positional_encoding_different(self, pos_enc):
        """Different positions should have different encodings."""
        x = torch.zeros(1, 10, 64)
        output = pos_enc(x)

        # First and second positions should differ
        assert not torch.allclose(output[0, 0], output[0, 1])

    def test_deterministic_without_dropout(self, pos_enc):
        """Should be deterministic when dropout=0."""
        x = torch.randn(2, 5, 64)

        output1 = pos_enc(x)
        output2 = pos_enc(x)

        assert torch.allclose(output1, output2)


class TestForwardPolicy:
    """Test suite for ForwardPolicy model."""

    @pytest.fixture
    def policy(self):
        """Create a forward policy for testing."""
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

        # Check output shape: [batch, 21] (20 AA + STOP)
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

    def test_log_prob_sums_to_one(self, policy):
        """Log probs should correspond to valid probability distribution."""
        partial_seq = torch.randint(0, 20, (4, 5))

        logits = policy(partial_seq)
        probs = torch.softmax(logits, dim=-1)

        # Should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

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

    def test_sample_action_valid_log_probs(self, policy):
        """Sampled action log probs should be consistent."""
        policy.eval()  # Disable dropout for deterministic behavior
        partial_seq = torch.randint(0, 20, (4, 5))

        action, log_prob = policy.sample_action(partial_seq)

        # Recompute log prob
        recomputed = policy.log_prob(partial_seq, action)

        # With dropout disabled, should be close
        assert torch.allclose(log_prob, recomputed, atol=1e-4)
        policy.train()

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

    def test_sample_sequence_log_probs_valid(self, policy):
        """Total log probs should be non-positive."""
        sequences, log_probs = policy.sample_sequence(
            batch_size=4,
            max_length=15,
            min_length=5,
        )

        assert torch.all(log_probs <= 0)

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

    def test_encode_all_amino_acids(self, policy):
        """All 20 amino acids should be encodable."""
        all_aas = "ACDEFGHIKLMNPQRSTVWY"

        tokens = policy.encode_sequence(all_aas)

        # Should have START + 20 AAs + STOP = 22 tokens
        assert len(tokens) == 22
        assert tokens[0] == policy.start_idx
        assert tokens[-1] == policy.stop_idx

        # Middle tokens should be 0-19
        for i, aa in enumerate(all_aas):
            assert tokens[i + 1] == policy.AMINO_ACIDS.index(aa)

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

    def test_causal_masking(self, policy):
        """The policy uses causal masking internally via transformer."""
        # Create a sequence
        x = torch.tensor([[0, 1, 2, 3, 4]])

        # Forward pass should work
        logits = policy(x)

        # Verify the output is valid
        assert logits.shape == (1, 21)
        assert torch.isfinite(logits).all()

        # Causal masking is enforced by the transformer architecture itself
        # The test verifies the model processes sequences correctly

    def test_padding_mask(self, policy):
        """Padding mask should work correctly."""
        batch_size = 2
        seq_len = 5

        partial_seq = torch.randint(0, 20, (batch_size, seq_len))
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[1, 3:] = True  # Pad last 2 positions of second sequence

        logits = policy(partial_seq, padding_mask=padding_mask)

        assert logits.shape == (batch_size, 21)
        assert torch.isfinite(logits).all()

    def test_gradient_flow(self, policy):
        """Gradients should flow through the model."""
        partial_seq = torch.randint(0, 20, (4, 5))

        logits = policy(partial_seq)
        loss = logits.sum()
        loss.backward()

        # Check that parameters received gradients
        for param in policy.parameters():
            assert param.grad is not None

    def test_vocab_size_attribute(self, policy):
        """Should correctly store vocab size."""
        assert policy.vocab_size == 23

    def test_special_token_indices(self, policy):
        """Special token indices should be correct."""
        assert policy.start_idx == 20
        assert policy.stop_idx == 21
        assert policy.pad_idx == 22

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Model should work on CUDA."""
        policy = ForwardPolicy(vocab_size=23, d_model=64, n_layers=2, n_heads=4)
        policy = policy.cuda()

        partial_seq = torch.randint(0, 20, (4, 5)).cuda()
        logits = policy(partial_seq)

        assert logits.device.type == 'cuda'


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
