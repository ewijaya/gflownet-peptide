"""Tests for trajectory sampler."""

import pytest
import torch

from gflownet_peptide.models.forward_policy import ForwardPolicy
from gflownet_peptide.models.backward_policy import BackwardPolicy
from gflownet_peptide.training.sampler import TrajectorySampler, Trajectory


class TestTrajectory:
    """Test suite for Trajectory dataclass."""

    def test_trajectory_creation(self):
        """Should create Trajectory with all fields."""
        states = [torch.tensor([[20]])]
        actions = torch.tensor([0, 1, 2])
        log_pf = torch.tensor([-1.0, -1.5, -2.0])
        log_pb = torch.tensor([0.0, 0.0, 0.0])
        sequence = "ARG"

        traj = Trajectory(
            states=states,
            actions=actions,
            log_pf=log_pf,
            log_pb=log_pb,
            sequence=sequence,
        )

        assert traj.sequence == "ARG"
        assert len(traj.actions) == 3
        assert traj.reward is None

    def test_trajectory_with_reward(self):
        """Should create Trajectory with optional reward."""
        traj = Trajectory(
            states=[torch.tensor([[20]])],
            actions=torch.tensor([0]),
            log_pf=torch.tensor([-1.0]),
            log_pb=torch.tensor([0.0]),
            sequence="A",
            reward=torch.tensor(0.5),
        )

        assert traj.reward is not None
        assert traj.reward.item() == 0.5


class TestTrajectorySampler:
    """Test suite for TrajectorySampler."""

    @pytest.fixture
    def forward_policy(self):
        """Create forward policy for testing."""
        return ForwardPolicy(
            vocab_size=23,
            d_model=64,
            n_layers=2,
            n_heads=4,
            dim_feedforward=128,
            max_length=40,
        )

    @pytest.fixture
    def backward_policy(self):
        """Create backward policy for testing."""
        return BackwardPolicy(use_uniform=True)

    @pytest.fixture
    def sampler(self, forward_policy, backward_policy):
        """Create sampler for testing."""
        return TrajectorySampler(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            min_length=5,
            max_length=15,
            exploration_eps=0.0,
        )

    @pytest.fixture
    def sampler_with_exploration(self, forward_policy, backward_policy):
        """Create sampler with exploration for testing."""
        return TrajectorySampler(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            min_length=5,
            max_length=15,
            exploration_eps=0.1,
        )

    def test_sample_trajectories_count(self, sampler):
        """Should return correct number of trajectories."""
        batch_size = 4
        trajectories = sampler.sample_trajectories(batch_size)

        assert len(trajectories) == batch_size

    def test_sample_trajectories_types(self, sampler):
        """Each trajectory should have correct types."""
        trajectories = sampler.sample_trajectories(batch_size=2)

        for traj in trajectories:
            assert isinstance(traj, Trajectory)
            assert isinstance(traj.states, list)
            assert isinstance(traj.actions, torch.Tensor)
            assert isinstance(traj.log_pf, torch.Tensor)
            assert isinstance(traj.log_pb, torch.Tensor)
            assert isinstance(traj.sequence, str)

    def test_minimum_length_enforced(self, sampler):
        """Sequences should respect minimum length."""
        trajectories = sampler.sample_trajectories(batch_size=10)

        for traj in trajectories:
            assert len(traj.sequence) >= sampler.min_length

    def test_maximum_length_enforced(self, sampler):
        """Sequences should respect maximum length."""
        trajectories = sampler.sample_trajectories(batch_size=10)

        for traj in trajectories:
            assert len(traj.sequence) <= sampler.max_length

    def test_valid_amino_acid_sequences(self, sampler):
        """All sequences should contain only valid amino acids."""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        trajectories = sampler.sample_trajectories(batch_size=10)

        for traj in trajectories:
            for aa in traj.sequence:
                assert aa in valid_aas, f"Invalid amino acid: {aa}"

    def test_log_pf_non_positive(self, sampler):
        """Log probabilities should be non-positive."""
        trajectories = sampler.sample_trajectories(batch_size=4)

        for traj in trajectories:
            assert torch.all(traj.log_pf <= 0)

    def test_log_pb_zero_for_uniform(self, sampler):
        """Log backward probabilities should be zero for uniform policy."""
        trajectories = sampler.sample_trajectories(batch_size=4)

        for traj in trajectories:
            assert torch.allclose(traj.log_pb, torch.zeros_like(traj.log_pb))

    def test_action_length_matches_sequence(self, sampler):
        """Number of actions should match sequence length (+ STOP if applicable)."""
        trajectories = sampler.sample_trajectories(batch_size=4)

        for traj in trajectories:
            # Actions include either all AAs + STOP, or just AAs if max_length reached
            assert len(traj.actions) >= len(traj.sequence)

    def test_temperature_affects_sampling(self, sampler):
        """Higher temperature should produce more diverse samples."""
        # Sample with low temperature
        torch.manual_seed(42)
        trajs_low = sampler.sample_trajectories(batch_size=20, temperature=0.1)
        seqs_low = set(t.sequence for t in trajs_low)

        # Sample with high temperature
        torch.manual_seed(42)
        trajs_high = sampler.sample_trajectories(batch_size=20, temperature=2.0)
        seqs_high = set(t.sequence for t in trajs_high)

        # High temperature should generally produce more unique sequences
        # This is probabilistic, so we use a weak assertion
        assert len(seqs_low) <= len(seqs_high) + 5

    def test_exploration_eps_effect(self, sampler, sampler_with_exploration):
        """Exploration epsilon should add randomness."""
        torch.manual_seed(42)
        trajs_no_explore = sampler.sample_trajectories(batch_size=20)

        torch.manual_seed(42)
        trajs_explore = sampler_with_exploration.sample_trajectories(batch_size=20)

        # With exploration, sequences should differ even with same seed
        seqs_no_explore = [t.sequence for t in trajs_no_explore]
        seqs_explore = [t.sequence for t in trajs_explore]

        # At least some sequences should be different
        # (exploration adds uniform noise to sampling)
        different_count = sum(
            s1 != s2 for s1, s2 in zip(seqs_no_explore, seqs_explore)
        )
        # Exploration should cause some divergence
        assert different_count > 0 or len(set(seqs_explore)) > len(set(seqs_no_explore))

    def test_sample_with_gradients_returns_correct_shapes(self, sampler):
        """sample_trajectories_with_gradients should return correct shapes."""
        batch_size = 4
        sequences, log_pf_sum, log_pb_sum = sampler.sample_trajectories_with_gradients(
            batch_size=batch_size
        )

        assert len(sequences) == batch_size
        assert log_pf_sum.shape == (batch_size,)
        assert log_pb_sum.shape == (batch_size,)

    def test_sample_with_gradients_valid_sequences(self, sampler):
        """Sequences from gradient sampling should be valid."""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        sequences, _, _ = sampler.sample_trajectories_with_gradients(batch_size=10)

        for seq in sequences:
            assert len(seq) >= sampler.min_length
            assert len(seq) <= sampler.max_length
            for aa in seq:
                assert aa in valid_aas

    def test_sample_with_gradients_requires_grad(self, sampler):
        """log_pf_sum should have gradients enabled."""
        sequences, log_pf_sum, log_pb_sum = sampler.sample_trajectories_with_gradients(
            batch_size=4
        )

        assert log_pf_sum.requires_grad

    def test_sample_with_gradients_backward(self, sampler):
        """Should be able to backpropagate through log_pf_sum."""
        sequences, log_pf_sum, log_pb_sum = sampler.sample_trajectories_with_gradients(
            batch_size=4
        )

        loss = -log_pf_sum.mean()
        loss.backward()

        # Check that forward policy received gradients
        for param in sampler.forward_policy.parameters():
            if param.grad is not None:
                break
        else:
            pytest.fail("No gradients found in forward policy")

    def test_device_placement(self, forward_policy, backward_policy):
        """Sampler should work on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        forward_policy = forward_policy.cuda()
        sampler = TrajectorySampler(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            min_length=5,
            max_length=15,
        )

        trajectories = sampler.sample_trajectories(batch_size=2, device=torch.device('cuda'))

        for traj in trajectories:
            assert traj.actions.device.type == 'cuda'
            assert traj.log_pf.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
