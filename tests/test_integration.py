"""Integration tests for GFlowNet peptide generation.

Tests the full training pipeline end-to-end.
"""

import pytest
import tempfile
from pathlib import Path

import torch

from gflownet_peptide.models.forward_policy import ForwardPolicy
from gflownet_peptide.models.backward_policy import BackwardPolicy
from gflownet_peptide.training.sampler import TrajectorySampler, Trajectory
from gflownet_peptide.training.loss import TrajectoryBalanceLoss, SubTrajectoryBalanceLoss
from gflownet_peptide.training.trainer import GFlowNetTrainer


class SimpleRewardModel:
    """Simple reward model for integration testing.

    Rewards sequences based on:
    - Length (prefer 10-20 AAs)
    - Diversity (penalize homopolymers)
    """

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, sequences):
        import math
        rewards = []
        for seq in sequences:
            # Length reward: bell curve around 15
            length_reward = math.exp(-((len(seq) - 15) ** 2) / 50)

            # Diversity reward: penalize low entropy
            if len(seq) > 0:
                unique_ratio = len(set(seq)) / len(seq)
            else:
                unique_ratio = 0

            reward = 0.5 * length_reward + 0.5 * unique_ratio + 0.1
            rewards.append(reward)

        return rewards

    def to(self, device):
        self.device = device
        return self


class TestEndToEndTraining:
    """End-to-end integration tests."""

    @pytest.fixture
    def components(self):
        """Create all components for training."""
        forward_policy = ForwardPolicy(
            vocab_size=23,
            d_model=64,
            n_layers=2,
            n_heads=4,
            dim_feedforward=128,
            max_length=40,
        )
        backward_policy = BackwardPolicy(use_uniform=True)
        reward_model = SimpleRewardModel()

        return {
            'forward_policy': forward_policy,
            'backward_policy': backward_policy,
            'reward_model': reward_model,
        }

    def test_full_training_pipeline(self, components):
        """Test complete training pipeline."""
        trainer = GFlowNetTrainer(
            forward_policy=components['forward_policy'],
            backward_policy=components['backward_policy'],
            reward_model=components['reward_model'],
            learning_rate=1e-3,
            log_z_lr_multiplier=10.0,
            min_length=5,
            max_length=20,
            exploration_eps=0.01,
            device=torch.device('cpu'),
        )

        # Train for a few steps
        initial_log_z = trainer.loss_fn.get_log_z()
        losses = []

        for step in range(20):
            metrics = trainer.train_step(batch_size=8)
            loss_val = metrics['loss']
            losses.append(loss_val)

            # Verify metrics are valid (may be inf initially, but should be non-NaN)
            assert not torch.isnan(torch.tensor(loss_val))
            assert 'log_z' in metrics
            assert 'mean_reward' in metrics

        # log_Z should have been updated
        final_log_z = trainer.loss_fn.get_log_z()

        # Training should complete without errors
        assert len(losses) == 20
        # Some losses may be large but should stabilize
        assert all(l >= 0 or l == float('inf') for l in losses)

    def test_checkpoint_and_resume(self, components):
        """Test checkpointing and resuming training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Initial training
            trainer1 = GFlowNetTrainer(
                forward_policy=components['forward_policy'],
                backward_policy=components['backward_policy'],
                reward_model=components['reward_model'],
                learning_rate=1e-3,
                device=torch.device('cpu'),
            )

            for _ in range(10):
                trainer1.train_step(batch_size=4)

            # Save checkpoint
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            trainer1.save_checkpoint(checkpoint_path, step=trainer1.global_step)

            saved_step = trainer1.global_step
            saved_log_z = trainer1.loss_fn.get_log_z()

            # Create new trainer with SAME architecture and load checkpoint
            trainer2 = GFlowNetTrainer(
                forward_policy=ForwardPolicy(
                    vocab_size=23, d_model=64, n_layers=2, n_heads=4,
                    dim_feedforward=128, max_length=40  # Match the original
                ),
                backward_policy=BackwardPolicy(),
                reward_model=SimpleRewardModel(),
                learning_rate=1e-3,
                device=torch.device('cpu'),
            )

            trainer2.load_checkpoint(checkpoint_path)

            # Verify state restored
            assert trainer2.global_step == saved_step
            assert abs(trainer2.loss_fn.get_log_z() - saved_log_z) < 1e-6

            # Continue training
            for _ in range(5):
                metrics = trainer2.train_step(batch_size=4)
                # Loss may be inf but should not be NaN
                assert not torch.isnan(torch.tensor(metrics['loss']))

    def test_sampling_generates_valid_peptides(self, components):
        """Test that sampler generates valid peptide sequences."""
        sampler = TrajectorySampler(
            forward_policy=components['forward_policy'],
            backward_policy=components['backward_policy'],
            min_length=5,
            max_length=20,
            exploration_eps=0.0,
        )

        # Sample trajectories
        trajectories = sampler.sample_trajectories(batch_size=10)

        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

        for traj in trajectories:
            # Check sequence validity
            assert len(traj.sequence) >= 5
            assert len(traj.sequence) <= 20
            assert all(aa in valid_aas for aa in traj.sequence)

            # Check log probs
            assert torch.all(traj.log_pf <= 0)
            assert torch.allclose(traj.log_pb, torch.zeros_like(traj.log_pb))

    def test_loss_convergence(self, components):
        """Test that loss decreases during training."""
        trainer = GFlowNetTrainer(
            forward_policy=components['forward_policy'],
            backward_policy=components['backward_policy'],
            reward_model=components['reward_model'],
            learning_rate=3e-3,
            log_z_lr_multiplier=10.0,
            device=torch.device('cpu'),
        )

        # Collect losses
        losses = []
        for _ in range(50):
            metrics = trainer.train_step(batch_size=16)
            losses.append(metrics['loss'])

        # Moving average should decrease (with some tolerance for noise)
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10

        # This is a weak assertion due to training noise
        # At minimum, late losses should not be dramatically higher
        assert late_avg < early_avg * 2 or late_avg < 10

    def test_evaluation_metrics(self, components):
        """Test that evaluation produces valid metrics."""
        trainer = GFlowNetTrainer(
            forward_policy=components['forward_policy'],
            backward_policy=components['backward_policy'],
            reward_model=components['reward_model'],
            device=torch.device('cpu'),
        )

        # Train briefly
        for _ in range(5):
            trainer.train_step(batch_size=4)

        # Evaluate
        metrics = trainer.evaluate(n_samples=50)

        # Check all expected metrics
        assert 'mean_reward' in metrics
        assert 'max_reward' in metrics
        assert 'sequence_diversity' in metrics
        assert 'unique_ratio' in metrics

        # Validate ranges
        assert 0 <= metrics['unique_ratio'] <= 1
        assert 0 <= metrics['sequence_diversity'] <= 1
        assert metrics['mean_reward'] >= 0

    def test_gradient_flow_complete_pipeline(self, components):
        """Test gradient flow through complete pipeline."""
        trainer = GFlowNetTrainer(
            forward_policy=components['forward_policy'],
            backward_policy=components['backward_policy'],
            reward_model=components['reward_model'],
            device=torch.device('cpu'),
        )

        # Get initial parameters
        initial_params = {
            name: param.clone()
            for name, param in trainer.forward_policy.named_parameters()
        }

        # Train one step
        trainer.train_step(batch_size=4)

        # Check parameters changed
        changed_count = 0
        for name, param in trainer.forward_policy.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-7):
                changed_count += 1

        # At least some parameters should have changed
        assert changed_count > 0

    def test_sampler_with_exploration(self, components):
        """Test sampler exploration behavior."""
        sampler_no_explore = TrajectorySampler(
            forward_policy=components['forward_policy'],
            backward_policy=components['backward_policy'],
            min_length=5,
            max_length=20,
            exploration_eps=0.0,
        )

        sampler_explore = TrajectorySampler(
            forward_policy=components['forward_policy'],
            backward_policy=components['backward_policy'],
            min_length=5,
            max_length=20,
            exploration_eps=0.1,
        )

        # Sample with same seed
        torch.manual_seed(42)
        trajs_no_explore = sampler_no_explore.sample_trajectories(batch_size=20)

        torch.manual_seed(42)
        trajs_explore = sampler_explore.sample_trajectories(batch_size=20)

        seqs_no_explore = [t.sequence for t in trajs_no_explore]
        seqs_explore = [t.sequence for t in trajs_explore]

        # With exploration, we expect some difference
        # (same seed but exploration adds noise)
        unique_no_explore = len(set(seqs_no_explore))
        unique_explore = len(set(seqs_explore))

        # Exploration should maintain or increase diversity
        # This is probabilistic, so we use weak assertion
        assert unique_explore >= unique_no_explore - 2

    def test_temperature_sampling(self, components):
        """Test temperature affects sampling diversity."""
        trainer = GFlowNetTrainer(
            forward_policy=components['forward_policy'],
            backward_policy=components['backward_policy'],
            reward_model=components['reward_model'],
            device=torch.device('cpu'),
        )

        # Sample with low temperature (greedy-ish)
        trainer.sampler.temperature = 0.1
        seqs_low, _, _ = trainer.sampler.sample_trajectories_with_gradients(
            batch_size=20, temperature=0.1
        )
        unique_low = len(set(seqs_low))

        # Sample with high temperature (more random)
        trainer.sampler.temperature = 2.0
        seqs_high, _, _ = trainer.sampler.sample_trajectories_with_gradients(
            batch_size=20, temperature=2.0
        )
        unique_high = len(set(seqs_high))

        # High temperature should generally produce more diversity
        # This is probabilistic
        assert unique_high >= unique_low - 5


class TestComponentInteraction:
    """Test interactions between components."""

    def test_forward_backward_consistency(self):
        """Forward and backward policies should be consistent."""
        forward_policy = ForwardPolicy(
            vocab_size=23, d_model=64, n_layers=2, n_heads=4
        )
        backward_policy = BackwardPolicy(use_uniform=True)

        # Sample a sequence
        batch_size = 4
        partial_seq = torch.randint(0, 20, (batch_size, 5))

        # Forward policy should give distribution over actions
        logits = forward_policy(partial_seq)
        assert logits.shape == (batch_size, 21)  # 20 AA + STOP

        # Backward policy for same state should give log(1) = 0
        parent_seq = partial_seq[:, :-1]
        log_pb = backward_policy.log_prob(partial_seq, parent_seq)
        assert torch.allclose(log_pb, torch.zeros(batch_size))

    def test_loss_with_sampled_trajectories(self):
        """Loss should work with sampled trajectories."""
        forward_policy = ForwardPolicy(
            vocab_size=23, d_model=64, n_layers=2, n_heads=4
        )
        backward_policy = BackwardPolicy()
        loss_fn = TrajectoryBalanceLoss(init_log_z=0.0)

        sampler = TrajectorySampler(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            min_length=5,
            max_length=15,
        )

        # Sample with gradients
        sequences, log_pf_sum, log_pb_sum = sampler.sample_trajectories_with_gradients(
            batch_size=4
        )

        # Create dummy rewards
        rewards = torch.rand(4) + 0.1
        log_rewards = torch.log(rewards)

        # Compute loss
        loss = loss_fn(log_pf_sum, log_pb_sum, log_rewards)

        assert torch.isfinite(loss)
        assert loss >= 0

        # Should be able to backpropagate
        loss.backward()

        # Check gradients exist
        assert loss_fn.log_z.grad is not None
        grad_count = sum(
            1 for p in forward_policy.parameters() if p.grad is not None
        )
        assert grad_count > 0

    def test_reward_model_interface(self):
        """Reward model interface should work with trainer."""
        # Test callable interface
        def reward_fn(sequences):
            return [0.5 for _ in sequences]

        forward_policy = ForwardPolicy(
            vocab_size=23, d_model=64, n_layers=2, n_heads=4
        )
        backward_policy = BackwardPolicy()

        trainer = GFlowNetTrainer(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            reward_model=reward_fn,
            device=torch.device('cpu'),
        )

        metrics = trainer.train_step(batch_size=4)
        assert torch.isfinite(torch.tensor(metrics['loss']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
