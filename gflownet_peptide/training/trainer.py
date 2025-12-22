"""
GFlowNet Trainer for Peptide Generation.

Main training loop that coordinates sampling, loss computation, and optimization.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from gflownet_peptide.models.forward_policy import ForwardPolicy
from gflownet_peptide.models.backward_policy import BackwardPolicy
from gflownet_peptide.models.reward_model import CompositeReward
from gflownet_peptide.training.sampler import TrajectorySampler
from gflownet_peptide.training.loss import TrajectoryBalanceLoss, SubTrajectoryBalanceLoss

logger = logging.getLogger(__name__)


class GFlowNetTrainer:
    """
    Trainer for GFlowNet peptide generation.

    Implements the training loop with trajectory sampling, loss computation,
    and gradient updates.
    """

    def __init__(
        self,
        forward_policy: ForwardPolicy,
        backward_policy: BackwardPolicy,
        reward_model: CompositeReward,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        loss_type: str = "trajectory_balance",
        min_length: int = 10,
        max_length: int = 30,
        reward_temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            forward_policy: Forward policy network
            backward_policy: Backward policy network
            reward_model: Reward model (frozen during GFlowNet training)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            loss_type: "trajectory_balance" or "sub_trajectory_balance"
            min_length: Minimum peptide length
            max_length: Maximum peptide length
            reward_temperature: Temperature for reward (R^beta)
            device: Device to use
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Models
        self.forward_policy = forward_policy.to(self.device)
        self.backward_policy = backward_policy.to(self.device)
        self.reward_model = reward_model.to(self.device)

        # Freeze reward model
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Sampler
        self.sampler = TrajectorySampler(
            forward_policy=self.forward_policy,
            backward_policy=self.backward_policy,
            min_length=min_length,
            max_length=max_length,
        )

        # Loss function
        if loss_type == "trajectory_balance":
            self.loss_fn = TrajectoryBalanceLoss().to(self.device)
        elif loss_type == "sub_trajectory_balance":
            self.loss_fn = SubTrajectoryBalanceLoss().to(self.device)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.reward_temperature = reward_temperature
        self.max_grad_norm = max_grad_norm

        # Optimizer
        params = (
            list(self.forward_policy.parameters())
            + list(self.loss_fn.parameters())  # Includes log_z
        )
        self.optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

        # Scheduler (optional)
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")

    def setup_scheduler(self, n_steps: int, warmup_steps: int = 1000):
        """Setup learning rate scheduler."""
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=n_steps - warmup_steps
        )
        self.warmup_steps = warmup_steps

    def train_step(
        self,
        batch_size: int = 64,
        temperature: float = 1.0,
    ) -> dict:
        """
        Perform one training step.

        Args:
            batch_size: Number of trajectories to sample
            temperature: Sampling temperature

        Returns:
            metrics: Dictionary of training metrics
        """
        self.forward_policy.train()

        # Sample trajectories with gradients
        sequences, log_pf_sum, log_pb_sum = self.sampler.sample_trajectories_with_gradients(
            batch_size=batch_size,
            temperature=temperature,
            device=self.device,
        )

        # Compute rewards
        with torch.no_grad():
            rewards = self.reward_model(sequences)

            # Apply temperature to rewards
            if self.reward_temperature != 1.0:
                rewards = rewards ** self.reward_temperature

            # Log rewards (with epsilon for stability)
            log_rewards = torch.log(rewards + 1e-8)

        # Compute loss
        loss = self.loss_fn(log_pf_sum, log_pb_sum, log_rewards)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.forward_policy.parameters(), self.max_grad_norm
            )

        # Optimizer step
        self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None and self.global_step >= self.warmup_steps:
            self.scheduler.step()

        self.global_step += 1

        # Compute metrics
        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "log_z": self.loss_fn.get_log_z(),
                "mean_reward": rewards.mean().item(),
                "max_reward": rewards.max().item(),
                "mean_log_pf": log_pf_sum.mean().item(),
                "mean_length": sum(len(s) for s in sequences) / len(sequences),
                "unique_sequences": len(set(sequences)) / len(sequences),
            }

        return metrics

    def train(
        self,
        n_steps: int,
        batch_size: int = 64,
        temperature: float = 1.0,
        log_every: int = 100,
        eval_every: int = 1000,
        save_every: int = 5000,
        checkpoint_dir: Optional[str] = None,
        wandb_run: Optional["wandb.Run"] = None,
    ) -> dict:
        """
        Run full training loop.

        Args:
            n_steps: Total number of training steps
            batch_size: Batch size
            temperature: Sampling temperature
            log_every: Log metrics every N steps
            eval_every: Run evaluation every N steps
            save_every: Save checkpoint every N steps
            checkpoint_dir: Directory for saving checkpoints
            wandb_run: Optional wandb run for logging

        Returns:
            final_metrics: Dictionary of final training metrics
        """
        logger.info(f"Starting training for {n_steps} steps")
        logger.info(f"Batch size: {batch_size}, Temperature: {temperature}")

        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.setup_scheduler(n_steps)

        all_metrics = []

        for step in range(n_steps):
            metrics = self.train_step(batch_size, temperature)

            # Logging
            if step % log_every == 0:
                logger.info(
                    f"Step {step}: loss={metrics['loss']:.4f}, "
                    f"log_z={metrics['log_z']:.2f}, "
                    f"mean_reward={metrics['mean_reward']:.4f}"
                )

                if wandb_run:
                    wandb_run.log(metrics, step=step)

            all_metrics.append(metrics)

            # Evaluation
            if step % eval_every == 0 and step > 0:
                eval_metrics = self.evaluate(n_samples=500)
                logger.info(
                    f"Eval: diversity={eval_metrics['sequence_diversity']:.4f}, "
                    f"mean_reward={eval_metrics['mean_reward']:.4f}"
                )

                if wandb_run:
                    wandb_run.log(
                        {f"eval/{k}": v for k, v in eval_metrics.items()},
                        step=step,
                    )

            # Checkpointing
            if checkpoint_dir and step % save_every == 0 and step > 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_{step}.pt",
                    step=step,
                )

                # Save best model
                if metrics["loss"] < self.best_loss:
                    self.best_loss = metrics["loss"]
                    self.save_checkpoint(
                        checkpoint_dir / "best.pt",
                        step=step,
                    )

        # Final checkpoint
        if checkpoint_dir:
            self.save_checkpoint(
                checkpoint_dir / "final.pt",
                step=n_steps,
            )

        return all_metrics[-1]

    @torch.no_grad()
    def evaluate(self, n_samples: int = 1000) -> dict:
        """
        Evaluate the trained model.

        Args:
            n_samples: Number of samples to generate

        Returns:
            metrics: Evaluation metrics
        """
        self.forward_policy.eval()

        # Sample sequences
        sequences, log_probs = self.forward_policy.sample_sequence(
            batch_size=n_samples,
            max_length=30,
            min_length=10,
            temperature=1.0,
            device=self.device,
        )

        # Compute rewards
        rewards = self.reward_model(sequences)

        # Compute diversity
        unique_seqs = set(sequences)
        unique_ratio = len(unique_seqs) / len(sequences)

        # Pairwise sequence identity
        def seq_identity(s1: str, s2: str) -> float:
            if len(s1) == 0 or len(s2) == 0:
                return 0.0
            matches = sum(a == b for a, b in zip(s1, s2))
            return matches / max(len(s1), len(s2))

        # Sample subset for diversity computation
        sample_seqs = list(unique_seqs)[:100]
        total_identity = 0
        count = 0
        for i, s1 in enumerate(sample_seqs):
            for s2 in sample_seqs[i + 1 :]:
                total_identity += seq_identity(s1, s2)
                count += 1

        mean_identity = total_identity / max(count, 1)
        sequence_diversity = 1 - mean_identity

        metrics = {
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "std_reward": rewards.std().item(),
            "unique_ratio": unique_ratio,
            "sequence_diversity": sequence_diversity,
            "mean_length": sum(len(s) for s in sequences) / len(sequences),
        }

        self.forward_policy.train()
        return metrics

    def save_checkpoint(self, path: str, step: int = 0):
        """Save training checkpoint."""
        checkpoint = {
            "step": step,
            "forward_policy_state_dict": self.forward_policy.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.forward_policy.load_state_dict(checkpoint["forward_policy_state_dict"])
        self.loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        logger.info(f"Loaded checkpoint from {path} at step {self.global_step}")
