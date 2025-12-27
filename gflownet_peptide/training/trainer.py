"""
GFlowNet Trainer for Peptide Generation.

Main training loop that coordinates sampling, loss computation, and optimization.

References:
    - Bengio et al. (2021): Flow Network based Generative Models
    - Malkin et al. (2022): Trajectory Balance: Improved Credit Assignment in GFlowNets
    - Jain et al. (2022): Biological Sequence Design with GFlowNets
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from gflownet_peptide.models.forward_policy import ForwardPolicy
from gflownet_peptide.models.backward_policy import BackwardPolicy
from gflownet_peptide.training.sampler import TrajectorySampler
from gflownet_peptide.training.loss import TrajectoryBalanceLoss, SubTrajectoryBalanceLoss

logger = logging.getLogger(__name__)


# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Optional tqdm import
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


class GFlowNetTrainer:
    """
    Trainer for GFlowNet peptide generation.

    Implements the training loop with trajectory sampling, loss computation,
    and gradient updates.

    Features:
        - Separate learning rate for log_Z (Malkin 2022 recommendation)
        - Exploration via uniform mixing (Bengio 2021)
        - W&B integration for experiment tracking
        - Checkpoint management (latest + final)
    """

    def __init__(
        self,
        forward_policy: ForwardPolicy,
        backward_policy: BackwardPolicy,
        reward_model: Union[nn.Module, Callable[[list[str]], torch.Tensor]],
        learning_rate: float = 3e-4,
        log_z_lr_multiplier: float = 10.0,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        loss_type: str = "trajectory_balance",
        init_log_z: float = 0.0,
        min_length: int = 10,
        max_length: int = 30,
        exploration_eps: float = 0.0,
        reward_temperature: float = 1.0,
        entropy_weight: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            forward_policy: Forward policy network P_F(a|s)
            backward_policy: Backward policy network P_B(s|s')
            reward_model: Reward model (frozen) - can be nn.Module or callable
            learning_rate: Base learning rate for policy parameters
            log_z_lr_multiplier: Multiplier for log_Z learning rate (Malkin 2022
                                 recommends ~10x policy LR)
            weight_decay: Weight decay for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            loss_type: "trajectory_balance" or "sub_trajectory_balance"
            init_log_z: Initial value for learnable log partition function
            min_length: Minimum peptide length
            max_length: Maximum peptide length
            exploration_eps: ε for uniform mixing (Bengio 2021, Eq. 10)
            reward_temperature: Temperature β for reward sharpening R(x)^β
            entropy_weight: Weight for entropy regularization to prevent mode collapse.
                           Recommended: 0.01-0.1. Set to 0.0 to disable.
            device: Device to use
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Models
        self.forward_policy = forward_policy.to(self.device)
        self.backward_policy = backward_policy.to(self.device)

        # Handle reward model (can be nn.Module or callable)
        if isinstance(reward_model, nn.Module):
            self.reward_model = reward_model.to(self.device)
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad = False
        else:
            # Callable reward function (e.g., ImprovedReward)
            self.reward_model = reward_model

        # Sampler with exploration
        self.sampler = TrajectorySampler(
            forward_policy=self.forward_policy,
            backward_policy=self.backward_policy,
            min_length=min_length,
            max_length=max_length,
            exploration_eps=exploration_eps,
        )

        # Loss function
        if loss_type == "trajectory_balance":
            self.loss_fn = TrajectoryBalanceLoss(
                init_log_z=init_log_z,
                entropy_weight=entropy_weight,
            ).to(self.device)
        elif loss_type == "sub_trajectory_balance":
            self.loss_fn = SubTrajectoryBalanceLoss(
                init_log_z=init_log_z,
                entropy_weight=entropy_weight,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.reward_temperature = reward_temperature
        self.max_grad_norm = max_grad_norm
        self.exploration_eps = exploration_eps

        # Optimizer with separate parameter groups (Malkin 2022)
        # Policy parameters get base LR, log_Z gets higher LR
        policy_params = list(self.forward_policy.parameters())
        log_z_params = [self.loss_fn.log_z]

        self.optimizer = AdamW([
            {'params': policy_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': log_z_params, 'lr': learning_rate * log_z_lr_multiplier, 'weight_decay': 0.0},
        ])

        # Scheduler (optional)
        self.scheduler = None
        self.warmup_steps = 0

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")
        self.best_reward = 0.0  # Track best mean reward for checkpoint selection

        # Store config for checkpointing
        self.config = {
            'learning_rate': learning_rate,
            'log_z_lr_multiplier': log_z_lr_multiplier,
            'weight_decay': weight_decay,
            'max_grad_norm': max_grad_norm,
            'loss_type': loss_type,
            'init_log_z': init_log_z,
            'min_length': min_length,
            'max_length': max_length,
            'exploration_eps': exploration_eps,
            'reward_temperature': reward_temperature,
            'entropy_weight': entropy_weight,
        }

        logger.info(f"GFlowNetTrainer initialized on {self.device}")
        logger.info(f"  Policy LR: {learning_rate}, log_Z LR: {learning_rate * log_z_lr_multiplier}")
        logger.info(f"  Exploration ε: {exploration_eps}")

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
            reward_result = self.reward_model(sequences)
            # Handle both tensor and list returns
            if isinstance(reward_result, torch.Tensor):
                rewards = reward_result
            else:
                rewards = torch.tensor(reward_result, device=self.device, dtype=torch.float32)

            # Apply temperature to rewards (β: reward sharpening)
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
        grad_norm = 0.0
        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.forward_policy.parameters(), self.max_grad_norm
            ).item()

        # Optimizer step
        self.optimizer.step()

        # Scheduler step (only after warmup)
        if self.scheduler is not None and self.global_step >= self.warmup_steps:
            self.scheduler.step()

        self.global_step += 1

        # Compute metrics
        with torch.no_grad():
            lengths = [len(s) for s in sequences]
            unique_seqs = set(sequences)
            metrics = {
                "step": self.global_step,
                "loss": loss.item(),
                "log_z": self.loss_fn.get_log_z(),
                "mean_reward": rewards.mean().item(),
                "max_reward": rewards.max().item(),
                "min_reward": rewards.min().item(),
                "mean_log_pf": log_pf_sum.mean().item(),
                "mean_log_pb": log_pb_sum.mean().item(),
                "mean_log_reward": log_rewards.mean().item(),
                "mean_length": sum(lengths) / len(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "unique_ratio": len(unique_seqs) / len(sequences),
                "grad_norm": grad_norm,
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
        run_name: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "gflownet-peptide",
        wandb_entity: Optional[str] = None,
        wandb_run: Optional[Any] = None,
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
            run_name: Name for this training run
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name
            wandb_entity: W&B entity (username/team)
            wandb_run: Optional existing W&B run (if None and use_wandb, creates new)

        Returns:
            final_metrics: Dictionary of final training metrics
        """
        logger.info(f"Starting training for {n_steps} steps")
        logger.info(f"Batch size: {batch_size}, Temperature: {temperature}")

        # Setup checkpoint directory
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup W&B
        if use_wandb and wandb_run is None:
            if not WANDB_AVAILABLE:
                logger.warning("W&B not available. Install with: pip install wandb")
                use_wandb = False
            else:
                wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    config={
                        **self.config,
                        'n_steps': n_steps,
                        'batch_size': batch_size,
                        'temperature': temperature,
                    },
                )
                logger.info(f"W&B run initialized: {wandb_run.url}")

        self.setup_scheduler(n_steps)

        all_metrics = []

        # Setup progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(range(n_steps), desc="Training", unit="step")
        else:
            pbar = range(n_steps)

        for step in pbar:
            metrics = self.train_step(batch_size, temperature)

            # Update progress bar
            if TQDM_AVAILABLE:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.2f}",
                    'log_z': f"{metrics['log_z']:.2f}",
                    'reward': f"{metrics['mean_reward']:.3f}",
                })

            # Logging
            if step % log_every == 0:
                if not TQDM_AVAILABLE:
                    logger.info(
                        f"Step {step}: loss={metrics['loss']:.4f}, "
                        f"log_z={metrics['log_z']:.2f}, "
                        f"mean_reward={metrics['mean_reward']:.4f}, "
                        f"grad_norm={metrics['grad_norm']:.4f}"
                    )

                if wandb_run:
                    wandb_run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

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

            # Checkpointing (policy: overwrite latest, keep final)
            if checkpoint_dir and step % save_every == 0 and step > 0:
                # Overwrite latest checkpoint
                latest_path = checkpoint_dir / f"{run_name or 'gflownet'}_latest.pt"
                self.save_checkpoint(latest_path, step=step)

                # Save best model by REWARD (not loss)
                # Rationale: TB loss is unreliable - can be low for degenerate policies
                # See docs/reward-comparison-analysis.md Section 4
                if metrics.get("mean_reward", 0) > self.best_reward:
                    self.best_reward = metrics["mean_reward"]
                    best_path = checkpoint_dir / f"{run_name or 'gflownet'}_best.pt"
                    self.save_checkpoint(best_path, step=step)
                    logger.info(
                        f"New best model at step {step}: "
                        f"mean_reward={metrics['mean_reward']:.4f} "
                        f"(loss={metrics['loss']:.2f})"
                    )

        # Final checkpoint
        if checkpoint_dir:
            final_path = checkpoint_dir / f"{run_name or 'gflownet'}_final.pt"
            self.save_checkpoint(final_path, step=n_steps)

        # Close W&B run if we created it
        if wandb_run and use_wandb:
            # Log final samples
            final_eval = self.evaluate(n_samples=100)
            if wandb_run:
                self._log_sample_table(wandb_run, step=n_steps)
            wandb_run.finish()

        return all_metrics[-1] if all_metrics else {}

    def _log_sample_table(self, wandb_run: Any, step: int, n_samples: int = 20):
        """Log sample sequences to W&B as a table."""
        if not WANDB_AVAILABLE or wandb_run is None:
            return

        self.forward_policy.eval()
        sequences, log_probs = self.forward_policy.sample_sequence(
            batch_size=n_samples,
            max_length=30,
            min_length=10,
            temperature=1.0,
            device=self.device,
        )

        # Get rewards
        reward_result = self.reward_model(sequences)
        if isinstance(reward_result, torch.Tensor):
            rewards = reward_result.cpu().tolist()
        else:
            rewards = reward_result

        # Create W&B table
        table = wandb.Table(columns=['sequence', 'reward', 'length', 'log_prob'])
        for seq, r, lp in zip(sequences, rewards, log_probs.cpu().tolist()):
            table.add_data(seq, r, len(seq), lp)

        wandb_run.log({'samples': table}, step=step)
        self.forward_policy.train()

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

        # Compute rewards (handle both tensor and list returns)
        reward_result = self.reward_model(sequences)
        if isinstance(reward_result, torch.Tensor):
            rewards = reward_result.cpu()
        else:
            rewards = torch.tensor(reward_result, dtype=torch.float32)

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

    def save_checkpoint(self, path: Union[str, Path], step: int = 0):
        """Save training checkpoint."""
        checkpoint = {
            "step": step,
            "forward_policy_state_dict": self.forward_policy.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "best_reward": self.best_reward,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.forward_policy.load_state_dict(checkpoint["forward_policy_state_dict"])
        self.loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.best_reward = checkpoint.get("best_reward", 0.0)

        # Optionally restore config
        if "config" in checkpoint:
            self.config = checkpoint["config"]

        logger.info(f"Loaded checkpoint from {path} at step {self.global_step}")
