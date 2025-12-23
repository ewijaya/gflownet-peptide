#!/usr/bin/env python
"""Training script for GRPO-D peptide generator.

This script trains a GRPO-D (Group Relative Policy Optimization with Diversity)
model for peptide generation using ESM-2 pseudo-likelihood as the reward function.

Example usage:
    # Full training
    python scripts/train_grpo.py --config configs/grpo.yaml

    # Dry run (10 iterations)
    python scripts/train_grpo.py --config configs/grpo.yaml --dry_run

    # Custom iterations
    python scripts/train_grpo.py --total_iterations 500 --eval_interval 50
"""

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gflownet_peptide.rewards.esm2_reward import ESM2Reward
from gflownet_peptide.training.grpo_trainer import GRPOTrainer
from gflownet_peptide.training.diversity import calculate_batch_diversity_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        # Model
        "model_name": "littleworth/protgpt2-distilled-medium",
        "hidden_dim": 256,
        # Peptide constraints
        "min_length": 10,
        "max_length": 30,
        "amino_acids": "ACDEFGHIKLMNPQRSTVWY",
        # GRPO hyperparameters
        "learning_rate": 3e-4,
        "batch_size": 16,
        "num_generations": 8,
        "beta": 0.04,
        "max_grad_norm": 1.0,
        # Diversity (GRPO-D)
        "diversity_weight": 0.15,
        "diversity_weight_aa": 0.7,
        "diversity_weight_seq": 0.3,
        # Training
        "total_iterations": 1000,
        "buffer_size": 500,
        "min_buffer_size": 16,
        "save_interval": 200,
        "eval_interval": 50,
        # Generation
        "temperature": 1.0,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        # ESM-2 reward
        "esm_model": "esm2_t12_35M_UR50D",
        # Paths
        "checkpoint_dir": "checkpoints/grpo",
        "results_dir": "results/grpo",
        # W&B
        "wandb_project": "gflownet-peptide",
        "wandb_entity": "ewijaya",
    }


def merge_configs(base: dict, override: dict) -> dict:
    """Merge override config into base config."""
    result = base.copy()
    for key, value in override.items():
        if value is not None:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description="Train GRPO-D peptide generator")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )

    # Override individual parameters
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--total_iterations", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--diversity_weight", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--min_length", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--esm_model", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Special flags
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run only 10 iterations for testing",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    args = parser.parse_args()

    # Load configuration
    config = get_default_config()

    if args.config:
        file_config = load_config(args.config)
        config = merge_configs(config, file_config)

    # Override with command line arguments
    cli_overrides = {
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "total_iterations": args.total_iterations,
        "save_interval": args.save_interval,
        "eval_interval": args.eval_interval,
        "diversity_weight": args.diversity_weight,
        "beta": args.beta,
        "temperature": args.temperature,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "checkpoint_dir": args.checkpoint_dir,
        "results_dir": args.results_dir,
        "esm_model": args.esm_model,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
    }
    config = merge_configs(config, cli_overrides)

    # Apply dry run settings
    if args.dry_run:
        config["total_iterations"] = 10
        config["eval_interval"] = 2
        config["save_interval"] = 5
        logger.info("Dry run mode: 10 iterations")

    # Create directories
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    # Create run name
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{current_date}_grpod_"
        f"it{config['total_iterations']}_"
        f"dw{config['diversity_weight']}_"
        f"beta{config['beta']}"
    )

    logger.info(f"Run name: {run_name}")
    logger.info(f"Configuration: {config}")

    # Initialize wandb if not disabled
    if not args.no_wandb:
        try:
            import wandb

            wandb.init(
                project=config["wandb_project"],
                entity=config["wandb_entity"],
                name=run_name,
                config=config,
            )
            use_wandb = True
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False
    else:
        use_wandb = False

    # Initialize reward function
    logger.info(f"Loading ESM-2 reward model: {config['esm_model']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_fn = ESM2Reward(
        model_name=config["esm_model"],
        device=device,
        normalize=True,
        temperature=1.0,
    )

    # Initialize trainer
    logger.info("Initializing GRPO-D trainer...")
    trainer = GRPOTrainer(
        config=config,
        reward_fn=reward_fn,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Training loop
    logger.info(f"Starting training for {config['total_iterations']} iterations...")

    all_peptides = []
    all_rewards = []

    for iteration in tqdm(range(trainer.iteration, config["total_iterations"])):
        # Training step
        metrics = trainer.train_step()

        # Log to wandb
        if use_wandb:
            wandb.log(
                {
                    "iteration": iteration,
                    "total_loss": metrics["total_loss"],
                    "policy_loss": metrics["policy_loss"],
                    "kl_loss": metrics["kl_loss"],
                    "mean_reward": metrics["mean_reward"],
                    "max_reward": metrics["max_reward"],
                    "mean_diversity": metrics["mean_diversity"],
                    "num_peptides": metrics["num_peptides"],
                }
            )

        # Evaluation logging
        if iteration % config["eval_interval"] == 0:
            logger.info(
                f"Iteration {iteration}: "
                f"Loss={metrics['total_loss']:.4f}, "
                f"Mean R={metrics['mean_reward']:.4f}, "
                f"Max R={metrics['max_reward']:.4f}, "
                f"Diversity={metrics['mean_diversity']:.4f}"
            )

            # Log top peptides
            top_peptides = trainer.get_top_peptides(5)
            logger.info("Top peptides:")
            for i, p in enumerate(top_peptides):
                logger.info(f"  {i+1}. {p['peptide']} (R={p['reward']:.4f})")

        # Save checkpoint
        if iteration > 0 and iteration % config["save_interval"] == 0:
            checkpoint_path = os.path.join(
                config["checkpoint_dir"],
                f"{run_name}_iter{iteration}.pt",
            )
            trainer.save_checkpoint(checkpoint_path)

    # Save final checkpoint
    final_checkpoint_path = os.path.join(
        config["checkpoint_dir"],
        f"{run_name}_final.pt",
    )
    trainer.save_checkpoint(final_checkpoint_path)

    # Save training statistics
    stats_df = pd.DataFrame(trainer.training_stats)
    stats_path = os.path.join(config["results_dir"], f"{run_name}_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Training statistics saved to {stats_path}")

    # Save top peptides
    top_peptides = trainer.get_top_peptides(1000)
    peptides_df = pd.DataFrame(top_peptides)
    peptides_path = os.path.join(config["results_dir"], f"{run_name}_peptides.csv")
    peptides_df.to_csv(peptides_path, index=False)
    logger.info(f"Top peptides saved to {peptides_path}")

    # Final summary
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Final checkpoint: {final_checkpoint_path}")
    logger.info(f"Statistics: {stats_path}")
    logger.info(f"Peptides: {peptides_path}")

    if trainer.training_stats["mean_reward"]:
        final_mean_reward = trainer.training_stats["mean_reward"][-1]
        final_max_reward = trainer.training_stats["max_reward"][-1]
        final_diversity = trainer.training_stats["mean_diversity"][-1]
        logger.info(f"Final mean reward: {final_mean_reward:.4f}")
        logger.info(f"Final max reward: {final_max_reward:.4f}")
        logger.info(f"Final mean diversity: {final_diversity:.4f}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
