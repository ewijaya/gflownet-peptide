#!/usr/bin/env python3
"""
Train GFlowNet for peptide generation.

This script trains a GFlowNet to generate diverse peptides
proportionally to their predicted fitness.

Usage:
    python scripts/train_gflownet.py --config configs/default.yaml
    python scripts/train_gflownet.py --reward_checkpoint checkpoints/reward/stability_best.pt
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GFlowNet")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--reward_checkpoint",
        type=str,
        default=None,
        help="Path to reward model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/gflownet/",
        help="Output directory",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="Number of training steps (overrides config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    if args.n_steps:
        config["training"]["n_steps"] = args.n_steps
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Import after parsing args to avoid slow imports
    from gflownet_peptide.models.forward_policy import ForwardPolicy
    from gflownet_peptide.models.backward_policy import BackwardPolicy
    from gflownet_peptide.models.reward_model import CompositeReward
    from gflownet_peptide.training.trainer import GFlowNetTrainer

    # Create forward policy
    policy_config = config.get("policy", {})
    forward_policy = ForwardPolicy(
        vocab_size=policy_config.get("vocab_size", 23),
        d_model=policy_config.get("d_model", 256),
        n_layers=policy_config.get("n_layers", 4),
        n_heads=policy_config.get("n_heads", 8),
        dim_feedforward=policy_config.get("dim_feedforward", 512),
        dropout=policy_config.get("dropout", 0.1),
        max_length=policy_config.get("max_length", 40),
    )
    logger.info(f"Created forward policy with {sum(p.numel() for p in forward_policy.parameters()):,} parameters")

    # Create backward policy (uniform for linear generation)
    backward_policy = BackwardPolicy(use_uniform=True)

    # Load or create reward model
    reward_config = config.get("reward", {})

    if args.reward_checkpoint:
        logger.info(f"Loading reward model from {args.reward_checkpoint}")
        reward_model = CompositeReward.from_pretrained(
            args.reward_checkpoint,
            esm_model=reward_config.get("esm_model", "esm2_t12_35M_UR50D"),
        )
    else:
        logger.info("Creating new composite reward model")
        reward_model = CompositeReward(
            esm_model=reward_config.get("esm_model", "esm2_t12_35M_UR50D"),
            stability_weight=reward_config.get("weights", {}).get("stability", 1.0),
            binding_weight=reward_config.get("weights", {}).get("binding", 1.0),
            naturalness_weight=reward_config.get("weights", {}).get("naturalness", 0.5),
            freeze_esm=True,
            share_backbone=True,
        )

    # Create trainer
    training_config = config.get("training", {})
    generation_config = config.get("generation", {})

    trainer = GFlowNetTrainer(
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        reward_model=reward_model,
        learning_rate=training_config.get("learning_rate", 3e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        loss_type=training_config.get("loss_type", "trajectory_balance"),
        min_length=generation_config.get("min_length", 10),
        max_length=generation_config.get("max_length", 30),
        reward_temperature=reward_config.get("temperature", 1.0),
        device=device,
    )

    # Setup wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_config = config.get("logging", {})
            wandb_run = wandb.init(
                project=wandb_config.get("wandb_project", "gflownet-peptide"),
                config=config,
            )
            logger.info("Wandb logging enabled")
        except ImportError:
            logger.warning("wandb not installed, skipping")

    # Train
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training for {training_config.get('n_steps', 100000)} steps")

    final_metrics = trainer.train(
        n_steps=training_config.get("n_steps", 100000),
        batch_size=training_config.get("batch_size", 64),
        temperature=generation_config.get("sample_temperature", 1.0),
        log_every=config.get("logging", {}).get("log_every", 100),
        eval_every=training_config.get("eval_every", 1000),
        save_every=training_config.get("save_every", 5000),
        checkpoint_dir=str(output_dir),
        wandb_run=wandb_run,
    )

    logger.info("Training complete!")
    logger.info(f"Final metrics: {final_metrics}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
