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
        "--run_name",
        type=str,
        default=None,
        help="W&B run name (e.g., 'baseline-10k', 'sweep-lr1e4')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="composite",
        choices=["composite", "esm2_pll", "improved", "trained"],
        help="Reward function type: composite (default, untrained MLP heads), "
             "esm2_pll (ESM pseudo-likelihood), improved (entropy gate + naturalness), "
             "trained (trained stability + entropy gate, requires --reward_checkpoint)",
    )
    parser.add_argument(
        "--esm_model",
        type=str,
        default=None,
        help="ESM-2 model for reward (default: from config or esm2_t6_8M_UR50D)",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_reward_model(args, reward_config, device):
    """Create reward model based on --reward_type flag.

    Args:
        args: Parsed command line arguments
        reward_config: Reward configuration from YAML
        device: torch device

    Returns:
        Reward model (nn.Module or callable)
    """
    esm_model = args.esm_model or reward_config.get("esm_model", "esm2_t6_8M_UR50D")

    if args.reward_type == "composite":
        # Original behavior - uses models/reward_model.py (untrained MLP heads)
        from gflownet_peptide.models.reward_model import CompositeReward
        if args.reward_checkpoint:
            logger.info(f"Loading composite reward from {args.reward_checkpoint}")
            return CompositeReward.from_pretrained(
                args.reward_checkpoint,
                esm_model=esm_model,
            )
        logger.info(f"Creating composite reward with untrained heads ({esm_model})")
        return CompositeReward(
            esm_model=esm_model,
            stability_weight=reward_config.get("weights", {}).get("stability", 1.0),
            binding_weight=reward_config.get("weights", {}).get("binding", 1.0),
            naturalness_weight=reward_config.get("weights", {}).get("naturalness", 0.5),
            freeze_esm=True,
            share_backbone=True,
        )

    elif args.reward_type == "improved":
        from gflownet_peptide.rewards.improved_reward import ImprovedReward
        logger.info(f"Using improved reward (entropy gate + naturalness) with {esm_model}")
        return ImprovedReward(
            model_name=esm_model,
            device=str(device),
            entropy_threshold=0.5,
            entropy_sharpness=10.0,
            min_length=10,
            normalize=True,
        )

    elif args.reward_type == "esm2_pll":
        from gflownet_peptide.rewards.esm2_reward import ESM2Reward
        logger.info(f"Using ESM-2 pseudo-likelihood reward with {esm_model}")
        return ESM2Reward(
            model_name=esm_model,
            device=str(device),
            normalize=True,
            temperature=reward_config.get("temperature", 1.0),
        )

    elif args.reward_type == "trained":
        from gflownet_peptide.rewards.composite_reward import CompositeReward
        if not args.reward_checkpoint:
            raise ValueError("--reward_type trained requires --reward_checkpoint")
        logger.info(f"Using trained stability reward from {args.reward_checkpoint}")
        return CompositeReward(
            stability_checkpoint=args.reward_checkpoint,
            weights={'stability': 1.0, 'naturalness': 0.5},
            esm_model=esm_model,
            device=str(device),
        )

    else:
        raise ValueError(f"Unknown reward type: {args.reward_type}")


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

    # Create reward model based on --reward_type
    reward_config = config.get("reward", {})
    reward_model = create_reward_model(args, reward_config, device)

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

            # Generate descriptive run name if not provided
            if args.run_name:
                run_name = args.run_name
            else:
                # Auto-generate: gflownet-{steps}-{batch}-{lr}
                n_steps = training_config.get("n_steps", 100000)
                batch_size = training_config.get("batch_size", 64)
                lr = training_config.get("learning_rate", 3e-4)
                run_name = f"gflownet-{n_steps//1000}k-b{batch_size}-lr{lr:.0e}"

            wandb_run = wandb.init(
                project=wandb_config.get("wandb_project", "gflownet-peptide"),
                entity=wandb_config.get("wandb_entity"),
                name=run_name,
                config=config,
            )
            logger.info(f"Wandb logging enabled: {run_name}")
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
