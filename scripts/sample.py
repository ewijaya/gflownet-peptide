#!/usr/bin/env python3
"""
Sample diverse peptides from trained GFlowNet.

Usage:
    python scripts/sample.py --checkpoint checkpoints/gflownet/best.pt --n_samples 1000
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample peptides from GFlowNet")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to GFlowNet checkpoint",
    )
    parser.add_argument(
        "--reward_checkpoint",
        type=str,
        default=None,
        help="Path to reward model checkpoint (for scoring)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of peptides to sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum peptide length",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=30,
        help="Maximum peptide length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="samples/peptides.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Import models
    from gflownet_peptide.models.forward_policy import ForwardPolicy
    from gflownet_peptide.models.reward_model import CompositeReward

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create and load forward policy
    # Try to infer config from checkpoint
    policy_config = checkpoint.get("config", {}).get("policy", {})

    forward_policy = ForwardPolicy(
        vocab_size=policy_config.get("vocab_size", 23),
        d_model=policy_config.get("d_model", 256),
        n_layers=policy_config.get("n_layers", 4),
        n_heads=policy_config.get("n_heads", 8),
    ).to(device)

    forward_policy.load_state_dict(checkpoint["forward_policy_state_dict"])
    forward_policy.eval()

    logger.info("Loaded forward policy")

    # Load reward model for scoring (optional)
    reward_model = None
    if args.reward_checkpoint:
        logger.info(f"Loading reward model from {args.reward_checkpoint}")
        reward_model = CompositeReward.from_pretrained(args.reward_checkpoint).to(device)
        reward_model.eval()

    # Sample peptides
    logger.info(f"Sampling {args.n_samples} peptides...")

    all_sequences = []
    all_log_probs = []

    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for i in range(n_batches):
            batch_size = min(args.batch_size, args.n_samples - len(all_sequences))

            sequences, log_probs = forward_policy.sample_sequence(
                batch_size=batch_size,
                max_length=args.max_length,
                min_length=args.min_length,
                temperature=args.temperature,
                device=device,
            )

            all_sequences.extend(sequences)
            all_log_probs.extend(log_probs.cpu().tolist())

            if (i + 1) % 10 == 0:
                logger.info(f"Sampled {len(all_sequences)}/{args.n_samples}")

    logger.info(f"Sampled {len(all_sequences)} peptides")

    # Compute rewards if model available
    rewards = None
    if reward_model is not None:
        logger.info("Computing rewards...")
        rewards = []

        for i in range(0, len(all_sequences), args.batch_size):
            batch = all_sequences[i:i + args.batch_size]
            with torch.no_grad():
                batch_rewards = reward_model(batch).cpu().tolist()
            rewards.extend(batch_rewards)

    # Create DataFrame
    df = pd.DataFrame({
        "sequence": all_sequences,
        "log_prob": all_log_probs,
        "length": [len(s) for s in all_sequences],
    })

    if rewards is not None:
        df["reward"] = rewards

    # Add uniqueness info
    df["is_unique"] = ~df["sequence"].duplicated()

    # Sort by reward (if available) or log_prob
    if rewards is not None:
        df = df.sort_values("reward", ascending=False)
    else:
        df = df.sort_values("log_prob", ascending=False)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(df)} peptides to {output_path}")

    # Print summary statistics
    print("\n=== Sample Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Unique sequences: {df['is_unique'].sum()} ({100 * df['is_unique'].mean():.1f}%)")
    print(f"Mean length: {df['length'].mean():.1f} +/- {df['length'].std():.1f}")

    if rewards is not None:
        print(f"Mean reward: {df['reward'].mean():.4f}")
        print(f"Max reward: {df['reward'].max():.4f}")
        print(f"Min reward: {df['reward'].min():.4f}")

    print("\n=== Top 10 Peptides ===")
    for i, row in df.head(10).iterrows():
        if rewards is not None:
            print(f"{row['sequence']} (R={row['reward']:.4f}, len={row['length']})")
        else:
            print(f"{row['sequence']} (log_p={row['log_prob']:.2f}, len={row['length']})")


if __name__ == "__main__":
    main()
