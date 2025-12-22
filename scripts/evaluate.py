#!/usr/bin/env python3
"""
Evaluate and compare peptide generation methods.

This script computes diversity and quality metrics for generated peptides
and optionally compares multiple methods.

Usage:
    python scripts/evaluate.py --gflownet_samples samples/gflownet.csv
    python scripts/evaluate.py --gflownet_samples samples/gflownet.csv --grpo_samples samples/grpo.csv
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate peptide samples")

    parser.add_argument(
        "--gflownet_samples",
        type=str,
        required=True,
        help="Path to GFlowNet samples CSV",
    )
    parser.add_argument(
        "--grpo_samples",
        type=str,
        default=None,
        help="Path to GRPO samples CSV (for comparison)",
    )
    parser.add_argument(
        "--reward_checkpoint",
        type=str,
        default=None,
        help="Path to reward model for scoring",
    )
    parser.add_argument(
        "--compute_embeddings",
        action="store_true",
        help="Compute ESM embeddings for diversity metrics",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="figures/",
        help="Directory for output figures",
    )

    return parser.parse_args()


def load_samples(path: str) -> pd.DataFrame:
    """Load samples from CSV."""
    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise ValueError(f"CSV must have 'sequence' column: {path}")
    return df


def compute_esm_embeddings(sequences: list[str], device: torch.device) -> np.ndarray:
    """Compute ESM-2 embeddings for sequences."""
    try:
        import esm
    except ImportError:
        logger.warning("ESM not installed, skipping embedding computation")
        return None

    logger.info("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    embeddings = []
    batch_size = 32

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12], return_contacts=False)
            token_embeddings = results["representations"][12]

        # Mean pool
        batch_embeddings = token_embeddings[:, 1:-1, :].mean(dim=1)
        embeddings.append(batch_embeddings.cpu().numpy())

        if (i + batch_size) % 100 == 0:
            logger.info(f"Computed embeddings for {min(i + batch_size, len(sequences))}/{len(sequences)}")

    return np.vstack(embeddings)


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import evaluation functions
    from gflownet_peptide.evaluation.metrics import (
        compute_sequence_diversity,
        compute_embedding_diversity,
        compute_cluster_metrics,
        compute_proportionality,
        compute_quality_metrics,
    )
    from gflownet_peptide.evaluation.visualize import (
        plot_umap_clusters,
        plot_reward_distribution,
        plot_proportionality,
        plot_comparison,
        create_summary_table,
    )

    # Load samples
    logger.info(f"Loading GFlowNet samples from {args.gflownet_samples}")
    gflownet_df = load_samples(args.gflownet_samples)
    gflownet_seqs = gflownet_df["sequence"].tolist()

    results = {}

    # Compute GFlowNet metrics
    logger.info("Computing GFlowNet metrics...")

    gflownet_metrics = {}

    # Sequence diversity
    seq_diversity = compute_sequence_diversity(gflownet_seqs)
    gflownet_metrics.update(seq_diversity)

    # Quality metrics (if rewards available)
    if "reward" in gflownet_df.columns:
        rewards = gflownet_df["reward"].values
        quality = compute_quality_metrics(rewards)
        gflownet_metrics.update(quality)

        # Proportionality
        prop = compute_proportionality(rewards)
        gflownet_metrics.update({f"prop_{k}": v for k, v in prop.items()
                                 if not isinstance(v, (list, np.ndarray))})

    # Embedding-based metrics
    if args.compute_embeddings:
        logger.info("Computing ESM embeddings...")
        embeddings = compute_esm_embeddings(gflownet_seqs, device)

        if embeddings is not None:
            embed_div = compute_embedding_diversity(embeddings)
            gflownet_metrics.update({f"embed_{k}": v for k, v in embed_div.items()})

            cluster = compute_cluster_metrics(embeddings)
            gflownet_metrics.update({f"cluster_{k}": v for k, v in cluster.items()
                                     if not isinstance(v, np.ndarray)})

            # Store for visualization
            gflownet_umap = cluster.get("umap_coords")
            gflownet_labels = cluster.get("cluster_labels")

    results["gflownet"] = gflownet_metrics

    # Compare with GRPO if provided
    if args.grpo_samples:
        logger.info(f"Loading GRPO samples from {args.grpo_samples}")
        grpo_df = load_samples(args.grpo_samples)
        grpo_seqs = grpo_df["sequence"].tolist()

        grpo_metrics = {}

        # Sequence diversity
        seq_diversity = compute_sequence_diversity(grpo_seqs)
        grpo_metrics.update(seq_diversity)

        # Quality metrics
        if "reward" in grpo_df.columns:
            rewards = grpo_df["reward"].values
            quality = compute_quality_metrics(rewards)
            grpo_metrics.update(quality)

        # Embedding metrics
        if args.compute_embeddings:
            logger.info("Computing GRPO embeddings...")
            embeddings = compute_esm_embeddings(grpo_seqs, device)

            if embeddings is not None:
                embed_div = compute_embedding_diversity(embeddings)
                grpo_metrics.update({f"embed_{k}": v for k, v in embed_div.items()})

                cluster = compute_cluster_metrics(embeddings)
                grpo_metrics.update({f"cluster_{k}": v for k, v in cluster.items()
                                     if not isinstance(v, np.ndarray)})

        results["grpo"] = grpo_metrics

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Create comparison if both methods available
    if "grpo" in results:
        print("\n" + "=" * 60)
        print("COMPARISON (GFlowNet vs GRPO)")
        print("=" * 60)

        # Key metrics to compare
        compare_keys = [
            "sequence_diversity",
            "unique_ratio",
            "mean_reward",
            "max_reward",
            "cluster_n_clusters",
        ]

        for key in compare_keys:
            if key in results["gflownet"] and key in results["grpo"]:
                gf_val = results["gflownet"][key]
                grpo_val = results["grpo"][key]
                if isinstance(gf_val, (int, float)) and isinstance(grpo_val, (int, float)):
                    ratio = gf_val / (grpo_val + 1e-8)
                    print(f"  {key}: GFlowNet={gf_val:.4f}, GRPO={grpo_val:.4f}, ratio={ratio:.2f}x")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_path}")

    # Create figures
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reward distribution
    if "reward" in gflownet_df.columns:
        rewards_dict = {"GFlowNet": gflownet_df["reward"].values}
        if args.grpo_samples and "reward" in grpo_df.columns:
            rewards_dict["GRPO"] = grpo_df["reward"].values

        plot_reward_distribution(
            rewards_dict,
            save_path=figures_dir / "reward_distribution.png",
        )
        logger.info(f"Saved reward distribution plot")

        # Proportionality plot for GFlowNet
        plot_proportionality(
            gflownet_df["reward"].values,
            save_path=figures_dir / "proportionality.png",
        )
        logger.info(f"Saved proportionality plot")

    # Comparison bar chart
    if "grpo" in results:
        plot_comparison(
            results,
            metrics_to_plot=["sequence_diversity", "unique_ratio", "mean_reward", "max_reward"],
            save_path=figures_dir / "comparison.png",
        )
        logger.info(f"Saved comparison plot")

        # Summary table
        table = create_summary_table(results, figures_dir / "summary.md")
        print("\n" + table)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
