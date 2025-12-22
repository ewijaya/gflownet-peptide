"""
Visualization Tools for GFlowNet Peptide Generation.

Provides plotting functions for evaluation and comparison.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def plot_umap_clusters(
    umap_coords: np.ndarray,
    cluster_labels: np.ndarray,
    rewards: Optional[np.ndarray] = None,
    title: str = "UMAP Projection with Clusters",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """
    Plot UMAP projection colored by cluster or reward.

    Args:
        umap_coords: UMAP coordinates [n, 2]
        cluster_labels: Cluster labels for each point
        rewards: Optional rewards for color coding
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if rewards is not None else 1, figsize=figsize)

    if rewards is None:
        axes = [axes]

    # Plot 1: Clusters
    ax = axes[0]
    scatter = ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=cluster_labels,
        cmap="tab20",
        s=10,
        alpha=0.7,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{title} - Clusters")

    # Count clusters
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    ax.text(
        0.02, 0.98, f"n_clusters = {n_clusters}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Plot 2: Rewards (if provided)
    if rewards is not None:
        ax = axes[1]
        scatter = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=rewards,
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax, label="Reward")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{title} - Rewards")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_reward_distribution(
    rewards_dict: dict[str, np.ndarray],
    title: str = "Reward Distribution Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Plot reward distributions for multiple methods.

    Args:
        rewards_dict: Dictionary mapping method names to reward arrays
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for name, rewards in rewards_dict.items():
        ax.hist(
            rewards,
            bins=50,
            alpha=0.5,
            label=f"{name} (mean={np.mean(rewards):.3f})",
            density=True,
        )

    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_proportionality(
    rewards: np.ndarray,
    n_bins: int = 10,
    title: str = "GFlowNet Proportionality Check",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
):
    """
    Plot log(frequency) vs log(reward) to verify proportional sampling.

    Args:
        rewards: Reward values
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    rewards = np.array(rewards)
    valid_mask = rewards > 0
    rewards = rewards[valid_mask]

    if len(rewards) < n_bins * 2:
        print("Not enough samples for proportionality plot")
        return

    log_rewards = np.log(rewards)
    bin_edges = np.percentile(log_rewards, np.linspace(0, 100, n_bins + 1))

    bin_counts = []
    bin_means = []

    for i in range(n_bins):
        mask = (log_rewards >= bin_edges[i]) & (log_rewards < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (log_rewards == bin_edges[i + 1])

        count = np.sum(mask)
        if count > 0:
            bin_counts.append(count)
            bin_means.append(np.mean(log_rewards[mask]))

    bin_counts = np.array(bin_counts)
    bin_means = np.array(bin_means)
    log_freq = np.log(bin_counts + 1)

    # Fit line
    slope, intercept = np.polyfit(bin_means, log_freq, 1)
    fit_line = slope * bin_means + intercept

    # R²
    ss_res = np.sum((log_freq - fit_line) ** 2)
    ss_tot = np.sum((log_freq - np.mean(log_freq)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(bin_means, log_freq, s=100, alpha=0.7, label="Observed")
    ax.plot(bin_means, fit_line, "r--", label=f"Fit (R²={r2:.3f})")

    # Ideal line (slope = 1 for P(x) ∝ R(x))
    ideal_line = bin_means + (np.mean(log_freq) - np.mean(bin_means))
    ax.plot(bin_means, ideal_line, "g:", alpha=0.5, label="Ideal (slope=1)")

    ax.set_xlabel("log(Reward)")
    ax.set_ylabel("log(Frequency)")
    ax.set_title(f"{title}\nSlope={slope:.2f}, R²={r2:.3f}")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison(
    metrics_dict: dict[str, dict],
    metrics_to_plot: list[str] = None,
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """
    Create bar plot comparing metrics across methods.

    Args:
        metrics_dict: Dictionary mapping method names to metric dicts
        metrics_to_plot: List of metrics to include (None = all common)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    methods = list(metrics_dict.keys())

    # Find common metrics
    if metrics_to_plot is None:
        common_metrics = set(metrics_dict[methods[0]].keys())
        for method in methods[1:]:
            common_metrics &= set(metrics_dict[method].keys())
        # Filter to numeric metrics
        metrics_to_plot = [
            m for m in common_metrics
            if isinstance(metrics_dict[methods[0]][m], (int, float))
        ]

    n_metrics = len(metrics_to_plot)
    x = np.arange(n_metrics)
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        values = [metrics_dict[method].get(m, 0) for m in metrics_to_plot]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=method, alpha=0.8)

    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_summary_table(
    metrics_dict: dict[str, dict],
    output_path: Optional[str] = None,
) -> str:
    """
    Create markdown table summarizing metrics across methods.

    Args:
        metrics_dict: Dictionary mapping method names to metric dicts
        output_path: Optional path to save markdown file

    Returns:
        table: Markdown table as string
    """
    methods = list(metrics_dict.keys())

    # Find common metrics
    common_metrics = set(metrics_dict[methods[0]].keys())
    for method in methods[1:]:
        common_metrics &= set(metrics_dict[method].keys())

    # Filter to numeric metrics
    numeric_metrics = [
        m for m in sorted(common_metrics)
        if isinstance(metrics_dict[methods[0]][m], (int, float))
    ]

    # Build table
    header = "| Metric | " + " | ".join(methods) + " |"
    separator = "|" + "|".join(["---"] * (len(methods) + 1)) + "|"

    rows = []
    for metric in numeric_metrics:
        values = []
        for method in methods:
            v = metrics_dict[method].get(metric, "N/A")
            if isinstance(v, float):
                values.append(f"{v:.4f}")
            else:
                values.append(str(v))
        row = f"| {metric} | " + " | ".join(values) + " |"
        rows.append(row)

    table = "\n".join([header, separator] + rows)

    if output_path:
        Path(output_path).write_text(table)

    return table
