"""
Evaluation Metrics for GFlowNet Peptide Generation.

Implements diversity, quality, and calibration metrics.
"""

from typing import Optional

import numpy as np
import torch


def compute_sequence_diversity(sequences: list[str]) -> dict:
    """
    Compute sequence-level diversity metrics.

    Args:
        sequences: List of peptide sequences

    Returns:
        metrics: Dictionary containing diversity metrics
    """
    n = len(sequences)
    if n == 0:
        return {"sequence_diversity": 0.0, "unique_ratio": 0.0}

    # Unique sequences
    unique_seqs = set(sequences)
    unique_ratio = len(unique_seqs) / n

    # Pairwise sequence identity
    def seq_identity(s1: str, s2: str) -> float:
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(s1, s2))
        return matches / max(len(s1), len(s2))

    # Sample for efficiency if too many sequences
    sample_seqs = list(unique_seqs)
    if len(sample_seqs) > 500:
        np.random.shuffle(sample_seqs)
        sample_seqs = sample_seqs[:500]

    # Compute mean pairwise identity
    total_identity = 0.0
    count = 0
    for i, s1 in enumerate(sample_seqs):
        for s2 in sample_seqs[i + 1:]:
            total_identity += seq_identity(s1, s2)
            count += 1

    mean_identity = total_identity / max(count, 1)
    sequence_diversity = 1 - mean_identity

    # Length statistics
    lengths = [len(s) for s in sequences]

    return {
        "sequence_diversity": sequence_diversity,
        "unique_ratio": unique_ratio,
        "mean_identity": mean_identity,
        "n_unique": len(unique_seqs),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }


def compute_embedding_diversity(
    embeddings: np.ndarray,
    sample_size: int = 500,
) -> dict:
    """
    Compute embedding-space diversity metrics.

    Args:
        embeddings: Sequence embeddings [n_sequences, embed_dim]
        sample_size: Max number of sequences for pairwise computation

    Returns:
        metrics: Dictionary containing embedding diversity metrics
    """
    n = len(embeddings)
    if n == 0:
        return {"embedding_diversity": 0.0}

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-8)

    # Sample if needed
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings_normalized = embeddings_normalized[indices]

    # Compute pairwise cosine distances
    # cos_dist = 1 - cos_sim = 1 - (a · b)
    similarity_matrix = embeddings_normalized @ embeddings_normalized.T
    n_sample = len(embeddings_normalized)

    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n_sample, k=1)
    similarities = similarity_matrix[triu_indices]

    mean_similarity = np.mean(similarities)
    embedding_diversity = 1 - mean_similarity

    return {
        "embedding_diversity": embedding_diversity,
        "mean_cosine_similarity": mean_similarity,
        "std_cosine_similarity": np.std(similarities),
    }


def compute_cluster_metrics(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 5,
) -> dict:
    """
    Compute clustering metrics using UMAP + HDBSCAN.

    Args:
        embeddings: Sequence embeddings [n_sequences, embed_dim]
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        min_cluster_size: HDBSCAN min_cluster_size

    Returns:
        metrics: Dictionary containing clustering metrics
    """
    try:
        import umap
        import hdbscan
    except ImportError:
        return {"n_clusters": -1, "error": "umap-learn and hdbscan required"}

    if len(embeddings) < min_cluster_size * 2:
        return {"n_clusters": 0, "noise_ratio": 1.0}

    # UMAP projection
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
    )
    umap_coords = reducer.fit_transform(embeddings)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
    )
    cluster_labels = clusterer.fit_predict(umap_coords)

    # Count clusters (excluding noise = -1)
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Noise ratio
    noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)

    # Cluster sizes
    cluster_sizes = []
    for label in unique_labels:
        if label != -1:
            cluster_sizes.append(np.sum(cluster_labels == label))

    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "mean_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
        "std_cluster_size": np.std(cluster_sizes) if cluster_sizes else 0,
        "umap_coords": umap_coords,
        "cluster_labels": cluster_labels,
    }


def compute_proportionality(
    rewards: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Check if sampling frequency is proportional to reward.

    For a well-trained GFlowNet: P(x) ∝ R(x)
    So log(frequency) should be linear with log(reward).

    Args:
        rewards: Reward values for sampled sequences
        n_bins: Number of bins for reward

    Returns:
        metrics: Dictionary containing proportionality metrics
    """
    if len(rewards) == 0:
        return {"proportionality_r2": 0.0}

    rewards = np.array(rewards)

    # Remove zero rewards
    valid_mask = rewards > 0
    rewards = rewards[valid_mask]

    if len(rewards) < n_bins * 2:
        return {"proportionality_r2": 0.0, "n_valid": len(rewards)}

    # Bin by reward
    log_rewards = np.log(rewards)
    bin_edges = np.percentile(log_rewards, np.linspace(0, 100, n_bins + 1))

    bin_counts = []
    bin_means = []

    for i in range(n_bins):
        mask = (log_rewards >= bin_edges[i]) & (log_rewards < bin_edges[i + 1])
        if i == n_bins - 1:  # Include last edge
            mask = mask | (log_rewards == bin_edges[i + 1])

        count = np.sum(mask)
        if count > 0:
            bin_counts.append(count)
            bin_means.append(np.mean(log_rewards[mask]))

    if len(bin_counts) < 3:
        return {"proportionality_r2": 0.0, "n_bins_used": len(bin_counts)}

    bin_counts = np.array(bin_counts)
    bin_means = np.array(bin_means)

    # Compute R² between log(frequency) and log(reward)
    log_freq = np.log(bin_counts + 1)  # +1 to avoid log(0)

    # Linear regression
    slope, intercept = np.polyfit(bin_means, log_freq, 1)
    predicted = slope * bin_means + intercept
    ss_res = np.sum((log_freq - predicted) ** 2)
    ss_tot = np.sum((log_freq - np.mean(log_freq)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return {
        "proportionality_r2": r2,
        "proportionality_slope": slope,
        "n_bins_used": len(bin_counts),
        "bin_counts": bin_counts.tolist(),
        "bin_means": bin_means.tolist(),
    }


def compute_quality_metrics(rewards: np.ndarray) -> dict:
    """
    Compute quality metrics from rewards.

    Args:
        rewards: Reward values

    Returns:
        metrics: Dictionary containing quality metrics
    """
    rewards = np.array(rewards)

    if len(rewards) == 0:
        return {
            "mean_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
            "std_reward": 0.0,
        }

    # Top-k statistics
    sorted_rewards = np.sort(rewards)[::-1]
    top_10_pct = int(len(rewards) * 0.1)
    top_10_pct = max(1, top_10_pct)

    return {
        "mean_reward": np.mean(rewards),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards),
        "std_reward": np.std(rewards),
        "median_reward": np.median(rewards),
        "top_10pct_mean": np.mean(sorted_rewards[:top_10_pct]),
        "top_10pct_min": sorted_rewards[top_10_pct - 1] if top_10_pct > 0 else 0,
    }


def compute_all_metrics(
    sequences: list[str],
    rewards: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    compute_clusters: bool = True,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        sequences: List of peptide sequences
        rewards: Reward values for sequences
        embeddings: Optional embeddings for diversity/clustering
        compute_clusters: Whether to compute cluster metrics

    Returns:
        metrics: Dictionary containing all metrics
    """
    metrics = {}

    # Sequence diversity
    seq_metrics = compute_sequence_diversity(sequences)
    metrics.update({f"seq_{k}": v for k, v in seq_metrics.items()})

    # Quality metrics
    quality_metrics = compute_quality_metrics(rewards)
    metrics.update(quality_metrics)

    # Proportionality
    prop_metrics = compute_proportionality(rewards)
    metrics.update({f"prop_{k}": v for k, v in prop_metrics.items()
                    if not isinstance(v, (list, np.ndarray))})

    # Embedding-based metrics
    if embeddings is not None:
        embed_metrics = compute_embedding_diversity(embeddings)
        metrics.update({f"embed_{k}": v for k, v in embed_metrics.items()})

        if compute_clusters:
            cluster_metrics = compute_cluster_metrics(embeddings)
            # Only include scalar metrics
            for k, v in cluster_metrics.items():
                if not isinstance(v, np.ndarray):
                    metrics[f"cluster_{k}"] = v

    return metrics
