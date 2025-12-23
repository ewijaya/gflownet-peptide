"""Diversity calculation functions for GRPO-D.

This module implements per-peptide diversity scoring following the user's
original GRPO-D implementation. Diversity is computed using two components:

1. Amino acid frequency diversity: Rewards peptides with rare AAs in the batch
2. Sequence dissimilarity: Levenshtein distance to other peptides

These are combined with configurable weights (default: 0.7 AA, 0.3 sequence).
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

import Levenshtein


def calculate_aa_frequency_diversity(peptides: List[str]) -> List[float]:
    """Calculate diversity based on amino acid frequency rarity.

    For each peptide, compute a rarity score based on how uncommon its
    amino acids are within the batch. Peptides with rare AAs get higher scores.

    The score for each peptide is:
        sum(1 / (aa_count + 1)) / length

    where aa_count is how many times that amino acid appears across all peptides.

    Args:
        peptides: List of peptide sequences

    Returns:
        List of diversity scores (one per peptide), higher = more diverse
    """
    if not peptides:
        return []

    # Count amino acid frequencies across all peptides
    aa_counts: Dict[str, int] = defaultdict(int)
    for peptide in peptides:
        for aa in peptide:
            aa_counts[aa] += 1

    # Calculate rarity scores for each peptide
    diversity_scores = []
    for peptide in peptides:
        if not peptide:
            diversity_scores.append(0.0)
            continue

        # Sum of rarity (inverse frequency) for each AA in peptide
        aa_rarity = 0.0
        for aa in peptide:
            aa_rarity += 1.0 / (aa_counts[aa] + 1)

        # Normalize by length
        diversity = aa_rarity / len(peptide)
        diversity_scores.append(diversity)

    return diversity_scores


def calculate_sequence_dissimilarity(
    peptides: List[str],
    max_compare: int = 50,
) -> List[float]:
    """Calculate diversity based on sequence dissimilarity (Levenshtein distance).

    For each peptide, compute the average normalized edit distance to other
    peptides in the batch. Higher distance = more different = higher diversity.

    Args:
        peptides: List of peptide sequences
        max_compare: Maximum number of reference peptides to compare against
                     (for efficiency with large batches)

    Returns:
        List of diversity scores (one per peptide), higher = more diverse
    """
    if not peptides:
        return []

    if len(peptides) == 1:
        return [1.0]  # Single peptide is maximally diverse

    # For large batches, sample a subset of peptides to compare against
    reference_peptides = peptides
    if len(peptides) > max_compare:
        reference_peptides = random.sample(peptides, max_compare)

    # Calculate average dissimilarity to other peptides
    diversity_scores = []
    for peptide in peptides:
        if not peptide:
            diversity_scores.append(0.0)
            continue

        total_distance = 0.0
        count = 0

        for ref_peptide in reference_peptides:
            if peptide == ref_peptide:
                continue

            # Calculate normalized Levenshtein distance
            distance = Levenshtein.distance(peptide, ref_peptide)
            max_len = max(len(peptide), len(ref_peptide))
            normalized_distance = distance / max_len if max_len > 0 else 0.0

            total_distance += normalized_distance
            count += 1

        avg_distance = total_distance / count if count > 0 else 0.0
        diversity_scores.append(avg_distance)

    return diversity_scores


def calculate_peptide_diversity(
    peptides: List[str],
    config: Optional[Dict] = None,
) -> List[float]:
    """Calculate combined diversity scores for peptides.

    Combines amino acid frequency diversity and sequence dissimilarity
    with configurable weights.

    Args:
        peptides: List of peptide sequences
        config: Configuration dict with optional keys:
            - diversity_weight_aa: Weight for AA frequency diversity (default: 0.7)
            - diversity_weight_seq: Weight for sequence dissimilarity (default: 0.3)
            - use_sequence_diversity: Whether to use sequence diversity (default: True)

    Returns:
        List of diversity scores in [0, 1] range (one per peptide)
    """
    if not peptides:
        return []

    config = config or {}

    # Get weights from config
    div_weight_aa = config.get("diversity_weight_aa", 0.7)
    div_weight_seq = config.get("diversity_weight_seq", 0.3)
    use_seq_diversity = config.get("use_sequence_diversity", True)

    # Compute AA frequency diversity
    aa_diversity = calculate_aa_frequency_diversity(peptides)

    # Compute sequence dissimilarity
    if use_seq_diversity:
        seq_diversity = calculate_sequence_dissimilarity(peptides)
    else:
        seq_diversity = [0.0] * len(peptides)

    # Combine diversity metrics with weights
    diversity_scores = []
    for i in range(len(peptides)):
        combined = div_weight_aa * aa_diversity[i] + div_weight_seq * seq_diversity[i]
        diversity_scores.append(combined)

    # Normalize to [0, 1] range using min-max normalization
    if diversity_scores:
        min_div = min(diversity_scores)
        max_div = max(diversity_scores)
        if max_div > min_div:
            diversity_scores = [
                (d - min_div) / (max_div - min_div) for d in diversity_scores
            ]
        else:
            # All same diversity, set to 0.5
            diversity_scores = [0.5] * len(diversity_scores)

    return diversity_scores


def calculate_batch_diversity_stats(peptides: List[str]) -> Dict[str, float]:
    """Calculate summary statistics for batch diversity.

    Useful for logging and monitoring during training.

    Args:
        peptides: List of peptide sequences

    Returns:
        Dictionary with diversity statistics
    """
    if not peptides:
        return {
            "min_diversity": 0.0,
            "max_diversity": 0.0,
            "mean_diversity": 0.0,
            "unique_ratio": 0.0,
        }

    diversity_scores = calculate_peptide_diversity(peptides)

    unique_peptides = set(peptides)
    unique_ratio = len(unique_peptides) / len(peptides)

    return {
        "min_diversity": min(diversity_scores),
        "max_diversity": max(diversity_scores),
        "mean_diversity": sum(diversity_scores) / len(diversity_scores),
        "unique_ratio": unique_ratio,
    }
