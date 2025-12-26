#!/usr/bin/env python3
"""Preprocess FLIP and Propedia datasets for reward model training.

This script handles data preprocessing for Phase 1 reward model training:
1. FLIP Stability: Filter, normalize, and create train/val/test splits
2. Propedia: Filter peptides and create splits (binary labels for binders)

Note: FLIP stability data contains mostly full proteins (median ~400 AA).
We use a relaxed length filter (20-500 AA) to get sufficient training data,
since ESM-2 embeddings can transfer patterns across lengths.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')


def validate_sequence(seq: str) -> bool:
    """Check if sequence contains only valid amino acids."""
    if not isinstance(seq, str) or len(seq) == 0:
        return False
    return all(aa in VALID_AAS for aa in seq.upper())


def preprocess_flip_stability(
    input_path: str,
    output_dir: str,
    min_length: int = 20,
    max_length: int = 500,
    seed: int = 42
) -> dict:
    """Preprocess FLIP stability dataset.

    Args:
        input_path: Path to stability.csv
        output_dir: Output directory for processed files
        min_length: Minimum sequence length (default 20)
        max_length: Maximum sequence length (default 500)
        seed: Random seed for splits

    Returns:
        Dictionary with preprocessing statistics
    """
    logger.info(f"Loading FLIP stability from {input_path}")
    df = pd.read_csv(input_path)

    original_count = len(df)
    logger.info(f"Original dataset size: {original_count}")

    # Identify columns
    seq_col = 'sequence'
    target_col = 'target'

    # Add length column
    df['length'] = df[seq_col].str.len()

    # Filter by length
    df = df[(df['length'] >= min_length) & (df['length'] <= max_length)]
    logger.info(f"After length filter ({min_length}-{max_length}): {len(df)}")

    # Validate sequences (canonical AAs only, no X or special chars)
    df = df[df[seq_col].apply(validate_sequence)]
    logger.info(f"After sequence validation: {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates(subset=[seq_col])
    logger.info(f"After deduplication: {len(df)}")

    if len(df) == 0:
        raise ValueError("No sequences remaining after filtering!")

    # Compute normalization parameters on full dataset
    mean_target = df[target_col].mean()
    std_target = df[target_col].std()
    df['target_normalized'] = (df[target_col] - mean_target) / std_target

    logger.info(f"Target stats: mean={mean_target:.2f}, std={std_target:.2f}")

    # Use existing FLIP splits if available
    if 'set' in df.columns:
        train_df = df[df['set'] == 'train'].copy()
        test_df = df[df['set'] == 'test'].copy()

        # Split train into train/val (90/10)
        if len(train_df) > 100:
            train_df, val_df = train_test_split(
                train_df, test_size=0.1, random_state=seed
            )
        else:
            val_df = train_df.sample(n=max(1, len(train_df)//10), random_state=seed)
            train_df = train_df.drop(val_df.index)
    else:
        # Create new splits (80/10/10)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save processed files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select columns to save
    save_cols = [seq_col, target_col, 'target_normalized', 'length']

    train_df[save_cols].to_csv(output_dir / 'train.csv', index=False)
    val_df[save_cols].to_csv(output_dir / 'val.csv', index=False)
    test_df[save_cols].to_csv(output_dir / 'test.csv', index=False)

    # Save normalization parameters
    norm_params = {
        'mean': float(mean_target),
        'std': float(std_target),
        'min_length': min_length,
        'max_length': max_length,
    }
    with open(output_dir / 'norm_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    stats = {
        'original_count': original_count,
        'filtered_count': len(df),
        'train_count': len(train_df),
        'val_count': len(val_df),
        'test_count': len(test_df),
        'mean_target': mean_target,
        'std_target': std_target,
        'mean_length': df['length'].mean(),
        'min_length_actual': df['length'].min(),
        'max_length_actual': df['length'].max(),
    }

    logger.info(f"Saved processed data to {output_dir}")
    return stats


def preprocess_propedia(
    input_path: str,
    output_dir: str,
    min_length: int = 10,
    max_length: int = 50,
    seed: int = 42
) -> dict:
    """Preprocess Propedia peptide-protein binding dataset.

    Note: Propedia only contains peptide sequences without binding affinity.
    All peptides are verified binders, so we use binary labels (1.0).
    This can be used for binder/non-binder classification if combined
    with negative examples.

    Args:
        input_path: Path to propedia.csv
        output_dir: Output directory for processed files
        min_length: Minimum peptide length
        max_length: Maximum peptide length
        seed: Random seed for splits

    Returns:
        Dictionary with preprocessing statistics
    """
    logger.info(f"Loading Propedia from {input_path}")
    df = pd.read_csv(input_path)

    original_count = len(df)
    logger.info(f"Original dataset size: {original_count}")

    # Identify sequence column
    seq_col = 'sequence'

    # Filter by length
    if 'length' in df.columns:
        df = df[(df['length'] >= min_length) & (df['length'] <= max_length)]
    else:
        df['length'] = df[seq_col].str.len()
        df = df[(df['length'] >= min_length) & (df['length'] <= max_length)]

    logger.info(f"After length filter ({min_length}-{max_length}): {len(df)}")

    # Validate sequences
    df = df[df[seq_col].apply(validate_sequence)]
    logger.info(f"After sequence validation: {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates(subset=[seq_col])
    logger.info(f"After deduplication: {len(df)}")

    if len(df) == 0:
        raise ValueError("No peptides remaining after filtering!")

    # All Propedia peptides are binders (binary label = 1.0)
    df['binding_label'] = 1.0

    # Create splits (80/10/10)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save processed files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_cols = ['pdb_id', seq_col, 'length', 'binding_label']
    save_cols = [c for c in save_cols if c in df.columns]

    train_df[save_cols].to_csv(output_dir / 'train.csv', index=False)
    val_df[save_cols].to_csv(output_dir / 'val.csv', index=False)
    test_df[save_cols].to_csv(output_dir / 'test.csv', index=False)

    # Save metadata
    metadata = {
        'min_length': min_length,
        'max_length': max_length,
        'label_type': 'binary',
        'label_description': 'All peptides are verified binders (label=1.0)',
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    stats = {
        'original_count': original_count,
        'filtered_count': len(df),
        'train_count': len(train_df),
        'val_count': len(val_df),
        'test_count': len(test_df),
        'mean_length': df['length'].mean(),
    }

    logger.info(f"Saved processed data to {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for reward model training")
    parser.add_argument('--flip_input', type=str, default='data/flip/stability/stability.csv',
                        help='Path to FLIP stability CSV')
    parser.add_argument('--flip_output', type=str, default='data/processed/flip_stability',
                        help='Output directory for processed FLIP data')
    parser.add_argument('--propedia_input', type=str, default='data/propedia/propedia.csv',
                        help='Path to Propedia CSV')
    parser.add_argument('--propedia_output', type=str, default='data/processed/propedia',
                        help='Output directory for processed Propedia data')
    parser.add_argument('--flip_min_length', type=int, default=20,
                        help='Minimum length for FLIP sequences')
    parser.add_argument('--flip_max_length', type=int, default=500,
                        help='Maximum length for FLIP sequences')
    parser.add_argument('--propedia_min_length', type=int, default=10,
                        help='Minimum length for Propedia peptides')
    parser.add_argument('--propedia_max_length', type=int, default=50,
                        help='Maximum length for Propedia peptides')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1: Data Preprocessing")
    print("=" * 60)

    # Process FLIP stability
    print("\n--- Processing FLIP Stability ---")
    flip_stats = preprocess_flip_stability(
        args.flip_input,
        args.flip_output,
        min_length=args.flip_min_length,
        max_length=args.flip_max_length,
        seed=args.seed
    )
    print(f"\nFLIP Statistics:")
    for k, v in flip_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Process Propedia
    print("\n--- Processing Propedia ---")
    propedia_stats = preprocess_propedia(
        args.propedia_input,
        args.propedia_output,
        min_length=args.propedia_min_length,
        max_length=args.propedia_max_length,
        seed=args.seed
    )
    print(f"\nPropedia Statistics:")
    for k, v in propedia_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
