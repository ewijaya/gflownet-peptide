"""FLIP benchmark data loading utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

CANONICAL_AA = set('ACDEFGHIKLMNPQRSTVWY')


def validate_sequence(seq: str) -> bool:
    """Check if sequence contains only canonical amino acids.

    Args:
        seq: Amino acid sequence string

    Returns:
        True if sequence contains only canonical amino acids
    """
    return all(aa in CANONICAL_AA for aa in seq.upper())


def _find_csv_file(data_path: Path, default_name: str) -> Path:
    """Find CSV file in data path."""
    if data_path.is_file():
        return data_path

    # Try default name
    csv_file = data_path / default_name
    if csv_file.exists():
        return csv_file

    # Try any CSV file
    csv_files = list(data_path.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(f"No CSV file found in {data_path}")


def _create_splits(
    df: pd.DataFrame,
    split: Optional[str],
    seed: int
) -> pd.DataFrame:
    """Create or filter train/val/test splits."""
    if 'set' in df.columns:
        # Use existing splits from FLIP
        if split is not None:
            if split == 'val':
                # FLIP uses 'validation' column for val set within train
                if 'validation' in df.columns:
                    return df[(df['set'] == 'train') & (df['validation'].notna())]
                else:
                    # Fallback: take 10% of train as val
                    train_df = df[df['set'] == 'train']
                    val_size = int(len(train_df) * 0.1)
                    return train_df.sample(n=val_size, random_state=seed)
            else:
                return df[df['set'] == split]
        return df

    # Create random splits if none exist
    np.random.seed(seed)
    n = len(df)
    indices = np.random.permutation(n)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    df = df.copy()
    df['_split'] = 'test'
    df.iloc[indices[:train_end], df.columns.get_loc('_split')] = 'train'
    df.iloc[indices[train_end:val_end], df.columns.get_loc('_split')] = 'val'

    if split is not None:
        df = df[df['_split'] == split]

    return df


def load_flip_stability(
    data_path: str,
    min_length: int = 10,
    max_length: int = 50,
    normalize: bool = True,
    split: Optional[str] = None,
    seed: int = 42
) -> Tuple[List[str], np.ndarray]:
    """
    Load FLIP stability (meltome) task dataset.

    Args:
        data_path: Path to FLIP stability directory or CSV file
        min_length: Minimum sequence length to include
        max_length: Maximum sequence length to include
        normalize: Whether to normalize labels to zero-mean, unit-variance
        split: Optional split ('train', 'val', 'test', or None for all)
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (sequences, labels) where sequences is a list of strings
        and labels is a numpy array of fitness values.
    """
    data_path = Path(data_path)
    csv_file = _find_csv_file(data_path, 'stability.csv')

    df = pd.read_csv(csv_file)

    # Determine sequence and target columns
    seq_col = 'sequence' if 'sequence' in df.columns else df.columns[0]
    target_col = 'target' if 'target' in df.columns else (
        'fitness' if 'fitness' in df.columns else df.columns[1]
    )

    # Filter by length
    df['_length'] = df[seq_col].str.len()
    df = df[(df['_length'] >= min_length) & (df['_length'] <= max_length)]

    # Filter for canonical amino acids only
    df = df[df[seq_col].apply(validate_sequence)]

    # Apply splits
    df = _create_splits(df, split, seed)

    sequences = df[seq_col].tolist()
    labels = df[target_col].values.astype(np.float32)

    if normalize and len(labels) > 0:
        mean = labels.mean()
        std = labels.std()
        if std > 0:
            labels = (labels - mean) / std

    return sequences, labels


def load_flip_gb1(
    data_path: str,
    min_length: int = 10,
    max_length: int = 300,
    normalize: bool = True,
    split: Optional[str] = None,
    seed: int = 42,
    split_file: str = 'one_vs_rest.csv'
) -> Tuple[List[str], np.ndarray]:
    """
    Load FLIP GB1 binding task dataset.

    Note: GB1 sequences are longer (~280 AA) as they include the full
    protein construct with the GB1 domain.

    Args:
        data_path: Path to FLIP GB1 directory or CSV file
        min_length: Minimum sequence length to include
        max_length: Maximum sequence length to include
        normalize: Whether to normalize labels to zero-mean, unit-variance
        split: Optional split ('train', 'val', 'test', or None for all)
        seed: Random seed for reproducible splits
        split_file: Which split file to use (default: 'one_vs_rest.csv')

    Returns:
        Tuple of (sequences, labels) where sequences is a list of strings
        and labels is a numpy array of binding fitness values.
    """
    data_path = Path(data_path)

    # Try specific split file first
    if data_path.is_dir():
        csv_file = data_path / split_file
        if not csv_file.exists():
            csv_file = _find_csv_file(data_path, 'gb1.csv')
    else:
        csv_file = data_path

    df = pd.read_csv(csv_file)

    # Determine sequence and target columns
    seq_col = 'sequence' if 'sequence' in df.columns else df.columns[0]
    target_col = 'target' if 'target' in df.columns else (
        'Fitness' if 'Fitness' in df.columns else df.columns[1]
    )

    # Filter by length
    df['_length'] = df[seq_col].str.len()
    df = df[(df['_length'] >= min_length) & (df['_length'] <= max_length)]

    # Filter for canonical amino acids only
    df = df[df[seq_col].apply(validate_sequence)]

    # Apply splits
    df = _create_splits(df, split, seed)

    sequences = df[seq_col].tolist()
    labels = df[target_col].values.astype(np.float32)

    if normalize and len(labels) > 0:
        mean = labels.mean()
        std = labels.std()
        if std > 0:
            labels = (labels - mean) / std

    return sequences, labels
