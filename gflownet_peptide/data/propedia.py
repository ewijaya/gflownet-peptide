"""Propedia/PepBDB protein-peptide binding data loading utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

from .flip import CANONICAL_AA, validate_sequence


def load_propedia(
    data_path: str,
    min_length: int = 10,
    max_length: int = 50,
    normalize: bool = True,
    split: Optional[str] = None,
    seed: int = 42
) -> Tuple[List[str], np.ndarray]:
    """
    Load Propedia/PepBDB protein-peptide binding dataset.

    This dataset contains peptide sequences extracted from the PepBDB database
    of peptide-protein structural complexes. Since these are experimentally
    verified binding peptides, we use a binary binding label (1.0) or can
    assign synthetic affinity scores based on structural properties.

    Args:
        data_path: Path to Propedia directory or CSV file
        min_length: Minimum peptide length to include
        max_length: Maximum peptide length to include
        normalize: Whether to normalize labels (no effect for binary labels)
        split: Optional split ('train', 'val', 'test', or None for all)
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (sequences, labels) where sequences is a list of peptide
        strings and labels is a numpy array of binding scores (1.0 for binders).
    """
    data_path = Path(data_path)

    # Find CSV file
    if data_path.is_file():
        csv_file = data_path
    else:
        csv_file = data_path / 'propedia.csv'
        if not csv_file.exists():
            # Try any CSV
            csv_files = list(data_path.glob("*.csv"))
            if csv_files:
                csv_file = csv_files[0]
            else:
                raise FileNotFoundError(f"No CSV file found in {data_path}")

    df = pd.read_csv(csv_file)

    # Determine sequence column
    seq_col = None
    for col in ['sequence', 'peptide_sequence', 'peptide']:
        if col in df.columns:
            seq_col = col
            break
    if seq_col is None:
        seq_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Filter by length
    if 'length' in df.columns:
        df = df[(df['length'] >= min_length) & (df['length'] <= max_length)]
    else:
        df['_length'] = df[seq_col].str.len()
        df = df[(df['_length'] >= min_length) & (df['_length'] <= max_length)]

    # Filter for canonical amino acids only
    df = df[df[seq_col].apply(validate_sequence)]

    # Remove duplicates
    df = df.drop_duplicates(subset=[seq_col])

    # Create splits
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

    sequences = df[seq_col].tolist()

    # Use binding affinity if available, otherwise binary label
    if 'binding_affinity' in df.columns:
        labels = df['binding_affinity'].values.astype(np.float32)
        if normalize and len(labels) > 0:
            mean = labels.mean()
            std = labels.std()
            if std > 0:
                labels = (labels - mean) / std
    else:
        # All peptides in PepBDB are verified binders, so label = 1.0
        labels = np.ones(len(sequences), dtype=np.float32)

    return sequences, labels
