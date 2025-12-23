#!/usr/bin/env python
"""Validate downloaded datasets for the GFlowNet peptide project."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


def validate_flip_stability(data_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Validate FLIP stability dataset."""
    from gflownet_peptide.data import load_flip_stability, validate_sequence

    results = {
        'name': 'FLIP Stability',
        'status': 'unknown',
        'issues': [],
        'stats': {}
    }

    try:
        # Check if file exists
        data_path = Path(data_path)
        if not data_path.exists():
            results['status'] = 'error'
            results['issues'].append(f"Path does not exist: {data_path}")
            return results

        # Load data - FLIP stability has full proteins, not just peptides
        # Use broader length range for validation
        sequences, labels = load_flip_stability(
            str(data_path),
            min_length=10,
            max_length=2000,  # Full proteins can be very long
            normalize=False
        )

        results['stats']['total_sequences'] = len(sequences)
        results['stats']['label_min'] = float(labels.min()) if len(labels) > 0 else None
        results['stats']['label_max'] = float(labels.max()) if len(labels) > 0 else None
        results['stats']['label_mean'] = float(labels.mean()) if len(labels) > 0 else None

        # Check sequence validity
        invalid_count = sum(1 for seq in sequences if not validate_sequence(seq))
        if invalid_count > 0:
            results['issues'].append(f"{invalid_count} sequences with non-canonical AA")

        # Length distribution
        lengths = [len(seq) for seq in sequences]
        results['stats']['length_min'] = min(lengths) if lengths else None
        results['stats']['length_max'] = max(lengths) if lengths else None
        results['stats']['length_mean'] = float(np.mean(lengths)) if lengths else None

        # Check minimum count (FLIP stability should have ~27K sequences)
        if len(sequences) < 20000:
            results['issues'].append(f"Only {len(sequences)} sequences (expected >= 20000)")

        # Determine status
        if len(results['issues']) == 0:
            results['status'] = 'pass'
        elif any('error' in issue.lower() for issue in results['issues']):
            results['status'] = 'error'
        else:
            results['status'] = 'warning'

        if verbose:
            print(f"  Sequences: {len(sequences)}")
            print(f"  Label range: [{results['stats']['label_min']:.2f}, {results['stats']['label_max']:.2f}]")
            print(f"  Length range: [{results['stats']['length_min']}, {results['stats']['length_max']}]")

    except Exception as e:
        results['status'] = 'error'
        results['issues'].append(f"Error loading data: {str(e)}")

    return results


def validate_flip_gb1(data_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Validate FLIP GB1 dataset."""
    from gflownet_peptide.data import load_flip_gb1, validate_sequence

    results = {
        'name': 'FLIP GB1',
        'status': 'unknown',
        'issues': [],
        'stats': {}
    }

    try:
        data_path = Path(data_path)
        if not data_path.exists():
            results['status'] = 'skipped'
            results['issues'].append(f"Path does not exist (optional): {data_path}")
            return results

        sequences, labels = load_flip_gb1(
            str(data_path),
            min_length=10,
            max_length=300,
            normalize=False
        )

        results['stats']['total_sequences'] = len(sequences)
        results['stats']['label_min'] = float(labels.min()) if len(labels) > 0 else None
        results['stats']['label_max'] = float(labels.max()) if len(labels) > 0 else None

        lengths = [len(seq) for seq in sequences]
        results['stats']['length_min'] = min(lengths) if lengths else None
        results['stats']['length_max'] = max(lengths) if lengths else None

        if len(results['issues']) == 0:
            results['status'] = 'pass'
        else:
            results['status'] = 'warning'

        if verbose:
            print(f"  Sequences: {len(sequences)}")
            print(f"  Label range: [{results['stats']['label_min']:.2f}, {results['stats']['label_max']:.2f}]")

    except Exception as e:
        results['status'] = 'error'
        results['issues'].append(f"Error loading data: {str(e)}")

    return results


def validate_propedia(data_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Validate Propedia/PepBDB dataset."""
    from gflownet_peptide.data import load_propedia, validate_sequence

    results = {
        'name': 'Propedia/PepBDB',
        'status': 'unknown',
        'issues': [],
        'stats': {}
    }

    try:
        data_path = Path(data_path)
        if not data_path.exists():
            results['status'] = 'error'
            results['issues'].append(f"Path does not exist: {data_path}")
            return results

        sequences, labels = load_propedia(
            str(data_path),
            min_length=10,
            max_length=50,
            normalize=False
        )

        results['stats']['total_sequences'] = len(sequences)
        results['stats']['unique_sequences'] = len(set(sequences))

        lengths = [len(seq) for seq in sequences]
        results['stats']['length_min'] = min(lengths) if lengths else None
        results['stats']['length_max'] = max(lengths) if lengths else None
        results['stats']['length_mean'] = float(np.mean(lengths)) if lengths else None

        # Check minimum count (Propedia should have ~4K unique peptides 10-50 AA)
        if len(sequences) < 3000:
            results['issues'].append(f"Only {len(sequences)} sequences (expected >= 3000)")

        if len(results['issues']) == 0:
            results['status'] = 'pass'
        else:
            results['status'] = 'warning'

        if verbose:
            print(f"  Sequences: {len(sequences)}")
            print(f"  Unique: {results['stats']['unique_sequences']}")
            print(f"  Length range: [{results['stats']['length_min']}, {results['stats']['length_max']}]")

    except Exception as e:
        results['status'] = 'error'
        results['issues'].append(f"Error loading data: {str(e)}")

    return results


def validate_esm2(verbose: bool = False) -> Dict[str, Any]:
    """Validate ESM-2 model loading."""
    results = {
        'name': 'ESM-2',
        'status': 'unknown',
        'issues': [],
        'stats': {}
    }

    try:
        import esm
        import torch

        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        # Test forward pass
        test_seq = "MKFLILFLPFASMGKLL"
        data = [("test", test_seq)]
        _, _, batch_tokens = batch_converter(data)

        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[12])
            embeddings = out['representations'][12]

        expected_shape = (1, len(test_seq) + 2, 480)
        if embeddings.shape != expected_shape:
            results['issues'].append(
                f"Unexpected embedding shape: {embeddings.shape} vs {expected_shape}"
            )

        results['stats']['model'] = 'esm2_t12_35M_UR50D'
        results['stats']['embedding_dim'] = model.embed_dim
        results['stats']['alphabet_size'] = len(alphabet)

        if len(results['issues']) == 0:
            results['status'] = 'pass'
        else:
            results['status'] = 'error'

        if verbose:
            print(f"  Model: {results['stats']['model']}")
            print(f"  Embedding dim: {results['stats']['embedding_dim']}")

    except ImportError as e:
        results['status'] = 'error'
        results['issues'].append(f"Import error: {str(e)}")
    except Exception as e:
        results['status'] = 'error'
        results['issues'].append(f"Error: {str(e)}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Validate project datasets')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json', help='Output JSON report to file')
    parser.add_argument('--skip-esm', action='store_true', help='Skip ESM-2 validation')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_results = []
    has_errors = False
    has_warnings = False

    print("=" * 60)
    print("GFlowNet Peptide Data Validation")
    print("=" * 60)

    # Validate FLIP Stability
    print("\n[1/4] Validating FLIP Stability...")
    result = validate_flip_stability(data_dir / 'flip' / 'stability', args.verbose)
    all_results.append(result)
    status_icon = {'pass': '✓', 'warning': '⚠', 'error': '✗', 'skipped': '○'}.get(result['status'], '?')
    print(f"  Status: {status_icon} {result['status'].upper()}")
    if result['issues']:
        for issue in result['issues']:
            print(f"    - {issue}")
    if result['status'] == 'error':
        has_errors = True
    elif result['status'] == 'warning':
        has_warnings = True

    # Validate FLIP GB1
    print("\n[2/4] Validating FLIP GB1 (optional)...")
    result = validate_flip_gb1(data_dir / 'flip' / 'gb1', args.verbose)
    all_results.append(result)
    status_icon = {'pass': '✓', 'warning': '⚠', 'error': '✗', 'skipped': '○'}.get(result['status'], '?')
    print(f"  Status: {status_icon} {result['status'].upper()}")
    if result['issues']:
        for issue in result['issues']:
            print(f"    - {issue}")

    # Validate Propedia
    print("\n[3/4] Validating Propedia/PepBDB...")
    result = validate_propedia(data_dir / 'propedia', args.verbose)
    all_results.append(result)
    status_icon = {'pass': '✓', 'warning': '⚠', 'error': '✗', 'skipped': '○'}.get(result['status'], '?')
    print(f"  Status: {status_icon} {result['status'].upper()}")
    if result['issues']:
        for issue in result['issues']:
            print(f"    - {issue}")
    if result['status'] == 'error':
        has_errors = True
    elif result['status'] == 'warning':
        has_warnings = True

    # Validate ESM-2
    if not args.skip_esm:
        print("\n[4/4] Validating ESM-2...")
        result = validate_esm2(args.verbose)
        all_results.append(result)
        status_icon = {'pass': '✓', 'warning': '⚠', 'error': '✗', 'skipped': '○'}.get(result['status'], '?')
        print(f"  Status: {status_icon} {result['status'].upper()}")
        if result['issues']:
            for issue in result['issues']:
                print(f"    - {issue}")
        if result['status'] == 'error':
            has_errors = True
    else:
        print("\n[4/4] Skipping ESM-2 validation...")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for result in all_results:
        status_icon = {'pass': '✓', 'warning': '⚠', 'error': '✗', 'skipped': '○'}.get(result['status'], '?')
        print(f"  {status_icon} {result['name']}: {result['status'].upper()}")
        if 'total_sequences' in result.get('stats', {}):
            print(f"      Sequences: {result['stats']['total_sequences']}")

    # Write JSON report
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nJSON report written to: {args.json}")

    # Exit code
    if has_errors:
        print("\n✗ Validation FAILED with errors")
        sys.exit(2)
    elif has_warnings:
        print("\n⚠ Validation completed with warnings")
        sys.exit(1)
    else:
        print("\n✓ All validations PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()
