#!/usr/bin/env python3
"""Validate Phase 0b peptides using ESMFold structure prediction.

This script runs ESMFold on generated peptides to assess structural quality.
pLDDT (predicted Local Distance Difference Test) scores indicate confidence
in the predicted structure - higher scores suggest more realistic structures.

pLDDT interpretation:
- >90: Very high confidence (well-folded)
- 70-90: High confidence (likely folded)
- 50-70: Low confidence (possibly disordered)
- <50: Very low confidence (likely disordered)

For therapeutic peptides, we expect pLDDT >50 for structurally plausible sequences.
"""

import argparse
import json
import logging
import os
from collections import Counter
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Natural amino acid frequencies from UniProt (approximate)
NATURAL_AA_FREQ = {
    'A': 0.0825, 'R': 0.0553, 'N': 0.0406, 'D': 0.0545, 'C': 0.0137,
    'Q': 0.0393, 'E': 0.0675, 'G': 0.0707, 'H': 0.0227, 'I': 0.0596,
    'L': 0.0966, 'K': 0.0584, 'M': 0.0242, 'F': 0.0386, 'P': 0.0470,
    'S': 0.0656, 'T': 0.0534, 'W': 0.0108, 'Y': 0.0292, 'V': 0.0687,
}

# APD3 antimicrobial peptide database frequencies (for comparison)
APD3_AA_FREQ = {
    'L': 0.089, 'G': 0.087, 'K': 0.082, 'A': 0.078, 'I': 0.061,
    'C': 0.054, 'V': 0.052, 'R': 0.049, 'S': 0.047, 'F': 0.044,
    'W': 0.039, 'T': 0.035, 'P': 0.033, 'N': 0.031, 'Y': 0.028,
    'Q': 0.024, 'H': 0.022, 'D': 0.018, 'E': 0.017, 'M': 0.017,
}


def compute_aa_composition(sequences):
    """Compute amino acid frequency distribution."""
    total_aa = 0
    aa_counts = Counter()

    for seq in sequences:
        aa_counts.update(seq)
        total_aa += len(seq)

    aa_freq = {aa: count / total_aa for aa, count in aa_counts.items()}

    # Ensure all 20 AAs are present
    for aa in NATURAL_AA_FREQ:
        if aa not in aa_freq:
            aa_freq[aa] = 0.0

    return aa_freq


def compute_kl_divergence(p, q):
    """Compute KL divergence D_KL(P || Q).

    Args:
        p: Generated distribution (dict)
        q: Reference distribution (dict)

    Returns:
        KL divergence value
    """
    kl_div = 0.0
    for aa in p:
        if p[aa] > 0 and q.get(aa, 0) > 0:
            kl_div += p[aa] * log2(p[aa] / q[aa])
    return kl_div


def compute_sequence_entropy(sequence):
    """Compute normalized Shannon entropy of a sequence."""
    if len(sequence) == 0:
        return 0.0

    aa_counts = Counter(sequence)
    total = len(sequence)

    entropy = 0.0
    for count in aa_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * log2(p)

    # Normalize by max entropy (log2(20))
    max_entropy = log2(20)
    return entropy / max_entropy


def check_repetitive_patterns(sequence):
    """Check for repetitive patterns in sequence.

    Returns:
        dict with homopolymer rate, dipeptide repeat rate, and pattern details
    """
    # Check for homopolymer runs (3+ same AA)
    max_run = 1
    current_run = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    has_homopolymer = max_run >= 3

    # Check for dipeptide repeats (e.g., ICIC, FYFY)
    has_dipeptide_repeat = False
    for i in range(len(sequence) - 3):
        if sequence[i:i+2] == sequence[i+2:i+4]:
            has_dipeptide_repeat = True
            break

    return {
        'max_homopolymer_run': max_run,
        'has_homopolymer': has_homopolymer,
        'has_dipeptide_repeat': has_dipeptide_repeat,
    }


def run_esmfold(sequences, batch_size=1, device='cuda'):
    """Run ESMFold to predict structures and get pLDDT scores.

    Uses Hugging Face transformers ESMFold implementation for Python 3.12 compatibility.

    Args:
        sequences: List of peptide sequences
        batch_size: Batch size (ESMFold is memory-intensive, use 1 for safety)
        device: Device to use

    Returns:
        List of dicts with pLDDT scores for each sequence
    """
    from transformers import EsmForProteinFolding, AutoTokenizer

    logger.info("Loading ESMFold model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.eval().to(device)

    # Disable gradient checkpointing for inference
    model.esm = model.esm.half()  # Use FP16 for ESM backbone
    model.trunk.set_chunk_size(64)  # Reduce memory usage

    results = []

    for seq in tqdm(sequences, desc="Running ESMFold"):
        try:
            with torch.no_grad():
                # Tokenize
                inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Run inference
                outputs = model(**inputs)

                # Extract pLDDT scores (per-residue confidence)
                # pLDDT is in outputs.plddt with shape [batch, seq_len]
                plddt = outputs.plddt[0].cpu().numpy()

                # Filter valid residues (exclude padding)
                seq_len = len(seq)
                plddt = plddt[:seq_len]

                mean_plddt = float(np.mean(plddt))
                min_plddt = float(np.min(plddt))
                max_plddt = float(np.max(plddt))

                results.append({
                    'sequence': seq,
                    'mean_plddt': mean_plddt,
                    'min_plddt': min_plddt,
                    'max_plddt': max_plddt,
                    'length': len(seq),
                })

        except Exception as e:
            logger.warning(f"ESMFold failed for sequence {seq[:20]}...: {e}")
            results.append({
                'sequence': seq,
                'mean_plddt': 0.0,
                'min_plddt': 0.0,
                'max_plddt': 0.0,
                'length': len(seq),
                'error': str(e),
            })

    return results


def analyze_peptides(peptides_csv, output_dir, run_esmfold_analysis=True, max_peptides=50):
    """Comprehensive analysis of generated peptides.

    Args:
        peptides_csv: Path to peptides CSV file
        output_dir: Output directory for results
        run_esmfold_analysis: Whether to run ESMFold (slow, ~1min per peptide)
        max_peptides: Maximum peptides to analyze with ESMFold
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load peptides
    df = pd.read_csv(peptides_csv)
    sequences = df['peptide'].tolist() if 'peptide' in df.columns else df['sequence'].tolist()

    logger.info(f"Analyzing {len(sequences)} peptides from {peptides_csv}")

    # Basic statistics
    lengths = [len(seq) for seq in sequences]
    entropies = [compute_sequence_entropy(seq) for seq in sequences]

    # AA composition analysis
    aa_freq = compute_aa_composition(sequences)
    kl_vs_uniprot = compute_kl_divergence(aa_freq, NATURAL_AA_FREQ)
    kl_vs_apd3 = compute_kl_divergence(aa_freq, APD3_AA_FREQ)

    # Pattern analysis
    pattern_results = [check_repetitive_patterns(seq) for seq in sequences]
    homopolymer_rate = sum(1 for r in pattern_results if r['has_homopolymer']) / len(sequences)
    dipeptide_repeat_rate = sum(1 for r in pattern_results if r['has_dipeptide_repeat']) / len(sequences)

    # Compile basic metrics
    basic_metrics = {
        'n_sequences': len(sequences),
        'length_stats': {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
        },
        'entropy_stats': {
            'mean': np.mean(entropies),
            'std': np.std(entropies),
            'min': min(entropies),
            'max': max(entropies),
        },
        'aa_composition': aa_freq,
        'kl_divergence': {
            'vs_uniprot': kl_vs_uniprot,
            'vs_apd3': kl_vs_apd3,
        },
        'pattern_analysis': {
            'homopolymer_rate': homopolymer_rate,
            'dipeptide_repeat_rate': dipeptide_repeat_rate,
        },
    }

    # Identify over/under-represented amino acids
    aa_bias = {}
    for aa in NATURAL_AA_FREQ:
        if aa in aa_freq:
            ratio = aa_freq[aa] / NATURAL_AA_FREQ[aa] if NATURAL_AA_FREQ[aa] > 0 else 0
            aa_bias[aa] = {
                'generated_freq': aa_freq.get(aa, 0),
                'natural_freq': NATURAL_AA_FREQ[aa],
                'ratio': ratio,
            }

    # Sort by ratio to find biased AAs
    over_represented = sorted(
        [(aa, data['ratio']) for aa, data in aa_bias.items() if data['ratio'] > 1.5],
        key=lambda x: -x[1]
    )
    under_represented = sorted(
        [(aa, data['ratio']) for aa, data in aa_bias.items() if data['ratio'] < 0.5],
        key=lambda x: x[1]
    )

    basic_metrics['aa_bias'] = {
        'over_represented': {aa: ratio for aa, ratio in over_represented},
        'under_represented': {aa: ratio for aa, ratio in under_represented},
    }

    # ESMFold analysis (optional, slow)
    esmfold_metrics = None
    if run_esmfold_analysis:
        # Sample sequences for ESMFold (it's slow)
        sample_size = min(max_peptides, len(sequences))
        sample_indices = np.random.choice(len(sequences), sample_size, replace=False)
        sample_seqs = [sequences[i] for i in sample_indices]

        logger.info(f"Running ESMFold on {sample_size} sequences...")
        esmfold_results = run_esmfold(sample_seqs)

        plddt_scores = [r['mean_plddt'] for r in esmfold_results if r['mean_plddt'] > 0]

        if plddt_scores:
            esmfold_metrics = {
                'n_analyzed': len(plddt_scores),
                'mean_plddt': np.mean(plddt_scores),
                'std_plddt': np.std(plddt_scores),
                'min_plddt': np.min(plddt_scores),
                'max_plddt': np.max(plddt_scores),
                'plddt_gt50_rate': sum(1 for p in plddt_scores if p > 50) / len(plddt_scores),
                'plddt_gt70_rate': sum(1 for p in plddt_scores if p > 70) / len(plddt_scores),
            }

            # Save detailed ESMFold results
            esmfold_df = pd.DataFrame(esmfold_results)
            esmfold_df.to_csv(output_dir / 'esmfold_results.csv', index=False)

    # Combine all metrics
    all_metrics = {
        'basic': basic_metrics,
        'esmfold': esmfold_metrics,
    }

    # Save metrics
    with open(output_dir / 'baseline_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Generate report
    report = generate_report(basic_metrics, esmfold_metrics, peptides_csv)
    with open(output_dir / 'baseline_report.md', 'w') as f:
        f.write(report)

    logger.info(f"Results saved to {output_dir}")

    return all_metrics


def generate_report(basic_metrics, esmfold_metrics, peptides_csv):
    """Generate markdown report of analysis."""
    report = f"""# GRPO-D Baseline Metrics Report

**Source**: `{peptides_csv}`
**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total sequences | {basic_metrics['n_sequences']} |
| Mean length | {basic_metrics['length_stats']['mean']:.1f} ± {basic_metrics['length_stats']['std']:.1f} |
| Mean entropy | {basic_metrics['entropy_stats']['mean']:.3f} ± {basic_metrics['entropy_stats']['std']:.3f} |

## Pattern Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Homopolymer rate | {basic_metrics['pattern_analysis']['homopolymer_rate']*100:.1f}% | Sequences with 3+ consecutive same AAs |
| Dipeptide repeat rate | {basic_metrics['pattern_analysis']['dipeptide_repeat_rate']*100:.1f}% | Sequences with repeating dipeptides (e.g., ICIC) |

## Amino Acid Composition

**KL Divergence**:
- vs UniProt natural proteins: {basic_metrics['kl_divergence']['vs_uniprot']:.3f}
- vs APD3 antimicrobial peptides: {basic_metrics['kl_divergence']['vs_apd3']:.3f}

**Over-represented AAs** (>1.5x natural frequency):
"""
    for aa, ratio in basic_metrics['aa_bias']['over_represented'].items():
        report += f"- {aa}: {ratio:.1f}x\n"

    report += """
**Under-represented AAs** (<0.5x natural frequency):
"""
    for aa, ratio in basic_metrics['aa_bias']['under_represented'].items():
        report += f"- {aa}: {ratio:.2f}x\n"

    if esmfold_metrics:
        report += f"""
## Structural Quality (ESMFold)

| Metric | Value | Target |
|--------|-------|--------|
| Sequences analyzed | {esmfold_metrics['n_analyzed']} | - |
| Mean pLDDT | {esmfold_metrics['mean_plddt']:.1f} | >50 |
| pLDDT >50 rate | {esmfold_metrics['plddt_gt50_rate']*100:.1f}% | >80% |
| pLDDT >70 rate | {esmfold_metrics['plddt_gt70_rate']*100:.1f}% | - |

**pLDDT Interpretation**:
- >90: Very high confidence (well-folded)
- 70-90: High confidence (likely folded)
- 50-70: Low confidence (possibly disordered)
- <50: Very low confidence (likely disordered)
"""
    else:
        report += "\n## Structural Quality (ESMFold)\n\n*ESMFold analysis not performed*\n"

    report += """
## Baseline for GFlowNet Comparison

These metrics establish the GRPO-D baseline for Phase 4 comparison:

| Metric | GRPO-D Value | GFlowNet Target |
|--------|--------------|-----------------|
"""
    report += f"| Sequence entropy | {basic_metrics['entropy_stats']['mean']:.3f} | ≥{basic_metrics['entropy_stats']['mean']:.3f} |\n"
    report += f"| Homopolymer rate | {basic_metrics['pattern_analysis']['homopolymer_rate']*100:.1f}% | <20% |\n"
    report += f"| AA KL divergence (vs UniProt) | {basic_metrics['kl_divergence']['vs_uniprot']:.3f} | ≤{basic_metrics['kl_divergence']['vs_uniprot']:.3f} |\n"

    if esmfold_metrics:
        report += f"| Mean pLDDT | {esmfold_metrics['mean_plddt']:.1f} | ≥{esmfold_metrics['mean_plddt']:.1f} |\n"

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate peptides with ESMFold")
    parser.add_argument('--peptides', type=str,
                        default='results/grpo/20251224_093311_grpod_it1000_dw0.15_beta0.04_peptides.csv',
                        help='Path to peptides CSV')
    parser.add_argument('--output_dir', type=str, default='outputs/grpod_baseline',
                        help='Output directory')
    parser.add_argument('--skip_esmfold', action='store_true',
                        help='Skip ESMFold analysis (faster)')
    parser.add_argument('--max_peptides', type=int, default=50,
                        help='Maximum peptides for ESMFold analysis')

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1: GRPO-D Peptide Validation")
    print("=" * 60)

    analyze_peptides(
        args.peptides,
        args.output_dir,
        run_esmfold_analysis=not args.skip_esmfold,
        max_peptides=args.max_peptides,
    )


if __name__ == '__main__':
    main()
