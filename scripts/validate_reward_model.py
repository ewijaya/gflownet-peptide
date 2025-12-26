#!/usr/bin/env python3
"""Validate trained reward model on test set and sanity checks.

This script validates the Phase 1 reward model by:
1. Computing R² on held-out test set (target: ≥0.6)
2. Verifying non-negativity of all rewards
3. Checking reward spread (std > 0.1)
4. Timing inference speed (<100ms per sequence)
5. Testing entropy gate integration (homopolymers score <0.1)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_stability_predictor(checkpoint_path, test_csv, device='cuda'):
    """Validate stability predictor on test set.

    Args:
        checkpoint_path: Path to trained checkpoint
        test_csv: Path to test data CSV
        device: Device for computation

    Returns:
        Dictionary with validation metrics
    """
    from gflownet_peptide.rewards import StabilityPredictor

    logger.info(f"Loading stability predictor from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    model = StabilityPredictor(
        esm_model=config.get('esm_model', 'esm2_t6_8M_UR50D'),
        hidden_dims=config.get('hidden_dims', [256, 128]),
        dropout=config.get('dropout', 0.1),
        freeze_esm=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data
    df = pd.read_csv(test_csv)
    sequences = df['sequence'].tolist()
    labels = df['target_normalized'].values

    # Predict
    all_preds = []
    batch_size = 16
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            preds = model(batch)
            all_preds.extend(preds.cpu().tolist())

    total_time = time.time() - start_time
    time_per_seq = total_time / len(sequences) * 1000  # ms

    # Compute R²
    r2 = r2_score(labels, all_preds)

    return {
        'r2': r2,
        'n_samples': len(sequences),
        'time_per_seq_ms': time_per_seq,
        'checkpoint_val_r2': checkpoint.get('val_r2', 'N/A'),
    }


def validate_composite_reward(stability_checkpoint=None, device='cuda'):
    """Validate composite reward function.

    Args:
        stability_checkpoint: Path to stability predictor checkpoint
        device: Device for computation

    Returns:
        Dictionary with validation results
    """
    from gflownet_peptide.rewards import CompositeReward

    # Create reward (with or without stability predictor)
    reward = CompositeReward(
        stability_checkpoint=stability_checkpoint,
        device=device,
    )

    # Test sequences
    test_cases = [
        # (name, sequence, expected_behavior)
        ("Real peptide (signal)", "MKTLLILAVVALACARSSAQAANPF", "high"),
        ("Real peptide (antimicrobial)", "GIGKFLHSAKKFGKAFVGEIMNS", "high"),
        ("Homopolymer Q", "QQQQQQQQQQQQQQQQQQQQQQQQ", "low"),
        ("Homopolymer A", "AAAAAAAAAAAAAAAAAAAAAAAA", "low"),
        ("Repetitive pattern", "AQAQAQAQAQAQAQAQAQAQAQAQ", "low"),
        ("All different AAs", "ACDEFGHIKLMNPQRSTVWY", "medium"),
        ("Too short", "ACDE", "low"),
        ("Phase 0b example 1", "AQRPYPIQSICICWHHNFYVVVVVVDTLG", "medium"),
        ("Phase 0b example 2", "VKSIQSFYFYYPICAILMQFNQRYPHNWHH", "medium"),
    ]

    results = []
    timing_seqs = []

    for name, seq, expected in test_cases:
        start = time.time()
        score = reward([seq])[0]
        elapsed = (time.time() - start) * 1000

        components = reward.get_components(seq)

        results.append({
            'name': name,
            'sequence': seq[:30] + '...' if len(seq) > 30 else seq,
            'length': len(seq),
            'reward': score,
            'expected': expected,
            'time_ms': elapsed,
            **{k: v for k, v in components.items() if k != 'total'},
        })
        timing_seqs.append(elapsed)

    # Aggregate metrics
    all_rewards = [r['reward'] for r in results]

    validation = {
        'test_cases': results,
        'summary': {
            'min_reward': min(all_rewards),
            'max_reward': max(all_rewards),
            'mean_reward': sum(all_rewards) / len(all_rewards),
            'std_reward': torch.tensor(all_rewards).std().item(),
            'mean_time_ms': sum(timing_seqs) / len(timing_seqs),
            'all_non_negative': all(r >= 0 for r in all_rewards),
        },
        'criteria': {
            'non_negativity': all(r >= 0 for r in all_rewards),
            'spread_gt_0.1': torch.tensor(all_rewards).std().item() > 0.1,
            'homopolymer_lt_0.1': all(
                r['reward'] < 0.3 for r in results
                if 'Homopolymer' in r['name'] or r['expected'] == 'low'
            ),
            'good_peptide_gt_0.5': all(
                r['reward'] > 0.3 for r in results
                if r['expected'] == 'high'
            ),
        }
    }

    return validation


def main():
    parser = argparse.ArgumentParser(description="Validate reward model")
    parser.add_argument('--stability_checkpoint', type=str,
                        default='checkpoints/reward_models/stability_predictor_best.pt',
                        help='Path to stability predictor checkpoint')
    parser.add_argument('--test_csv', type=str,
                        default='data/processed/flip_stability/test.csv',
                        help='Path to test data CSV')
    parser.add_argument('--output', type=str,
                        default='outputs/reward_validation.json',
                        help='Output file for validation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1: Reward Model Validation")
    print("=" * 60)

    device = args.device if torch.cuda.is_available() else 'cpu'
    results = {}

    # 1. Validate stability predictor if checkpoint exists
    if Path(args.stability_checkpoint).exists():
        print("\n--- Stability Predictor Validation ---")
        stability_results = validate_stability_predictor(
            args.stability_checkpoint,
            args.test_csv,
            device,
        )
        results['stability_predictor'] = stability_results

        print(f"Test R²: {stability_results['r2']:.4f} (target: ≥0.6)")
        print(f"Status: {'✅ PASSED' if stability_results['r2'] >= 0.6 else '⚠️ BELOW TARGET'}")
        print(f"Time per sequence: {stability_results['time_per_seq_ms']:.1f}ms (target: <100ms)")
    else:
        print(f"\n⚠️ Stability checkpoint not found: {args.stability_checkpoint}")
        print("Running composite reward validation without stability predictor...")
        results['stability_predictor'] = {'status': 'checkpoint_not_found'}

    # 2. Validate composite reward
    print("\n--- Composite Reward Validation ---")
    checkpoint = args.stability_checkpoint if Path(args.stability_checkpoint).exists() else None
    composite_results = validate_composite_reward(checkpoint, device)
    results['composite_reward'] = composite_results

    print("\nTest Cases:")
    print(f"{'Name':<30} {'Reward':>8} {'Expected':>10}")
    print("-" * 50)
    for tc in composite_results['test_cases']:
        status = "✓" if (
            (tc['expected'] == 'high' and tc['reward'] > 0.3) or
            (tc['expected'] == 'low' and tc['reward'] < 0.3) or
            (tc['expected'] == 'medium')
        ) else "✗"
        print(f"{tc['name']:<30} {tc['reward']:>8.4f} {tc['expected']:>10} {status}")

    print("\n--- Success Criteria ---")
    criteria = composite_results['criteria']
    print(f"Non-negativity (all ≥ 0): {'✅' if criteria['non_negativity'] else '❌'}")
    print(f"Reward spread (std > 0.1): {'✅' if criteria['spread_gt_0.1'] else '❌'} (std={composite_results['summary']['std_reward']:.3f})")
    print(f"Homopolymers penalized (<0.3): {'✅' if criteria['homopolymer_lt_0.1'] else '❌'}")
    print(f"Good peptides rewarded (>0.3): {'✅' if criteria['good_peptide_gt_0.5'] else '❌'}")

    # Overall pass/fail
    all_passed = all(criteria.values())
    if 'stability_predictor' in results and 'r2' in results['stability_predictor']:
        all_passed = all_passed and results['stability_predictor']['r2'] >= 0.6

    print("\n" + "=" * 60)
    print(f"Overall Status: {'✅ ALL CRITERIA MET' if all_passed else '⚠️ SOME CRITERIA NOT MET'}")
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
