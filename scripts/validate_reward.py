#!/usr/bin/env python3
"""Validate improved reward function on known good and bad examples.

This script tests that:
1. R(real_peptide) > R(repetitive) for all pairs
2. R(homopolymer) < 0.1 for all homopolymers
3. R(real_peptide) > 0.5 for all real peptides

Usage:
    python scripts/validate_reward.py
    python scripts/validate_reward.py --device cpu
"""

import argparse
import sys
from typing import List, Tuple


# Test cases: (name, sequence, category)
TEST_CASES = [
    # Real peptides (should score HIGH)
    ("Signal peptide", "MKTLLILAVVALACARSSAQAANPF", "real"),
    ("Antimicrobial", "GIGKFLHSAKKFGKAFVGEIMNS", "real"),
    ("Defensin-like", "GFGCPLNQGACHNHCRSIRRRGGYC", "real"),
    ("Insulin B-chain", "FVNQHLCGSHLVEALYLVCGERGFF", "real"),
    ("GLP-1 analog", "HAEGTFTSDVSSYLEGQAAKEFIAWL", "real"),

    # Repetitive homopolymers (should score LOW)
    ("Poly-Q", "QQQQQQQQQQQQQQQQQQQQQQQQQQ", "homopolymer"),
    ("Poly-N", "NNNNNNNNNNNNNNNNNNNNNNNNNN", "homopolymer"),
    ("Poly-G", "GGGGGGGGGGGGGGGGGGGGGGGGGG", "homopolymer"),
    ("Poly-A", "AAAAAAAAAAAAAAAAAAAAAAAAAA", "homopolymer"),
    ("Poly-S", "SSSSSSSSSSSSSSSSSSSSSSSSSS", "homopolymer"),

    # Repetitive patterns (should score LOW)
    ("Alternating QN", "QNQNQNQNQNQNQNQNQNQNQNQN", "pattern"),
    ("Alternating AQ", "AQAQAQAQAQAQAQAQAQAQAQAQ", "pattern"),
    ("Triplet repeat", "CAGCAGCAGCAGCAGCAGCAGCAG", "pattern"),

    # Edge cases (should score MODERATE)
    ("All different", "ACDEFGHIKLMNPQRSTVWY", "diverse"),
    ("Short real", "GIGKFLHSAK", "short_real"),
    ("Very short", "ACDE", "too_short"),
]


def run_validation(device: str = "cuda") -> Tuple[bool, List[dict]]:
    """Run validation tests on improved reward.

    Args:
        device: Device to run on (cuda or cpu)

    Returns:
        Tuple of (all_passed, results_list)
    """
    from gflownet_peptide.rewards.improved_reward import ImprovedReward

    print(f"Loading ImprovedReward on {device}...")
    reward = ImprovedReward(device=device, normalize=False)
    print("Model loaded.\n")

    results = []
    all_passed = True

    print("=" * 80)
    print("REWARD VALIDATION RESULTS")
    print("=" * 80)

    # Compute rewards for all test cases
    for name, sequence, category in TEST_CASES:
        components = reward.get_components(sequence)
        result = {
            "name": name,
            "sequence": sequence[:30] + "..." if len(sequence) > 30 else sequence,
            "category": category,
            **components,
        }
        results.append(result)

        print(f"\n{name} ({category})")
        print(f"  Sequence: {result['sequence']}")
        print(f"  Entropy:      {components['entropy']:.4f}")
        print(f"  Entropy gate: {components['entropy_gate']:.4f}")
        print(f"  Length gate:  {components['length_gate']:.4f}")
        print(f"  Naturalness:  {components['naturalness']:.4f}")
        print(f"  TOTAL:        {components['total']:.4f}")

    # Run validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    # Get rewards by category
    real_results = [r for r in results if r["category"] == "real"]
    homopolymer_results = [r for r in results if r["category"] == "homopolymer"]
    pattern_results = [r for r in results if r["category"] == "pattern"]
    repetitive_results = homopolymer_results + pattern_results

    # Check 1: R(real) > R(repetitive) for all pairs
    print("\nCheck 1: R(real_peptide) > R(repetitive) for all pairs")
    check1_passed = True
    check1_count = 0
    check1_total = 0

    for real in real_results:
        for rep in repetitive_results:
            check1_total += 1
            if real["total"] > rep["total"]:
                check1_count += 1
            else:
                check1_passed = False
                print(f"  FAIL: {real['name']} ({real['total']:.4f}) <= "
                      f"{rep['name']} ({rep['total']:.4f})")

    if check1_passed:
        print(f"  PASS: {check1_count}/{check1_total} pairs correct (100%)")
    else:
        print(f"  FAIL: {check1_count}/{check1_total} pairs correct "
              f"({100*check1_count/check1_total:.1f}%)")
        all_passed = False

    # Check 2: R(homopolymer) < 0.1
    print("\nCheck 2: R(homopolymer) < 0.1 for all homopolymers")
    check2_passed = True

    for r in homopolymer_results:
        if r["total"] < 0.1:
            print(f"  PASS: {r['name']} = {r['total']:.4f}")
        else:
            print(f"  FAIL: {r['name']} = {r['total']:.4f} (expected < 0.1)")
            check2_passed = False
            all_passed = False

    # Check 3: R(real_peptide) > 0.5
    print("\nCheck 3: R(real_peptide) > 0.5 for all real peptides")
    check3_passed = True

    for r in real_results:
        if r["total"] > 0.5:
            print(f"  PASS: {r['name']} = {r['total']:.4f}")
        else:
            print(f"  FAIL: {r['name']} = {r['total']:.4f} (expected > 0.5)")
            check3_passed = False
            all_passed = False

    # Check 4: Edge cases behave reasonably
    print("\nCheck 4: Edge cases")
    diverse_results = [r for r in results if r["category"] == "diverse"]
    for r in diverse_results:
        if 0.3 < r["total"] < 0.8:
            print(f"  PASS: {r['name']} = {r['total']:.4f} (moderate as expected)")
        else:
            print(f"  WARN: {r['name']} = {r['total']:.4f} (expected moderate)")

    too_short = [r for r in results if r["category"] == "too_short"]
    for r in too_short:
        if r["total"] < 0.3:
            print(f"  PASS: {r['name']} = {r['total']:.4f} (penalized as expected)")
        else:
            print(f"  WARN: {r['name']} = {r['total']:.4f} (expected low due to length)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    checks = [
        ("Check 1: R(real) > R(repetitive)", check1_passed),
        ("Check 2: R(homopolymer) < 0.1", check2_passed),
        ("Check 3: R(real) > 0.5", check3_passed),
    ]

    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {check_name}: {status}")

    if all_passed:
        print("\n✓ ALL VALIDATION CHECKS PASSED")
    else:
        print("\n✗ SOME VALIDATION CHECKS FAILED")

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(description="Validate improved reward function")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)"
    )
    args = parser.parse_args()

    passed, _ = run_validation(device=args.device)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
