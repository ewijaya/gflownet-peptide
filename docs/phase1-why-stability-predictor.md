# Phase 1: Why Use a Stability Predictor Instead of Real Experimental Values?

## The Core Problem: We Need to Score NEW Sequences

During GFlowNet training, we generate **millions of novel peptide sequences** that have never been synthesized in a lab. For each one, we need a reward score.

| Approach | Can Score Novel Sequences? | Speed | Cost |
|----------|---------------------------|-------|------|
| Real experiments | No - need to synthesize each peptide | Weeks per peptide | $$$$ |
| Public data lookup | No - only has known sequences | N/A | Free |
| **Predictor** | Yes - generalizes to new sequences | ~10ms per peptide | Free |

## Analogy

Imagine you're training an AI to design new cars:
- **Real experiments** = Build each car design and crash-test it (impossible at scale)
- **Public data lookup** = Only score cars that already exist (can't score new designs)
- **Predictor** = Train a physics simulator that predicts crash safety for ANY design

## The Training Loop Requires Fast Scoring

```
GFlowNet Training Loop (millions of iterations):
1. Generate a new peptide sequence (e.g., "MKTLLILAVV...")
2. Score it with reward function  ← NEEDS TO BE FAST
3. Update the model
4. Repeat
```

If step 2 required a lab experiment, training would take centuries.

## What the Predictor Actually Does

```
Training Data (FLIP):           Predictor:
Known sequences → Stability  →  Learns the pattern
"ACDEF..." → 0.8
"GHIKL..." → 0.3               New sequence → Predicted stability
"MNPQR..." → 0.9               "XYZAB..." → 0.7 (estimated)
```

The predictor learns **what makes a peptide stable** from real experimental data, then applies that knowledge to score novel sequences.

## Why R² = 0.65 is Good Enough

- We don't need perfect predictions
- We just need the predictor to **rank** peptides correctly (stable > unstable)
- GFlowNet will sample proportionally to predicted reward
- Top candidates go to real experiments for validation

## Summary

| Question | Answer |
|----------|--------|
| Why not real experiments? | Too slow and expensive for millions of sequences |
| Why not lookup tables? | Novel sequences aren't in any database |
| Why predictor works? | Learns patterns from real data, generalizes to new sequences |

The predictor is a **proxy** for real stability - good enough to guide generation, with real experiments reserved for final validation of top candidates.
