# ESM-2 Pseudo-Likelihood Reward: How It Works

**Created**: 2025-12-24
**Purpose**: Explain how ESM-2 is used as the reward function in GRPO-D training

---

## Overview

The reward function used in `scripts/train_grpo.py` is based on **ESM-2 pseudo-likelihood scoring**. This document explains the mechanism and how it connects to the W&B metrics.

For the full mathematical formulation, see: [docs/reward_formulation.md](reward_formulation.md), specifically:
- Section 2: ESM-2 Pseudo-Likelihood Reward
- Section 2.2: Exact Pseudo-Likelihood Formulation
- Section 2.3: Fast Approximation
- Section 2.5: Normalization

---

## The Core Idea

ESM-2 is a **masked language model** trained on millions of protein sequences. The reward measures "how natural does this peptide look?" according to ESM-2.

### The Math

For a peptide `x = (x₁, x₂, ..., xₗ)` of length `L`:

```
R_PLL(x) = (1/L) Σᵢ log P_ESM(xᵢ | x_masked_at_i)
```

**In plain English:**
1. For each position `i`, mask it with `[MASK]`
2. Ask ESM-2: "What amino acid should be here?"
3. Get the probability ESM-2 assigns to the **true** amino acid
4. Average the log-probabilities across all positions

Higher score = ESM-2 thinks the sequence is more "protein-like"

---

## Implementation

### Source Files

| Component | File | Key Lines |
|-----------|------|-----------|
| ESM-2 Reward Class | `gflownet_peptide/rewards/esm2_reward.py` | 55-221 |
| Training Script | `scripts/train_grpo.py` | 226-231 |
| Trainer Integration | `gflownet_peptide/training/grpo_trainer.py` | 424, 534-535 |

### Exact Method (slow but accurate)

From `esm2_reward.py:97-134`:

```python
def _compute_pseudo_likelihood(self, sequence: str) -> float:
    for pos in range(seq_len):
        # Mask position
        masked_tokens[0, token_pos] = self.mask_idx

        # Get ESM-2 prediction
        results = self.model(masked_tokens)
        log_probs = F.log_softmax(logits[0, token_pos], dim=-1)

        # Score = probability of the TRUE amino acid
        total_log_prob += log_probs[true_token].item()

    return total_log_prob / seq_len
```

### Fast Approximation (used in training)

From `esm2_reward.py:136-167`:
- Single forward pass without masking
- Uses ESM-2's bidirectional context as approximation
- Much faster: O(1) forward pass vs O(L) for exact method

---

## The Complete Reward Flow

```
train_grpo.py:226-231
┌─────────────────────────────────────────────────┐
│ reward_fn = ESM2Reward(                         │
│     model_name="esm2_t6_8M_UR50D",              │
│     normalize=True,    ← Maps to [0, 1]         │
│     temperature=1.0                             │
│ )                                               │
└─────────────────────────────────────────────────┘
                    │
                    ▼
grpo_trainer.py:424
┌─────────────────────────────────────────────────┐
│ reward = self.reward_fn(peptide)                │
│                                                 │
│ This calls ESM2Reward.__call__() which:         │
│ 1. Computes R_PLL = mean log P(aa_i | context)  │
│ 2. Normalizes to [0, 1] using min-max           │
└─────────────────────────────────────────────────┘
                    │
                    ▼
grpo_trainer.py:534-535
┌─────────────────────────────────────────────────┐
│ self.training_stats["mean_reward"].append(      │
│     np.mean(rewards)    ← This goes to W&B      │
│ )                                               │
│ self.training_stats["max_reward"].append(       │
│     np.max(rewards)     ← This goes to W&B      │
│ )                                               │
└─────────────────────────────────────────────────┘
```

---

## W&B Metrics Explained

| Metric | What it is |
|--------|------------|
| `mean_reward` | Average normalized ESM-2 pseudo-likelihood across batch |
| `max_reward` | Highest normalized ESM-2 pseudo-likelihood in batch |

**Normalization formula** (from `esm2_reward.py:206-213`):

```
R_norm = (raw_score - min_seen) / (max_seen - min_seen)
```

This maps rewards to `[0, 1]`, so values like 0.93, 0.82 are normalized pseudo-likelihood scores.

---

## Why This Causes Reward Hacking

The fundamental problem: **ESM-2 gives HIGH scores to repetitive sequences**.

### Example

| Sequence | Score | Why |
|----------|-------|-----|
| `QQQQQQQQQQQQQQQQQQ` | ~0.95 | Each Q is highly predictable given context of other Qs |
| `MKTLLILAVVALACAG` | ~0.70 | Realistic signal peptide but less predictable |

### Explanation

ESM-2 learned that **context predicts the next residue**. Repetitive patterns are trivially predictable:

```
P(Q | QQQQQQQQ[MASK]QQQQQQQQ) ≈ 0.99  → high log-prob
```

So the pseudo-likelihood score rewards degenerate sequences that exploit this predictability.

### Evidence from Phase 0 Training

From `results/grpo/*_peptides.csv`, top sequences show clear reward hacking:

```
MRQQQQQQQQQQQQQQQQNNNNNNNNNNNN  R=0.932
MPGNNNNNNNNQQQQQQQQQQQQQQQQQQQ  R=0.929
MRSSSSSSSSSSSSSSSSDDDDDDDDEEEE  R=0.926
```

97% of generated sequences contain repeats of 3+ identical amino acids.

---

## Related Documentation

| Document | Relevant Sections |
|----------|-------------------|
| [reward_formulation.md](reward_formulation.md) | Section 2 (full mathematical derivation) |
| [reward_formulation.md](reward_formulation.md) | Section 4 (diversity-augmented reward) |
| [reward_formulation.md](reward_formulation.md) | Section 5 (integration with GRPO) |
| [phase0_decision.md](phase0_decision.md) | Evidence of reward hacking |
| [prd-phase-0-validation.md](prd-phase-0-validation.md) | Training configuration |

---

## Implications for GFlowNet

GFlowNet solves the reward hacking problem by sampling **proportionally** to reward rather than maximizing:

```
P_GFlowNet(x) ∝ R(x)^β
```

This means:
- High-reward sequences are sampled more often
- But moderate-reward diverse sequences **still get sampled**
- No collapse to degenerate solutions

This is why Phase 0 concluded with a **GO decision** for GFlowNet development despite GRPO-D achieving high rewards.

---

*End of document*
