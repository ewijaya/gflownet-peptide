# Reward Options A/B/C vs GRPO Phase 0/0b: Are They Redundant?

**Date**: 2025-12-26

---

## Question

Option B (`esm2_pll`) and Option C (`improved`) use the same reward functions as GRPO Phase 0 and Phase 0b. Are they redundant?

## Answer: No, They Serve Different Purposes

| Aspect | GRPO Phase 0/0b | GFlowNet A/B/C |
|--------|-----------------|----------------|
| **Algorithm** | GRPO (reward maximization) | GFlowNet (proportional sampling) |
| **Goal** | Find best sequences | Sample diverse high-reward sequences |
| **Outcome** | Validated reward functions work | Validate GFlowNet learns with these rewards |

### What GRPO Phase 0/0b Proved

- ESM-2 PLL can be hacked (Phase 0)
- Entropy gate fixes the hacking (Phase 0b)
- The reward functions themselves are sound

### What GFlowNet A/B/C Proves

- GFlowNet can actually learn from discriminative rewards (vs flat baseline)
- The reward signal is strong enough for TB loss convergence
- GFlowNet maintains diversity while optimizing reward

### The Real Comparison for the Paper

- GFlowNet + ImprovedReward (C) vs GRPO + ImprovedReward
- Same reward, different algorithms → isolates the algorithm contribution

So B/C in GFlowNet isn't re-doing the reward validation - it's validating that **GFlowNet can learn** with rewards we already know work. The baseline failure (flat 0.445 rewards) showed this isn't guaranteed.

---

## Option Summary

| Option | `--reward_type` | GRPO Equivalent | Notes |
|--------|-----------------|-----------------|-------|
| **C** | `improved` | Phase 0b | Entropy gate + naturalness, anti-hacking |
| **B** | `esm2_pll` | Phase 0 | Pure ESM-2 PLL, vulnerable to hacking |
| **A** | `trained` | Phase 1 | Trained stability predictor, publication-ready |

---

## Compute Optimization (If Needed)

If you want to save compute, you could **skip Option B** (`esm2_pll`) since:

1. We know it's hackable
2. Option C (`improved`) is strictly better
3. Option A (`trained`) is the publication-ready one

But running all three gives a complete comparison for the paper's appendix.

---

## Implementation: Factorized Code

The code is factorized - GFlowNet's B/C options directly import and use the same reward classes that were implemented for GRPO Phase 0/0b. No code duplication.

| Training Script | Reward Type | Import | Line |
|-----------------|-------------|--------|------|
| `scripts/train_grpo.py` | `esm2_pll` | `ESM2Reward` | 242 |
| `scripts/train_grpo.py` | `improved` | `ImprovedReward` | 251 |
| `scripts/train_gflownet.py` | `esm2_pll` | `ESM2Reward` | 145 |
| `scripts/train_gflownet.py` | `improved` | `ImprovedReward` | 133 |

**GRPO** (`scripts/train_grpo.py`):

```python
# Line 242 - Phase 0
from gflownet_peptide.rewards.esm2_reward import ESM2Reward
reward_fn = ESM2Reward(...)

# Line 251 - Phase 0b
from gflownet_peptide.rewards.improved_reward import ImprovedReward
reward_fn = ImprovedReward(...)
```

**GFlowNet** (`scripts/train_gflownet.py`):

```python
# Line 145 - Option B
from gflownet_peptide.rewards.esm2_reward import ESM2Reward
return ESM2Reward(...)

# Line 133 - Option C
from gflownet_peptide.rewards.improved_reward import ImprovedReward
return ImprovedReward(...)
```

Both `ImprovedReward` and `ESM2Reward` live in `gflownet_peptide/rewards/` and are shared between GRPO and GFlowNet training pipelines. Same classes, same module paths.

This factorization enables the fair comparison for the paper: **same reward implementation, different optimization algorithm**.

---

## Conclusion

Running all three options (A, B, C) is valuable because:

1. **Validates GFlowNet learning** - proves the algorithm works with discriminative rewards
2. **Complete comparison** - useful for paper's appendix/supplementary material
3. **Isolates contributions** - same reward functions as GRPO enables fair algorithm comparison
