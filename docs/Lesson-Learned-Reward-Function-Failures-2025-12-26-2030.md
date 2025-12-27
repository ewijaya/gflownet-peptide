# GFlowNet Reward Function Comparison: Lessons Learned

**Date**: 2025-12-26
**Runs Analyzed**: 4 experimental runs comparing different reward functions
**Author**: Analysis conducted via Claude Code

## Executive Summary

We conducted a systematic comparison of four reward function designs for GFlowNet-based peptide generation. The experiments revealed critical insights about reward function design, training dynamics, and checkpoint selection that are essential for successful GFlowNet training.

**Key Finding**: The choice of reward function is the single most important factor determining whether GFlowNet training succeeds or fails catastrophically.

| Run | Reward Type | Outcome | Recommendation |
|-----|-------------|---------|----------------|
| Baseline | Random MLP heads | ❌ Failed | Abandon |
| Option A | Trained stability | ⚠️ Partial | Revisit later |
| Option B | ESM2 pseudo-likelihood | ❌ Catastrophic failure | Abandon |
| **Option C** | Improved (entropy gate) | ✅ **Success** | **Use this** |

---

## 1. Experimental Setup

### 1.1 Background

The baseline GFlowNet run (`gflownet-baseline-10k`) showed concerning training dynamics:
- Loss exploding from 67 to 4100 after step 4000
- Mean reward stuck at ~0.44 throughout training
- log_Z growing unboundedly to 17.7

We hypothesized the issue was the reward function and designed three alternatives to test.

### 1.2 Reward Function Designs

#### Baseline: Composite Reward (Random MLP Heads)
```python
# From gflownet_peptide/models/reward_model.py
R(x) = stability^w1 × binding^w2 × naturalness^w3

# Where each component uses UNTRAINED randomly-initialized MLP heads:
stability = exp(random_mlp(esm_embedding))
binding = softplus(random_mlp(esm_embedding))
naturalness = sigmoid(random_mlp(esm_embedding))
```

#### Option A: Trained Stability Predictor
```python
# Uses StabilityPredictor trained on FLIP meltome data
R(x) = sigmoid(trained_stability(x)) × entropy_gate × naturalness^0.5
```

#### Option B: ESM-2 Pseudo-Likelihood
```python
# For each position, compute P(true_aa | context)
R(x) = normalize(mean(log P(aa_i | context)))
```

#### Option C: Improved Reward (Entropy Gate + Naturalness)
```python
# From gflownet_peptide/rewards/improved_reward.py
naturalness = sigmoid(embedding_norm / temperature)
entropy_gate = sigmoid(sharpness × (entropy - threshold))
length_gate = sigmoid(sharpness × (length - min_length))

R(x) = naturalness × entropy_gate × length_gate
```

### 1.3 Training Configuration

All runs used identical hyperparameters:
- Steps: 10,000
- Batch size: 64
- Learning rate: 3e-4
- log_Z LR multiplier: 10x
- Loss: Trajectory Balance
- ESM model: esm2_t6_8M_UR50D
- Sequence length: 10-30 amino acids

---

## 2. Results

### 2.1 Final Metrics Comparison

| Run | Loss | Mean Reward | Diversity | Unique Ratio | log_Z |
|-----|------|-------------|-----------|--------------|-------|
| Baseline | 4130 | 0.45 | 0.92 | 100% | 17.7 |
| Option A | 3771 | 0.37 | 0.92 | 100% | 18.9 |
| Option B | **0.00** | 0.96 | 1.00* | **0.2%** | -0.04 |
| **Option C** | 3378 | 0.94 | 0.90 | **100%** | 16.2 |

*Option B's "diversity" is misleading - see Section 3.2

### 2.2 Sample Quality Analysis

We sampled 100 sequences from each trained model:

#### Baseline (8rflp7l6)
```
Unique sequences: 100/100
AA diversity: 16/20
Sample: ALTIMPFQNTFETMPLMTDYMYFNQFFLAL
Issue: Reward signal is random noise - no meaningful learning
```

#### Option A - Trained Stability (6qsqq6wz)
```
Unique sequences: 100/100
AA diversity: 18/20 (best!)
Sample: LNKCRIALITTQPMTERSQKMAWCEMDLYE
Issue: Low rewards (~0.37) but sequences are actually good
```

#### Option B - ESM2-PLL (3fr3yzn0)
```
Unique sequences: 1/100 ❌
AA diversity: 1/20 ❌
Sample: MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM (all 100 samples!)
Issue: CATASTROPHIC MODE COLLAPSE
```

#### Option C - Improved (zcb95gyl)
```
Unique sequences: 100/100 ✅
AA diversity: 15/20 ✅
Sample: FYNPEIIESDTTLFSPFLPMYIDRTIIQEL
Result: Diverse, high-quality sequences
```

### 2.3 Reward Scores for Generated Sequences

| Source | Sequence Example | Entropy | Entropy Gate | Naturalness | Total Reward |
|--------|------------------|---------|--------------|-------------|--------------|
| Option B | `MMMMMMMMMM...` | 0.00 | 0.007 | 0.69 | **0.005** |
| Option C | `FYNPEIIESDTTLF...` | 0.82 | 0.959 | 0.65 | **0.619** |
| Natural (Insulin B) | `GIVEQCCTSICS...` | 0.77 | 0.939 | 0.65 | **0.609** |

The entropy gate correctly penalizes degenerate sequences by 100x.

---

## 3. Detailed Analysis

### 3.1 Baseline Failure: Random Reward Landscape

**Problem**: The CompositeReward uses randomly initialized MLP heads that were never trained. This creates a reward function that is essentially random noise.

**Training Dynamics**:
```
Step 0:    mean_reward = 0.4417
Step 4000: mean_reward = 0.4446  (+0.0029 over 4000 steps)
Step 9000: mean_reward = 0.4465  (+0.0048 total)
```

The reward barely changes because random MLPs assign similar scores to all sequences. The policy has no gradient signal to follow.

**Lesson**: Never use untrained reward heads. Either train them on real data or use a principled reward like ESM-2 embeddings.

### 3.2 Option B Failure: ESM-2 Pseudo-Likelihood Exploitation

**Problem**: ESM-2's masked language model is trivially exploitable by repetitive sequences.

**The Math**:
```python
# For poly-M sequence "MMMMMMMMMM..."
# At each position, context is all M's
# P(M | MMMM...MMM) ≈ 1.0 (trivially predictable)
# Mean log-prob ≈ -0.03 (nearly perfect)

# For natural sequence "GIVEQCCTSICS..."
# Each position has diverse context
# P(true_aa | context) varies
# Mean log-prob ≈ -0.61 (much worse!)
```

**Training Collapse Timeline**:
```
Step 0:   loss=848, mean_reward=0.51, unique_ratio=1.00
Step 500: loss=0.02, mean_reward=0.96, unique_ratio=0.015  ← COLLAPSE
Step 1000+: Everything flatlined - model only generates "MMM..."
```

The model discovered that poly-M gets reward 0.96 while diverse sequences get 0.51. It immediately collapsed to generating only poly-M.

**Lesson**: Pseudo-likelihood from masked language models is NOT suitable as a reward function. It rewards predictability, not quality.

### 3.3 Option C Success: The Entropy Gate Solution

**Key Innovation**: The entropy gate creates a multiplicative penalty for low-diversity sequences:

```python
entropy_gate = sigmoid(10 × (entropy - 0.5))

# For MMMMMMMM: entropy=0.0 → gate=0.007 (99.3% penalty)
# For FYNPEIIE: entropy=0.82 → gate=0.96 (4% penalty)
```

**Training Dynamics - Two Phases**:

**Phase 1 (Steps 0-6000): Exploration**
```
Step    Loss    Mean Reward   Diversity   Status
0       864     0.59          N/A         Starting
1000    46      0.03          0.31        Stuck in DL-trap
5000    32      0.06          0.49        Still exploring
6000    21      0.06          0.50        Minimum loss (but bad samples!)
```

The policy initially got stuck generating only D and L amino acids. Loss was low because the policy was confident in a small subspace.

**Phase 2 (Steps 6000-10000): Mode Discovery**
```
Step    Loss    Mean Reward   Diversity   Status
6500    502     0.37          N/A         Escaping trap
7000    1327    0.79          0.72        Finding good modes
9000    3139    0.95          0.90        High-quality samples
```

After step 6000, the policy escaped the DL-trap and found diverse, high-reward sequences. Loss exploded but sample quality improved dramatically.

### 3.4 The Loss Explosion Paradox

**Observation**: Option C's loss increased from 21 to 3378, yet sample quality improved from 2 amino acids to 15+ amino acids.

**Explanation**: The Trajectory Balance loss is:
```
L_TB = (log_Z + log_P_F - log_R)²
```

At step 6000 (DL-only, low loss):
- log_P_F = -21 (only 2 choices per position → high confidence)
- log_R = -2.8 (low reward due to entropy gate)
- log_Z = 13.4
- Imbalance = -4.6 → Loss = 21

At step 9000 (diverse, high loss):
- log_P_F = -72 (15+ choices → very confident in specific good sequences)
- log_R = -0.05 (high reward)
- log_Z = 16.1
- Imbalance = -56 → Loss = 3139

**The Core Issue**: log_Z is a global partition function estimate, but when the policy concentrates on high-reward modes, it becomes very confident (low log_P_F). log_Z can't increase fast enough to balance the equation.

**Lesson**: Loss is not a reliable quality metric for GFlowNet. Use sample-based metrics (diversity, reward, uniqueness) instead.

---

## 4. Critical Bug: Wrong Checkpoint Selection

### 4.1 The Problem

The trainer saves the "best" checkpoint based on lowest loss:

```python
# trainer.py line 382-385
if metrics["loss"] < self.best_loss:
    self.best_loss = metrics["loss"]
    save_checkpoint(best_path)
```

For Option C:
- **"Best" checkpoint** (step 5000, loss=21): Generates only D,L → reward 0.04
- **Final checkpoint** (step 10000, loss=3378): Generates 15+ AAs → reward 0.62

The "best" checkpoint is actually 15x worse!

### 4.2 Verification

```
Best Checkpoint (step 5000):
  Samples: DDLDDDLDDDDDDLLLDDDDLDDDLDLLLL
  AA diversity: 2/20
  Entropy gate: 0.06
  Total reward: 0.04

Final Checkpoint (step 10000):
  Samples: FYNPEIIESDTTLFSPFLPMYIDRTIIQEL
  AA diversity: 15/20
  Entropy gate: 0.96
  Total reward: 0.62
```

### 4.3 The Fix

Change checkpoint selection criterion from loss to reward:

```python
# trainer.py - BEFORE
if metrics["loss"] < self.best_loss:
    self.best_loss = metrics["loss"]
    save_checkpoint(best_path)

# trainer.py - AFTER
if metrics["mean_reward"] > self.best_reward:
    self.best_reward = metrics["mean_reward"]
    save_checkpoint(best_path)
```

---

## 5. Lessons Learned

### 5.1 Reward Function Design

1. **Never use untrained neural network heads as rewards**. Random weights provide no learning signal.

2. **ESM-2 pseudo-likelihood is exploitable**. Repetitive sequences achieve near-perfect likelihood because each position is trivially predictable from identical neighbors.

3. **Multiplicative gates are essential**. The entropy gate ensures ALL criteria must be satisfied:
   ```python
   R(x) = component1 × component2 × component3
   # If any component is near zero, total reward is near zero
   ```

4. **Embedding-based naturalness works**. Using `sigmoid(embedding_norm)` as a naturalness proxy is simple and effective.

### 5.2 Training Dynamics

5. **Low loss ≠ good samples**. The policy can achieve low TB loss by being confident in a degenerate subspace.

6. **Loss explosion can indicate success**. When the policy finds good modes and becomes confident, loss may increase due to log_Z tracking lag.

7. **Two-phase training is normal**. Expect:
   - Phase 1: Exploration, decreasing loss, low diversity
   - Phase 2: Mode discovery, increasing loss, high diversity

8. **log_Z learning rate matters**. The 10x multiplier may be insufficient when the policy sharpens rapidly. Consider:
   - Lower multiplier (3x) for stability
   - Sub-Trajectory Balance loss for better credit assignment

### 5.3 Evaluation

9. **Use sample-based metrics, not loss**:
   - Unique ratio (should be 100%)
   - Amino acid diversity (should be 15+/20)
   - Sequence diversity (should be >0.8)
   - Mean reward (should be 0.5-0.8, not >0.95)

10. **Beware of deceptive metrics**. Option B showed "diversity=1.0" in W&B but actually generated identical sequences. The metric was computed incorrectly.

### 5.4 Checkpoint Management

11. **Save based on reward, not loss**. The lowest-loss checkpoint may be the worst model.

12. **Always validate checkpoints by sampling**. Before using any checkpoint, generate samples and verify quality.

---

## 6. Recommendations

### 6.1 Immediate Actions

1. **Use Option C's final checkpoint**:
   ```bash
   cp checkpoints/gflownet/reward-comparison/improved/gflownet_final.pt \
      checkpoints/gflownet/best_model.pt
   ```

2. **Fix checkpoint selection in trainer.py** (see Section 4.3)

### 6.2 Future Training Runs

```bash
python scripts/train_gflownet.py \
    --reward_type improved \
    --n_steps 10000 \
    --loss_type sub_trajectory_balance \
    --log_z_lr_multiplier 3 \
    --eval_every 500 \
    --save_every 1000
```

### 6.3 Potential Improvements

1. **Combine Option C with Option A**:
   ```python
   R(x) = improved_reward(x) × stability(x)^0.3
   ```
   This adds biological relevance while maintaining diversity.

2. **Add entropy regularization**:
   ```python
   L_total = L_TB + 0.01 × mean(-log_P_F)
   ```
   This prevents the policy from becoming overconfident.

3. **Implement early stopping based on sample quality**, not loss.

---

## 7. Run Reference

| ID | Name | Reward Type | Final Samples | Verdict |
|----|------|-------------|---------------|---------|
| 8rflp7l6 | gflownet-baseline-10k | Random MLP | Diverse but meaningless | ❌ |
| 6qsqq6wz | gflownet-reward-A-trained-10k | Trained stability | Diverse, low reward | ⚠️ |
| 3fr3yzn0 | gflownet-reward-B-esm2pll-10k | ESM2-PLL | `MMMMMMMM...` | ❌ |
| zcb95gyl | gflownet-reward-C-improved-10k | Improved | Diverse, high-quality | ✅ |

**Winner**: Option C (Improved Reward with Entropy Gate)

---

## Appendix A: Trajectory Balance Loss Decomposition

For Option C at key steps:

| Step | log_Z | log_P_F | log_R | Required log_Z | Imbalance | Loss |
|------|-------|---------|-------|----------------|-----------|------|
| 0 | 0.00 | -27.91 | -0.94 | 26.97 | -26.97 | 727 |
| 6000 | 13.39 | -20.74 | -2.76 | 17.98 | -4.59 | 21 |
| 9000 | 16.14 | -72.05 | -0.05 | 72.00 | -55.86 | 3120 |

The required log_Z at step 9000 is 72, but actual log_Z is only 16 - a gap of 56 that causes the loss explosion.

## Appendix B: ESM-2 PLL Exploitation Proof

| Sequence | Entropy | ESM2-PLL Score | Improved Reward |
|----------|---------|----------------|-----------------|
| `MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM` | 0.00 | **-0.03** | 0.005 |
| `GIVEQCCTSICSLYQLENYCN` (Insulin) | 0.77 | -0.61 | **0.609** |

ESM2-PLL incorrectly ranks poly-M 20x higher than natural insulin. The Improved Reward correctly ranks insulin 120x higher than poly-M.
