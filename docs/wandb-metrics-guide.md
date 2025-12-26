# W&B Metrics Guide for GFlowNet Training

This guide explains how to interpret the key metrics logged during GFlowNet training.

---

## Top 3 Metrics (TL;DR)

### `train/` - Training Health

| Rank | Metric | Why |
|------|--------|-----|
| 1 | `train/loss` | Core objective - if this doesn't decrease, nothing works |
| 2 | `train/log_z` | Must stabilize - divergence means training failure |
| 3 | `train/unique_ratio` | Early warning for mode collapse |

### `eval/` - Model Quality

| Rank | Metric | Why |
|------|--------|-----|
| 1 | `eval/diversity` | **The whole point of GFlowNet** - must stay high |
| 2 | `eval/mean_reward` | Quality check - should increase while diversity stays high |
| 3 | `eval/max_reward` | Ceiling performance - shows best the model can do |

### The One Plot That Matters Most

If you could only look at **one thing**, watch:

```
eval/diversity  vs  eval/mean_reward
```

**Success**: Both high (diversity > 0.6, reward > 0.5)
**Failure**: High reward but low diversity (that's just GRPO, not GFlowNet)

This is what distinguishes GFlowNet from reward-maximizing methods - it maintains diversity while improving quality.

---

## Key Metrics to Monitor

### 1. `train/loss` (Trajectory Balance Loss)

**What it measures**: How well the flow equation is satisfied: `L = (log Z + Σlog P_F - log R - Σlog P_B)²`

**Ideal shape**:
- Starts high (100-1000+), drops rapidly in first 1000 steps
- Gradually decreases and plateaus near 0-5
- Should be **monotonically decreasing** with some noise

**Good**: Loss < 10 after 5K steps, < 5 after convergence
**Bad**: Diverging (→∞), oscillating wildly, stuck at high value

```
Good:                          Bad:
│█                             │    ██
│ █                            │   █  █
│  ██                          │  █    █
│    ███                       │ █      █
│       ████████               │█        ████
└────────────────►             └────────────────►
```

---

### 2. `train/log_z` (Log Partition Function)

**What it measures**: Learned estimate of log(Z) where Z = Σ R(x) over all sequences

**Ideal shape**:
- Starts at 0 (initialization)
- Gradually increases/decreases to find true value
- **Stabilizes** at some constant value

**Good**: Settles to a stable value (could be -10 to +10 depending on reward scale)
**Bad**: Diverging to ±∞, oscillating without settling

```
Good:                          Bad:
│       ████████               │              █
│     ██                       │            ██
│   ██                         │          ██
│ ██                           │        ██
│█                             │██████████
└────────────────►             └────────────────►
   (stabilizes)                   (diverging up)
```

---

### 3. `train/mean_reward`

**What it measures**: Average reward of sampled sequences in each batch

**Ideal shape**:
- Starts around 0.4-0.5 (random policy)
- Gradually increases as policy learns to sample better sequences
- Plateaus at higher value (0.6-0.8+)

**Good**: Increasing trend, higher than initial
**Bad**: Decreasing, stuck at initial value, or very low (<0.3)

```
Good:                          Bad:
│           ████████           │████████████████
│       ████                   │
│   ████                       │
│████                          │
│                              │
└────────────────►             └────────────────►
   (improving)                    (not learning)
```

---

### 4. `train/mean_log_pf` (Mean Log Forward Probability)

**What it measures**: Average log probability of actions taken by the policy

**Ideal shape**:
- Starts very negative (-30 to -50, random choices)
- Becomes less negative as policy becomes more confident
- Stabilizes at moderate negative value (-10 to -20)

**Good**: Increasing (less negative) but not to 0
**Bad**: Collapsing to 0 (mode collapse - always same sequence)

```
Good:                          Bad (mode collapse):
│           ████████           │               █
│       ████                   │              █
│   ████                       │            ██
│████                          │         ███
│-30                           │█████████
└────────────────►             └────────────────►
   (-30 → -15)                    (-30 → 0)
```

---

### 5. `train/grad_norm`

**What it measures**: Magnitude of gradients during training

**Ideal shape**:
- High initially (100-1000+)
- Decreases as training progresses
- Stabilizes at low value (1-10)

**Good**: Decreasing, bounded, no spikes to infinity
**Bad**: NaN, exploding (→∞), always at max clip value

---

### 6. `train/unique_ratio` (Diversity)

**What it measures**: Fraction of unique sequences in each batch

**Ideal shape**:
- Should stay **high** throughout training (0.8-1.0)
- Slight decrease is OK as policy becomes more focused

**Good**: > 0.5, ideally > 0.8
**Bad**: Collapsing to 0 (mode collapse - generating same sequence)

```
Good:                          Bad:
│████████████████              │████
│                              │    ████
│                              │        ████
│                              │            ████
│ ~0.9 stable                  │                ████ → 0
└────────────────►             └────────────────►
```

---

## Evaluation Metrics (`eval/`)

Evaluation metrics are computed periodically (every `eval_every` steps) on a fresh batch of samples. They provide a cleaner signal than training metrics since they're not affected by gradient updates.

### 7. `eval/mean_reward`

**What it measures**: Average reward of evaluation samples (not used for training)

**Why it matters**: More reliable quality indicator than `train/mean_reward` since it's computed on independent samples without exploration noise.

**Ideal shape**: Same as `train/mean_reward` - should increase over time

**Good**: Higher than `train/mean_reward` (policy is learning), trending upward
**Bad**: Lower than training reward (overfitting to training dynamics), flat

**Relationship to train/mean_reward**:
- `eval/mean_reward` ≈ `train/mean_reward`: Healthy
- `eval/mean_reward` >> `train/mean_reward`: Exploration is hurting training samples
- `eval/mean_reward` << `train/mean_reward`: Possible issue with evaluation

---

### 8. `eval/max_reward`

**What it measures**: Maximum reward among evaluation samples

**Why it matters**: Shows the best sequence the model can generate - indicates ceiling performance.

**Good**: Increasing, approaching theoretical maximum (1.0 for normalized rewards)
**Bad**: Stuck at low value, decreasing

---

### 9. `eval/min_reward`

**What it measures**: Minimum reward among evaluation samples

**Why it matters**: Shows the worst sequence being generated - indicates floor performance.

**Good**: Increasing over time (even bad samples are getting better)
**Bad**: Very low or zero (model still generating junk sequences)

---

### 10. `eval/diversity`

**What it measures**: Sequence diversity of evaluation samples (1 - mean pairwise sequence identity)

**Why it matters**: This is the **key metric for GFlowNet success** - we want high diversity while maintaining quality.

**Ideal shape**:
- Should stay **high** (0.7-0.9) throughout training
- Slight decrease OK as model focuses on high-reward regions

**Good**: > 0.6, stable or slowly decreasing
**Bad**: Collapsing toward 0 (mode collapse)

```
Good:                          Bad:
│████████████████              │████████
│                              │        ████
│  ~0.8 stable                 │            ████
│                              │                ████
│                              │                    ██ → 0
└────────────────►             └────────────────►
```

---

### 11. `eval/mean_length` / `eval/min_length` / `eval/max_length`

**What it measures**: Statistics on generated sequence lengths

**Why it matters**: Ensures model generates sequences in the valid range (10-30 AA typically)

**Good**:
- `mean_length` between min_length and max_length config values
- Reasonable spread (not all same length)

**Bad**:
- All sequences at max_length (model not learning to stop)
- All sequences at min_length (model stopping too early)

---

## Eval vs Train Metrics Comparison

| Aspect | `train/*` | `eval/*` |
|--------|-----------|----------|
| **Frequency** | Every step | Every `eval_every` steps |
| **Samples** | Used for gradient update | Fresh samples, no gradients |
| **Exploration** | May include exploration noise | Pure policy sampling |
| **Purpose** | Monitor training dynamics | Measure true performance |
| **Reliability** | Noisier | Cleaner signal |

**Key insight**: If `train/mean_reward` looks good but `eval/mean_reward` is bad, the model may be overfitting to training dynamics rather than learning the reward landscape.

---

## Eval Metrics Quick Reference

| Metric | Good Sign | Warning Sign | Target |
|--------|-----------|--------------|--------|
| `eval/mean_reward` | Increasing | Flat, decreasing | > 0.5 |
| `eval/max_reward` | Approaching 1.0 | Stuck low | > 0.7 |
| `eval/min_reward` | Increasing | Stuck at 0 | > 0.3 |
| `eval/diversity` | High, stable | Collapsing | > 0.6 |
| `eval/mean_length` | In valid range | At boundaries | 15-25 |

---

## Quick Reference Table

| Metric | Good Sign | Warning Sign | Target |
|--------|-----------|--------------|--------|
| `loss` | Decreasing, plateaus | Diverging, NaN | < 5 |
| `log_z` | Stabilizes | Diverging ±∞ | Stable value |
| `mean_reward` | Increasing | Stuck, decreasing | > 0.5 |
| `mean_log_pf` | Less negative | Collapsing to 0 | -20 to -10 |
| `grad_norm` | Decreasing | NaN, exploding | < 10 |
| `unique_ratio` | High (>0.8) | Collapsing to 0 | > 0.5 |

---

## What to Watch For

### Healthy Training
1. Loss drops rapidly then slowly decreases
2. log_Z settles to stable value
3. mean_reward trends upward
4. unique_ratio stays high

### Mode Collapse (Bad)
- `unique_ratio` → 0
- `mean_log_pf` → 0
- All sequences become identical

**Fix**: Increase `exploration_eps`, raise temperature

### Divergence (Bad)
- `loss` → ∞ or NaN
- `log_z` → ±∞
- `grad_norm` explodes

**Fix**: Reduce learning rate, increase gradient clipping

---

## Example: Interpreting Early Training

At step 100, if you see:
- `loss`: 847 → 9.4 ✅ (good rapid drop)
- `log_z`: 0.22 ✅ (starting to move)
- `mean_reward`: 0.46 ✅ (reasonable starting point)

This indicates healthy training - continue monitoring.

---

## W&B Dashboard Setup

Recommended panels to create:

### Top Priority (Pin These)
1. **Loss Panel**: `train/loss` (log scale Y-axis)
2. **Partition Function**: `train/log_z`
3. **Quality Panel**: `train/mean_reward`, `eval/mean_reward`
4. **Diversity Panel**: `train/unique_ratio`, `eval/diversity`

### Secondary (Training Health)
5. **Gradient Health**: `train/grad_norm`
6. **Policy Confidence**: `train/mean_log_pf`

### Evaluation Details
7. **Eval Quality**: `eval/mean_reward`, `eval/max_reward`, `eval/min_reward`
8. **Eval Diversity**: `eval/diversity`
9. **Sequence Lengths**: `eval/mean_length`, `eval/min_length`, `eval/max_length`

---

## References

- Bengio et al. (2021): Flow Network based Generative Models
- Malkin et al. (2022): Trajectory Balance: Improved Credit Assignment in GFlowNets
