# W&B Metrics Guide for GFlowNet Training

This guide explains how to interpret the key metrics logged during GFlowNet training.

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

1. **Loss Panel**: `train/loss` (log scale Y-axis)
2. **Partition Function**: `train/log_z`
3. **Quality Panel**: `train/mean_reward`, `train/max_reward`
4. **Diversity Panel**: `train/unique_ratio`
5. **Training Health**: `train/grad_norm`, `train/mean_log_pf`

---

## References

- Bengio et al. (2021): Flow Network based Generative Models
- Malkin et al. (2022): Trajectory Balance: Improved Credit Assignment in GFlowNets
