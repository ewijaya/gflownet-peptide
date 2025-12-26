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

### 1. We Need Ranking, Not Exact Values

The predictor doesn't need to perfectly predict stability values. It just needs to correctly **rank** peptides:

```
Good enough:                    Not needed:
"Peptide A > Peptide B"         "Peptide A = 0.823 exactly"
```

GFlowNet samples **proportionally** to reward - so if the predictor correctly identifies that A is better than B, GFlowNet will sample A more often. The exact value matters less.

### 2. It's a Filter, Not the Final Answer

```
Pipeline:
                                         Real experiments
Millions of    Predictor     Top 100     (ground truth)      Final
sequences  →   scores    →   candidates  →    validation  →  winners
               (R²=0.65)     (cheap)          (expensive)
```

The predictor is a **cheap filter** to narrow down candidates. The top candidates still go through real experimental validation. We're not trusting the predictor blindly - we're using it to avoid wasting lab resources on obviously bad sequences.

### 3. R² = 0.65 Means Strong Correlation

| R² Value | Interpretation |
|----------|----------------|
| 0.0 | Random guess |
| 0.5 | Moderate correlation |
| **0.65** | **Strong correlation - ranking is reliable** |
| 1.0 | Perfect prediction |

With R² = 0.65, the predictor explains 65% of the variance in stability. That's enough to reliably separate "stable" from "unstable" peptides.

### Bottom Line

> R² = 0.65 is enough because GFlowNet only needs **relative ranking** to explore the right regions of peptide space. The predictor acts as a **cheap filter**, and real experiments validate the final candidates.

## Summary

| Question | Answer |
|----------|--------|
| Why not real experiments? | Too slow and expensive for millions of sequences |
| Why not lookup tables? | Novel sequences aren't in any database |
| Why predictor works? | Learns patterns from real data, generalizes to new sequences |

The predictor is a **proxy** for real stability - good enough to guide generation, with real experiments reserved for final validation of top candidates.

---

## Technical Details

### Regression, Not Classification

The predictor is a **regression** model, not binary classification:

| Type | Output | Example |
|------|--------|---------|
| Binary Classification | 0 or 1 (stable/unstable) | "This peptide is stable" |
| **Regression** (what we use) | Continuous value | "This peptide has stability = 0.73" |

Why regression is better for GFlowNet:

```
Binary (bad for GFlowNet):
Peptide A → "stable"     (1)
Peptide B → "stable"     (1)
→ GFlowNet treats A and B equally

Regression (good for GFlowNet):
Peptide A → 0.95  (very stable)
Peptide B → 0.60  (somewhat stable)
→ GFlowNet samples A more than B
```

### What is "Stability" Exactly?

The stability score is based on **melting point (Tm)** from the FLIP database:

| Higher Melting Point | Lower Melting Point |
|---------------------|---------------------|
| More stable | Less stable |
| Needs more heat to unfold | Unfolds at lower temperature |
| Protein holds its shape longer | Protein denatures easily |

**Analogy:** Think of ice - ice that melts at higher temperature is "more stable."

For proteins:
- **Tm = 37°C** → Unfolds at body temperature (unstable, bad)
- **Tm = 65°C** → Needs much more heat to unfold (stable, good)

### The FLIP Data Source

```
FLIP Database (Real Experiments)
/data/flip/stability/

Files:
├── full_dataset.json      # Raw experimental data (221K entries)
│   └── Contains: meltingPoint, meltingBehaviour curves
│
├── stability.csv          # Simplified version (what we use)
│   └── Contains: sequence, target (meltingPoint)
│
└── Processed splits:
    /data/processed/flip_stability/
    ├── train.csv  (11,367 sequences)
    ├── val.csv    (1,263 sequences)
    └── test.csv   (2,006 sequences)
```

### Multi-Species Training Data

The FLIP stability dataset contains proteins from **multiple species and cell types** (221K total entries):

| Source | Percentage | Examples |
|--------|------------|----------|
| Human cell lines | ~78% | HepG2, Jurkat, K562, HEK293T, HL60, U937 |
| Mouse | ~5% | BMDC lysate, liver lysate |
| Other model organisms | ~5% | *C. elegans*, *Arabidopsis*, *Drosophila*, zebrafish |
| Bacteria | ~5% | *E. coli*, *Bacillus*, *Thermus thermophilus* |
| Yeast | ~1% | *S. cerevisiae* |

**Why multi-species training is beneficial:**

1. **Physics is universal**: Protein stability depends on amino acid sequence → 3D structure → thermodynamic properties. The same physical principles (hydrogen bonds, hydrophobic packing, etc.) apply across all life.

2. **More data = better generalization**: Training on diverse species helps the predictor learn fundamental stability patterns rather than species-specific quirks.

3. **Standard practice**: The original FLIP paper uses all species. Most protein ML papers train on cross-species data (ESM-2 was trained on all of UniRef across all domains of life).

4. **For therapeutic peptides**: We want general stability principles, not species-specific ones. A peptide designed for humans should be stable based on chemistry, not because it copies a particular species' sequence patterns.

### Predictor Output: Normalized Melting Point

The predictor outputs a **normalized** stability score:

```
Raw FLIP data:
  meltingPoint: 37.96, 54.42, 52.89 (in °C)

After normalization (z-score):
  target_normalized: -0.52, 0.02, -0.05 (centered around 0)

Predictor output:
  0.637  →  Higher than average stability
 -0.52   →  Lower than average stability
  0.0    →  Average stability
```

Higher score = higher melting point = more stable = better therapeutic candidate.

---

## The R² = 0.65 Metric Explained

### Two R² Values from Training

| Metric | Value | Meaning |
|--------|-------|---------|
| `best_val_r2` | 0.616 | Best R² on validation set (at epoch 16) |
| `test_r2` | **0.651** | R² on test set using the best model |

### R² is Correlation Between Real and Predicted

```
R² = 0.65 means:

Correlation between:
├── Real stability (from FLIP experiments)
└── Predicted stability (our model's output)

Measured on 2,006 TEST sequences the model
has NEVER seen during training.
```

---

## Model Checkpoint Details

### What's Saved in `stability_predictor_best.pt`

The checkpoint contains **ONLY the best epoch's weights** (epoch 16), not all epochs:

```
stability_predictor_best.pt
└── Only epoch 16 model weights (the best one)

NOT this:
├── epoch 1 weights
├── epoch 2 weights
├── ...
└── epoch 30 weights
```

### How the Saving Works

```python
# During training:
best_val_r2 = -inf

for epoch in range(30):
    train(...)
    val_r2 = validate(...)

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        # OVERWRITE the file with current weights
        torch.save(model.state_dict(), "stability_predictor_best.pt")

# Result: File contains only the best epoch (16)
```

### Training Timeline

```
Epoch 1:  val R² = 0.546 → Save (first best)
Epoch 2:  val R² = 0.566 → Save (new best, overwrites)
...
Epoch 16: val R² = 0.616 → Save (new best) ← THIS IS KEPT
Epoch 17: val R² = 0.605 → No save (not better)
...
Epoch 30: val R² = 0.562 → No save (overfitting)

Final: stability_predictor_best.pt = epoch 16 weights only
```

### Why Test R² (0.651) > Validation R² (0.616)?

This is normal - test and validation sets are different random samples. The test set might be slightly "easier" by chance. Both values indicate strong performance.

### Using the Model in Production

```python
# For downstream phases (ablation with CompositeReward):
model.load_state_dict(torch.load("stability_predictor_best.pt"))
# That's it! Already the best model, no epoch selection needed.
```

The checkpoint is already "production ready" - the best model was saved automatically during training.
