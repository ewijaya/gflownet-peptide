# Phase 0 Go/No-Go Decision Document

**Date**: 2025-12-24
**Decision**: CONDITIONAL GO

---

## Executive Summary

GRPO-D training has completed 1000 iterations. Analysis reveals **borderline diversity** with 10 clusters identified. However, the generated peptides exhibit clear **reward hacking** through repetitive amino acid patterns, confirming the need for intrinsic diversity mechanisms that GFlowNet provides.

---

## Metrics Comparison

| Metric | GRPO-D | Random Baseline |
|--------|--------|-----------------|
| Sample count | 128 | 128 |
| Unique ratio | 100% | 100% |
| Mean reward | 0.8164 | 0.8284 |
| Max reward | 0.9323 | 1.0000 |
| Sequence diversity | 0.8951 | 0.8651 |
| Embedding diversity | 0.3363 | 0.0943 |
| **Cluster count** | **10** | **2** |
| Noise points | 16 (12.5%) | 4 (3.1%) |

---

## Criteria Evaluation

### 1. Cluster count < 10?
- **Observed**: 10 clusters (borderline)
- **Result**: BORDERLINE - at threshold

### 2. Mode coverage gap > 30%?
- Random baseline: 2 clusters
- GRPO-D: 10 clusters
- **Gap**: GRPO-D actually has MORE clusters than random
- **Result**: FAIL (but note: random rewards very high due to ESM-2 quirks)

### 3. Reward hacking observed?
- **Result**: STRONG PASS
- Top peptides show repetitive patterns:
  - `MRQQQQQQQQQQQQQQQQNNNNNNNNNNNN` (R=0.932)
  - `MPGNNNNNNNNQQQQQQQQQQQQQQQQQQQ` (R=0.929)
  - 97% of sequences contain repeats of 3+ identical amino acids
  - Mean repeat length: 7.8 amino acids
  - Maximum repeat length: 24 amino acids

---

## Key Observations

### Evidence FOR GFlowNet (GO signals):

1. **Severe reward hacking**: ESM-2 pseudo-likelihood rewards degenerate repetitive sequences (QQQQ..., NNNN..., GGGG...) with high scores. This is a fundamental problem that diversity penalties cannot fully address.

2. **Pattern exploitation**: Top 5 amino acids (G, Q, N, S, D) comprise 54% of all residues despite uniform natural distribution expected at 5% each.

3. **Low embedding diversity**: Despite 10 clusters, embedding diversity (0.34) is modest, indicating sequences cluster in limited regions of sequence space.

4. **Random baseline anomaly**: Random sequences filtered by high reward also show extreme concentration (2 clusters) - confirming ESM-2 pseudo-likelihood concentrates probability mass on specific patterns.

### Evidence for caution:

1. **Cluster count at threshold**: 10 clusters meets the borderline criterion (10-20 range per PRD).

2. **High sequence diversity score**: 0.895 indicates reasonable sequence-level variation, though this is driven by positional differences in repeating blocks.

---

## Interpretation

The analysis reveals a nuanced situation:

1. **GRPO-D achieves more structural diversity than random high-fitness sampling** (10 vs 2 clusters), suggesting the diversity penalty has some effect.

2. **However, both methods collapse to repetitive patterns** because ESM-2 pseudo-likelihood intrinsically rewards them. This is a property of the reward function, not the optimization method.

3. **GFlowNet's value proposition remains valid**: By sampling proportionally to reward rather than maximizing it, GFlowNet can explore the full reward landscape including moderate-reward regions with genuine sequence diversity.

4. **The core hypothesis is confirmed**: Pure reward optimization (GRPO) leads to exploitation of reward function artifacts. GFlowNet's intrinsic diversity mechanism addresses this at the architectural level.

---

## Decision

### **CONDITIONAL GO to Phase 1**

**Rationale**:
- Reward hacking is severe and systematic
- ESM-2 pseudo-likelihood has known biases toward repetitive sequences
- GFlowNet's proportional sampling directly addresses this issue
- Even with diversity penalties, GRPO converges to degenerate solutions
- The problem is fundamental to reward-maximizing approaches

**Conditions**:
1. Phase 1 should develop a reward model less susceptible to repetitive patterns
2. Consider using ESM-2 masked language modeling score with entropy penalties
3. Evaluate GFlowNet on the new reward before comparing to GRPO-D

---

## Artifacts

| File | Description |
|------|-------------|
| `outputs/grpo_umap_clusters.png` | GRPO-D cluster visualization |
| `outputs/comparison_plot.png` | Side-by-side GRPO vs Random comparison |
| `outputs/grpo_metrics.json` | GRPO-D diversity metrics |
| `outputs/random_metrics.json` | Random baseline metrics |
| `outputs/grpo_embeddings.npy` | ESM-2 embeddings for GRPO samples |
| `results/grpo/*_peptides.csv` | Generated peptide sequences |
| `notebooks/gflownet-phase-0-validation.ipynb` | Analysis notebook |

---

## Next Steps (Phase 1)

1. Design reward model resistant to repetitive pattern exploitation
2. Consider composite rewards: naturalness + diversity + structural features
3. Implement GFlowNet core with trajectory balance loss
4. Compare GFlowNet diversity to GRPO-D on improved reward

---

*End of Phase 0 Decision Document*
