# Phase 0 Go/No-Go Decision Document

**Date**: 2025-12-24
**Decision**: GO

---

## Summary

Diversity problem confirmed. GRPO-D shows mode collapse.

## Metrics Comparison

| Metric | GRPO-D | Random Baseline |
|--------|--------|----------------|
| Sample count | 128 | 128 |
| Mean reward | 0.8164 | 0.8284 |
| Sequence diversity | 0.8951 | 0.8651 |
| Embedding diversity | 0.3363 | 0.0943 |
| **Cluster count** | **3** | **2** |

## Criteria Evaluation

1. **Cluster count < 10**: PASS (observed: 3)
2. **Mode coverage gap > 30%**: FAIL (observed: -50.0%)
3. **Reward hacking observed**: PASS (repetitive patterns in top peptides)

## Observations

1. GRPO-D achieves high rewards (0.816) but through repetitive patterns
2. Top peptides show clear mode collapse (e.g., QQQQ..., NNNN..., GGGG... blocks)
3. ESM-2 pseudo-likelihood rewards these degenerate sequences
4. Random baseline achieves lower rewards but higher diversity

## Recommendation

Proceed to Phase 1 (Reward Model Development) to develop GFlowNet with intrinsic diversity.

## Artifacts

- Visualization: `outputs/grpo_umap_clusters.png`
- Comparison: `outputs/comparison_plot.png`
- Metrics: `outputs/grpo_metrics.json`, `outputs/random_metrics.json`
- Embeddings: `outputs/grpo_embeddings.npy`
