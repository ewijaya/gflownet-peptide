# GFlowNet Papers: Training Loss Curves Survey

## Summary

**Key Finding**: Most GFlowNet papers do NOT plot raw training loss values (TB loss, FM loss, DB loss) over iterations. Instead, they primarily show evaluation metrics like L1 distance, mode discovery rates, and reward-based metrics.

---

## Papers in This Repository (`docs/papers/`)

### 1. Bengio2021_GFlowNet.pdf (NeurIPS 2021)
**"Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation"**

- **Shows actual loss curves**: No
- **What figures show**:
  - Figure 2: **L1 error** between true and sampled distributions over "states visited"
  - Figure 2: **Number of modes found** vs states visited
  - Figures 3-5: Empirical density of rewards, average reward of top-k over molecules visited
- **Loss type introduced**: Flow Matching (FM) loss
- **Note**: Original GFlowNet paper; focuses on sample quality comparisons vs MCMC and PPO

### 2. Malkin2022_TrajectoryBalance.pdf (NeurIPS 2022)
**"Trajectory Balance: Improved Credit Assignment in GFlowNets"**

- **Shows actual loss curves**: No
- **What figures show**:
  - Figure 2: **Empirical L1 error** vs states visited (comparing TB, DB, FM)
  - Figure 3: **Correlation** between log p(x) and log R(x)
  - Figure 3: **Tanimoto similarity** over molecules seen
  - Figure 4: **Spearman correlation** and **number of modes** over training steps
- **Loss type introduced**: Trajectory Balance (TB) loss
- **Note**: Introduces TB loss but evaluates via distribution matching metrics, not raw loss values

### 3. Jain2022_BiologicalSequenceDesign.pdf (ICML 2022)
**"Biological Sequence Design with GFlowNets"**

- **Shows actual loss curves**: No
- **What figures show**:
  - Figure 4: **TopK reward** over training steps (closest to a "training curve")
  - Tables: Performance/diversity/novelty metrics
  - Figure 3: Distribution of amino acids in generated sequences
- **Loss type used**: Trajectory Balance (TB) loss
- **Note**: Application paper; focuses on biological sequence quality metrics

---

## Additional Papers from Literature Search

### 4. "Learning GFlowNets from Partial Episodes" (Madan et al., ICML 2023)
**SubTB Paper** - [arXiv:2209.12782](https://arxiv.org/abs/2209.12782)

- **Shows actual loss curves**: No
- **What figures show**:
  - Figure 3: **L1 distance** between empirical and target distributions
  - Figure A.2: Effect of SubTB lambda parameter on L1 curves
- **Loss type introduced**: Sub-Trajectory Balance (SubTB) loss
- **Note**: Compares TB, DB, and SubTB but plots L1 error, not raw training loss

### 5. "Towards Understanding and Improving GFlowNet Training" (Shen et al., ICML 2023)
[arXiv:2305.07170](https://arxiv.org/abs/2305.07170)

- **Shows actual loss curves**: Partially
- **What figures show**:
  - Figure 1: Training curves showing **reward progression** over rounds
  - Convergence analysis comparing different methods
- **Loss type introduced**: Guided TB
- **Note**: Shows training progression but focuses on reward/quality metrics rather than raw loss values. This paper is the closest to showing actual training dynamics.

### 6. "GFlowNet Foundations" (Bengio et al., JMLR 2023)
[arXiv:2111.09266](https://arxiv.org/abs/2111.09266)

- **Shows actual loss curves**: No
- **What figures show**:
  - Theoretical diagrams: Flow network structure, DAG visualizations
  - No empirical training curves
- **Loss types discussed**: FM, DB, TB (theoretical foundations)
- **Note**: Primarily a theoretical/foundational paper

### 7. "Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks" (Hu et al., ICLR 2025)
[arXiv:2410.02596](https://arxiv.org/abs/2410.02596)

- **Shows actual loss curves**: No
- **What figures show**:
  - Figure 3: **L1 distance** between P_T and P_R across iterations
  - Figure 4: **Number of modes found** during training
  - Figure 5: Average reward and pairwise similarities
- **Loss types introduced**: Linex losses, Shifted-Cosh loss
- **Note**: Explores alternative loss functions but still evaluates via L1 distance

### 8. "Curriculum-Augmented GFlowNets for mRNA Sequence Generation" (2024)
[arXiv:2510.03811](https://arxiv.org/abs/2510.03811)

- **Shows actual loss curves**: Yes (partial)
- **What figures show**:
  - Figures 9(a)-9(c): Compare different loss functions
  - Training stability analysis
- **Note**: Found that SubTB loss is critical for numerical stability with sequences >55 amino acids; standard TB loss failed

---

## Summary Table

| Paper | Year | Shows Raw Training Loss? | Primary Metrics Shown |
|-------|------|-------------------------|----------------------|
| Original GFlowNet (Bengio) | 2021 | No | L1 error, modes found, rewards |
| Trajectory Balance (Malkin) | 2022 | No | L1 error, correlation, Tanimoto |
| Biological Sequence Design (Jain) | 2022 | No | TopK reward, diversity, novelty |
| GFlowNet Foundations | 2023 | No | Theoretical diagrams only |
| SubTB Paper (Madan) | 2023 | No | L1 distance to target |
| Improving Training (Shen) | 2023 | Partial | Reward curves, convergence |
| Beyond Squared Error (Hu) | 2025 | No | L1 distance, modes, rewards |
| Curriculum GFlowNets | 2024 | Partial | Loss comparison for stability |

---

## Why Raw Loss Curves Are Rarely Shown

The GFlowNet community evaluates success through **distribution matching metrics** rather than raw loss values because:

1. **Interpretability**: The loss value itself (e.g., TB loss = `(log Z + Σ log P_F - log R - Σ log P_B)²`) is less interpretable than how well the learned distribution matches the target

2. **True Objective**: GFlowNet success is measured by `P(x) ∝ R(x)`, not by minimizing loss. A lower loss doesn't always mean better sampling

3. **Scale Variability**: Loss magnitudes vary significantly across tasks and can be very large (TB loss can range to 10^6 in some settings)

4. **Standard Practice**: The community has converged on:
   - **L1 distance** between learned and target distributions
   - **Total Variation (TV)** distance
   - **Number of modes discovered**
   - **Sample reward/quality metrics**

---

## Implications for Our Project

When monitoring training in `gflownet-peptide`:

1. **Don't rely solely on TB loss values** - they may not directly indicate sampling quality
2. **Track functional metrics**:
   - Mean reward of generated samples
   - Diversity metrics (sequence identity, embedding diversity)
   - Mode coverage (if ground truth is known)
3. **Use loss for debugging**: Large loss spikes or NaN values indicate training instability
4. **Consider SubTB for long sequences**: Standard TB may have numerical issues for sequences >50 amino acids

---

## References

- [Original GFlowNet Paper](https://arxiv.org/abs/2106.04399)
- [Trajectory Balance Paper](https://arxiv.org/abs/2201.13259)
- [GFlowNet Foundations](https://arxiv.org/abs/2111.09266)
- [SubTB Paper](https://arxiv.org/abs/2209.12782)
- [Towards Understanding and Improving GFlowNet Training](https://arxiv.org/abs/2305.07170)
- [Beyond Squared Error](https://arxiv.org/abs/2410.02596)
- [torchgfn Library](https://github.com/GFNOrg/torchgfn)
