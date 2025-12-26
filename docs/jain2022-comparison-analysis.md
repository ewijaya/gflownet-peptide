# Jain et al. 2022 vs Our Project: Comparison Analysis

## Summary: Jain et al. 2022 "Biological Sequence Design with GFlowNets"

### What They Did

1. **GFlowNet-AL**: Active learning framework with GFlowNet as diverse candidate generator
2. **Tasks**: AMP design (≤50 AA), TF-Bind-8 (DNA, length 8), GFP (length 237)
3. **Architecture**: Simple MLPs for both proxy and GFlowNet policy
4. **Key contributions**:
   - Off-policy training with existing datasets (γ mixing parameter)
   - Epistemic uncertainty via ensembles/MC Dropout + acquisition functions (UCB/EI)
   - Trajectory balance loss

## Where Our Project Can Contribute

**Yes, there's significant opportunity.** Key differentiators:

| Aspect | Jain 2022 | Our Project |
|--------|-----------|-------------|
| **Backbone** | MLP with one-hot encoding | ESM-2 pretrained embeddings |
| **Policy** | MLP | Causal Transformer |
| **Reward** | Single property (AMP activity) | Multi-objective composite (stability + binding + naturalness) |
| **Stability** | Not addressed | Explicit stability predictor from FLIP data |
| **Structure** | None | ESMFold pLDDT integration planned |
| **Length** | Fixed/padded | Variable with learned STOP |

## Concrete Research Angles

1. **Better representations**: ESM-2 embeddings capture evolutionary/structural info that one-hot MLPs miss entirely

2. **Multi-objective optimization**: Real therapeutics need stability, binding, AND manufacturability - not just activity

3. **Structural awareness**: pLDDT from ESMFold provides structural validity signal they lack

4. **Modern architecture**: Transformer policy should better capture long-range dependencies in peptides

5. **Reproducibility/benchmarking**: Their code exists but using newer FLIP benchmarks would be valuable

## Honest Assessment

The Jain paper established GFlowNets work for biological sequences. Our contribution would be:

- **Not**: "First to use GFlowNets for peptides"
- **Yes**: "First to combine GFlowNets with PLM representations and multi-objective therapeutic design"

This is a valid and publishable direction - improving upon foundational work with better representations and more realistic objectives is standard practice in ML for biology.

## Technical Details from Jain 2022

### Their Architecture
- Proxy: MLP with 2 hidden layers (dim 2048), ReLU activation
- GFlowNet: MLP with 2 hidden layers (dim 2048)
- Training: Adam optimizer, trajectory balance loss
- Offline data mixing: γ = 0.5 optimal

### Their Results (AMP Task, K=100)
| Method | Performance | Diversity | Novelty |
|--------|-------------|-----------|---------|
| GFlowNet-AL | 0.932 ± 0.002 | 22.34 ± 1.24 | 28.44 ± 1.32 |
| DynaPPO | 0.938 ± 0.009 | 12.12 ± 1.71 | 9.31 ± 0.69 |
| COMs | 0.761 ± 0.009 | 19.38 ± 0.14 | 26.47 ± 1.3 |

### Key Insight
GFlowNet-AL achieves comparable performance to RL (DynaPPO) while generating **2x more diverse** and **3x more novel** candidates.

## References

- Paper: Jain et al., "Biological Sequence Design with GFlowNets", ICML 2022
- Code: https://github.com/MJ10/BioSeq-GFN-AL
