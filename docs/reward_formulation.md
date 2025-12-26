# Reward Formulation for GFlowNet Peptide Generation

This document provides a comprehensive mathematical description of the reward functions used in the GFlowNet-based therapeutic peptide generation system. The reward formulation is central to guiding the generative model toward producing peptides with desirable properties.

## Table of Contents

1. [Overview](#1-overview)
2. [ESM-2 Pseudo-Likelihood Reward](#2-esm-2-pseudo-likelihood-reward)
3. [Improved Reward with Entropy Gate](#3-improved-reward-with-entropy-gate)
4. [Composite Multi-Objective Reward](#4-composite-multi-objective-reward)
5. [Diversity-Augmented Reward (GRPO-D)](#5-diversity-augmented-reward-grpo-d)
6. [Integration with GRPO Training](#6-integration-with-grpo-training)
7. [Hyperparameter Summary](#7-hyperparameter-summary)
8. [References](#8-references)

---

## 1. Overview

The reward function $R: \mathcal{X} \rightarrow \mathbb{R}^+$ maps peptide sequences to non-negative scalar values, guiding the generative policy to sample sequences proportionally to their reward:

$$P(x) \propto R(x)^\beta$$

where $\beta$ is the inverse temperature controlling the sharpness of the distribution.

Our system implements three reward formulations, selectable via `--reward_type` in `train_gflownet.py`:

| Option | `--reward_type` | Description | Use Case |
|--------|-----------------|-------------|----------|
| **C** | `improved` | Entropy gate + length gate + ESM naturalness | Fast validation, anti-hacking |
| **B** | `esm2_pll` | ESM-2 pseudo-likelihood scoring | Biologically grounded baseline |
| **A** | `trained` | Trained stability predictor + entropy gate | Publication-ready, uses real data |

Additionally, `--reward_type composite` provides the original (untrained) CompositeReward for backward compatibility.

All formulations leverage ESM-2 (Evolutionary Scale Modeling) as the backbone encoder for extracting sequence representations.

---

## 2. ESM-2 Pseudo-Likelihood Reward

### 2.1 Theoretical Foundation

The ESM-2 pseudo-likelihood reward quantifies how "natural" or protein-like a sequence appears according to a masked language model trained on evolutionary protein sequences. This approach is grounded in the principle that evolutionarily conserved sequences exhibit statistical regularities captured by large-scale protein language models.

### 2.2 Exact Pseudo-Likelihood Formulation

For a peptide sequence $x = (x_1, x_2, \ldots, x_L)$ of length $L$, the exact pseudo-likelihood score is defined as:

$$R_{\text{PLL}}(x) = \frac{1}{L} \sum_{i=1}^{L} \log P_{\text{ESM}}(x_i \mid x_{\setminus i})$$

where:
- $x_i \in \mathcal{A}$ is the amino acid at position $i$
- $\mathcal{A} = \{\text{A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y}\}$ is the 20-letter amino acid alphabet
- $x_{\setminus i} = (x_1, \ldots, x_{i-1}, \texttt{[MASK]}, x_{i+1}, \ldots, x_L)$ is the sequence with position $i$ masked
- $P_{\text{ESM}}(x_i \mid x_{\setminus i})$ is the probability assigned by ESM-2 to the true amino acid given the masked context

**Derivation**: The pseudo-likelihood approximates the joint probability $P(x)$ by assuming conditional independence:

$$\log P(x) \approx \sum_{i=1}^{L} \log P(x_i \mid x_{\setminus i})$$

This avoids the computational intractability of computing the true joint probability while preserving local sequence context.

### 2.3 Fast Approximation

Computing the exact pseudo-likelihood requires $L$ forward passes (one per masked position). We employ a single-pass approximation:

$$\hat{R}_{\text{PLL}}(x) = \frac{1}{L} \sum_{i=1}^{L} \log P_{\text{ESM}}(x_i \mid x)$$

where $P_{\text{ESM}}(x_i \mid x)$ uses the contextual representation from a single forward pass without masking. This approximation is valid because ESM-2's bidirectional attention captures global context at each position.

**Computational Complexity**:
- Exact: $\mathcal{O}(L \cdot T_{\text{forward}})$
- Fast: $\mathcal{O}(T_{\text{forward}})$

where $T_{\text{forward}}$ is the time for one ESM-2 forward pass.

### 2.4 Score Computation

Given the ESM-2 output logits $\mathbf{z}_i \in \mathbb{R}^{|\mathcal{A}|}$ at position $i$:

$$P_{\text{ESM}}(x_i \mid x) = \text{softmax}(\mathbf{z}_i)_{x_i} = \frac{\exp(z_{i,x_i})}{\sum_{a \in \mathcal{A}} \exp(z_{i,a})}$$

The log-probability is:

$$\log P_{\text{ESM}}(x_i \mid x) = z_{i,x_i} - \log \sum_{a \in \mathcal{A}} \exp(z_{i,a})$$

### 2.5 Normalization

Raw pseudo-likelihood scores are normalized using running min-max statistics:

$$R_{\text{norm}}(x) = \frac{R_{\text{PLL}}(x) - R_{\min}}{R_{\max} - R_{\min}}$$

where $R_{\min}$ and $R_{\max}$ are updated online after observing $N > 10$ sequences:

$$R_{\min} \leftarrow \min(R_{\min}, R_{\text{PLL}}(x))$$
$$R_{\max} \leftarrow \max(R_{\max}, R_{\text{PLL}}(x))$$

This maps rewards to $[0, 1]$, facilitating combination with diversity scores.

### 2.6 Temperature Scaling

An optional temperature parameter $\tau$ sharpens or smooths the reward distribution:

$$R_{\text{scaled}}(x) = R_{\text{norm}}(x)^\tau$$

- $\tau > 1$: Sharpens distribution (favors high-reward sequences)
- $\tau < 1$: Smooths distribution (encourages exploration)
- $\tau = 1$: No scaling (default)

### 2.7 Edge Cases

Sequences shorter than 3 amino acids receive a penalty score:

$$R_{\text{PLL}}(x) = -10.0 \quad \text{if } |x| < 3$$

This prevents degenerate short sequences from receiving artificially high scores.

### 2.8 Limitations: Reward Hacking

**Critical Issue**: ESM-2 pseudo-likelihood is vulnerable to reward hacking via repetitive sequences. Homopolymers like `QQQQQQQQQQ` receive high scores because each position is trivially predictable from context:

$$P_{\text{ESM}}(Q \mid \text{QQQQQ[MASK]QQQQ}) \approx 0.99$$

This leads to:
- 97% of generated sequences containing repetitive patterns
- Mode collapse to degenerate sequences
- High sequence diversity (edit distance) but low embedding diversity

**Evidence from Phase 0a Training**:

| Sequence | Normalized Reward |
|----------|-------------------|
| `MRQQQQQQQQQQQQQQQQNNNNNNNNNNNN` | 0.932 |
| `MPGNNNNNNNNQQQQQQQQQQQQQQQQQQQ` | 0.929 |
| `MKTLLILAVVALACARSSAQAANPF` (real) | ~0.70 |

The improved reward (Section 3) addresses this fundamental flaw.

---

## 3. Improved Reward with Entropy Gate

### 3.1 Motivation

The ESM-2 pseudo-likelihood reward (Section 2) suffers from reward hacking: repetitive sequences receive high scores because each amino acid is predictable from its homogeneous context. The improved reward addresses this by:

1. Replacing pseudo-likelihood with embedding-based naturalness
2. Adding an entropy gate to penalize low-complexity sequences
3. Adding a length gate to penalize too-short sequences

### 3.2 Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         Peptide Sequence x          │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
           │ ESM-2        │  │ Entropy      │  │ Length       │
           │ Embedding    │  │ Computation  │  │ Check        │
           │ + Norm       │  │              │  │              │
           └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                  │                 │                 │
                  ▼                 ▼                 ▼
           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
           │ Sigmoid      │  │ Sigmoid      │  │ Sigmoid      │
           │ Naturalness  │  │ Entropy Gate │  │ Length Gate  │
           │ [0, 1]       │  │ [0, 1]       │  │ [0, 1]       │
           └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                  │                 │                 │
                  └────────────────┬┴─────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  R = naturalness × entropy_gate     │
                    │                  × length_gate      │
                    └─────────────────────────────────────┘
```

### 3.3 Component 1: Embedding Naturalness

Instead of pseudo-likelihood, we use the ESM-2 embedding norm as a proxy for sequence quality:

$$R_{\text{nat}}(x) = \sigma\left(\frac{\|\mathbf{e}(x)\|_2}{\tau_{\text{emb}}}\right)$$

where:
- $\mathbf{e}(x) = \frac{1}{L} \sum_{i=1}^{L} \mathbf{h}_i^{(l)}$ is the mean-pooled embedding
- $\|\cdot\|_2$ is the L2 norm
- $\tau_{\text{emb}} = 10.0$ is the temperature parameter
- $\sigma$ is the sigmoid function

**Rationale**: Real proteins occupy a well-defined region in ESM-2's embedding space with consistent norms. Degenerate sequences (repetitive or random) have abnormal embedding characteristics.

**Key Difference from Pseudo-Likelihood**:
- Pseudo-likelihood asks: "Is each amino acid predictable?" → Repetition is predictable → High score
- Embedding norm asks: "Does this look like a real protein in embedding space?" → Repetition is abnormal → Lower score

### 3.4 Component 2: Entropy Gate

The entropy gate penalizes low-complexity sequences using Shannon entropy of the amino acid distribution:

$$H(x) = -\sum_{a \in \mathcal{A}} p_a \log_2 p_a$$

where $p_a = \frac{\text{count}(a, x)}{|x|}$ is the frequency of amino acid $a$ in sequence $x$.

**Normalized Entropy**:

$$\hat{H}(x) = \frac{H(x)}{\log_2 20}$$

This normalizes to $[0, 1]$ where:
- $\hat{H} = 0$: All same amino acid (e.g., `QQQQQQQQQQ`)
- $\hat{H} = 1$: Uniform distribution of all 20 amino acids

**Soft Gate**:

$$G_{\text{ent}}(x) = \sigma\left(k_{\text{ent}} \cdot (\hat{H}(x) - \theta_{\text{ent}})\right)$$

where:
- $\theta_{\text{ent}} = 0.5$ is the entropy threshold (50% of maximum)
- $k_{\text{ent}} = 10.0$ is the sigmoid sharpness

**Behavior**:
- $\hat{H} < \theta_{\text{ent}}$: Gate → 0 (penalize)
- $\hat{H} > \theta_{\text{ent}}$: Gate → 1 (allow)

### 3.5 Component 3: Length Gate

The length gate penalizes sequences shorter than a minimum length:

$$G_{\text{len}}(x) = \sigma\left(k_{\text{len}} \cdot (|x| - L_{\min})\right)$$

where:
- $L_{\min} = 10$ is the minimum peptide length
- $k_{\text{len}} = 0.5$ is the sigmoid sharpness

**Behavior**:
- $|x| < L_{\min}$: Gate → 0 (penalize short sequences)
- $|x| \geq L_{\min}$: Gate → 1 (allow)

### 3.6 Combined Improved Reward

The three components are combined multiplicatively:

$$R_{\text{improved}}(x) = R_{\text{nat}}(x) \times G_{\text{ent}}(x) \times G_{\text{len}}(x)$$

**Properties**:
- All components in $[0, 1]$ → Combined reward in $[0, 1]$
- Multiplicative combination: ALL components must be good for high reward
- If any gate → 0, total reward → 0

### 3.7 Validation Results

| Sequence Type | Example | Entropy | Entropy Gate | Total Reward |
|---------------|---------|---------|--------------|--------------|
| Real peptide | `MKTLLILAVVALACARSSAQAANPF` | 0.78 | 0.94 | **0.60** |
| Homopolymer | `QQQQQQQQQQQQQQQQQQQQQQQQQQ` | 0.00 | 0.007 | **0.005** |
| Alternating | `AQAQAQAQAQAQAQAQAQAQAQAQ` | 0.23 | 0.06 | **0.04** |
| All different | `ACDEFGHIKLMNPQRSTVWY` | 1.00 | 0.99 | **0.64** |

**Pass Criteria Met**:
- R(real) > R(repetitive) for 100% of test pairs ✓
- R(homopolymer) < 0.1 for all homopolymers ✓
- R(real) > 0.5 for all real peptides ✓

### 3.8 Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| ESM model | - | `esm2_t6_8M_UR50D` | Backbone for embeddings |
| Embedding temperature | $\tau_{\text{emb}}$ | 10.0 | Sigmoid temperature for naturalness |
| Entropy threshold | $\theta_{\text{ent}}$ | 0.5 | Minimum normalized entropy |
| Entropy sharpness | $k_{\text{ent}}$ | 10.0 | Sigmoid slope for entropy gate |
| Min length | $L_{\min}$ | 10 | Minimum peptide length |
| Length sharpness | $k_{\text{len}}$ | 0.5 | Sigmoid slope for length gate |

### 3.9 Implementation Reference

| Component | Source File | Key Lines |
|-----------|-------------|-----------|
| ImprovedReward class | `gflownet_peptide/rewards/improved_reward.py` | 1-250 |
| Entropy computation | `gflownet_peptide/rewards/improved_reward.py` | 97-116 |
| Entropy gate | `gflownet_peptide/rewards/improved_reward.py` | 118-131 |
| Length gate | `gflownet_peptide/rewards/improved_reward.py` | 133-145 |
| Embedding naturalness | `gflownet_peptide/rewards/improved_reward.py` | 147-180 |

---

## 4. Composite Multi-Objective Reward

### 4.1 Architecture Overview

The composite reward combines three property-specific predictions using a shared ESM-2 backbone:

```
                    ┌─────────────────────────────────────┐
                    │         Peptide Sequence x          │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     ESM-2 Backbone (Frozen)         │
                    │   h_i^(l) for i = 1, ..., L         │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Mean Pooling (excl. special)   │
                    │   e(x) = (1/L) Σ h_i^(l)            │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
           │ Stability    │  │ Binding      │  │ Naturalness  │
           │ MLP Head     │  │ MLP Head     │  │ Head         │
           │ exp(·)       │  │ softplus(·)  │  │ σ(‖·‖/τ)     │
           └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                  │                 │                 │
                  ▼                 ▼                 ▼
                 R_S               R_B               R_N
                  │                 │                 │
                  └────────────────┬┴─────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  R = R_S^w_S · R_B^w_B · R_N^w_N    │
                    │       (Geometric Mean)              │
                    └─────────────────────────────────────┘
```

### 4.2 Embedding Extraction

ESM-2 produces contextualized representations $\mathbf{h}_i^{(l)} \in \mathbb{R}^d$ for each position $i$ at layer $l$. The sequence embedding is computed via mean pooling, excluding special tokens:

$$\mathbf{e}(x) = \frac{1}{L} \sum_{i=1}^{L} \mathbf{h}_i^{(l)}$$

where:
- $L$ is the sequence length (excluding CLS and EOS tokens)
- $l$ is the final transformer layer
- $d$ is the embedding dimension (model-dependent)

**Supported ESM-2 Models**:

| Model | Layers ($l$) | Embedding Dim ($d$) | Parameters |
|-------|--------------|---------------------|------------|
| `esm2_t6_8M_UR50D` | 6 | 320 | 8M |
| `esm2_t12_35M_UR50D` | 12 | 480 | 35M |
| `esm2_t30_150M_UR50D` | 30 | 640 | 150M |
| `esm2_t33_650M_UR50D` | 33 | 1280 | 650M |
| `esm2_t36_3B_UR50D` | 36 | 2560 | 3B |

### 4.3 Reward Head Architecture

Each property-specific reward is predicted by an MLP head:

$$\text{MLP}(\mathbf{e}) = \mathbf{W}_n \cdot \sigma(\mathbf{W}_{n-1} \cdot \sigma(\cdots \sigma(\mathbf{W}_1 \cdot \mathbf{e} + \mathbf{b}_1) \cdots) + \mathbf{b}_{n-1}) + \mathbf{b}_n$$

where:
- $\sigma$ is the activation function (ReLU or GELU)
- $n$ is the number of layers (default: 2)
- Hidden dimension: 256 (default)

### 4.4 Stability Reward

The stability reward predicts thermal stability, trained on FLIP benchmark data:

$$R_S(x) = \exp\left(\text{MLP}_S(\mathbf{e}(x))\right)$$

**Properties**:
- Output range: $(0, +\infty)$
- Exponential transform ensures strict positivity
- Higher values indicate greater predicted thermal stability

**Training Target**: $\Delta\Delta G$ (change in Gibbs free energy upon mutation)

### 4.5 Binding Affinity Reward

The binding reward predicts peptide-protein binding affinity, trained on Propedia data:

$$R_B(x) = \text{softplus}\left(\text{MLP}_B(\mathbf{e}(x))\right) = \log\left(1 + \exp(\text{MLP}_B(\mathbf{e}(x)))\right)$$

**Properties**:
- Output range: $(0, +\infty)$
- Softplus provides smooth, differentiable non-negativity
- Approximates ReLU for large inputs: $\text{softplus}(z) \approx z$ for $z \gg 0$

**Derivation**: Softplus is chosen over exponential to prevent numerical overflow for high-affinity predictions while maintaining gradient flow for low values.

### 4.6 Naturalness Reward

The naturalness reward uses embedding norm as a proxy for sequence quality:

$$R_N(x) = \sigma\left(\frac{\|\mathbf{e}(x)\|_2}{\tau_N}\right) = \frac{1}{1 + \exp\left(-\|\mathbf{e}(x)\|_2 / \tau_N\right)}$$

where:
- $\|\mathbf{e}(x)\|_2 = \sqrt{\sum_{j=1}^{d} e_j^2}$ is the L2 norm
- $\tau_N$ is the temperature parameter (default: 10.0)
- $\sigma$ is the sigmoid function

**Rationale**: Sequences well-represented in ESM-2's embedding space (higher norm) are more "natural" according to evolutionary patterns. The sigmoid bounds the output to $(0, 1)$.

**Properties**:
- Output range: $(0, 1)$
- Monotonically increasing in embedding norm
- Temperature $\tau_N$ controls sensitivity

### 4.7 Composite Reward Formulation

Individual rewards are combined using a weighted geometric mean:

$$R_{\text{composite}}(x) = R_S(x)^{w_S} \cdot R_B(x)^{w_B} \cdot R_N(x)^{w_N}$$

**Default Weights**:
- $w_S = 1.0$ (stability)
- $w_B = 1.0$ (binding)
- $w_N = 0.5$ (naturalness)

**Justification for Geometric Mean**:

1. **Multiplicative Penalty**: If any component approaches zero, the composite reward approaches zero. This ensures all objectives must be satisfied:
   $$\lim_{R_i \to 0} R_{\text{composite}} = 0$$

2. **Scale Invariance**: The geometric mean is invariant to rescaling of individual components:
   $$(\alpha R_S)^{w_S} \cdot R_B^{w_B} \cdot R_N^{w_N} = \alpha^{w_S} \cdot R_{\text{composite}}$$

3. **Pareto Optimality**: Maximizing the geometric mean tends to find Pareto-optimal solutions that balance all objectives.

**Logarithmic Form** (for numerical stability):

$$\log R_{\text{composite}}(x) = w_S \log R_S(x) + w_B \log R_B(x) + w_N \log R_N(x)$$

### 4.8 Non-Negativity Transforms Summary

| Transform | Formula | Range | Use Case | Gradient at 0 |
|-----------|---------|-------|----------|---------------|
| Exponential | $\exp(z)$ | $(0, +\infty)$ | Stability | $\exp(0) = 1$ |
| Softplus | $\log(1 + e^z)$ | $(0, +\infty)$ | Binding | $\sigma(0) = 0.5$ |
| Sigmoid | $(1 + e^{-z})^{-1}$ | $(0, 1)$ | Naturalness | $0.25$ |
| ReLU + $\epsilon$ | $\max(0, z) + 10^{-6}$ | $[\epsilon, +\infty)$ | Fallback | 0 or 1 |

---

## 5. Diversity-Augmented Reward (GRPO-D)

### 5.1 Motivation

Pure reward maximization can lead to mode collapse, where the generator produces repetitive high-reward sequences. Diversity-augmented rewards explicitly encourage exploration of the sequence space.

### 5.2 Per-Peptide Diversity Score

For a batch of peptides $\mathcal{B} = \{x_1, x_2, \ldots, x_n\}$, each peptide receives a diversity score:

$$D(x_i) = \lambda_{\text{AA}} \cdot D_{\text{AA}}(x_i) + \lambda_{\text{seq}} \cdot D_{\text{seq}}(x_i)$$

where:
- $\lambda_{\text{AA}} = 0.7$ (amino acid frequency weight)
- $\lambda_{\text{seq}} = 0.3$ (sequence dissimilarity weight)

### 5.3 Amino Acid Frequency Diversity

This component rewards peptides containing rare amino acids within the batch:

$$D_{\text{AA}}(x_i) = \frac{1}{|x_i|} \sum_{a \in x_i} \frac{1}{C(a) + 1}$$

where:
- $|x_i|$ is the length of peptide $x_i$
- $C(a) = \sum_{j=1}^{n} \sum_{k=1}^{|x_j|} \mathbb{1}[x_{j,k} = a]$ is the count of amino acid $a$ across all peptides in the batch
- The $+1$ in the denominator prevents division by zero

**Intuition**: Peptides with uncommon amino acid compositions receive higher diversity scores, encouraging exploration of underrepresented regions of sequence space.

**Derivation**: This formulation is analogous to inverse document frequency (IDF) in information retrieval, where rare terms receive higher weights.

### 5.4 Sequence Dissimilarity

This component measures how different a peptide is from others in the batch using normalized Levenshtein distance:

$$D_{\text{seq}}(x_i) = \frac{1}{|\mathcal{B}| - 1} \sum_{x_j \in \mathcal{B}, j \neq i} \frac{d_{\text{Lev}}(x_i, x_j)}{\max(|x_i|, |x_j|)}$$

where:
- $d_{\text{Lev}}(x_i, x_j)$ is the Levenshtein (edit) distance between sequences
- Normalization by maximum length bounds the score to $[0, 1]$

**Levenshtein Distance Definition**:

$$d_{\text{Lev}}(x, y) = \min \text{ number of single-character edits (insert, delete, substitute)}$$

**Computational Optimization**: For large batches, we sample a subset of $m = 50$ reference peptides to reduce complexity from $\mathcal{O}(n^2 L^2)$ to $\mathcal{O}(nm L^2)$.

### 5.5 Normalization

Raw diversity scores are normalized within each batch using min-max scaling:

$$\hat{D}(x_i) = \frac{D(x_i) - D_{\min}}{D_{\max} - D_{\min}}$$

where $D_{\min} = \min_{j} D(x_j)$ and $D_{\max} = \max_{j} D(x_j)$.

If all diversity scores are identical ($D_{\max} = D_{\min}$), we set $\hat{D}(x_i) = 0.5$.

### 5.6 Combined Reward

The final reward combines the base reward with the diversity bonus:

$$R_{\text{combined}}(x) = (1 - \omega_D) \cdot R_{\text{base}}(x) + \omega_D \cdot \hat{D}(x)$$

where:
- $R_{\text{base}}(x)$ is either $R_{\text{PLL}}$, $R_{\text{improved}}$, or $R_{\text{composite}}$
- $\omega_D = 0.15$ is the diversity weight (default)

**Properties**:
- Linear interpolation preserves reward scale
- $\omega_D = 0$: Pure reward maximization
- $\omega_D = 1$: Pure diversity maximization
- $\omega_D = 0.15$: Balanced exploration-exploitation

---

## 6. Integration with GRPO Training

### 6.1 Group Relative Policy Optimization

GRPO extends REINFORCE with group-wise normalization and KL regularization. The combined reward feeds into advantage computation.

### 6.2 Advantage Computation

For peptides grouped by prompt $g$, advantages are computed as:

$$A(x) = \frac{R_{\text{combined}}(x) - \mu_g}{\sigma_g + \epsilon}$$

where:
- $\mu_g = \frac{1}{|g|} \sum_{x \in g} R_{\text{combined}}(x)$
- $\sigma_g = \sqrt{\frac{1}{|g|} \sum_{x \in g} (R_{\text{combined}}(x) - \mu_g)^2}$
- $\epsilon = 10^{-8}$ (numerical stability)

**Derivation**: Group normalization reduces variance in advantage estimates, stabilizing training. Centering around the group mean ensures both positive and negative advantages, enabling contrastive learning.

### 6.3 Policy Gradient Loss

The policy loss encourages actions leading to high-advantage outcomes:

$$\mathcal{L}_{\text{policy}} = -\mathbb{E}_{x \sim \pi_\theta}\left[\log \pi_\theta(x) \cdot A(x)\right]$$

In practice, with a batch of samples:

$$\mathcal{L}_{\text{policy}} = -\frac{1}{n} \sum_{i=1}^{n} \left(\frac{1}{|x_i|} \sum_{t=1}^{|x_i|} \log \pi_\theta(x_{i,t} \mid x_{i,<t})\right) \cdot A(x_i)$$

### 6.4 KL Divergence Penalty

To prevent the policy from deviating too far from the reference model:

$$\mathcal{L}_{\text{KL}} = \beta \cdot D_{\text{KL}}(\pi_{\text{ref}} \| \pi_\theta)$$

The KL divergence is estimated using the k3 estimator:

$$\hat{D}_{\text{KL}} = \mathbb{E}_{x \sim \pi_\theta}\left[\frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} - \log \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} - 1\right]$$

where $\beta = 0.04$ (default).

### 6.5 Total GRPO Loss

$$\mathcal{L}_{\text{GRPO}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{KL}}$$

### 6.6 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GRPO Training Loop                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. GENERATION                                                      │
│     ┌─────────────────┐                                             │
│     │  ProtGPT2       │  Generate n peptides per prompt             │
│     │  Policy π_θ     │  using nucleus sampling (top-p)             │
│     └────────┬────────┘                                             │
│              │                                                      │
│              ▼                                                      │
│     ┌─────────────────┐                                             │
│     │  Peptide Batch  │  x_1, x_2, ..., x_n                         │
│     └────────┬────────┘                                             │
│              │                                                      │
│  2. REWARD COMPUTATION                                              │
│              │                                                      │
│     ┌────────┴────────┐                                             │
│     ▼                 ▼                                             │
│  ┌──────────────┐  ┌──────────────┐                                 │
│  │ ESM-2 Reward │  │ Diversity    │                                 │
│  │ R_base(x)    │  │ D(x)         │                                 │
│  └──────┬───────┘  └──────┬───────┘                                 │
│         │                 │                                         │
│         └────────┬────────┘                                         │
│                  ▼                                                  │
│         ┌──────────────────┐                                        │
│         │ Combined Reward  │                                        │
│         │ R = (1-ω)R + ωD  │                                        │
│         └────────┬─────────┘                                        │
│                  │                                                  │
│  3. ADVANTAGE COMPUTATION                                           │
│                  │                                                  │
│                  ▼                                                  │
│         ┌──────────────────┐                                        │
│         │ Group-Normalize  │  A = (R - μ_g) / (σ_g + ε)             │
│         └────────┬─────────┘                                        │
│                  │                                                  │
│  4. POLICY UPDATE                                                   │
│                  │                                                  │
│     ┌────────────┴────────────┐                                     │
│     ▼                         ▼                                     │
│  ┌──────────────┐      ┌──────────────┐                             │
│  │ Policy Loss  │      │ KL Penalty   │                             │
│  │ -E[log π · A]│      │ β · D_KL     │                             │
│  └──────┬───────┘      └──────┬───────┘                             │
│         │                     │                                     │
│         └──────────┬──────────┘                                     │
│                    ▼                                                │
│         ┌──────────────────┐                                        │
│         │ Total Loss       │  L = L_policy + L_KL                   │
│         └────────┬─────────┘                                        │
│                  │                                                  │
│                  ▼                                                  │
│         ┌──────────────────┐                                        │
│         │ Gradient Update  │  θ ← θ - η∇L                           │
│         │ (Adam + clip)    │                                        │
│         └──────────────────┘                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Hyperparameter Summary

### 7.1 ESM-2 Pseudo-Likelihood Reward

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Model | - | `esm2_t12_35M_UR50D` | ESM-2 variant |
| Normalize | - | `True` | Min-max normalization |
| Temperature | $\tau$ | 1.0 | Reward sharpening |
| Min observations | $N$ | 10 | Before normalization starts |
| Min sequence length | - | 3 | Penalty for shorter |

### 7.2 Improved Reward (Entropy Gate)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| ESM model | - | `esm2_t6_8M_UR50D` | Backbone for embeddings |
| Embedding temperature | $\tau_{\text{emb}}$ | 10.0 | Sigmoid temperature for naturalness |
| Entropy threshold | $\theta_{\text{ent}}$ | 0.5 | Minimum normalized entropy |
| Entropy sharpness | $k_{\text{ent}}$ | 10.0 | Sigmoid slope for entropy gate |
| Min length | $L_{\min}$ | 10 | Minimum peptide length |
| Length sharpness | $k_{\text{len}}$ | 0.5 | Sigmoid slope for length gate |

### 7.3 Composite Reward

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| ESM model | - | `esm2_t33_650M_UR50D` | Backbone model |
| Freeze ESM | - | `True` | No backbone fine-tuning |
| Share backbone | - | `True` | Single ESM pass for all heads |
| Hidden dim | - | 256 | MLP hidden layer size |
| MLP layers | $n$ | 2 | Number of MLP layers |
| Stability weight | $w_S$ | 1.0 | Geometric mean weight |
| Binding weight | $w_B$ | 1.0 | Geometric mean weight |
| Naturalness weight | $w_N$ | 0.5 | Geometric mean weight |
| Naturalness temp | $\tau_N$ | 10.0 | Sigmoid temperature |

### 7.4 Diversity (GRPO-D)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Diversity weight | $\omega_D$ | 0.15 | Combined reward interpolation |
| AA frequency weight | $\lambda_{\text{AA}}$ | 0.7 | Diversity component weight |
| Sequence weight | $\lambda_{\text{seq}}$ | 0.3 | Diversity component weight |
| Max compare | $m$ | 50 | Reference peptides for Levenshtein |

### 7.5 GRPO Training

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Learning rate | $\eta$ | $3 \times 10^{-4}$ | Adam optimizer |
| KL coefficient | $\beta$ | 0.04 | KL penalty weight |
| Batch size | - | 16 | Prompts per iteration |
| Generations per prompt | - | 8 | Peptides per prompt |
| Max gradient norm | - | 1.0 | Gradient clipping |
| Buffer size | - | 500 | Experience buffer capacity |

---

## 8. References

### 8.1 ESM-2

Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.

### 8.2 Pseudo-Likelihood Scoring

Salazar, J., et al. (2020). Masked language model scoring. *Proceedings of ACL*, 2699-2712.

### 8.3 GRPO

Shao, Z., et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*.

### 8.4 GFlowNets

Bengio, E., et al. (2021). Flow network based generative models for non-iterative diverse candidate generation. *NeurIPS*, 34, 27381-27394.

---

## Appendix A: Code References

| Component | Source File | Key Lines |
|-----------|-------------|-----------|
| ESM-2 Pseudo-Likelihood | `gflownet_peptide/rewards/esm2_reward.py` | 55-278 |
| Improved Reward | `gflownet_peptide/rewards/improved_reward.py` | 1-250 |
| Trained Composite Reward | `gflownet_peptide/rewards/composite_reward.py` | 1-257 |
| Stability Predictor | `gflownet_peptide/rewards/stability_predictor.py` | - |
| ESM Backbone | `gflownet_peptide/models/reward_model.py` | 18-101 |
| Reward Head (MLP) | `gflownet_peptide/models/reward_model.py` | 104-162 |
| Stability Reward | `gflownet_peptide/models/reward_model.py` | 165-202 |
| Binding Reward | `gflownet_peptide/models/reward_model.py` | 205-235 |
| Naturalness Reward | `gflownet_peptide/models/reward_model.py` | 238-285 |
| Composite Reward (untrained) | `gflownet_peptide/models/reward_model.py` | 288-410 |
| AA Frequency Diversity | `gflownet_peptide/training/diversity.py` | 19-61 |
| Sequence Dissimilarity | `gflownet_peptide/training/diversity.py` | 64-117 |
| Combined Diversity | `gflownet_peptide/training/diversity.py` | 120-176 |
| GRPO Advantages | `gflownet_peptide/training/grpo_trainer.py` | 101-149 |
| GRPO Loss | `gflownet_peptide/training/grpo_trainer.py` | 211-306 |
| Training Loop | `scripts/train_grpo.py` | 246-329 |

---

## Appendix B: Reward Type Selection in Training

The `--reward_type` flag in `scripts/train_gflownet.py` selects the reward function:

```bash
# Option C: Improved Reward (fastest, anti-hacking)
python scripts/train_gflownet.py --reward_type improved --esm_model esm2_t6_8M_UR50D

# Option B: ESM-2 Pseudo-likelihood (biologically grounded)
python scripts/train_gflownet.py --reward_type esm2_pll --esm_model esm2_t6_8M_UR50D

# Option A: Trained Stability (requires checkpoint)
python scripts/train_gflownet.py --reward_type trained \
    --reward_checkpoint checkpoints/reward_models/stability_predictor_best.pt

# Legacy: Untrained composite (not recommended - flat rewards)
python scripts/train_gflownet.py --reward_type composite
```

### Reward Comparison Summary

| Reward Type | Log Reward Range | Discriminative Power | Training Signal |
|-------------|------------------|---------------------|-----------------|
| `composite` (untrained) | ~0.03 | Very Low | Insufficient |
| `improved` | ~2-3 | Good | Sufficient |
| `esm2_pll` | ~3-5 | Good | Sufficient |
| `trained` | ~2-4 | Good | Sufficient |

The baseline training with `composite` (untrained) produced flat rewards (~0.445) because the MLP heads were never trained. Options A, B, and C provide discriminative rewards that enable GFlowNet learning.

---

*Document generated for scientific publication. Last updated: 2025-12-26*
