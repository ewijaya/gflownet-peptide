# Reward Formulation for GFlowNet Peptide Generation

This document provides a comprehensive mathematical description of the reward functions used in the GFlowNet-based therapeutic peptide generation system. The reward formulation is central to guiding the generative model toward producing peptides with desirable properties.

## Table of Contents

1. [Overview](#1-overview)
2. [ESM-2 Pseudo-Likelihood Reward](#2-esm-2-pseudo-likelihood-reward)
3. [Composite Multi-Objective Reward](#3-composite-multi-objective-reward)
4. [Diversity-Augmented Reward (GRPO-D)](#4-diversity-augmented-reward-grpo-d)
5. [Integration with GRPO Training](#5-integration-with-grpo-training)
6. [Hyperparameter Summary](#6-hyperparameter-summary)
7. [References](#7-references)

---

## 1. Overview

The reward function $R: \mathcal{X} \rightarrow \mathbb{R}^+$ maps peptide sequences to non-negative scalar values, guiding the generative policy to sample sequences proportionally to their reward:

$$P(x) \propto R(x)^\beta$$

where $\beta$ is the inverse temperature controlling the sharpness of the distribution.

Our system implements two complementary reward formulations:

1. **ESM-2 Pseudo-Likelihood Reward**: Measures sequence naturalness using a pretrained protein language model
2. **Composite Multi-Objective Reward**: Combines stability, binding affinity, and naturalness predictions

Both formulations leverage ESM-2 (Evolutionary Scale Modeling) as the backbone encoder for extracting sequence representations.

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

---

## 3. Composite Multi-Objective Reward

### 3.1 Architecture Overview

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

### 3.2 Embedding Extraction

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

### 3.3 Reward Head Architecture

Each property-specific reward is predicted by an MLP head:

$$\text{MLP}(\mathbf{e}) = \mathbf{W}_n \cdot \sigma(\mathbf{W}_{n-1} \cdot \sigma(\cdots \sigma(\mathbf{W}_1 \cdot \mathbf{e} + \mathbf{b}_1) \cdots) + \mathbf{b}_{n-1}) + \mathbf{b}_n$$

where:
- $\sigma$ is the activation function (ReLU or GELU)
- $n$ is the number of layers (default: 2)
- Hidden dimension: 256 (default)

### 3.4 Stability Reward

The stability reward predicts thermal stability, trained on FLIP benchmark data:

$$R_S(x) = \exp\left(\text{MLP}_S(\mathbf{e}(x))\right)$$

**Properties**:
- Output range: $(0, +\infty)$
- Exponential transform ensures strict positivity
- Higher values indicate greater predicted thermal stability

**Training Target**: $\Delta\Delta G$ (change in Gibbs free energy upon mutation)

### 3.5 Binding Affinity Reward

The binding reward predicts peptide-protein binding affinity, trained on Propedia data:

$$R_B(x) = \text{softplus}\left(\text{MLP}_B(\mathbf{e}(x))\right) = \log\left(1 + \exp(\text{MLP}_B(\mathbf{e}(x)))\right)$$

**Properties**:
- Output range: $(0, +\infty)$
- Softplus provides smooth, differentiable non-negativity
- Approximates ReLU for large inputs: $\text{softplus}(z) \approx z$ for $z \gg 0$

**Derivation**: Softplus is chosen over exponential to prevent numerical overflow for high-affinity predictions while maintaining gradient flow for low values.

### 3.6 Naturalness Reward

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

### 3.7 Composite Reward Formulation

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

### 3.8 Non-Negativity Transforms Summary

| Transform | Formula | Range | Use Case | Gradient at 0 |
|-----------|---------|-------|----------|---------------|
| Exponential | $\exp(z)$ | $(0, +\infty)$ | Stability | $\exp(0) = 1$ |
| Softplus | $\log(1 + e^z)$ | $(0, +\infty)$ | Binding | $\sigma(0) = 0.5$ |
| Sigmoid | $(1 + e^{-z})^{-1}$ | $(0, 1)$ | Naturalness | $0.25$ |
| ReLU + $\epsilon$ | $\max(0, z) + 10^{-6}$ | $[\epsilon, +\infty)$ | Fallback | 0 or 1 |

---

## 4. Diversity-Augmented Reward (GRPO-D)

### 4.1 Motivation

Pure reward maximization can lead to mode collapse, where the generator produces repetitive high-reward sequences. Diversity-augmented rewards explicitly encourage exploration of the sequence space.

### 4.2 Per-Peptide Diversity Score

For a batch of peptides $\mathcal{B} = \{x_1, x_2, \ldots, x_n\}$, each peptide receives a diversity score:

$$D(x_i) = \lambda_{\text{AA}} \cdot D_{\text{AA}}(x_i) + \lambda_{\text{seq}} \cdot D_{\text{seq}}(x_i)$$

where:
- $\lambda_{\text{AA}} = 0.7$ (amino acid frequency weight)
- $\lambda_{\text{seq}} = 0.3$ (sequence dissimilarity weight)

### 4.3 Amino Acid Frequency Diversity

This component rewards peptides containing rare amino acids within the batch:

$$D_{\text{AA}}(x_i) = \frac{1}{|x_i|} \sum_{a \in x_i} \frac{1}{C(a) + 1}$$

where:
- $|x_i|$ is the length of peptide $x_i$
- $C(a) = \sum_{j=1}^{n} \sum_{k=1}^{|x_j|} \mathbb{1}[x_{j,k} = a]$ is the count of amino acid $a$ across all peptides in the batch
- The $+1$ in the denominator prevents division by zero

**Intuition**: Peptides with uncommon amino acid compositions receive higher diversity scores, encouraging exploration of underrepresented regions of sequence space.

**Derivation**: This formulation is analogous to inverse document frequency (IDF) in information retrieval, where rare terms receive higher weights.

### 4.4 Sequence Dissimilarity

This component measures how different a peptide is from others in the batch using normalized Levenshtein distance:

$$D_{\text{seq}}(x_i) = \frac{1}{|\mathcal{B}| - 1} \sum_{x_j \in \mathcal{B}, j \neq i} \frac{d_{\text{Lev}}(x_i, x_j)}{\max(|x_i|, |x_j|)}$$

where:
- $d_{\text{Lev}}(x_i, x_j)$ is the Levenshtein (edit) distance between sequences
- Normalization by maximum length bounds the score to $[0, 1]$

**Levenshtein Distance Definition**:

$$d_{\text{Lev}}(x, y) = \min \text{ number of single-character edits (insert, delete, substitute)}$$

**Computational Optimization**: For large batches, we sample a subset of $m = 50$ reference peptides to reduce complexity from $\mathcal{O}(n^2 L^2)$ to $\mathcal{O}(nm L^2)$.

### 4.5 Normalization

Raw diversity scores are normalized within each batch using min-max scaling:

$$\hat{D}(x_i) = \frac{D(x_i) - D_{\min}}{D_{\max} - D_{\min}}$$

where $D_{\min} = \min_{j} D(x_j)$ and $D_{\max} = \max_{j} D(x_j)$.

If all diversity scores are identical ($D_{\max} = D_{\min}$), we set $\hat{D}(x_i) = 0.5$.

### 4.6 Combined Reward

The final reward combines the base reward with the diversity bonus:

$$R_{\text{combined}}(x) = (1 - \omega_D) \cdot R_{\text{base}}(x) + \omega_D \cdot \hat{D}(x)$$

where:
- $R_{\text{base}}(x)$ is either $R_{\text{PLL}}$ or $R_{\text{composite}}$
- $\omega_D = 0.15$ is the diversity weight (default)

**Properties**:
- Linear interpolation preserves reward scale
- $\omega_D = 0$: Pure reward maximization
- $\omega_D = 1$: Pure diversity maximization
- $\omega_D = 0.15$: Balanced exploration-exploitation

---

## 5. Integration with GRPO Training

### 5.1 Group Relative Policy Optimization

GRPO extends REINFORCE with group-wise normalization and KL regularization. The combined reward feeds into advantage computation.

### 5.2 Advantage Computation

For peptides grouped by prompt $g$, advantages are computed as:

$$A(x) = \frac{R_{\text{combined}}(x) - \mu_g}{\sigma_g + \epsilon}$$

where:
- $\mu_g = \frac{1}{|g|} \sum_{x \in g} R_{\text{combined}}(x)$
- $\sigma_g = \sqrt{\frac{1}{|g|} \sum_{x \in g} (R_{\text{combined}}(x) - \mu_g)^2}$
- $\epsilon = 10^{-8}$ (numerical stability)

**Derivation**: Group normalization reduces variance in advantage estimates, stabilizing training. Centering around the group mean ensures both positive and negative advantages, enabling contrastive learning.

### 5.3 Policy Gradient Loss

The policy loss encourages actions leading to high-advantage outcomes:

$$\mathcal{L}_{\text{policy}} = -\mathbb{E}_{x \sim \pi_\theta}\left[\log \pi_\theta(x) \cdot A(x)\right]$$

In practice, with a batch of samples:

$$\mathcal{L}_{\text{policy}} = -\frac{1}{n} \sum_{i=1}^{n} \left(\frac{1}{|x_i|} \sum_{t=1}^{|x_i|} \log \pi_\theta(x_{i,t} \mid x_{i,<t})\right) \cdot A(x_i)$$

### 5.4 KL Divergence Penalty

To prevent the policy from deviating too far from the reference model:

$$\mathcal{L}_{\text{KL}} = \beta \cdot D_{\text{KL}}(\pi_{\text{ref}} \| \pi_\theta)$$

The KL divergence is estimated using the k3 estimator:

$$\hat{D}_{\text{KL}} = \mathbb{E}_{x \sim \pi_\theta}\left[\frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} - \log \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} - 1\right]$$

where $\beta = 0.04$ (default).

### 5.5 Total GRPO Loss

$$\mathcal{L}_{\text{GRPO}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{KL}}$$

### 5.6 Training Pipeline

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

## 6. Hyperparameter Summary

### 6.1 ESM-2 Pseudo-Likelihood Reward

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Model | - | `esm2_t12_35M_UR50D` | ESM-2 variant |
| Normalize | - | `True` | Min-max normalization |
| Temperature | $\tau$ | 1.0 | Reward sharpening |
| Min observations | $N$ | 10 | Before normalization starts |
| Min sequence length | - | 3 | Penalty for shorter |

### 6.2 Composite Reward

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

### 6.3 Diversity (GRPO-D)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Diversity weight | $\omega_D$ | 0.15 | Combined reward interpolation |
| AA frequency weight | $\lambda_{\text{AA}}$ | 0.7 | Diversity component weight |
| Sequence weight | $\lambda_{\text{seq}}$ | 0.3 | Diversity component weight |
| Max compare | $m$ | 50 | Reference peptides for Levenshtein |

### 6.4 GRPO Training

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Learning rate | $\eta$ | $3 \times 10^{-4}$ | Adam optimizer |
| KL coefficient | $\beta$ | 0.04 | KL penalty weight |
| Batch size | - | 16 | Prompts per iteration |
| Generations per prompt | - | 8 | Peptides per prompt |
| Max gradient norm | - | 1.0 | Gradient clipping |
| Buffer size | - | 500 | Experience buffer capacity |

---

## 7. References

### 7.1 ESM-2

Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.

### 7.2 Pseudo-Likelihood Scoring

Salazar, J., et al. (2020). Masked language model scoring. *Proceedings of ACL*, 2699-2712.

### 7.3 GRPO

Shao, Z., et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*.

### 7.4 GFlowNets

Bengio, E., et al. (2021). Flow network based generative models for non-iterative diverse candidate generation. *NeurIPS*, 34, 27381-27394.

---

## Appendix A: Code References

| Component | Source File | Key Lines |
|-----------|-------------|-----------|
| ESM-2 Pseudo-Likelihood | `gflownet_peptide/rewards/esm2_reward.py` | 55-221 |
| ESM Backbone | `gflownet_peptide/models/reward_model.py` | 18-101 |
| Reward Head (MLP) | `gflownet_peptide/models/reward_model.py` | 104-162 |
| Stability Reward | `gflownet_peptide/models/reward_model.py` | 165-202 |
| Binding Reward | `gflownet_peptide/models/reward_model.py` | 205-235 |
| Naturalness Reward | `gflownet_peptide/models/reward_model.py` | 238-285 |
| Composite Reward | `gflownet_peptide/models/reward_model.py` | 288-410 |
| AA Frequency Diversity | `gflownet_peptide/training/diversity.py` | 19-61 |
| Sequence Dissimilarity | `gflownet_peptide/training/diversity.py` | 64-117 |
| Combined Diversity | `gflownet_peptide/training/diversity.py` | 120-176 |
| GRPO Advantages | `gflownet_peptide/training/grpo_trainer.py` | 101-149 |
| GRPO Loss | `gflownet_peptide/training/grpo_trainer.py` | 211-306 |
| Training Loop | `scripts/train_grpo.py` | 246-329 |

---

*Document generated for scientific publication. Last updated: 2024*
