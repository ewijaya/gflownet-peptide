# Product Requirements Document: GFlowNet for Diverse Therapeutic Peptide Generation

**Project Code:** Idea F / P16 (proposed)
**Version:** 1.0
**Date:** December 22, 2025
**Author:** Computational ISDD Team
**Status:** Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Goals, Non-Goals, Success Criteria](#3-goals-non-goals-success-criteria)
4. [Technical Architecture](#4-technical-architecture)
5. [Implementation Phases & Phase-Gate Criteria](#5-implementation-phases--phase-gate-criteria)
6. [Data & Compute Requirements](#6-data--compute-requirements)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Risks & Mitigations](#8-risks--mitigations)
9. [Publication Strategy](#9-publication-strategy)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary

### 1.1 Overview

This project develops a GFlowNet-based peptide generation system that produces **diverse, high-quality therapeutic peptide candidates** by sampling proportionally to predicted fitness. Unlike reward-maximizing methods (GRPO/PPO), GFlowNet inherently maintains diversity without requiring explicit penalties or constraints.

### 1.2 Key Value Proposition

| Current State (P14 GRPO) | Proposed State (GFlowNet) |
|--------------------------|---------------------------|
| Optimizes for best candidate | Samples proportionally from all good candidates |
| Diversity requires explicit penalty (hacky) | Diversity is intrinsic to the objective |
| Converges to ~5 similar scaffolds | Explores full high-fitness landscape |
| Trade-off: diversity vs. quality | No trade-off: diversity by design |

### 1.3 Expected Outcomes

- **2-3× higher diversity** at equivalent quality levels compared to GRPO
- **Publishable methodology** (first GFlowNet for therapeutic peptides)
- **Complementary tool** to existing P14 pipeline for wet-lab candidate generation

### 1.4 Investment Summary

| Resource | Estimate |
|----------|----------|
| Development time | 8-10 weeks |
| Compute | 1× A100 GPU (or equivalent) |
| Personnel | 1 computational researcher |
| External dependencies | Public datasets only (FLIP, Propedia, ProteinGym) |

---

## 2. Problem Statement & Motivation

### 2.1 The Diversity Problem in Peptide Generation

**Current P14 (GRPO) behavior:**

P14 uses Group Relative Policy Optimization to maximize expected reward E[R(x)]. This objective fundamentally drives convergence:

```
Objective: max E[R(x)]
         = max ∫ P(x) R(x) dx
         → P(x) concentrates on arg max R(x)
```

**Result:** The generator converges to a narrow region of sequence space, producing ~5-20 structurally similar peptides, all variations of the same "winning" scaffold.

**Why this matters for drug discovery:**

| What GRPO Produces | What Wet-Lab Needs |
|-------------------|-------------------|
| 20 similar high-scoring peptides | 20 *diverse* high-scoring peptides |
| All from one structural family | Multiple structural families |
| One backup if lead fails | Several independent backup scaffolds |
| Optimized for predicted fitness | Robust to prediction errors |

### 2.2 Current Workaround Limitations

P14 includes a diversity penalty (GRPO-D) that explicitly penalizes similarity:

```python
reward = PEM(x) - λ * max_similarity(x, batch)
```

**Problems with this approach:**

1. **Trade-off is explicit:** Increasing λ increases diversity but decreases mean fitness
2. **Hyperparameter sensitivity:** Optimal λ varies by task, requires tuning
3. **Doesn't guarantee mode coverage:** Can still miss isolated fitness peaks
4. **Philosophically unsatisfying:** Diversity is a hack, not part of the objective

### 2.3 GFlowNet Solution

GFlowNet optimizes a fundamentally different objective:

```
Objective: P(x) ∝ R(x)
```

Instead of maximizing expected reward, GFlowNet learns to **sample sequences proportionally to their reward**:

- Peptide with R=0.9: sampled 3× more often than peptide with R=0.3
- But peptide with R=0.3 is **still sampled** (not discarded)
- All fitness peaks are discovered proportional to their height
- Diversity emerges naturally from the proportional sampling

### 2.4 Strategic Fit

| Portfolio Project | Status | Gap GFlowNet Fills |
|-------------------|--------|-------------------|
| P2 (ProtGPT2 Generator) | Active | Sequence-only, no fitness optimization |
| P14 (GRPO Generator) | Active | Mode collapse, diversity penalty needed |
| **P16 (GFlowNet Generator)** | Proposed | Inherent diversity, proportional sampling |

GFlowNet is the **natural evolution** of P14—same peptide generation goal, but with diversity built into the objective rather than patched in afterward.

---

## 3. Goals, Non-Goals, Success Criteria

### 3.1 Goals

| ID | Goal | Priority |
|----|------|----------|
| G1 | Build GFlowNet that generates peptides ∝ fitness | P0 (Must) |
| G2 | Achieve higher diversity than GRPO at equivalent quality | P0 (Must) |
| G3 | Use only public data for reward model (publishable) | P0 (Must) |
| G4 | Direct comparison: GFlowNet vs GRPO on same task | P0 (Must) |
| G5 | Modular reward system (swap in proprietary later) | P1 (Should) |
| G6 | Multi-property composite reward (stability + binding) | P1 (Should) |
| G7 | Publish at NeurIPS GenBio or equivalent venue | P2 (Could) |

### 3.2 Non-Goals

| ID | Non-Goal | Rationale |
|----|----------|-----------|
| NG1 | Novel GFlowNet training algorithm | We apply existing methods (TB/SubTB), not invent new ones |
| NG2 | Structure-conditioned generation | V1 is sequence-only; structure-aware is V2 |
| NG3 | Wet-lab validation | Out of scope; computational proof-of-concept only |
| NG4 | Replace GRPO in production | GFlowNet complements GRPO, doesn't replace it |
| NG5 | Real-time generation (<1s) | Batch generation is acceptable |

### 3.3 Success Criteria (Project-Level)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Diversity gain** | ≥2× improvement vs GRPO | Sequence diversity metric (see §7) |
| **Quality parity** | ≤5% drop in mean fitness | Mean predicted fitness score |
| **Mode coverage** | ≥3× more clusters discovered | HDBSCAN cluster count |
| **Proportionality** | R² ≥ 0.8 between log(frequency) and log(reward) | Calibration check |
| **Training stability** | <5% of runs diverge | Loss convergence rate |
| **Publication** | Accepted at workshop/journal | Peer review |

---

## 4. Technical Architecture

### 4.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GFlowNet Peptide Generation System                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        TRAINING LOOP                                │ │
│  │                                                                     │ │
│  │    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐      │ │
│  │    │  Forward     │     │  Trajectory  │     │  Backward    │      │ │
│  │    │  Policy P_F  │────►│  Sampling    │────►│  Policy P_B  │      │ │
│  │    │ (Transformer)│     │              │     │  (Uniform)   │      │ │
│  │    └──────────────┘     └──────────────┘     └──────────────┘      │ │
│  │           │                    │                    │               │ │
│  │           │                    ▼                    │               │ │
│  │           │            ┌──────────────┐             │               │ │
│  │           │            │   Terminal   │             │               │ │
│  │           │            │   Sequence   │             │               │ │
│  │           │            │     x        │             │               │ │
│  │           │            └──────┬───────┘             │               │ │
│  │           │                   │                     │               │ │
│  │           │                   ▼                     │               │ │
│  │           │            ┌──────────────┐             │               │ │
│  │           │            │   Reward     │             │               │ │
│  │           │            │   Model R(x) │             │               │ │
│  │           │            │  (Frozen)    │             │               │ │
│  │           │            └──────┬───────┘             │               │ │
│  │           │                   │                     │               │ │
│  │           ▼                   ▼                     ▼               │ │
│  │    ┌─────────────────────────────────────────────────────────┐     │ │
│  │    │              Trajectory Balance Loss                     │     │ │
│  │    │                                                          │     │ │
│  │    │  L_TB = (log Z + Σ log P_F(s_t|s_{t-1})                 │     │ │
│  │    │         - log R(x) - Σ log P_B(s_{t-1}|s_t))²           │     │ │
│  │    │                                                          │     │ │
│  │    └─────────────────────────────────────────────────────────┘     │ │
│  │                               │                                     │ │
│  │                               ▼                                     │ │
│  │                        Gradient Update                              │ │
│  │                        (Adam optimizer)                             │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        INFERENCE                                    │ │
│  │                                                                     │ │
│  │    Sample from P_F autoregressively → Diverse peptide set          │ │
│  │    Optionally with temperature β: P(x) ∝ R(x)^β                    │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Specifications

#### 4.2.1 Forward Policy (P_F)

| Attribute | Specification |
|-----------|---------------|
| Architecture | Transformer Encoder (causal) |
| Layers | 4-6 |
| Hidden dim | 256-512 |
| Heads | 8 |
| Vocab | 20 amino acids + START + STOP |
| Output | Softmax over next amino acid |
| Pre-training | Optional: UniRef50 peptides (<50 AA) |

```python
class ForwardPolicy(nn.Module):
    def __init__(self, vocab_size=22, d_model=256, n_layers=4, n_heads=8):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=512),
            num_layers=n_layers
        )
        self.action_head = nn.Linear(d_model, vocab_size)

    def forward(self, partial_seq):  # [B, L]
        x = self.embedding(partial_seq) + self.pos_enc(partial_seq.size(1))
        x = self.transformer(x, mask=generate_causal_mask(x.size(1)))
        logits = self.action_head(x[:, -1, :])  # [B, vocab_size]
        return logits
```

#### 4.2.2 Backward Policy (P_B)

| Attribute | Specification |
|-----------|---------------|
| Type | Uniform (for linear generation) |
| Rationale | Only one parent state exists (remove last AA) |
| log P_B | Always 0 (log 1 = 0) |

For linear autoregressive generation, the backward policy is deterministic: the only parent of state `[A, B, C]` is `[A, B]`. Thus P_B = 1 for the correct parent.

#### 4.2.3 Reward Model (R)

| Attribute | Specification |
|-----------|---------------|
| Backbone | ESM-2 (esm2_t33_650M or esm2_t12_35M) |
| Head | MLP (1280 → 256 → 1) or Linear |
| Training data | FLIP Stability + Propedia Binding |
| Output transform | exp(·) or softplus(·) to ensure R ≥ 0 |
| Inference | Frozen during GFlowNet training |

**Composite Reward:**

```python
class CompositeReward(nn.Module):
    def __init__(self):
        self.stability = StabilityPredictor()   # FLIP-trained
        self.binding = BindingPredictor()       # Propedia-trained
        self.lm = PeptideLM()                   # ProtGPT2 or ESM
        self.weights = {'stability': 1.0, 'binding': 1.0, 'naturalness': 0.5}

    def forward(self, sequence):
        s = self.stability(sequence)        # [0, ∞)
        b = self.binding(sequence)          # [0, ∞)
        n = torch.exp(-self.lm.perplexity(sequence) / 10)  # (0, 1]

        return (s ** self.weights['stability'] *
                b ** self.weights['binding'] *
                n ** self.weights['naturalness'])
```

#### 4.2.4 Partition Function (Z)

| Attribute | Specification |
|-----------|---------------|
| Representation | Learnable scalar (log Z) |
| Initialization | log Z = 0 |
| Optimization | Jointly with P_F via Adam |

### 4.3 State Space

| Element | Definition |
|---------|------------|
| State | Partial sequence s = [a₁, a₂, ..., aₜ], aᵢ ∈ {20 AA} |
| Initial state | s₀ = [START] |
| Terminal state | Complete sequence of length L (or until STOP token) |
| Actions | Append one of 20 amino acids (or STOP) |
| Trajectory | τ = (s₀ → s₁ → ... → sₙ = x) |

### 4.4 Training Objective

**Trajectory Balance (TB) Loss:**

$$\mathcal{L}_{TB} = \left( \log Z + \sum_{t=0}^{n-1} \log P_F(s_{t+1}|s_t) - \log R(x) - \sum_{t=1}^{n} \log P_B(s_{t-1}|s_t) \right)^2$$

For uniform P_B, this simplifies to:

$$\mathcal{L}_{TB} = \left( \log Z + \sum_{t=0}^{n-1} \log P_F(s_{t+1}|s_t) - \log R(x) \right)^2$$

**Alternative: Sub-Trajectory Balance (SubTB)**

More stable for long sequences; computes loss on sub-trajectories rather than full trajectories.

---

## 5. Implementation Phases & Phase-Gate Criteria

### Phase 0: Validation (Is GFlowNet Needed?)

**Duration:** 1 week
**Objective:** Confirm GRPO-D has genuine diversity limitations that GFlowNet can solve

#### 5.0.1 Activities

| ID | Activity | Output |
|----|----------|--------|
| 0.1 | Generate 1000 peptides with GRPO-D (existing P14) | Peptide set |
| 0.2 | Compute diversity metrics (see §7) | Metrics report |
| 0.3 | Cluster sequences (UMAP + HDBSCAN) | Cluster visualization |
| 0.4 | Compare to random high-fitness sampling | Baseline comparison |
| 0.5 | Stakeholder interview: diversity needs | Requirements doc |

#### 5.0.2 Success Criteria (Phase Gate)

| Criterion | Target | Go/No-Go |
|-----------|--------|----------|
| GRPO-D cluster count | <10 distinct clusters | Go if <10 (diversity problem exists) |
| Mode coverage gap | Misses >30% of known fitness peaks | Go if >30% |
| Stakeholder need | Diversity explicitly requested | Go if yes |

**Decision:** If GRPO-D already achieves sufficient diversity, stop here. GFlowNet is not needed.

---

### Phase 1: Reward Model Development

**Duration:** 2 weeks
**Objective:** Train and validate public reward model suitable for GFlowNet

#### 5.1.1 Activities

| ID | Activity | Output |
|----|----------|--------|
| 1.1 | Download FLIP dataset (Stability task) | Raw data |
| 1.2 | Download Propedia dataset | Raw data |
| 1.3 | Preprocess: filter to peptide lengths 10-50 AA | Cleaned data |
| 1.4 | Train ESM-2 → Stability predictor | Model checkpoint |
| 1.5 | Train ESM-2 → Binding predictor | Model checkpoint |
| 1.6 | Implement composite reward | Reward function |
| 1.7 | Validate on held-out test set | Validation metrics |
| 1.8 | Implement non-negative transform (exp/softplus) | Final reward |

#### 5.1.2 Success Criteria (Phase Gate)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Stability predictor R² | ≥0.6 on FLIP test set | Sklearn r2_score |
| Binding predictor R² | ≥0.5 on Propedia test set | Sklearn r2_score |
| Reward non-negativity | 100% outputs ≥ 0 | min(R) ≥ 0 |
| Reward range | Spread across [0, max] | std(R) > 0.1 |
| Inference speed | <100ms per sequence | Timing benchmark |

**Decision:** If reward model R² < 0.5, revisit data preprocessing or model architecture before proceeding.

---

### Phase 2: GFlowNet Core Implementation

**Duration:** 2 weeks
**Objective:** Implement GFlowNet components and training loop

#### 5.2.1 Activities

| ID | Activity | Output |
|----|----------|--------|
| 2.1 | Set up environment (torch, transformers, esm) | requirements.txt |
| 2.2 | Implement ForwardPolicy (Transformer) | forward_policy.py |
| 2.3 | Implement BackwardPolicy (Uniform) | backward_policy.py |
| 2.4 | Implement trajectory sampling | sampler.py |
| 2.5 | Implement TB loss computation | loss.py |
| 2.6 | Implement training loop | trainer.py |
| 2.7 | Add logging (wandb) | Integrated |
| 2.8 | Unit tests for each component | tests/ |

#### 5.2.2 Code Structure

```
gflownet_peptide/
├── models/
│   ├── forward_policy.py      # P_F transformer
│   ├── backward_policy.py     # P_B (uniform)
│   └── reward_model.py        # Composite reward
├── training/
│   ├── sampler.py             # Trajectory sampling
│   ├── loss.py                # TB/SubTB loss
│   └── trainer.py             # Training loop
├── evaluation/
│   ├── metrics.py             # Diversity, quality metrics
│   └── visualize.py           # UMAP, plots
├── data/
│   ├── flip.py                # FLIP data loading
│   └── propedia.py            # Propedia data loading
├── configs/
│   └── default.yaml           # Hyperparameters
├── scripts/
│   ├── train_reward.py        # Phase 1
│   ├── train_gflownet.py      # Phase 3
│   └── evaluate.py            # Phase 4
├── tests/
│   └── test_*.py              # Unit tests
└── requirements.txt
```

#### 5.2.3 Success Criteria (Phase Gate)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Unit test coverage | ≥80% | pytest-cov |
| P_F forward pass | Correct shapes | Manual check |
| Trajectory sampling | Valid sequences | Decode and validate |
| TB loss computation | Finite, non-negative | Check for NaN/Inf |
| Training step runs | No errors | Single batch test |

**Decision:** All unit tests pass, single training step completes without error.

---

### Phase 3: Training & Hyperparameter Tuning

**Duration:** 2 weeks
**Objective:** Train GFlowNet to convergence and tune hyperparameters

#### 5.3.1 Activities

| ID | Activity | Output |
|----|----------|--------|
| 3.1 | Initial training run (default hyperparameters) | Baseline model |
| 3.2 | Monitor: loss, log_Z, sample diversity, sample quality | wandb dashboard |
| 3.3 | Hyperparameter sweep (learning rate, batch size, model size) | Best config |
| 3.4 | Address training instabilities (if any) | Fixes applied |
| 3.5 | Train to convergence with best config | Final model |
| 3.6 | Generate 10,000 peptides from trained model | Peptide dataset |

#### 5.3.2 Hyperparameter Grid

| Hyperparameter | Values to Try |
|----------------|---------------|
| Learning rate | 1e-4, 3e-4, 1e-3 |
| Batch size | 32, 64, 128 |
| P_F layers | 4, 6 |
| P_F hidden dim | 256, 512 |
| Max sequence length | 20, 30, 40 |
| Training steps | 10K, 50K, 100K |

#### 5.3.3 Success Criteria (Phase Gate)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Training loss | Converged (plateaued) | Loss curve |
| log_Z stability | No divergence | log_Z vs step |
| Sample validity | ≥99% valid AA sequences | Validation check |
| Sample diversity | Higher than random init | Diversity metric |
| Sample quality | Mean R > 0.5 | Reward evaluation |

**Decision:** Training converges, samples are valid, show improvement over random.

---

### Phase 4: Evaluation & GRPO Comparison

**Duration:** 2 weeks
**Objective:** Rigorously compare GFlowNet to GRPO on identical task

#### 5.4.1 Activities

| ID | Activity | Output |
|----|----------|--------|
| 4.1 | Train GRPO baseline on same reward model | GRPO model |
| 4.2 | Generate 1000 peptides from GFlowNet | GFlowNet samples |
| 4.3 | Generate 1000 peptides from GRPO | GRPO samples |
| 4.4 | Compute all metrics (see §7) for both | Metrics table |
| 4.5 | Proportionality check: P(x) vs R(x) | Calibration plot |
| 4.6 | Cluster analysis: mode coverage | UMAP + clusters |
| 4.7 | Statistical significance tests | p-values |
| 4.8 | Ablation studies (reward components, temperature) | Ablation table |

#### 5.4.2 Experiment Design

| Experiment | Question | Method |
|------------|----------|--------|
| **Exp 1: Diversity** | Does GFlowNet produce more diverse peptides? | Compare diversity metrics |
| **Exp 2: Quality** | Is quality comparable? | Compare mean/max reward |
| **Exp 3: Mode Coverage** | Does GFlowNet find more modes? | Count clusters |
| **Exp 4: Proportionality** | Does sampling match R(x)? | Bin by R, compare frequency |
| **Exp 5: Pareto Front** | Multi-objective coverage? | Hypervolume metric |
| **Exp 6: Temperature** | Effect of R^β? | Sweep β ∈ [0.5, 2.0] |

#### 5.4.3 Success Criteria (Phase Gate)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Diversity gain | ≥2× over GRPO | Sequence diversity |
| Quality parity | ≤5% drop in mean R | Mean reward |
| Cluster count | ≥3× over GRPO | HDBSCAN clusters |
| Proportionality | R² ≥ 0.8 | log(freq) vs log(R) |
| Statistical significance | p < 0.05 | Wilcoxon signed-rank |

**Decision:** If GFlowNet shows ≥2× diversity with ≤5% quality drop, project is successful.

---

### Phase 5: Documentation & Publication

**Duration:** 2 weeks
**Objective:** Prepare codebase for public release and write paper

#### 5.5.1 Activities

| ID | Activity | Output |
|----|----------|--------|
| 5.1 | Code cleanup and documentation | Docstrings, README |
| 5.2 | Sanitization check (no proprietary refs) | Audit report |
| 5.3 | Create reproducible experiments | run_all.sh |
| 5.4 | Write paper draft | paper.tex |
| 5.5 | Prepare figures | figures/ |
| 5.6 | Internal review | Feedback |
| 5.7 | Submit to venue | Submission |

#### 5.5.2 Success Criteria (Phase Gate)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Code passes sanitization | No proprietary terms | grep audit |
| README completeness | Covers install, train, evaluate | Checklist |
| Reproducibility | Fresh clone → results match | Test run |
| Paper completeness | All sections filled | Checklist |
| Submission | Accepted at venue | Review outcome |

---

## 6. Data & Compute Requirements

### 6.1 Datasets

| Dataset | Source | Size | Use |
|---------|--------|------|-----|
| **FLIP Stability** | [FLIP benchmark](https://benchmark.protein.properties/) | 53K sequences | Stability reward |
| **FLIP GB1** | FLIP benchmark | 150K sequences | Binding reward (optional) |
| **Propedia** | [propedia.org](http://propedia.org) | 19K complexes | Binding reward |
| **UniRef50** | UniProt | 50M sequences | Pre-training P_F (optional) |
| **ProteinGym** | [proteingym.org](https://proteingym.org) | 2.5M variants | Validation / analysis |

### 6.2 Data Processing Pipeline

```
FLIP Stability (53K)
    │
    ▼
Filter: length 10-50 AA
    │
    ▼
Split: 80% train / 10% val / 10% test
    │
    ▼
Tokenize: AA → integer
    │
    ▼
Ready for ESM-2 encoding
```

### 6.3 Compute Requirements

| Component | GPU Memory | Time Estimate |
|-----------|------------|---------------|
| Reward model training | 16GB | 2-4 hours |
| GFlowNet training (100K steps) | 16GB | 8-12 hours |
| Sample generation (10K seqs) | 8GB | 1-2 hours |
| Evaluation | 8GB | 1 hour |

**Recommended Setup:** 1× NVIDIA A100 40GB or 1× NVIDIA A6000 48GB

**Alternative:** 1× NVIDIA RTX 3090 24GB (slower, but sufficient)

### 6.4 Storage

| Item | Size |
|------|------|
| Raw datasets | ~10 GB |
| ESM-2 model (t33_650M) | 2.5 GB |
| Trained reward models | ~500 MB |
| Trained GFlowNet | ~500 MB |
| Generated samples | ~100 MB |
| Logs and checkpoints | ~5 GB |
| **Total** | ~20 GB |

---

## 7. Evaluation Framework

### 7.1 Metrics

#### 7.1.1 Diversity Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Sequence Diversity** | 1 - mean pairwise sequence identity | ≥0.7 |
| **Embedding Diversity** | Mean pairwise cosine distance in ESM space | ≥0.5 |
| **Cluster Count** | Number of HDBSCAN clusters | ≥10 |
| **Mode Coverage** | % of known fitness peaks discovered | ≥0.8 |
| **Unique Sequences** | # unique / # total generated | ≥0.95 |

#### 7.1.2 Quality Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Mean Reward** | Average R(x) over generated set | ≥0.6 |
| **Max Reward** | Highest R(x) in generated set | ≥0.9 |
| **Top-10% Mean** | Mean of top 10% by reward | ≥0.8 |
| **Validity** | % of valid amino acid sequences | ≥0.99 |

#### 7.1.3 Calibration Metrics (GFlowNet-Specific)

| Metric | Definition | Target |
|--------|------------|--------|
| **Proportionality R²** | Correlation: log(frequency) vs log(R) | ≥0.8 |
| **Bin Accuracy** | Frequency matches R in each reward bin | KL < 0.1 |

#### 7.1.4 Multi-Objective Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Hypervolume** | Volume dominated by Pareto front | Higher is better |
| **Pareto Size** | # of Pareto-optimal solutions | ≥50 |
| **Spread** | Coverage along each objective axis | Full range |

### 7.2 Baselines

| Baseline | Description |
|----------|-------------|
| **GRPO** | P14-style reward maximization |
| **GRPO-D** | GRPO with diversity penalty |
| **Random** | Uniform random AA sequences |
| **Random (filtered)** | Random sequences with R > threshold |
| **Rejection Sampling** | Sample GRPO, reject similar sequences |

### 7.3 Evaluation Protocol

```python
def evaluate_generator(generator, reward_model, n_samples=1000):
    # Generate samples
    sequences = generator.sample(n_samples)

    # Compute rewards
    rewards = [reward_model(seq) for seq in sequences]

    # Quality metrics
    mean_reward = np.mean(rewards)
    max_reward = np.max(rewards)

    # Diversity metrics
    embeddings = esm_embed(sequences)
    seq_diversity = 1 - mean_pairwise_identity(sequences)
    emb_diversity = mean_pairwise_cosine_distance(embeddings)

    # Clustering
    umap_coords = UMAP().fit_transform(embeddings)
    clusters = HDBSCAN().fit_predict(umap_coords)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    # Proportionality (GFlowNet only)
    freq_by_reward = bin_and_count(sequences, rewards)
    proportionality_r2 = compute_r2(freq_by_reward)

    return {
        'mean_reward': mean_reward,
        'max_reward': max_reward,
        'seq_diversity': seq_diversity,
        'emb_diversity': emb_diversity,
        'n_clusters': n_clusters,
        'proportionality_r2': proportionality_r2
    }
```

---

## 8. Risks & Mitigations

### 8.1 Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|------------|--------|------------|
| R1 | GRPO-D already sufficient (GFlowNet not needed) | Medium | High | Phase 0 validation before investment |
| R2 | Reward model doesn't generalize to peptides | High | High | Validate on peptide-specific data; use FLIP + Propedia |
| R3 | GFlowNet training unstable (log_Z diverges) | Medium | Medium | Use SubTB; gradient clipping; smaller learning rate |
| R4 | Proportional sampling not optimal for task | Low | Medium | Temperature parameter (R^β) for control |
| R5 | Scooped by Bengio lab | Medium | Medium | Execute quickly; differentiate on GRPO comparison |
| R6 | Wet-lab prefers exploitation over exploration | Medium | Low | Position as complementary tool, not replacement |
| R7 | Compute constraints | Low | Low | Use smaller ESM model; cloud GPU if needed |
| R8 | Public reward ≠ therapeutic value | High | Medium | Accept for publication; use PEM internally |

### 8.2 Contingency Plans

| Trigger | Contingency |
|---------|-------------|
| Phase 0 shows GRPO-D sufficient | Stop project; document findings |
| Reward model R² < 0.4 | Switch to self-supervised reward (perplexity + pLDDT) |
| Training diverges repeatedly | Switch from TB to SubTB; use torchgfn library |
| GFlowNet underperforms GRPO | Publish as negative result with analysis |
| Scooped before submission | Pivot to "comparison study" framing |

---

## 9. Publication Strategy

### 9.1 Positioning

**Primary Contribution:** First empirical comparison of GFlowNet vs GRPO for therapeutic peptide generation

**Secondary Contributions:**
- First GFlowNet for therapeutic peptides
- Multi-property composite reward design
- Analysis of when proportional sampling helps

### 9.2 Target Venues (Priority Order)

| Venue | Type | Deadline | Fit |
|-------|------|----------|-----|
| **NeurIPS GenBio Workshop** | Workshop | Oct 2026 | Excellent (application papers welcome) |
| **MLCB (ICML Workshop)** | Workshop | May 2026 | Excellent |
| **Nature Machine Intelligence** | Journal | Rolling | Good (application + clinical angle) |
| **Bioinformatics** | Journal | Rolling | Good (lower novelty bar) |
| **ICLR** | Conference | Sep 2026 | Medium (need stronger method delta) |

### 9.3 Paper Outline

```
Title: Diverse Therapeutic Peptide Generation with GFlowNet:
       A Comparison to Reward-Maximizing Reinforcement Learning

1. Introduction
   - Peptide generation challenge
   - Diversity problem in RL
   - GFlowNet solution

2. Background
   - GFlowNet fundamentals
   - GRPO for peptide generation

3. Methods
   - GFlowNet architecture for peptides
   - Reward model design
   - Training procedure

4. Experiments
   - Setup: same reward, same data
   - Exp 1: Diversity comparison
   - Exp 2: Quality comparison
   - Exp 3: Mode coverage
   - Exp 4: Proportionality verification
   - Exp 5: Ablations

5. Results
   - GFlowNet achieves 2-3× diversity
   - Quality parity maintained
   - Proportional sampling verified

6. Discussion
   - When to use GFlowNet vs GRPO
   - Limitations
   - Future work

7. Conclusion
```

### 9.4 Novelty Claims (Defensible)

| Claim | Evidence |
|-------|----------|
| First GFlowNet for therapeutic peptides | Literature search (see prior art analysis) |
| First GFlowNet vs GRPO comparison for peptides | Unique to our P14 baseline |
| Multi-property reward design for GFlowNet | Novel combination |

### 9.5 Sanitization Checklist

| Private (This Workspace) | Public (Paper/Code) |
|--------------------------|---------------------|
| StemRIM, ISDD | "biopharma R&D" |
| LRP1, Redasemtide | "therapeutic target" |
| PEM | "efficacy metric" (or use only public reward) |
| Project numbers (P14, P16) | "baseline RL generator", "proposed method" |

---

## 10. Appendices

### 10.1 Glossary

| Term | Definition |
|------|------------|
| **GFlowNet** | Generative Flow Network: learns to sample x with P(x) ∝ R(x) |
| **GRPO** | Group Relative Policy Optimization: RL method maximizing E[R(x)] |
| **TB Loss** | Trajectory Balance: GFlowNet training objective |
| **SubTB** | Sub-Trajectory Balance: more stable TB variant |
| **P_F** | Forward policy: probability of next action given state |
| **P_B** | Backward policy: probability of previous state given current |
| **Z** | Partition function: total flow (normalizing constant) |
| **FLIP** | Fitness Landscape Inference for Proteins: benchmark dataset |
| **ESM-2** | Evolutionary Scale Modeling: protein language model |

### 10.2 Key References

1. Bengio et al. (2021). "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation." NeurIPS.
2. Malkin et al. (2022). "Trajectory Balance: Improved Credit Assignment in GFlowNets." NeurIPS.
3. Jain et al. (2022). "Biological Sequence Design with GFlowNets." ICML.
4. Notin et al. (2023). "ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction." NeurIPS.

### 10.3 Checkpoint Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Phase 0 complete | Go/no-go decision |
| 3 | Phase 1 complete | Validated reward model |
| 5 | Phase 2 complete | Working GFlowNet code |
| 7 | Phase 3 complete | Trained GFlowNet model |
| 9 | Phase 4 complete | Comparison results |
| 11 | Phase 5 complete | Paper submitted |

### 10.4 Success Summary

**Project succeeds if:**

1. GFlowNet achieves ≥2× diversity vs GRPO at ≤5% quality drop
2. Proportional sampling is verified (R² ≥ 0.8)
3. Paper accepted at target venue

**Project fails if:**

1. Phase 0 shows GRPO-D sufficient (acceptable failure: saves resources)
2. GFlowNet underperforms GRPO on both diversity AND quality
3. Training never converges despite mitigations

---

*End of PRD*

**Document Control:**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-22 | Initial draft |
