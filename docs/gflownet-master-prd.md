# Product Requirements Document: GFlowNet for Diverse Therapeutic Peptide Generation

**Project Code:** Idea F / P16 (proposed)
**Version:** 1.1
**Date:** December 24, 2025
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

**Reward Options (Updated based on Phase 0 findings):**

| Reward Type | `--reward_type` | Description | Status |
|-------------|-----------------|-------------|--------|
| ESM-2 Pseudo-likelihood | `esm2_pll` | `R = (1/L) Σ log P(aa_i \| context)` | ⚠️ Use with caution - can reward repetitive sequences |
| Improved (Entropy-gated) | `improved` | `R = R_nat × G_ent × G_len` | ✅ **Option C - Fast validation** |
| Trained Composite | `trained` | Stability predictor + Entropy gate | ✅ **Option A - Publication-ready** |
| Untrained Composite | `composite` | Random MLP heads (legacy) | ❌ Not recommended - flat rewards |

**Warning:** The original `composite` reward (untrained MLP heads) produces near-constant rewards (~0.48-0.52), insufficient for GFlowNet learning. Use `improved`, `esm2_pll`, or `trained` instead.

### Reward Decision for Publication (Updated 2025-12-26)

**Main Benchmark: Use `ImprovedReward` for both GFlowNet and GRPO-D**

| Comparison | Reward Function | Rationale |
|------------|-----------------|-----------|
| GFlowNet vs GRPO-D Improved | Both use `ImprovedReward` | Fair comparison - same reward |
| Ablation study | `CompositeReward` | Shows results hold with richer reward |

**Why `ImprovedReward` (not `CompositeReward`) for main benchmark:**
1. **Fair comparison**: Any diversity improvement is due to GFlowNet's sampling algorithm, not a different reward function
2. **Simpler story**: One reward, two algorithms - isolates the algorithm contribution
3. **Reviewer-friendly**: Avoids questions about whether improvements come from reward engineering

**`CompositeReward` (with trained stability predictor) is available for:**
- Ablation studies showing GFlowNet works with data-driven rewards
- Future work extending to multi-objective optimization
- Post-publication extensions with binding predictor

See `docs/reward_formulation.md` for full mathematical specification of all reward types.

**ImprovedReward (Primary - for benchmark):**

```python
class ImprovedReward:
    """Used for main GFlowNet vs GRPO-D comparison."""
    def __call__(self, sequence):
        # Entropy gate: penalizes low-complexity sequences
        entropy_gate = sigmoid((entropy(seq) - threshold) * sharpness)
        # Length gate: penalizes too-short sequences
        length_gate = sigmoid((len(seq) - min_length) * length_sharpness)
        # Naturalness: ESM-2 embedding similarity to natural proteins
        naturalness = embedding_score(sequence)

        return entropy_gate * length_gate * naturalness
```

**CompositeReward (Secondary - for ablation):**

```python
class CompositeReward:
    """Used for ablation studies with trained stability predictor."""
    def __call__(self, sequence):
        # Reuses ImprovedReward components
        entropy_gate = self.improved_reward.entropy_gate(sequence)
        naturalness = self.improved_reward.naturalness(sequence)
        # Adds trained stability prediction (R²=0.65 on FLIP)
        stability = self.stability_predictor(sequence)

        return entropy_gate * stability * naturalness
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

### Phase -1: Data Acquisition & Infrastructure

**Duration:** 1 week (prerequisite)
**Objective:** Download, validate, and prepare all datasets and infrastructure before development begins

#### 5.-1.1 Dataset Inventory

| Dataset | Source | Size | Format | Required | Purpose |
|---------|--------|------|--------|----------|---------|
| **FLIP Stability** | [benchmark.protein.properties](https://benchmark.protein.properties/) | 53K sequences | CSV | Yes | Stability reward model |
| **FLIP GB1** | [benchmark.protein.properties](https://benchmark.protein.properties/) | 150K sequences | CSV | Optional | Binding fitness (alternative) |
| **Propedia** | [propedia.org](http://bioinfo.dcc.ufmg.br/propedia/) | 19K complexes | CSV/JSON | Yes | Binding reward model |
| **ESM-2 Models** | `fair-esm` package | 0.5-2.5 GB | PyTorch | Yes | Protein embeddings |
| **ProteinGym** | [proteingym.org](https://proteingym.org/) | 2.5M variants | CSV | Optional | Validation/benchmarking |

#### 5.-1.2 Activities

| ID | Activity | Output |
|----|----------|--------|
| -1.1 | Set up data directory structure (`data/flip/`, `data/propedia/`) | Directory tree |
| -1.2 | Download FLIP Stability dataset | `data/flip/stability/` |
| -1.3 | Download FLIP GB1 dataset (optional) | `data/flip/gb1/` |
| -1.4 | Download Propedia binding data | `data/propedia/` |
| -1.5 | Verify ESM-2 model download via `fair-esm` | Model loads successfully |
| -1.6 | Implement data loader: `gflownet_peptide/data/flip.py` | Working module |
| -1.7 | Implement data loader: `gflownet_peptide/data/propedia.py` | Working module |
| -1.8 | Create data validation script | `scripts/validate_data.py` |
| -1.9 | Run full data pipeline test | All tests pass |

#### 5.-1.3 Download Procedures

**FLIP Benchmark:**
```bash
# Option 1: Direct download from benchmark website
wget https://benchmark.protein.properties/downloads/stability.csv -O data/flip/stability.csv

# Option 2: Using the FLIP Python package
pip install flip-benchmark
python -c "from flip import load_dataset; load_dataset('stability', save_dir='data/flip/')"
```

**Propedia:**
```bash
# Download from Propedia website (requires manual navigation or API)
# Expected files: propedia_peptides.csv with columns: sequence, binding_affinity
wget http://bioinfo.dcc.ufmg.br/propedia/download/propedia_v2.csv -O data/propedia/propedia.csv
```

**ESM-2 (Auto-download on first use):**
```python
import esm
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()  # Downloads automatically
```

#### 5.-1.4 Data Format Specifications

**FLIP Stability Format:**
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| sequence | string | Amino acid sequence (uppercase) | `MKFLILFLPFAS` |
| fitness | float | ΔΔG stability score | `-1.23` |

**Propedia Format:**
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| peptide_sequence | string | Peptide amino acid sequence | `RKKRRQRRR` |
| binding_affinity | float | Binding strength (pKd or similar) | `7.5` |
| target_protein | string | Target protein ID (optional) | `P12345` |

**Amino Acid Vocabulary:**
- 20 standard amino acids: `ACDEFGHIKLMNPQRSTVWY`
- Special tokens: START (20), STOP (21), PAD (22)

#### 5.-1.5 Preprocessing Requirements

| Requirement | Specification |
|-------------|---------------|
| Length filter | 10-50 amino acids |
| Sequence validation | Only canonical 20 AA allowed |
| Label normalization | Zero-mean, unit-variance for regression |
| Train/Val/Test split | 80% / 10% / 10% |
| Deduplication | Remove exact sequence duplicates |

#### 5.-1.6 Data Loading Infrastructure

**Files to Create:**

1. `gflownet_peptide/data/__init__.py`:
```python
from .flip import load_flip_stability, load_flip_gb1
from .propedia import load_propedia
```

2. `gflownet_peptide/data/flip.py`:
```python
def load_flip_stability(data_path: str) -> tuple[list[str], np.ndarray]:
    """Load FLIP stability task. Returns (sequences, labels)."""

def load_flip_gb1(data_path: str) -> tuple[list[str], np.ndarray]:
    """Load FLIP GB1 binding task. Returns (sequences, labels)."""
```

3. `gflownet_peptide/data/propedia.py`:
```python
def load_propedia(data_path: str) -> tuple[list[str], np.ndarray]:
    """Load Propedia binding data. Returns (sequences, labels)."""
```

#### 5.-1.7 Storage Requirements

| Component | Size |
|-----------|------|
| FLIP datasets (raw) | ~500 MB |
| Propedia dataset (raw) | ~200 MB |
| ESM-2 t12_35M model | ~500 MB |
| ESM-2 t33_650M model | ~2.5 GB |
| Processed/cached data | ~1 GB |
| **Total** | ~5 GB |

#### 5.-1.8 Success Criteria (Phase Gate)

| Criterion | Target | Measurement | Go/No-Go |
|-----------|--------|-------------|----------|
| FLIP Stability downloaded | ≥50K sequences | `wc -l data/flip/stability.csv` | Go if ≥50K |
| Propedia downloaded | ≥15K sequences | `wc -l data/propedia/propedia.csv` | Go if ≥15K |
| Sequences valid | 100% canonical AA | Validation script | Go if 100% |
| Data loaders work | No import errors | `python -c "from gflownet_peptide.data import *"` | Go if passes |
| ESM-2 loads | Forward pass succeeds | Test script | Go if passes |
| Pipeline test | End-to-end success | `pytest tests/test_data_loading.py` | Go if passes |

**Decision:** If any required dataset is unavailable or data loaders fail, resolve before proceeding to Phase 0.

---

### Phase 0: Validation (Is GFlowNet Needed?)

**Duration:** 2 weeks (expanded from 1 week due to reward investigation)
**Objective:** Confirm GRPO-D has genuine diversity limitations that GFlowNet can solve

**Note:** Phase 0 has been subdivided into 0a and 0b based on initial findings.

#### 5.0.1 Phase 0a: Initial GRPO-D Evaluation (Completed)

| ID | Activity | Output | Status |
|----|----------|--------|--------|
| 0a.1 | Train GRPO-D with ESM-2 pseudo-likelihood reward | Trained model | ✅ Complete |
| 0a.2 | Generate 128 top peptides | Peptide set | ✅ Complete |
| 0a.3 | Compute diversity metrics | Metrics report | ✅ Complete |
| 0a.4 | Cluster sequences (UMAP + HDBSCAN) | Visualization | ✅ Complete |
| 0a.5 | Analyze reward hacking | Analysis report | ✅ Complete |

**Phase 0a Findings:**

| Metric | GRPO-D Result | Baseline |
|--------|---------------|----------|
| Mean reward | 0.816 | 0.828 |
| Cluster count | 3 | 2 |
| Embedding diversity | 0.336 | 0.094 |
| **Sequences with repeats** | **97%** | - |

**Critical Finding:** ESM-2 pseudo-likelihood reward is fundamentally broken for this use case. It rewards predictability, not biological viability. Sequences like `QQQQQQQQQQQQQQQQ` receive high scores (~0.93) because each position is trivially predictable from context.

**Phase 0a Decision:** CONDITIONAL GO - Need to fix reward before determining if diversity problem is from reward or from GRPO-D.

#### 5.0.2 Phase 0b: Improved Reward Validation (In Progress)

| ID | Activity | Output | Status |
|----|----------|--------|--------|
| 0b.1 | Design improved reward (entropy gate) | `reward-design-analysis_2025-12-24.md` | ✅ Complete |
| 0b.2 | Implement `ImprovedReward` class | `improved_reward.py` | ✅ Complete |
| 0b.3 | Validate reward on known examples | `validate_reward.py` | ✅ Complete |
| 0b.4 | Re-train GRPO-D with improved reward | Trained model | ⏳ Pending |
| 0b.5 | Compare Phase 0a vs 0b results | Comparison report | ⏳ Pending |
| 0b.6 | Update go/no-go decision | Decision doc | ⏳ Pending |

**Improved Reward Design:**

The improved reward addresses ESM-2 pseudo-likelihood's reward hacking vulnerability:

```
R_improved = R_naturalness × G_entropy × G_length
```

Where:
- `R_naturalness`: ESM-2 embedding norm (protein-likeness in embedding space)
- `G_entropy`: Sigmoid gate that zeros reward for low-entropy (repetitive) sequences
- `G_length`: Sigmoid gate that zeros reward for too-short sequences

**Validation Results (Phase 0b.3):**
- R(real_peptide) > R(repetitive) for 100% of test pairs ✅
- R(homopolymer) < 0.1 for all homopolymers ✅
- R(real_peptide) > 0.5 for all real peptides ✅

See: `docs/reward_formulation.md` Section 3 for full mathematical specification.

#### 5.0.3 Success Criteria (Phase Gate)

| Criterion | Target | Go/No-Go |
|-----------|--------|----------|
| GRPO-D cluster count (with improved reward) | <15 distinct clusters | Go if <15 (diversity problem persists) |
| Repeat rate (with improved reward) | <20% sequences with 3+ AA repeats | No-Go if >20% (reward still broken) |
| Embedding diversity (with improved reward) | <0.5 | Go if <0.5 (GRPO-D still limited) |
| Quality maintained | Mean R > 0.5 | No-Go if <0.5 (reward too strict) |

**Decision Logic:**
- If improved reward + GRPO-D shows low diversity (<15 clusters, emb_div <0.5): **CLEAR GO** for GFlowNet (problem is GRPO-D, not reward)
- If improved reward + GRPO-D shows high diversity (>15 clusters, emb_div >0.5): **NO-GO** for GFlowNet (problem was reward, GRPO-D is sufficient)
- If improved reward causes quality collapse (mean R <0.5): **REVISIT** reward design

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

**Key Decision:** Both GFlowNet and GRPO-D use `ImprovedReward` for fair comparison.

#### 5.4.1 Activities

| ID | Activity | Output |
|----|----------|--------|
| 4.1 | Use existing GRPO-D Improved baseline (Phase 0b) | GRPO-D samples (128 peptides) |
| 4.2 | Generate 1000 peptides from GFlowNet using `ImprovedReward` | GFlowNet samples |
| 4.3 | Compute all metrics (see §7) for both | Metrics table |
| 4.4 | Proportionality check: P(x) vs R(x) | Calibration plot |
| 4.5 | Cluster analysis: mode coverage | UMAP + clusters |
| 4.6 | Statistical significance tests | p-values |
| 4.7 | **Ablation:** GFlowNet with `CompositeReward` | Shows results hold with trained reward |
| 4.8 | Temperature sweep (β ∈ [0.5, 2.0]) | Ablation table |

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
| **R2a** | **ESM-2 pseudo-likelihood rewards repetitive sequences** | **Confirmed** | **High** | **Use entropy-gated improved reward (Phase 0b)** |
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
| 0 | Phase -1 complete | Data ready, loaders working |
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
| 1.1 | 2025-12-24 | Updated Phase 0 with 0a/0b subdivision; Added ESM-2 reward hacking findings; Added improved reward specification; Updated risk register with R2a |
