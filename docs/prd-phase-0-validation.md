# Phase 0: Validation (Is GFlowNet Needed?) - Detailed PRD

**Generated from**: docs/gflownet-master-prd.md Section 5.0
**Date**: 2025-12-23
**Status**: Implementation Complete - Ready for Training
**Last Updated**: 2025-12-23

---

## 1. Executive Summary

- **Objective**: Confirm that GRPO-D (Group Relative Policy Optimization with Diversity penalty) has genuine diversity limitations that justify developing a GFlowNet alternative. This phase serves as a go/no-go gate to prevent investing in GFlowNet development if existing methods already achieve sufficient diversity.

- **Duration**: 1-2 weeks (includes GRPO-D training)

- **Key Deliverables**:
  - Trained GRPO-D baseline model
  - GRPO-D generated peptide set (1000 samples)
  - Comprehensive diversity metrics report
  - UMAP + HDBSCAN cluster visualization
  - Baseline comparison against random high-fitness sampling
  - Go/no-go decision document

- **Prerequisites**:
  - ESM-2 model available (used as reward function via pseudo-likelihood)
  - ProtGPT2 model available from Hugging Face (used as base policy)
  - Python environment with PyTorch, transformers, fair-esm, python-Levenshtein

---

## 2. Objectives & Scope

### 2.1 In-Scope Goals

1. **Implement and train GRPO-D baseline**: Build a self-contained GRPO-D implementation for fair comparison
2. **Quantify GRPO-D diversity limitations**: Generate peptides with trained GRPO-D and measure diversity using multiple metrics
3. **Establish baseline metrics**: Create a reference point for all future GFlowNet comparisons
4. **Visualize mode collapse**: Produce UMAP visualizations showing clustering behavior
5. **Validate problem hypothesis**: Confirm that diversity is genuinely limited (<10 clusters)
6. **Document stakeholder needs**: Capture explicit diversity requirements from wet-lab or downstream users
7. **Random baseline comparison**: Compare GRPO-D against random high-fitness sampling to understand the trade-off

### 2.2 Out-of-Scope (Deferred)

| Item | Deferred To |
|------|-------------|
| Training new GRPO model | Use existing P14 or quick replica |
| GFlowNet implementation | Phase 2 |
| Reward model development | Phase 1 (use existing or simple proxy) |
| Statistical hypothesis testing | Phase 4 |
| Publication-quality figures | Phase 5 |

### 2.3 Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| ESM-2 model | `fair-esm` package | ✅ Available (esm2_t6_8M_UR50D) |
| ProtGPT2 backbone | Hugging Face | ✅ Available (protgpt2-distilled-tiny) |
| GRPO-D implementation | New (self-contained) | ✅ Complete |
| python-Levenshtein | pip install | ✅ Installed |
| Visualization libraries | pip install | Pending (umap-learn, hdbscan) |

---

## 3. Detailed Activities

### Activity 0.0: Implement and Train GRPO-D Baseline

**Description**: Implement a self-contained GRPO-D (Group Relative Policy Optimization with Diversity penalty) baseline using ESM-2 pseudo-likelihood as the reward function.

**Status**: ✅ Implementation Complete

**Architecture Decisions**:
- **Base model**: `littleworth/protgpt2-distilled-tiny` (protein-specific, fast for testing)
- **Reward function**: ESM-2 pseudo-likelihood (replaces proprietary PEM)
- **Diversity calculation**: Two-component score (AA frequency 70% + Levenshtein 30%)
- **Advantage computation**: Group-wise normalization with diversity integration

**Implemented Files**:

| File | Purpose | Status |
|------|---------|--------|
| `gflownet_peptide/rewards/esm2_reward.py` | ESM-2 pseudo-likelihood reward | ✅ Complete |
| `gflownet_peptide/training/diversity.py` | Diversity calculation functions | ✅ Complete |
| `gflownet_peptide/models/grpo_policy.py` | PolicyValueNetwork (GPT2 + value head) | ✅ Complete |
| `gflownet_peptide/training/grpo_trainer.py` | GRPO-D training loop | ✅ Complete |
| `scripts/train_grpo.py` | CLI training script | ✅ Complete |
| `configs/grpo.yaml` | Hyperparameter configuration | ✅ Complete |

**Key Implementation Details**:

1. **ESM-2 Reward** (`gflownet_peptide/rewards/esm2_reward.py`):
```python
class ESM2Reward:
    """ESM-2 pseudo-likelihood as reward function."""
    def __init__(self, model_name="esm2_t6_8M_UR50D", device="cuda"):
        self.model, self.alphabet, self.batch_converter = _load_esm_model(model_name, device)

    def __call__(self, sequences: List[str]) -> List[float]:
        """Compute pseudo-likelihood scores (higher = more natural)."""
        # For each position, compute P(true_aa | context)
        # Return normalized score in [0, 1] range
```

2. **Diversity Calculation** (`gflownet_peptide/training/diversity.py`):
```python
def calculate_peptide_diversity(peptides: List[str], config: dict) -> List[float]:
    """Combined diversity score per peptide."""
    # Component 1: AA frequency diversity (weight=0.7)
    aa_div = calculate_aa_frequency_diversity(peptides)

    # Component 2: Levenshtein dissimilarity (weight=0.3)
    seq_div = calculate_sequence_dissimilarity(peptides)

    # Combine and normalize to [0, 1]
    combined = [0.7 * a + 0.3 * s for a, s in zip(aa_div, seq_div)]
    return min_max_normalize(combined)
```

3. **GRPO-D Advantage** (`gflownet_peptide/training/grpo_trainer.py`):
```python
def compute_grpo_advantages(rewards_by_group, diversity_by_group, config):
    diversity_weight = config.get("diversity_weight", 0.15)  # 15% diversity

    for prompt, rewards in rewards_by_group.items():
        # Combine: combined = (1 - dw) * reward + dw * diversity
        combined = [(1 - diversity_weight) * r + diversity_weight * d
                    for r, d in zip(rewards, diversity_by_group[prompt])]

        # Normalize within group
        advantages = (combined - mean) / (std + 1e-8)
```

**Configuration** (`configs/grpo.yaml`):
```yaml
model_name: "littleworth/protgpt2-distilled-tiny"
esm_model: "esm2_t6_8M_UR50D"
min_length: 10
max_length: 30
learning_rate: 3.0e-4
batch_size: 16
num_generations: 8
beta: 0.04                    # KL penalty coefficient
diversity_weight: 0.15        # Balance: 85% reward, 15% diversity
diversity_weight_aa: 0.7      # AA frequency component
diversity_weight_seq: 0.3     # Levenshtein component
total_iterations: 1000
```

**Training Commands**:
```bash
# Dry run (10 iterations, test setup)
python scripts/train_grpo.py --config configs/grpo.yaml --dry_run --no_wandb

# Full training (1000 iterations)
python scripts/train_grpo.py --config configs/grpo.yaml

# Custom configuration
python scripts/train_grpo.py --total_iterations 500 --diversity_weight 0.2
```

**Output**:
- `checkpoints/grpo/<run_name>_final.pt` - Trained model checkpoint
- `results/grpo/<run_name>_stats.csv` - Training statistics
- `results/grpo/<run_name>_peptides.csv` - Top generated peptides

---

### Activity 0.1: Generate 1000 peptides with GRPO-D

**Description**: Use the trained GRPO-D model from Activity 0.0 to generate a set of 1000 peptide sequences. This establishes the baseline we're trying to improve upon.

**Steps**:
1. Load trained GRPO-D model from Activity 0.0
2. Configure generation parameters:
   - Number of samples: 1000
   - Length range: 10-50 amino acids
   - Diversity penalty λ value (document what's used)
3. Run generation and save results
4. Validate all sequences contain only canonical amino acids

**Implementation Notes**:
- If P14 codebase is unavailable, implement a minimal GRPO with diversity penalty
- Use the same reward model that will be used for GFlowNet comparison
- Document the exact λ value used for diversity penalty
- Save generation parameters for reproducibility

**Verification**:
```bash
# Check sample count
wc -l outputs/grpo_samples.csv
# Expected: 1001 (header + 1000 samples)

# Validate sequences
python -c "
import pandas as pd
df = pd.read_csv('outputs/grpo_samples.csv')
AA = set('ACDEFGHIKLMNPQRSTVWY')
valid = all(set(seq).issubset(AA) for seq in df['sequence'])
print(f'All valid: {valid}, Count: {len(df)}')
"
```

**Output**: `outputs/grpo_samples.csv` with columns: `sequence`, `reward`, `generation_step`

---

### Activity 0.2: Compute diversity metrics

**Description**: Calculate comprehensive diversity metrics on the GRPO-D generated set to quantify the diversity problem.

**Steps**:
1. Compute pairwise sequence identity for all pairs
2. Calculate sequence diversity: 1 - mean pairwise identity
3. Generate ESM-2 embeddings for all sequences
4. Calculate embedding diversity: mean pairwise cosine distance
5. Count unique sequences
6. Summarize all metrics in a report

**Implementation Notes**:
- For 1000 sequences, pairwise computation is O(n²) = 500K pairs - manageable
- Use batch processing for ESM-2 embeddings (batch size 32-64)
- Store embeddings for reuse in clustering activity

**Code Example**:
```python
from gflownet_peptide.evaluation.metrics import (
    sequence_diversity,
    embedding_diversity,
    compute_esm_embeddings
)

# Load samples
sequences = load_sequences('outputs/grpo_samples.csv')

# Sequence diversity
seq_div = sequence_diversity(sequences)
print(f"Sequence diversity: {seq_div:.3f}")

# Embedding diversity
embeddings = compute_esm_embeddings(sequences)
emb_div = embedding_diversity(embeddings)
print(f"Embedding diversity: {emb_div:.3f}")

# Unique ratio
unique_ratio = len(set(sequences)) / len(sequences)
print(f"Unique ratio: {unique_ratio:.3f}")
```

**Verification**:
```bash
# Run metrics computation
python scripts/compute_metrics.py --input outputs/grpo_samples.csv --output outputs/grpo_metrics.json

# Check output exists and contains required fields
python -c "
import json
with open('outputs/grpo_metrics.json') as f:
    m = json.load(f)
required = ['sequence_diversity', 'embedding_diversity', 'unique_ratio']
assert all(k in m for k in required), 'Missing metrics'
print('Metrics computed successfully')
"
```

**Output**: `outputs/grpo_metrics.json` with diversity metrics

---

### Activity 0.3: Cluster sequences (UMAP + HDBSCAN)

**Description**: Visualize the distribution of generated peptides in embedding space to assess mode coverage and identify cluster collapse.

**Steps**:
1. Load ESM-2 embeddings from Activity 0.2
2. Apply UMAP dimensionality reduction (2D)
3. Run HDBSCAN clustering on UMAP coordinates
4. Create visualization with cluster coloring
5. Count number of distinct clusters (excluding noise)
6. Analyze cluster sizes and distribution

**Implementation Notes**:
- UMAP parameters: n_neighbors=15, min_dist=0.1, metric='cosine'
- HDBSCAN parameters: min_cluster_size=10, min_samples=5
- Color noise points (-1 label) in gray
- Include cluster count in plot title

**Code Example**:
```python
import umap
import hdbscan
import matplotlib.pyplot as plt
import numpy as np

# Load embeddings
embeddings = np.load('outputs/grpo_embeddings.npy')

# UMAP reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
umap_coords = reducer.fit_transform(embeddings)

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
cluster_labels = clusterer.fit_predict(umap_coords)

# Count clusters (excluding noise label -1)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of clusters: {n_clusters}")

# Visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(umap_coords[:, 0], umap_coords[:, 1],
                      c=cluster_labels, cmap='tab20', s=10, alpha=0.7)
plt.colorbar(scatter)
plt.title(f'GRPO-D Generated Peptides: {n_clusters} clusters')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('outputs/grpo_umap_clusters.png', dpi=150)
plt.close()
```

**Verification**:
```bash
# Check visualization exists
ls -la outputs/grpo_umap_clusters.png

# Verify cluster count
python -c "
import json
with open('outputs/grpo_metrics.json') as f:
    m = json.load(f)
print(f'Cluster count: {m.get(\"n_clusters\", \"not computed\")}')
"
```

**Output**:
- `outputs/grpo_umap_clusters.png` - visualization
- `outputs/grpo_cluster_labels.npy` - cluster assignments
- Updated `outputs/grpo_metrics.json` with cluster count

---

### Activity 0.4: Compare to random high-fitness sampling

**Description**: Generate a baseline set of random peptides filtered by reward threshold to understand what "random" diversity looks like at comparable quality.

**Steps**:
1. Generate 10,000 random peptide sequences (uniform AA distribution)
2. Score all sequences with the reward model
3. Filter to top 1000 by reward (or filter by threshold)
4. Compute same diversity metrics as Activity 0.2
5. Create comparative visualization
6. Document the diversity-quality trade-off

**Implementation Notes**:
- Random generation should match GRPO length distribution
- Use same reward model for fair comparison
- This baseline shows maximum achievable diversity at given quality

**Code Example**:
```python
import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'

def generate_random_peptide(min_len=10, max_len=50):
    length = np.random.randint(min_len, max_len + 1)
    return ''.join(np.random.choice(list(AA)) for _ in range(length))

# Generate candidates
n_candidates = 10000
candidates = [generate_random_peptide() for _ in range(n_candidates)]

# Score with reward model
rewards = [reward_model(seq) for seq in candidates]

# Filter to top 1000
top_indices = np.argsort(rewards)[-1000:]
random_filtered = [candidates[i] for i in top_indices]
random_rewards = [rewards[i] for i in top_indices]

# Compare mean rewards
print(f"Random filtered mean reward: {np.mean(random_rewards):.3f}")
print(f"GRPO mean reward: {grpo_mean_reward:.3f}")
```

**Verification**:
```bash
# Compare metrics
python -c "
import json
with open('outputs/grpo_metrics.json') as f:
    grpo = json.load(f)
with open('outputs/random_metrics.json') as f:
    rand = json.load(f)

print('Metric comparison:')
print(f'Sequence diversity - GRPO: {grpo[\"sequence_diversity\"]:.3f}, Random: {rand[\"sequence_diversity\"]:.3f}')
print(f'Cluster count - GRPO: {grpo[\"n_clusters\"]}, Random: {rand[\"n_clusters\"]}')
print(f'Mean reward - GRPO: {grpo[\"mean_reward\"]:.3f}, Random: {rand[\"mean_reward\"]:.3f}')
"
```

**Output**:
- `outputs/random_samples.csv` - random filtered samples
- `outputs/random_metrics.json` - diversity metrics
- `outputs/comparison_plot.png` - side-by-side UMAP

---

### Activity 0.5: Stakeholder interview (diversity needs)

**Description**: Document explicit diversity requirements from downstream users (wet-lab, computational team, or decision-makers) to validate the need for GFlowNet.

**Steps**:
1. Identify key stakeholders (wet-lab lead, project manager, etc.)
2. Prepare interview questions:
   - How many distinct scaffold classes are needed for lead optimization?
   - What happens if generated peptides are too similar?
   - Current pain points with GRPO-D diversity?
   - Acceptable quality trade-off for increased diversity?
3. Conduct interviews (async or sync)
4. Document responses and synthesize requirements

**Interview Template**:
```markdown
## Stakeholder: [Name/Role]
## Date: [Date]

### Questions:

1. How many structurally distinct peptide scaffolds do you need for a typical lead optimization campaign?
   Response: ___

2. What problems arise when generated peptides are too similar?
   Response: ___

3. Have you observed mode collapse or lack of diversity with current (GRPO-D) generation?
   Response: ___

4. Would you accept a 5% drop in mean predicted fitness for 2x diversity?
   Response: ___

5. What is the minimum number of distinct clusters you would consider acceptable?
   Response: ___
```

**Verification**:
```bash
# Check requirements document exists
ls -la docs/stakeholder_requirements.md
```

**Output**: `docs/stakeholder_requirements.md` - documented diversity requirements

---

## 4. Technical Specifications

### 4.1 Architecture

This phase does not introduce new architecture. It uses:
- Existing GRPO-D model (or replica)
- ESM-2 for embeddings (esm2_t12_35M or esm2_t33_650M)
- UMAP for dimensionality reduction
- HDBSCAN for clustering

### 4.2 Code Structure

Files to create/modify for Phase 0:

```
notebooks/
└── gflownet-phase-0-validation.ipynb    # Main implementation notebook

gflownet_peptide/
├── evaluation/
│   ├── metrics.py                       # Add diversity metrics (if not present)
│   └── visualize.py                     # Add UMAP/clustering visualization

scripts/
├── compute_metrics.py                   # CLI for metrics computation
└── generate_random_baseline.py          # Random baseline generator

outputs/
├── grpo_samples.csv                     # Generated GRPO samples
├── grpo_metrics.json                    # Diversity metrics
├── grpo_embeddings.npy                  # ESM embeddings
├── grpo_umap_clusters.png               # UMAP visualization
├── random_samples.csv                   # Random baseline samples
├── random_metrics.json                  # Random baseline metrics
└── comparison_plot.png                  # Side-by-side comparison

docs/
├── stakeholder_requirements.md          # Interview results
└── phase0_decision.md                   # Go/no-go decision document
```

### 4.3 Configuration

**UMAP Parameters**:
```yaml
umap:
  n_neighbors: 15
  min_dist: 0.1
  metric: cosine
  n_components: 2
  random_state: 42
```

**HDBSCAN Parameters**:
```yaml
hdbscan:
  min_cluster_size: 10
  min_samples: 5
  cluster_selection_method: eom  # Excess of Mass
```

**Generation Parameters**:
```yaml
generation:
  n_samples: 1000
  min_length: 10
  max_length: 50
  diversity_penalty_lambda: [document actual value from GRPO-D]
```

### 4.4 Code Examples

**Diversity Metrics Module** (`gflownet_peptide/evaluation/metrics.py`):

```python
import numpy as np
from typing import List
from Bio import pairwise2

def sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two sequences."""
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    if not alignments:
        return 0.0
    alignment = alignments[0]
    matches = sum(a == b for a, b in zip(alignment[0], alignment[1]) if a != '-' and b != '-')
    return matches / max(len(seq1), len(seq2))

def sequence_diversity(sequences: List[str], n_pairs: int = 5000) -> float:
    """
    Compute sequence diversity as 1 - mean pairwise identity.
    Uses sampling for large datasets.
    """
    n = len(sequences)
    if n < 2:
        return 0.0

    # Sample pairs for efficiency
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    identities = []

    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        identities.append(sequence_identity(sequences[i], sequences[j]))

    return 1.0 - np.mean(identities)

def embedding_diversity(embeddings: np.ndarray) -> float:
    """
    Compute embedding diversity as mean pairwise cosine distance.
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Compute cosine similarities
    sim_matrix = normalized @ normalized.T

    # Extract upper triangle (excluding diagonal)
    n = len(embeddings)
    upper_indices = np.triu_indices(n, k=1)
    similarities = sim_matrix[upper_indices]

    # Return mean cosine distance
    return 1.0 - np.mean(similarities)
```

---

## 5. Success Criteria

| ID | Criterion | Target | Measurement Method | Verification Command |
|----|-----------|--------|-------------------|---------------------|
| SC0.1 | GRPO-D cluster count | <10 distinct clusters | HDBSCAN on UMAP coords | `python -c "import json; print(json.load(open('outputs/grpo_metrics.json'))['n_clusters'])"` |
| SC0.2 | Mode coverage gap | Misses >30% of fitness landscape | Compare to random baseline clusters | Manual analysis of cluster coverage |
| SC0.3 | Stakeholder need confirmed | Diversity explicitly requested | Stakeholder interviews | Review `docs/stakeholder_requirements.md` |
| SC0.4 | Sample generation complete | 1000 valid peptides | Count and validate | `wc -l outputs/grpo_samples.csv` |
| SC0.5 | All metrics computed | 5 diversity metrics | JSON contains all fields | Check `outputs/grpo_metrics.json` |
| SC0.6 | Visualization complete | UMAP plot saved | File exists | `ls outputs/grpo_umap_clusters.png` |

---

## 6. Deliverables Checklist

**GRPO-D Implementation:**
- [x] `gflownet_peptide/rewards/__init__.py` - Rewards module init
- [x] `gflownet_peptide/rewards/esm2_reward.py` - ESM-2 pseudo-likelihood reward
- [x] `gflownet_peptide/training/diversity.py` - Diversity calculation functions
- [x] `gflownet_peptide/models/grpo_policy.py` - PolicyValueNetwork
- [x] `gflownet_peptide/training/grpo_trainer.py` - GRPO-D training loop
- [x] `scripts/train_grpo.py` - Training script with CLI
- [x] `configs/grpo.yaml` - Hyperparameter configuration
- [ ] `checkpoints/grpo/<run>_final.pt` - Trained GRPO-D model (pending training)

**Analysis Outputs:**
- [ ] `results/grpo/<run>_peptides.csv` - Top GRPO-D generated peptides
- [ ] `results/grpo/<run>_stats.csv` - Training statistics
- [ ] `outputs/grpo_embeddings.npy` - ESM-2 embeddings
- [ ] `outputs/grpo_metrics.json` - Complete diversity metrics
- [ ] `outputs/grpo_umap_clusters.png` - UMAP visualization with clusters
- [ ] `outputs/random_samples.csv` - Random baseline samples
- [ ] `outputs/random_metrics.json` - Random baseline metrics
- [ ] `outputs/comparison_plot.png` - Side-by-side comparison

**Documentation:**
- [ ] `docs/stakeholder_requirements.md` - Documented diversity needs
- [ ] `docs/phase0_decision.md` - Go/no-go decision with rationale
- [ ] `notebooks/gflownet-phase-0-validation.ipynb` - Analysis notebook

**Review:**
- [ ] All success criteria verified
- [ ] Phase gate review completed

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|------------|--------|------------|-------------|
| GRPO-D implementation unavailable | Medium | High | Contact P14 team; prepare minimal replica | Implement basic GRPO with diversity penalty |
| GRPO-D achieves high diversity (no problem) | Low | High | Proceed with analysis; document findings | Stop project; document as acceptable outcome |
| ESM-2 embedding computation too slow | Low | Medium | Use smaller model (t12_35M); batch processing | Sample subset (500 sequences) |
| Clustering produces spurious results | Medium | Medium | Validate with multiple clustering algorithms | Use alternative metrics (silhouette score) |
| Stakeholders unavailable for interviews | Medium | Low | Use async communication; document assumptions | Proceed with computational analysis alone |
| Reward model quality too low | Low | High | Validate reward model before starting | Use perplexity-based proxy reward |

---

## 8. Phase Gate Review

### 8.1 Go/No-Go Criteria

**GO to Phase 1 if ALL of the following are true**:
1. GRPO-D cluster count < 10 (diversity problem confirmed)
2. Mode coverage gap > 30% compared to random baseline
3. Stakeholder explicitly confirms need for more diversity
4. All deliverables completed

**NO-GO (Stop Project) if ANY of the following are true**:
1. GRPO-D cluster count ≥ 20 (sufficient diversity already)
2. Stakeholders indicate current diversity is acceptable
3. Technical blockers prevent valid analysis

**CONDITIONAL GO if**:
- Cluster count is 10-20 (borderline): Discuss with stakeholders; proceed if they prefer GFlowNet exploration
- One minor criterion fails: Proceed with documented risk

### 8.2 Review Checklist

- [ ] All deliverables completed
- [ ] All success criteria met (or documented exceptions)
- [ ] Documentation updated
- [ ] Decision document written with clear rationale
- [ ] Stakeholder sign-off (if available)

### 8.3 Decision

**Status**: Pending | Go | No-Go (circle one)

**Decision Date**: ___________

**Cluster Count Observed**: ___________

**Stakeholder Confirmation**: Yes / No / N/A

**Notes**: ___________

---

## 9. Implementation Code

This phase uses a **hybrid approach**: scripts for GRPO-D training, notebooks for analysis.

### 9.1 Phase Format Rationale

| Use Case | Format | Rationale |
|----------|--------|-----------|
| GRPO-D implementation | Python modules | Reusable code |
| GRPO-D training | Python script | Long-running |
| Diversity analysis | Notebook | Interactive exploration |
| UMAP/clustering visualization | Notebook | Visual inspection needed |
| Baseline comparisons | Notebook | Side-by-side analysis |

### 9.2 Implementation Files (Complete)

**Modules** (production code):

| Module | Purpose | Status |
|--------|---------|--------|
| `gflownet_peptide/rewards/__init__.py` | Rewards module exports | ✅ Complete |
| `gflownet_peptide/rewards/esm2_reward.py` | ESM-2 pseudo-likelihood reward | ✅ Complete |
| `gflownet_peptide/training/diversity.py` | Diversity calculation (AA freq + Levenshtein) | ✅ Complete |
| `gflownet_peptide/models/grpo_policy.py` | PolicyValueNetwork (ProtGPT2 + value head) | ✅ Complete |
| `gflownet_peptide/training/grpo_trainer.py` | GRPO-D training loop | ✅ Complete |

**Scripts** (training):

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/train_grpo.py` | Train GRPO-D baseline model | ✅ Complete |

**Configuration**:

| File | Purpose | Status |
|------|---------|--------|
| `configs/grpo.yaml` | Hyperparameters | ✅ Complete |

**Notebooks** (analysis):

| Notebook | Purpose | Status |
|----------|---------|--------|
| `notebooks/gflownet-phase-0-validation.ipynb` | Analysis: metrics, clustering, comparison | [ ] Pending |

### 9.3 Training Script Usage

The `scripts/train_grpo.py` script supports:

**Command Line Arguments**:
```bash
python scripts/train_grpo.py --help

# Key arguments:
--config PATH           # YAML config file (default: None, uses defaults)
--model_name STR        # HuggingFace model (default: protgpt2-distilled-medium)
--esm_model STR         # ESM-2 model for reward (default: esm2_t12_35M_UR50D)
--learning_rate FLOAT   # Learning rate (default: 3e-4)
--batch_size INT        # Prompts per iteration (default: 16)
--num_generations INT   # Peptides per prompt (default: 8)
--total_iterations INT  # Training iterations (default: 1000)
--diversity_weight FLOAT # Diversity weight 0-1 (default: 0.15)
--beta FLOAT            # KL penalty coefficient (default: 0.04)
--dry_run               # Run only 10 iterations for testing
--no_wandb              # Disable wandb logging
--resume PATH           # Resume from checkpoint
```

**Example Invocations**:
```bash
# Quick test (10 iterations)
python scripts/train_grpo.py --config configs/grpo.yaml --dry_run --no_wandb

# Full training with defaults
python scripts/train_grpo.py --config configs/grpo.yaml

# Custom hyperparameters
python scripts/train_grpo.py \
    --total_iterations 2000 \
    --diversity_weight 0.2 \
    --beta 0.05

# Resume from checkpoint
python scripts/train_grpo.py --resume checkpoints/grpo/run_iter500.pt
```

**Training Output**:
```
2025-12-23 10:00:00 - INFO - Run name: 20251223_100000_grpod_it1000_dw0.15_beta0.04
2025-12-23 10:00:05 - INFO - Loading ESM-2 reward model: esm2_t6_8M_UR50D
2025-12-23 10:00:10 - INFO - Initializing GRPO-D trainer...
2025-12-23 10:00:15 - INFO - Starting training for 1000 iterations...
Iteration 50: Loss=0.2340, Mean R=0.6512, Max R=0.8234, Diversity=0.4523
Top peptides:
  1. MLKFQRSTVAGC (R=0.8234)
  2. ACDEFGHIKLMN (R=0.7891)
  ...
```

### 9.4 Notebook Structure

The analysis notebook `notebooks/gflownet-phase-0-validation.ipynb` should have:

1. **0.1 Setup and Imports**
2. **0.2 Load Training Results** (from `results/grpo/<run>_peptides.csv`)
3. **0.3 Diversity Metrics Computation**
4. **0.4 ESM-2 Embedding Generation**
5. **0.5 UMAP + HDBSCAN Clustering**
6. **0.6 Random Baseline Generation and Comparison**
7. **0.7 Summary and Go/No-Go Decision**

### 9.5 Notebook Requirements

1. **Include minimal, self-sufficient descriptions** before each code cell

2. **Configure all plots to display inline AND save to files**:
   ```python
   plt.savefig('outputs/grpo_umap_clusters.png', dpi=150, bbox_inches='tight')
   plt.show()
   ```

3. **Ensure fully executable from top to bottom** (after training script has run)

4. **Include verification cells** at the end of each major section:
   ```python
   assert os.path.exists('results/grpo/'), "Training not complete"
   print("✓ Section complete")
   ```

### 9.6 Key Hyperparameters Reference

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Base model | `model_name` | `protgpt2-distilled-tiny` | ProtGPT2 variant for policy |
| ESM-2 model | `esm_model` | `esm2_t6_8M_UR50D` | ESM-2 variant for reward |
| Learning rate | `learning_rate` | `3e-4` | AdamW learning rate |
| Batch size | `batch_size` | `16` | Prompts per iteration |
| Generations/prompt | `num_generations` | `8` | Peptides per prompt |
| KL penalty | `beta` | `0.04` | KL divergence coefficient |
| Diversity weight | `diversity_weight` | `0.15` | Combined reward balance |
| AA diversity weight | `diversity_weight_aa` | `0.7` | AA frequency component |
| Seq diversity weight | `diversity_weight_seq` | `0.3` | Levenshtein component |
| Total iterations | `total_iterations` | `1000` | Training steps |
| Min peptide length | `min_length` | `10` | Minimum AA |
| Max peptide length | `max_length` | `30` | Maximum AA |

---

## 10. Notes & References

### Master PRD Reference
- Master PRD: `docs/gflownet-master-prd.md`
- Phase 0 section: 5.0 (lines 462-486)

### Related Documents
- Phase -1 PRD: `docs/prd-phase--1-data-acquisition.md` (prerequisite)
- Stakeholder requirements template: `docs/stakeholder_requirements.md`

### External References
- UMAP documentation: https://umap-learn.readthedocs.io/
- HDBSCAN documentation: https://hdbscan.readthedocs.io/
- ESM-2 repository: https://github.com/facebookresearch/esm

### Key Metrics Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Sequence Diversity | 1 - mean(pairwise_identity) | Higher = more diverse (0-1 scale) |
| Embedding Diversity | mean(pairwise_cosine_distance) | Higher = more diverse (0-1 scale) |
| Cluster Count | HDBSCAN clusters (excl. noise) | Higher = more modes discovered |
| Mode Coverage | clusters_found / total_possible_modes | Higher = better exploration |

### Decision Logic

```
IF cluster_count < 10:
    -> Diversity problem CONFIRMED
    -> GO to Phase 1

ELIF cluster_count >= 10 AND cluster_count < 20:
    -> Borderline case
    -> Discuss with stakeholders
    -> GO if stakeholders want more diversity

ELSE (cluster_count >= 20):
    -> Sufficient diversity already exists
    -> NO-GO: Stop project
    -> Document findings as acceptable outcome
```

---

*End of Phase 0 PRD*
