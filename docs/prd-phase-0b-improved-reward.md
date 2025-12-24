# Phase 0b: Improved Reward Function - Detailed PRD

**Generated from**: docs/prd-phase-0-validation.md (extension)
**Date**: 2025-12-24
**Status**: Ready for Implementation
**Last Updated**: 2025-12-24

---

## 1. Executive Summary

- **Objective**: Fix the ESM-2 pseudo-likelihood reward hacking problem identified in Phase 0a by implementing an improved reward function with entropy-based low-complexity filtering. Re-run GRPO-D training to determine if the diversity problem persists with a proper reward.

- **Context**: Phase 0a revealed that 97% of generated peptides contain repetitive patterns (e.g., `QQQQQQQQ`, `NNNNNNNN`) because ESM-2 pseudo-likelihood rewards sequences where each amino acid is predictable from context. This is a fundamental flaw in the reward function, not the optimization method.

- **Key Question**: Does GRPO-D still have diversity limitations when using a reward function that doesn't reward degenerate sequences?

- **Key Deliverables**:
  - Improved reward function with entropy gate
  - Re-trained GRPO-D model with improved reward
  - Comparative analysis: old reward vs new reward
  - Updated go/no-go decision for GFlowNet

- **Prerequisites**:
  - Phase 0a completed (ESM-2 pseudo-likelihood training done)
  - Reward design analysis document (`docs/reward-design-analysis_2025-12-24.md`)

---

## 2. Background: Why This Phase Is Needed

### 2.1 Phase 0a Findings

| Metric | GRPO-D (Phase 0a) | Random Baseline |
|--------|-------------------|-----------------|
| Mean reward | 0.816 | 0.828 |
| Cluster count | 10 | 2 |
| Embedding diversity | 0.336 | 0.094 |
| Sequences with repeats | **97%** | - |

### 2.2 Root Cause Analysis

**ESM-2 pseudo-likelihood is broken for this use case:**

```
R_PLL(x) = (1/L) Σᵢ log P_ESM(xᵢ | context)
```

For a repetitive sequence like `QQQQQQQQQQ`:
- P(Q | QQQQQ[MASK]QQQQ) ≈ 0.99 → high log-prob
- Each position is trivially predictable → high total score

This rewards **predictability**, not **biological viability**.

### 2.3 The Fix

Replace pseudo-likelihood with a compositional reward:

```python
R_improved = R_naturalness × entropy_gate × length_gate
```

Where:
- `R_naturalness`: ESM-2 embedding norm (protein-likeness in embedding space)
- `entropy_gate`: Sigmoid gate that zeros reward for low-entropy (repetitive) sequences
- `length_gate`: Sigmoid gate that zeros reward for too-short sequences

---

## 3. Objectives & Scope

### 3.1 In-Scope Goals

1. **Implement improved reward function**: Create `ImprovedReward` class with three components
2. **Validate reward on known examples**: Test that R(real_peptides) > R(repetitive_garbage)
3. **Re-train GRPO-D**: Run same training config but with improved reward
4. **Compare results**: Analyze diversity metrics with old vs new reward
5. **Update go/no-go decision**: Determine if GFlowNet is still needed

### 3.2 Out-of-Scope (Deferred)

| Item | Deferred To |
|------|-------------|
| Structure-based rewards (pLDDT) | Phase 1 (too slow for now) |
| FLIP/Propedia trained reward | Phase 1 |
| GFlowNet implementation | Phase 2 (after reward is validated) |

### 3.3 Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| ESM-2 model | `fair-esm` | ✅ Available |
| Phase 0a results | `results/grpo/`, `outputs/` | ✅ Complete |
| Reward design document | `docs/reward-design-analysis_2025-12-24.md` | ✅ Complete |

---

## 4. Detailed Activities

### Activity 0b.1: Implement Improved Reward Function

**Description**: Create a new reward class that combines ESM-2 embedding naturalness with entropy-based filtering.

**File**: `gflownet_peptide/rewards/improved_reward.py`

**Implementation**:

```python
"""Improved reward function with entropy gate to prevent reward hacking."""

import torch
import torch.nn.functional as F
from typing import List, Union
from collections import Counter
from math import log2
import logging

logger = logging.getLogger(__name__)


class ImprovedReward:
    """
    Improved reward function for peptide generation.

    Components:
    1. ESM-2 embedding naturalness (NOT pseudo-likelihood)
    2. Sequence entropy gate (penalize low-complexity)
    3. Length gate (penalize too-short sequences)

    This addresses the reward hacking issue where ESM-2 pseudo-likelihood
    rewards repetitive sequences like QQQQQQQQ.
    """

    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        device: str = "cuda",
        # Entropy gate parameters
        entropy_threshold: float = 0.5,
        entropy_sharpness: float = 10.0,
        # Length gate parameters
        min_length: int = 10,
        length_sharpness: float = 0.5,
        # Embedding parameters
        embedding_temperature: float = 10.0,
        # Normalization
        normalize: bool = True,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.normalize = normalize

        # Gate parameters
        self.entropy_threshold = entropy_threshold
        self.entropy_sharpness = entropy_sharpness
        self.min_length = min_length
        self.length_sharpness = length_sharpness
        self.embedding_temperature = embedding_temperature

        # Load ESM-2
        self._load_model()

        # Running statistics for normalization
        self._min_score = float("inf")
        self._max_score = float("-inf")
        self._score_count = 0

    def _load_model(self):
        """Load ESM-2 model."""
        import esm

        logger.info(f"Loading ESM-2 model: {self.model_name}")

        if self.model_name == "esm2_t6_8M_UR50D":
            self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.repr_layer = 6
        elif self.model_name == "esm2_t12_35M_UR50D":
            self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.repr_layer = 12
        elif self.model_name == "esm2_t33_650M_UR50D":
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.repr_layer = 33
        else:
            raise ValueError(f"Unknown ESM model: {self.model_name}")

        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        logger.info(f"ESM-2 model loaded on {self.device}")

    def _compute_entropy(self, sequence: str) -> float:
        """
        Compute normalized Shannon entropy of amino acid distribution.

        Returns value in [0, 1] where:
        - 0 = all same amino acid (e.g., QQQQQQQQ)
        - 1 = uniform distribution of all 20 amino acids
        """
        if len(sequence) == 0:
            return 0.0

        aa_counts = Counter(sequence)
        total = len(sequence)

        entropy = 0.0
        for count in aa_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * log2(p)

        # Normalize by max possible entropy (log2(20) for 20 amino acids)
        max_entropy = log2(20)
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def _compute_entropy_gate(self, sequence: str) -> float:
        """
        Compute soft entropy gate.

        Returns ~1.0 for high-entropy (diverse) sequences.
        Returns ~0.0 for low-entropy (repetitive) sequences.
        """
        entropy = self._compute_entropy(sequence)

        # Sigmoid gate: high entropy → 1.0, low entropy → 0.0
        gate = 1.0 / (1.0 + pow(2.718281828,
            -self.entropy_sharpness * (entropy - self.entropy_threshold)))

        return gate

    def _compute_length_gate(self, sequence: str) -> float:
        """
        Compute soft length gate.

        Returns ~1.0 for sequences >= min_length.
        Returns ~0.0 for sequences << min_length.
        """
        length = len(sequence)
        gate = 1.0 / (1.0 + pow(2.718281828,
            -self.length_sharpness * (length - self.min_length)))

        return gate

    def _compute_embedding_naturalness(self, sequence: str) -> float:
        """
        Compute naturalness score based on ESM-2 embedding.

        Uses embedding norm as proxy for "how well ESM-2 represents this".
        Real proteins have consistent embedding norms; garbage has abnormal norms.
        """
        if len(sequence) < 3:
            return 0.0

        data = [("seq", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False
            )

            # Get embeddings (exclude BOS and EOS tokens)
            embeddings = results["representations"][self.repr_layer]
            seq_len = len(sequence)
            seq_embeddings = embeddings[0, 1:seq_len+1, :]  # (L, d)

            # Mean pooling
            mean_embedding = seq_embeddings.mean(dim=0)  # (d,)

            # Compute norm
            emb_norm = torch.norm(mean_embedding).item()

        # Sigmoid to map to [0, 1]
        naturalness = 1.0 / (1.0 + pow(2.718281828,
            -emb_norm / self.embedding_temperature))

        return naturalness

    def __call__(
        self,
        sequences: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """
        Compute improved reward for one or more sequences.

        Args:
            sequences: Single sequence string or list of sequences

        Returns:
            Reward score(s) in [0, 1] - higher is better
        """
        single_input = isinstance(sequences, str)
        if single_input:
            sequences = [sequences]

        scores = []
        for seq in sequences:
            # Component 1: Embedding naturalness
            naturalness = self._compute_embedding_naturalness(seq)

            # Component 2: Entropy gate
            entropy_gate = self._compute_entropy_gate(seq)

            # Component 3: Length gate
            length_gate = self._compute_length_gate(seq)

            # Combine: multiplicative (all must be good)
            score = naturalness * entropy_gate * length_gate

            # Update statistics
            self._score_count += 1
            self._min_score = min(self._min_score, score)
            self._max_score = max(self._max_score, score)

            scores.append(score)

        # Normalize if enabled and enough samples seen
        if self.normalize and self._score_count > 10:
            score_range = self._max_score - self._min_score
            if score_range > 0:
                scores = [(s - self._min_score) / score_range for s in scores]
            else:
                scores = [0.5] * len(scores)

        return scores[0] if single_input else scores

    def get_components(self, sequence: str) -> dict:
        """
        Get individual reward components for debugging.

        Returns dict with each component's value.
        """
        return {
            "naturalness": self._compute_embedding_naturalness(sequence),
            "entropy": self._compute_entropy(sequence),
            "entropy_gate": self._compute_entropy_gate(sequence),
            "length_gate": self._compute_length_gate(sequence),
            "total": self(sequence),
        }

    def reset_statistics(self):
        """Reset running normalization statistics."""
        self._min_score = float("inf")
        self._max_score = float("-inf")
        self._score_count = 0
```

**Verification**:
```bash
python -c "
from gflownet_peptide.rewards.improved_reward import ImprovedReward

reward = ImprovedReward(device='cuda')

# Test on known good peptide
good = 'MKTLLILAVVALACARSSAQAANPF'
r_good = reward(good)
print(f'Good peptide: {r_good:.4f}')
print(reward.get_components(good))

# Test on repetitive garbage
bad = 'QQQQQQQQQQQQQQQQQQQQQQQQQQ'
r_bad = reward(bad)
print(f'Repetitive: {r_bad:.4f}')
print(reward.get_components(bad))

# Verify good > bad
assert r_good > r_bad, 'Reward should prefer good peptides!'
print('✓ Reward validation passed')
"
```

**Output**: `gflownet_peptide/rewards/improved_reward.py`

---

### Activity 0b.2: Create Validation Test Suite

**Description**: Create a test script that validates the improved reward on known good and bad examples.

**File**: `scripts/validate_reward.py`

**Test Cases**:

| Category | Example | Expected Reward |
|----------|---------|-----------------|
| Real peptide (signal) | `MKTLLILAVVALACARSSAQAANPF` | High (>0.7) |
| Real peptide (antimicrobial) | `GIGKFLHSAKKFGKAFVGEIMNS` | High (>0.7) |
| Repetitive homopolymer | `QQQQQQQQQQQQQQQQQQQQQQQQ` | Low (<0.1) |
| Repetitive pattern | `AQAQAQAQAQAQAQAQAQAQAQAQ` | Low (<0.2) |
| All different AAs | `ACDEFGHIKLMNPQRSTVWY` | Moderate (0.4-0.7) |
| Too short | `ACDE` | Low (<0.2) |

**Pass Criteria**:
1. R(real_peptide) > R(repetitive) for 100% of pairs
2. R(homopolymer) < 0.1 for all homopolymers
3. R(real_peptide) > 0.5 for all real peptides

---

### Activity 0b.3: Update train_grpo.py to Support New Reward

**Description**: Modify training script to accept a `--reward_type` flag.

**Changes to `scripts/train_grpo.py`**:

```python
# Add new argument
parser.add_argument(
    "--reward_type",
    type=str,
    default="improved",
    choices=["esm2_pll", "improved"],
    help="Reward function type: esm2_pll (original) or improved (with entropy gate)"
)

# In main():
if args.reward_type == "esm2_pll":
    from gflownet_peptide.rewards.esm2_reward import ESM2Reward
    reward_fn = ESM2Reward(
        model_name=config["esm_model"],
        device=device,
        normalize=True,
    )
elif args.reward_type == "improved":
    from gflownet_peptide.rewards.improved_reward import ImprovedReward
    reward_fn = ImprovedReward(
        model_name=config["esm_model"],
        device=device,
        entropy_threshold=config.get("entropy_threshold", 0.5),
        entropy_sharpness=config.get("entropy_sharpness", 10.0),
        min_length=config.get("min_length", 10),
    )
```

---

### Activity 0b.4: Create Improved Reward Config

**Description**: Create a new config file for training with improved reward.

**File**: `configs/grpo_improved.yaml`

```yaml
# GRPO-D with Improved Reward (Entropy Gate)
# Phase 0b Configuration

# Model
model_name: "nferruz/ProtGPT2"
esm_model: "esm2_t6_8M_UR50D"

# Generation
min_length: 10
max_length: 30

# Training
learning_rate: 3.0e-4
batch_size: 16
num_generations: 8
beta: 0.04
total_iterations: 1000

# Diversity (GRPO-D)
diversity_weight: 0.15
diversity_weight_aa: 0.7
diversity_weight_seq: 0.3

# Improved Reward Parameters
reward_type: "improved"
entropy_threshold: 0.5      # Minimum normalized entropy (50% of max)
entropy_sharpness: 10.0     # Sigmoid slope for entropy gate
embedding_temperature: 10.0 # Sigmoid temperature for embedding norm

# Logging
log_interval: 50
checkpoint_interval: 200
wandb_project: "gflownet-peptide"
wandb_run_prefix: "grpod_improved"
```

---

### Activity 0b.5: Train GRPO-D with Improved Reward

**Description**: Run full training with the improved reward function.

**Command**:
```bash
python scripts/train_grpo.py \
    --config configs/grpo_improved.yaml \
    --reward_type improved
```

**Expected Training Time**: ~9-10 hours (same as Phase 0a)

**Output**:
- `checkpoints/grpo/<run>_final.pt`
- `results/grpo/<run>_peptides.csv`
- `results/grpo/<run>_stats.csv`

**Monitoring**:
```bash
# Watch training progress
tail -f logs/train_grpo.log

# Check W&B dashboard
# https://wandb.ai/ewijaya/gflownet-peptide
```

---

### Activity 0b.6: Analyze Results and Compare

**Description**: Run the same analysis as Phase 0a on the new results.

**Steps**:
1. Load new peptides from `results/grpo/`
2. Compute diversity metrics (entropy, clustering)
3. Compare to Phase 0a results
4. Create side-by-side visualization

**Notebook**: Update `notebooks/gflownet-phase-0-validation.ipynb` or create `notebooks/phase-0b-comparison.ipynb`

**Key Comparisons**:

| Metric | Phase 0a (ESM-2 PLL) | Phase 0b (Improved) | Target |
|--------|----------------------|---------------------|--------|
| Mean reward | 0.816 | ? | Similar or higher |
| Cluster count | 10 | ? | >10 (more diverse) |
| Embedding diversity | 0.336 | ? | >0.5 |
| Sequences with repeats | 97% | ? | <20% |
| Mean sequence entropy | ~0.3 | ? | >0.6 |

---

### Activity 0b.7: Update Go/No-Go Decision

**Description**: Based on comparison results, update the Phase 0 decision.

**Decision Logic**:

```
IF improved_reward + GRPO-D shows:
    - Low repetition (<20%)
    - High embedding diversity (>0.5)
    - Meaningful clusters (>15)
THEN:
    → NO-GO for GFlowNet
    → GRPO-D with improved reward is sufficient
    → Proceed to Phase 1 (reward model training) only

ELIF improved_reward + GRPO-D shows:
    - Reduced repetition but still mode collapse
    - Embedding diversity improved but still limited
    - Cluster count increased but <15
THEN:
    → CLEAR GO for GFlowNet
    → The problem is architectural (reward maximization)
    → GFlowNet's proportional sampling is needed

ELSE (unexpected results):
    → Further investigation required
```

**Update**: `docs/phase0_decision.md` with Phase 0b findings.

---

## 5. Technical Specifications

### 5.1 Improved Reward Architecture

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

### 5.2 Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| ESM-2 model | - | `esm2_t6_8M_UR50D` | Backbone for embeddings |
| Embedding temperature | τ_emb | 10.0 | Sigmoid temperature for naturalness |
| Entropy threshold | θ_ent | 0.5 | Minimum normalized entropy |
| Entropy sharpness | k_ent | 10.0 | Sigmoid slope for entropy gate |
| Min length | L_min | 10 | Minimum peptide length |
| Length sharpness | k_len | 0.5 | Sigmoid slope for length gate |

### 5.3 Code Structure

**New Files**:
```
gflownet_peptide/
└── rewards/
    ├── __init__.py           # Update exports
    ├── esm2_reward.py        # Original (unchanged)
    └── improved_reward.py    # NEW: Improved reward

scripts/
├── train_grpo.py             # Update with --reward_type
└── validate_reward.py        # NEW: Reward validation

configs/
├── grpo.yaml                 # Original config
└── grpo_improved.yaml        # NEW: Improved reward config
```

**Modified Files**:
```
scripts/train_grpo.py         # Add --reward_type argument
gflownet_peptide/rewards/__init__.py  # Export ImprovedReward
```

---

## 6. Success Criteria

### 6.1 Implementation Success Criteria

| ID | Criterion | Target | Measurement | Status |
|----|-----------|--------|-------------|--------|
| SC0b.1 | Reward validation | R(good) > R(bad) for 100% of test pairs | `scripts/validate_reward.py` | ✅ Pass |
| SC0b.2 | Repetitive sequences penalized | R(homopolymer) < 0.1 | Manual check | ✅ Pass |
| SC0b.3 | Training completes | 1000 iterations without crash | Log file | ⏳ Pending |
| SC0b.4 | Repeat rate reduced | <20% sequences with 3+ AA repeats | Analysis notebook | ⏳ Pending |
| SC0b.5 | Entropy improved | Mean sequence entropy > 0.6 | Analysis notebook | ⏳ Pending |
| SC0b.6 | Quality maintained | Mean reward > 0.5 | Analysis notebook | ⏳ Pending |

### 6.2 Go/No-Go Decision Criteria

The Phase 0b results will determine whether to proceed with GFlowNet implementation:

| Outcome | Criteria | Decision | Next Step |
|---------|----------|----------|-----------|
| **CLEAR GO** | Cluster count <15 AND embedding diversity <0.5 AND repeat rate <20% | Proceed to GFlowNet | Phase 1: GFlowNet implementation |
| **NO-GO** | Cluster count ≥15 AND embedding diversity ≥0.5 | GRPO-D is sufficient | Use GRPO-D with improved reward |
| **REVISIT** | Mean reward <0.5 OR repeat rate >20% | Reward needs tuning | Adjust entropy threshold/sharpness |

### 6.3 Comparison Metrics (Phase 0a vs Phase 0b)

| Metric | Phase 0a (ESM-2 PLL) | Phase 0b Target | Improvement |
|--------|----------------------|-----------------|-------------|
| Mean reward | 0.816 | >0.5 | May decrease (expected) |
| Cluster count | 3 | >10 | 3× improvement |
| Embedding diversity | 0.336 | >0.4 | 20% improvement |
| Sequences with repeats | 97% | <20% | 80% reduction |
| Mean sequence entropy | ~0.3 | >0.6 | 2× improvement |

### 6.4 Final Success Definition

**Phase 0b is successful if:**
1. ✅ Improved reward correctly penalizes repetitive sequences (validated)
2. ⏳ Training completes without divergence
3. ⏳ Generated peptides show significantly reduced repetition (<20%)
4. ⏳ A clear GO or NO-GO decision can be made for GFlowNet

---

## 7. Deliverables Checklist

**Implementation**:
- [x] `gflownet_peptide/rewards/improved_reward.py` - Improved reward class
- [x] `gflownet_peptide/rewards/__init__.py` - Updated exports
- [x] `scripts/validate_reward.py` - Reward validation script
- [x] `scripts/train_grpo.py` - Updated with `--reward_type`
- [x] `configs/grpo_improved.yaml` - New config file

**Training Outputs** (pending training):
- [ ] `checkpoints/grpo/<run>_final.pt` - Trained model
- [ ] `results/grpo/<run>_peptides.csv` - Generated peptides
- [ ] `results/grpo/<run>_stats.csv` - Training statistics
- [ ] `logs/train_grpo_improved.log` - Training log

**Analysis Outputs** (pending training):
- [ ] `outputs/grpo_improved_embeddings.npy` - ESM-2 embeddings
- [ ] `outputs/grpo_improved_metrics.json` - Diversity metrics
- [ ] `outputs/grpo_improved_umap.png` - UMAP visualization
- [ ] `outputs/phase0_comparison.png` - Side-by-side comparison

**Documentation** (pending training):
- [ ] `docs/phase0_decision.md` - Updated with Phase 0b findings
- [ ] `notebooks/phase-0b-comparison.ipynb` - Comparison analysis

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Entropy gate too strict | Medium | Medium | Tune threshold (start at 0.4 instead of 0.5) |
| Embedding norm not discriminative | Low | High | Add pseudo-likelihood as secondary signal |
| Training slower with new reward | Low | Low | Same ESM-2 forward pass, minimal overhead |
| New reward has different failure modes | Medium | Medium | Run validation suite before training |
| Results inconclusive | Low | Medium | Define clear metrics before training |

---

## 9. Timeline

| Activity | Estimated Duration | Dependencies |
|----------|-------------------|--------------|
| 0b.1: Implement improved reward | 1-2 hours | None |
| 0b.2: Create validation suite | 30 min | 0b.1 |
| 0b.3: Update train_grpo.py | 30 min | 0b.1 |
| 0b.4: Create config | 15 min | 0b.3 |
| 0b.5: Train GRPO-D | 9-10 hours | 0b.1-0b.4 |
| 0b.6: Analyze results | 1-2 hours | 0b.5 |
| 0b.7: Update decision | 30 min | 0b.6 |

**Total**: ~12-15 hours (mostly training time)

---

## 10. References

### Internal Documents
- `docs/prd-phase-0-validation.md` - Original Phase 0 PRD
- `docs/reward-design-analysis_2025-12-24.md` - Three-perspective reward analysis
- `docs/esm2-reward-explainer.md` - How ESM-2 pseudo-likelihood works
- `docs/phase0_decision.md` - Current decision (to be updated)
- `docs/reward_formulation.md` - Reward math documentation

### External References
- [Low Complexity Regions in Proteins](https://en.wikipedia.org/wiki/Low_complexity_regions_in_proteins)
- [SEG Algorithm for LCR Detection](https://academic.oup.com/bioinformatics/article/21/2/160/187330)
- [ESM-2 Repository](https://github.com/facebookresearch/esm)

---

*End of Phase 0b PRD*
