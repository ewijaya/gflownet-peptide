# GFlowNet Public Reward Options for Peptide Generation

*Collected: 2025-12-22 10:51:42*

---

## 1. For Idea F (GFlowNet Diverse Generation), we need a reward score. If we use public data, what reward score can we use? My GRPO approach (Project 14) uses PEM as reward model, which is proprietary.

You're correct — GFlowNet requires a reward function R(x) ≥ 0, and your PEM is proprietary. Let me analyze what public reward signals are available.

## Step 0: Context Loaded

Your P14 uses PEM (Peptide Efficacy Metric) trained on internal bioactivity data. This cannot be shared publicly.

## Step 1: Mode Selection

**Selected Mode: FEASIBILITY**

The question is about practical data availability for implementation — "what reward score can we use?" with public data constraints.

**Expert Trio:** Computational Pragmatist → Data Curator → Portfolio Manager

---

## Step 2: Three Perspectives

### Perspective 1: Computational Pragmatist
*Implementation reality check*

**Key Observations:**

1. **Any supervised signal can be a reward.** You don't need "therapeutic efficacy" — you need any measurable property that correlates with peptide quality. Options:
   - Predicted binding affinity (regression target → reward)
   - Predicted stability (classification/regression → reward)
   - Predicted solubility (regression → reward)
   - Multi-property composite (product of normalized scores)

2. **ESM-2 embeddings + simple head = universal reward.** Train a linear probe or MLP on top of frozen ESM-2 for any property with labels. This is cheap and generalizable.

3. **Self-supervised rewards exist.** You can reward:
   - Low perplexity under a peptide language model (naturalness)
   - High confidence from structure predictor (foldability)
   - Similarity to known functional peptides (embedding distance)

4. **The reward doesn't need to be perfect.** GFlowNet samples proportionally — even a noisy reward gives useful diversity in high-scoring regions.

5. **Composite rewards are straightforward.** R(x) = stability(x) × binding(x) × solubility(x) — all trainable from public data.

**Unique Concern:** You might train a reward model that doesn't correlate with your actual therapeutic goals. Public proxies ≠ PEM.

**Recommended Actions:**
1. Start with FLIP — it has multiple peptide properties with labels
2. Train ESM-2 → property predictors for each FLIP task
3. Combine into composite reward: R = ∏ property_i^{weight_i}

---

### Perspective 2: Data Curator
*What's actually available*

**Key Observations:**

1. **FLIP Benchmark (Best Option)**

| Task | Property | Size | Reward Use |
|------|----------|------|------------|
| AAV | Viral packaging fitness | 280K | Fitness score |
| GB1 | Protein G binding | 150K | Binding affinity |
| Meltome | Thermal stability | 28K | Melting temperature |
| Stability | Protein stability | 53K | ΔΔG prediction |
| Fluorescence | GFP fluorescence | 54K | Fluorescence intensity |

**Direct use:** Train regressor → exp(prediction) = reward

2. **ProteinGym (Large but Proteins)**

| Content | Size | Reward Use |
|---------|------|------------|
| Deep mutational scanning | 2.5M variants | Fitness scores |
| 217 protein assays | Mixed | Assay-specific |

**Caveat:** These are full proteins (100-500 AA), not short peptides. Domain shift risk.

3. **Propedia (Structure + Binding)**

| Content | Size | Reward Use |
|---------|------|------------|
| Peptide-protein complexes | 19K | Binding affinity proxy |
| Structural data | 19K | Contact-based scoring |

**Reward option:** Predict binding affinity or contact score from sequence.

4. **PDBBind (Affinity Labels)**

| Content | Size | Reward Use |
|---------|------|------------|
| Protein-ligand affinities | 23K | Kd/Ki values |
| Peptide subset | ~2K | Direct affinity |

**Direct use:** Train affinity predictor on peptide subset.

5. **ChEMBL (Bioactivity)**

| Content | Size | Reward Use |
|---------|------|------------|
| Peptide bioassays | ~50K | IC50, EC50 |
| Target-specific | Variable | Activity scores |

**Caveat:** Requires curation to extract peptide-specific data.

**Unique Concern:** FLIP is the cleanest option but covers specific proteins (AAV, GB1, GFP), not general therapeutic peptides. Generalization is uncertain.

**Recommended Actions:**
1. **Primary:** Use FLIP Stability task — most general, 53K examples
2. **Secondary:** Add Propedia binding signal
3. **Composite:** Combine stability + binding + low-perplexity

---

### Perspective 3: Portfolio Manager
*Fit with your existing work*

**Key Observations:**

1. **Your internal advantage is PEM.** Public rewards make the project publishable but remove your competitive edge. The public version is a "methods paper," not a drug discovery advantage.

2. **Two-track strategy makes sense:**
   - **Public track:** GFlowNet + FLIP reward → publishable, showcase-able
   - **Internal track:** GFlowNet + PEM reward → actual therapeutic value

3. **P14's GRPO already works.** If you build GFlowNet with public reward, you need to show it beats GRPO on that same public reward. Otherwise, what's the contribution?

4. **Portfolio gap this fills:** P2 and P14 are sequence-only generators. A structure-aware or binding-aware reward would add new capability.

5. **Reusable components:** Whatever reward model you train on public data can later be combined with PEM internally. R = PEM × stability × binding.

**Unique Concern:** If public reward doesn't correlate with PEM, the public version of GFlowNet won't help your actual drug discovery. It's just an academic exercise.

**Recommended Actions:**
1. Validate public reward correlation with PEM (if possible)
2. Build modular system: plug-and-play reward models
3. Publish methods paper on public reward, use internally with PEM

---

## Step 3: Synthesis

### Consensus (All Three Agree)

1. **FLIP is the best public reward source.** It has peptide/protein fitness data with clean labels, ready for training reward models.

2. **Composite rewards are better than single-property.** Combining stability + binding + naturalness gives richer signal than any one property.

3. **The public version is a methods contribution, not a drug discovery tool.** Real therapeutic value requires your proprietary PEM.

### Tensions & Resolutions

| Tension | Resolution |
|---------|------------|
| Public reward ≠ therapeutic value | Accept this for publication; use PEM internally |
| FLIP proteins ≠ short peptides | Fine-tune on Propedia peptides; validate on peptide subsets |
| Multiple reward options, which to choose? | Start with FLIP Stability (largest, most general) |

---

## Concrete Public Reward Options

### Option A: FLIP Stability Reward (Recommended Start)

```python
class FLIPStabilityReward(nn.Module):
    """Reward based on predicted protein/peptide stability"""
    def __init__(self):
        self.esm = ESM2.from_pretrained("esm2_t33_650M")
        self.head = nn.Linear(1280, 1)
        # Train on FLIP stability task (53K examples)

    def forward(self, sequence):
        emb = self.esm.encode(sequence).mean(dim=0)
        stability = self.head(emb)
        return torch.exp(stability)  # Non-negative reward
```

**Data:** FLIP Stability task, 53K labeled examples
**Property:** ΔΔG (stability change upon mutation)
**Training:** ~2 hours on single GPU

### Option B: Propedia Binding Reward

```python
class PropediaBindingReward(nn.Module):
    """Reward based on predicted binding affinity"""
    def __init__(self):
        self.esm = ESM2.from_pretrained("esm2_t33_650M")
        self.head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Train on Propedia with binding labels (or contact score proxy)

    def forward(self, sequence):
        emb = self.esm.encode(sequence).mean(dim=0)
        affinity = self.head(emb)
        return torch.softplus(affinity)  # Non-negative
```

**Data:** Propedia 19K complexes, derive binding labels from structure
**Property:** Predicted binding affinity or contact score
**Training:** ~3 hours

### Option C: Composite Multi-Property Reward (Best)

```python
class CompositeReward(nn.Module):
    """Combine multiple properties into single reward"""
    def __init__(self):
        self.stability_model = FLIPStabilityReward()
        self.binding_model = PropediaBindingReward()
        self.lm = load_peptide_lm()  # For naturalness

        # Learnable or fixed weights
        self.weights = {'stability': 1.0, 'binding': 1.0, 'naturalness': 0.5}

    def forward(self, sequence):
        stability = self.stability_model(sequence)
        binding = self.binding_model(sequence)
        naturalness = torch.exp(-self.lm.perplexity(sequence))

        # Geometric mean (all must be good)
        reward = (stability ** self.weights['stability'] *
                  binding ** self.weights['binding'] *
                  naturalness ** self.weights['naturalness'])

        return reward
```

**Properties:** Stability + Binding + Naturalness
**Advantage:** Peptides must score well on all axes

### Option D: Self-Supervised Reward (No Labels Needed)

```python
class SelfSupervisedReward(nn.Module):
    """Reward without any labels - just peptide quality signals"""
    def __init__(self):
        self.peptide_lm = load_protgpt2()  # Or ESM
        self.structure_pred = ESMFold()

    def forward(self, sequence):
        # Low perplexity = natural sequence
        perplexity = self.peptide_lm.perplexity(sequence)
        naturalness = torch.exp(-perplexity / 10)

        # High pLDDT = confident structure
        structure = self.structure_pred.predict(sequence)
        foldability = structure.plddt.mean() / 100

        return naturalness * foldability
```

**Properties:** Naturalness + Foldability
**Advantage:** Zero labels needed, fully public
**Caveat:** Doesn't capture therapeutic activity, just "peptide-ness"

---

## Action Plan

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | **Train FLIP Stability reward** | Largest dataset, most general property, direct labels |
| 2 | **Add Propedia binding reward** | Peptide-specific, adds binding dimension |
| 3 | **Combine into composite** | R = stability × binding × naturalness |
| 4 | **Validate on held-out FLIP** | Ensure reward model is accurate before GFlowNet |
| 5 | **Train GFlowNet with composite reward** | Now you have a public, publishable system |

---

## Summary Table: Public Reward Options

| Reward Source | Property | Data Size | Peptide-Specific? | Effort |
|--------------|----------|-----------|-------------------|--------|
| FLIP Stability | ΔΔG | 53K | Partial (proteins) | Low |
| FLIP GB1 | Binding | 150K | No (one protein) | Low |
| Propedia | Binding pose | 19K | Yes | Medium |
| PDBBind peptides | Kd/Ki | ~2K | Yes | Medium |
| ChEMBL peptides | IC50/EC50 | ~50K | Yes | High (curation) |
| Self-supervised | Naturalness + fold | Unlimited | Yes | Low |

**Recommendation:** Start with **FLIP Stability + Propedia Binding + Self-supervised naturalness** as a composite reward. This is fully public, covers multiple properties, and is peptide-relevant.
