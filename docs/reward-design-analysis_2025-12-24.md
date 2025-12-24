# Three-Angle Perspective Analysis: Reward Function Improvement

**Created**: 2025-12-24
**Topic**: Fixing ESM-2 pseudo-likelihood reward hacking in GFlowNet peptide generation
**Framework**: Based on `.claude/commands/reward-design.md`

---

## Step 0: Frame the Problem

### Task Domain
**Generative model / GFlowNet** for therapeutic peptide design. The goal is to sample diverse, high-quality peptide sequences proportionally to their reward.

### Current Reward
**ESM-2 pseudo-likelihood**: Measures "naturalness" by computing the average log-probability of each amino acid given its context.

```
R_PLL(x) = (1/L) Σᵢ log P_ESM(xᵢ | context)
```

### Desired Behavior
Generate peptides that are:
1. **Biologically viable** - realistic protein sequences
2. **Structurally diverse** - explore different folds/motifs
3. **Therapeutically relevant** - stable, potentially binding

### Failure Modes (Observed in Phase 0)
1. **Reward hacking via repetition**: Sequences like `QQQQQQQQQQ` get high scores because each position is trivially predictable
2. **Mode collapse**: 97% of generated sequences contain repetitive patterns
3. **Low embedding diversity**: Despite high sequence diversity (0.895), embedding diversity is only 0.336
4. **Degenerate clusters**: 10 clusters found, but all contain repetitive junk

---

## Step 1: Detect Mode and Select Expert Trio

**Selected Mode: FORMULATION**

Trigger: We need to **design/create/define a new reward** that doesn't suffer from the repetition exploitation problem.

**Expert Trio:**
- **Yoshua Bengio** (Causality & GFlowNets)
- **Richard Sutton** (RL Foundations)
- **Andrew Ng** (Reward Shaping & Practical Design)

---

## Step 2: Three Perspectives

### Perspective 1: Yoshua Bengio (Causal & Compositional Rewards)

**Key Observations:**

1. **ESM-2 pseudo-likelihood captures correlation, not causation.** It measures P(amino_acid | context) but this correlation exploits repetitive patterns. The reward does not capture the *causal* relationship between sequence → structure → function.

2. **The reward is not compositional.** Current R_PLL is a single monolithic score. A proper reward should decompose into interpretable components: naturalness + structural viability + no-degeneracy + diversity. Each can be debugged independently.

3. **GFlowNet needs a reward that supports mode diversity.** If R(repetitive) >> R(diverse), even proportional sampling will heavily favor degenerate sequences. The reward itself must not have a single dominant mode. Per [GFlowNet foundations](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf), the reward should reflect the true utility landscape.

4. **Low-complexity detection is a causal filter.** Repetitive sequences are *causally* non-functional (they don't fold, don't bind). This is domain knowledge that should be hard-coded, not learned. Per [LCR research](https://academic.oup.com/bioinformatics/article/21/2/160/187330), SEG-style entropy filtering can detect these regions.

5. **Entropy of the reward distribution matters.** If Shannon entropy of P(x) ∝ R(x) is low, GFlowNet will not explore. The reward should be designed to have high entropy over viable sequences.

**Unique Concerns Others Might Miss:**
- The reward should support *amortized inference* - the relationship between features and reward should be learnable
- [ESM-2 stores motif statistics](https://www.pnas.org/doi/10.1073/pnas.2406285121), not functional relationships - using it raw conflates statistical regularity with biological function

**Recommended Reward Components:**
```
R_Bengio = R_naturalness × R_no_LCR × R_structural
```
Where:
- `R_naturalness`: ESM-2 embedding quality (NOT pseudo-likelihood)
- `R_no_LCR`: Hard penalty for low-complexity regions (entropy-based)
- `R_structural`: pLDDT or predicted fold confidence

---

### Perspective 2: Richard Sutton (RL Foundations)

**Key Observations:**

1. **The Bitter Lesson applies: simpler rewards + more data > complex engineered rewards.** But "simpler" doesn't mean "broken." The current reward is simple *and* broken. A slightly more complex reward that actually works is preferable.

2. **The reward should be Markovian.** ESM-2 pseudo-likelihood is fine here - it's a function of the sequence alone, no hidden state. Good.

3. **Scalability: will this reward remain meaningful as the model improves?** ESM-2 pseudo-likelihood fails here - as the generator learns to exploit repetitive patterns, the reward gives 0.95+ to garbage. A robust reward should *increase* in difficulty as the generator improves.

4. **The value function should be learnable.** If R(x) = 0.95 for both garbage and good sequences, the value function cannot distinguish states. Need reward separation: R(good) >> R(garbage).

5. **Don't over-engineer.** The fix might be simpler than expected: (a) add entropy penalty, (b) add LCR filter, (c) done. Don't build a 10-component reward when 3 will suffice.

**Unique Concerns Others Might Miss:**
- Reward shaping should preserve optimal policy (potential-based shaping theorem)
- Adding too many components may introduce credit assignment problems
- The reward must remain efficiently computable (ESM-2 forward passes are expensive)

**Recommended Reward Components:**
```
R_Sutton = R_ESM2 × R_entropy_bonus
```
Where:
- `R_ESM2`: ESM-2 pseudo-likelihood (kept for naturalness signal)
- `R_entropy_bonus`: exp(sequence_entropy / τ) - rewards diverse amino acid usage

---

### Perspective 3: Andrew Ng (Reward Shaping & Practical Design)

**Key Observations:**

1. **Diagnose before prescribing.** The reward hacking is *known* - 97% repetitive patterns. Any fix must directly target this. Proposed diagnostic: compute reward for known good peptides vs. known garbage peptides. If R(garbage) > R(good), the reward is broken.

2. **Potential-based shaping for LCR penalty.** Adding `-λ × LCR_score` directly changes optimal policy. Instead, use: `R_shaped = R_base × sigmoid(LCR_threshold - LCR_score)`. This is a soft gate that zeroes reward for high-LCR sequences.

3. **Numerical stability matters.** Combining log-probabilities (pseudo-likelihood) with entropy scores requires careful scaling. Normalize all components to [0, 1] before combining. Use geometric mean for multiplicative combination.

4. **Unintended optima to watch for:**
   - Sequences with all 20 amino acids equally (max entropy, but nonsensical)
   - Very short sequences (high pLDDT, low complexity, but not useful)
   - Sequences that "game" the structural predictor

5. **Efficient computation.** Each training step calls reward 128+ times. ESM-2 forward pass: ~100ms. Adding ESMFold for pLDDT: ~500ms. Budget carefully. Prefer ESM-2 embedding-based rewards over structure prediction.

**Unique Concerns Others Might Miss:**
- The reward should be *differentiable* if possible (for future gradient-based methods)
- Need validation set of known-good and known-bad peptides to test reward accuracy
- Reward model should be checkpointed and versioned (reproducibility)

**Recommended Reward Components:**
```
R_Ng = R_base × LCR_gate × length_gate
```
Where:
- `R_base`: ESM-2 pseudo-likelihood OR embedding naturalness
- `LCR_gate`: sigmoid(τ × (entropy_threshold - sequence_entropy)), 0 for repetitive
- `length_gate`: sigmoid(length - min_length), 0 for too-short sequences

---

## Step 3: Synthesize Into Unified Reward Design

### Consensus (All Three Agree)

1. **ESM-2 pseudo-likelihood alone is broken.** It rewards repetitive patterns. All three perspectives agree this must be fixed.

2. **Entropy/LCR penalty is essential.** Low-complexity sequences should receive zero or near-zero reward. This is non-negotiable.

3. **Keep it simple and efficient.** Don't over-engineer. 2-3 components maximum. Must be computable in <200ms per sequence.

### Tensions and Resolutions

| Tension | Bengio View | Sutton View | Ng View | Resolution |
|---------|-------------|-------------|---------|------------|
| Use structural signal? | Yes (pLDDT) | Avoid complexity | Expensive, avoid | **Skip for now** - add in Phase 1b if needed |
| Hard vs soft LCR penalty | Hard threshold | Prefer simple | Soft gate (differentiable) | **Soft sigmoid gate** |
| Replace or augment ESM-2? | Replace with embedding | Augment with entropy | Augment with gates | **Augment** - ESM-2 embedding + entropy |

### Proposed Reward Function

```python
def reward(sequence: str) -> float:
    """
    Improved reward function for GFlowNet peptide generation.

    Components:
    1. ESM-2 embedding naturalness (NOT pseudo-likelihood)
    2. Sequence entropy gate (penalize low-complexity)
    3. Length gate (penalize too-short sequences)
    """

    # Component 1: ESM-2 Embedding Naturalness
    # Use embedding norm as proxy for "how well ESM-2 represents this"
    embedding = esm2.encode(sequence)  # Shape: (L, 320)
    emb_mean = embedding.mean(dim=0)   # Mean pooling
    emb_norm = torch.norm(emb_mean)
    r_naturalness = torch.sigmoid(emb_norm / tau_emb)  # [0, 1]

    # Component 2: Sequence Entropy Gate
    # Shannon entropy of amino acid distribution
    aa_counts = Counter(sequence)
    total = len(sequence)
    probs = [count / total for count in aa_counts.values()]
    entropy = -sum(p * log2(p) for p in probs)
    max_entropy = log2(20)  # Max possible with 20 amino acids
    normalized_entropy = entropy / max_entropy  # [0, 1]

    # Soft gate: high entropy → 1.0, low entropy → 0.0
    entropy_gate = torch.sigmoid(k * (normalized_entropy - threshold))

    # Component 3: Length Gate
    # Penalize sequences shorter than min_length
    length_gate = torch.sigmoid(0.5 * (len(sequence) - min_length))

    # Combine: multiplicative (all must be good)
    reward = r_naturalness * entropy_gate * length_gate

    return reward
```

### Design Rationale

| Component | What It Does | Failure Mode It Prevents |
|-----------|--------------|--------------------------|
| `r_naturalness` | Measures how "protein-like" the embedding is | Rewards realistic sequences, not just predictable ones |
| `entropy_gate` | Zero reward for repetitive sequences | Directly prevents the reward hacking we observed |
| `length_gate` | Zero reward for too-short sequences | Prevents degenerate short outputs |

**Key Change:** Replacing pseudo-likelihood with embedding norm eliminates the core vulnerability. Pseudo-likelihood asks "is each AA predictable?" (bad - repetition is predictable). Embedding norm asks "does this look like a real protein in ESM-2's learned space?" (good - repetitive garbage has abnormal embeddings).

### Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Embedding temp | τ_emb | 10.0 | Sigmoid temperature for embedding norm |
| Entropy threshold | threshold | 0.5 | Minimum normalized entropy (50% of max) |
| Entropy sharpness | k | 10.0 | Sigmoid slope for entropy gate |
| Min length | min_length | 10 | Minimum peptide length |

### Validation Plan

1. **Diagnostic 1: Known-Good vs Known-Bad Test**
   - Compute reward for 100 real therapeutic peptides from DrugBank
   - Compute reward for 100 generated repetitive sequences
   - **Pass criterion:** R(real) > R(repetitive) for >95% of pairs

2. **Diagnostic 2: Entropy Distribution**
   - Generate 1000 peptides with new reward
   - Compute entropy for each
   - **Pass criterion:** Mean entropy > 0.6, no sequence with entropy < 0.3

3. **Red-Team Scenario: Adversarial Sequences**
   - Test: `AQAQAQAQAQAQAQAQ` (alternating, low-entropy but not homopolymer)
   - Test: `ACDEFGHIKLMNPQRS` (all different AAs, max entropy but unnatural)
   - **Pass criterion:** Both should get moderate rewards (not extremes)

---

## Implementation Recommendation

Based on this analysis, the recommended implementation order:

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | Add entropy gate to existing ESM-2 reward | Immediate fix for reward hacking |
| 2 | Replace pseudo-likelihood with embedding norm | Addresses root cause |
| 3 | Train composite reward on FLIP + Propedia | Long-term robust solution |

---

## Sources

- [Biological Sequence Design with GFlowNets](https://proceedings.mlr.press/v162/jain22a/jain22a.pdf)
- [GFlowNet Foundations](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf)
- [LCR Detection Algorithm](https://academic.oup.com/bioinformatics/article/21/2/160/187330)
- [Low Complexity Regions Wikipedia](https://en.wikipedia.org/wiki/Low_complexity_regions_in_proteins)
- [ESM-2 Pseudo-perplexity Issues](https://arxiv.org/html/2407.07265v1)
- [Protein Language Models Learn Motif Statistics](https://www.pnas.org/doi/10.1073/pnas.2406285121)
- [ProtGPT2](https://www.nature.com/articles/s41467-022-32007-7)
- [Towards Robust Evaluation of Protein Generative Models](https://www.biorxiv.org/content/10.1101/2024.10.25.620213v1.full.pdf)
- [Awesome GFlowNets](https://github.com/zdhNarsil/Awesome-GFlowNets)

---

## Related Documents

- [docs/reward_formulation.md](reward_formulation.md) - Current reward math
- [docs/esm2-reward-explainer.md](esm2-reward-explainer.md) - How ESM-2 pseudo-likelihood works
- [docs/phase0_decision.md](phase0_decision.md) - Evidence of reward hacking
- [docs/gflownet-public-reward-options_2025-12-22_105142.md](gflownet-public-reward-options_2025-12-22_105142.md) - Public reward alternatives

---

*End of document*
