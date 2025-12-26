# GFlowNet Peptide Project: ELI5 Explanation

## The Big Picture Story

**Problem:** We want to generate therapeutic peptides (short protein sequences) using AI. The current method (GRPO-D) is like a "greedy optimizer" - it finds ONE good answer and keeps making tiny variations of it. You ask for 100 peptides, you get 100 nearly identical ones.

**Solution:** GFlowNet is a different approach - instead of finding THE BEST answer, it explores ALL GOOD answers proportionally. You ask for 100 peptides, you get 100 diverse ones from different "families."

**Analogy:**
- **GRPO-D** = A restaurant critic who finds the best pizza in town and only eats there
- **GFlowNet** = A food explorer who visits ALL good restaurants, spending more time at better ones

**Publication Goal:** Show that GFlowNet produces 2-3x more diverse peptides than GRPO-D at the same quality level.

---

## Phase Breakdown

### Phase -1: Data Acquisition
**What:** Download public datasets (FLIP for stability, Propedia for binding)

**Why:** We need training data. Can't use proprietary data for publication.

**Status:** Done

---

### Phase 0a: GRPO-D Vanilla (Baseline #1)
**What:** Run GRPO-D with simple ESM-2 reward

**Why:** Establish the "worst case" baseline. Shows the problem we're solving.

**Result:** 97% homopolymers (QQQQQQQ...) - complete failure due to reward hacking

**Status:** Done

---

### Phase 0b: GRPO-D Improved (Baseline #2)
**What:** Run GRPO-D with better reward (entropy gate + naturalness)

**Why:** Fix the reward hacking problem. This is the "fair" baseline for GFlowNet comparison.

**Result:** 19.5% homopolymers, much better diversity

**Status:** Done

---

### Phase 1: Reward Model & Baseline Metrics
**What:**
1. Measure GRPO-D peptides quality (pLDDT structure scores, AA composition)
2. Train a stability predictor (optional, for ablation studies)

**Why:**
1. Establish the benchmark numbers GFlowNet must beat
2. Have a data-driven reward available for future experiments

**Why a predictor instead of real experimental values?** We need to score millions of novel sequences during training - real experiments are too slow and expensive. The predictor learns patterns from real data (FLIP dataset) and generalizes to new sequences. See [Phase 1: Why Use a Stability Predictor?](phase1-why-stability-predictor.md) for detailed explanation.

**Key Decision:** For fair comparison, GFlowNet will use the SAME reward (`ImprovedReward`) as GRPO-D Improved.

**Status:** Done
- Stability predictor: R² = 0.65
- Baseline metrics: pLDDT ~56-60, entropy 0.83, etc.

---

### Phase 2: GFlowNet Implementation (NEXT)
**What:** Build the GFlowNet code - forward policy, trajectory sampling, loss function

**Why:** This is the core algorithm we're testing

**Status:** Not started

---

### Phase 3: GFlowNet Training
**What:** Train GFlowNet using `ImprovedReward` (same reward as GRPO-D)

**Why:** Generate diverse peptides

**Status:** Not started

---

### Phase 4: Evaluation & Comparison
**What:** Compare GFlowNet vs GRPO-D on all metrics

**Why:** This is the paper's main result

**Key Comparisons:**

| Metric | GRPO-D Improved | GFlowNet Target |
|--------|-----------------|-----------------|
| Diversity | X | >=2x X |
| Quality | Y | >=0.95x Y |
| Modes/Clusters | 15 | >=45 |

**Status:** Not started

---

### Phase 5: Publication
**What:** Write paper, clean code, submit

**Status:** Not started

---

## The One-Sentence Summary

> We're proving that GFlowNet finds MORE DIVERSE good peptides than GRPO-D, using the SAME reward function, so any improvement is due to the algorithm, not reward engineering.
