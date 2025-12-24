# GFlowNet vs GRPO-D: Strategic Analysis

**Created**: 2025-12-24
**Context**: Evaluating whether improving the reward function for GRPO-D undermines the case for GFlowNet

---

## 1. The Strategic Question

> "By using improved reward for GRPO-D, will it make our GFlowNet results look bad in the future?"

This is a valid concern. If GRPO-D + Improved Reward produces diverse, high-quality peptides, then:
- GFlowNet may not show a significant advantage over GRPO-D
- The paper narrative becomes weaker ("why use GFlowNet if GRPO-D works?")

---

## 2. Scenario Analysis

### Scenario 1: GRPO-D + Improved Reward Shows High Diversity

**If Phase 0b succeeds** (clusters >15, embedding diversity >0.5):
- This means the **reward was the problem, not GRPO-D**
- GFlowNet's advantage (proportional sampling) becomes less compelling
- **Honest conclusion**: GRPO-D is sufficient with a proper reward

### Scenario 2: GRPO-D + Improved Reward Still Shows Limited Diversity

**If Phase 0b shows persistent mode collapse** (clusters <15, diversity <0.5):
- This proves GRPO-D has **architectural limitations** (reward maximization → mode collapse)
- GFlowNet's proportional sampling becomes the clear solution
- **Strong narrative**: Even with a fixed reward, GRPO-D can't match GFlowNet's diversity

---

## 3. The Right Approach

**We should NOT handicap GRPO-D to make GFlowNet look better.** That would be:
1. Scientifically dishonest
2. Bad for the paper (reviewers would catch it)
3. Bad for actual peptide generation (we want the best tool)

---

## 4. Why GFlowNet Will Still Outperform

### 4.1 The Fundamental Difference

| GRPO-D | GFlowNet |
|--------|----------|
| **Maximizes** E[R(x)] | **Samples** P(x) ∝ R(x) |
| Gradient pushes toward highest reward | Learns to sample all good regions proportionally |
| Diversity is a penalty term (fights the objective) | Diversity emerges naturally from the objective |

### 4.2 Why GRPO-D Will Still Struggle

Even with improved reward, GRPO-D's core objective is:

```
max E[R(x)] - β·KL(π||π_ref) + λ·Diversity
```

The **diversity term fights the main objective**. The optimizer wants to:
1. Find high-reward sequences ✓
2. Stay close to reference ✓
3. Be diverse... but this **reduces expected reward**

So GRPO-D will naturally converge to a **few high-reward modes** and stop exploring.

### 4.3 What We Expect to See

**GRPO-D + Improved Reward (Phase 0b)**:
- ✅ No more repetitive garbage (entropy gate works)
- ✅ Real protein-like sequences
- ⚠️ Still limited to ~10-20 structural clusters
- ⚠️ Will find the "best" modes but miss isolated peaks

**GFlowNet (Phase 2)**:
- ✅ No repetitive garbage (same improved reward)
- ✅ Real protein-like sequences
- ✅ **30-50+ structural clusters** (proportional sampling)
- ✅ Finds isolated fitness peaks others miss
- ✅ Calibrated: can estimate R(x) from sampling frequency

### 4.4 Concrete Predictions

| Metric | Phase 0a (ESM-2 PLL) | Phase 0b (Improved) | GFlowNet (Predicted) |
|--------|----------------------|---------------------|----------------------|
| Repeat rate | 97% | <20% | <5% |
| Cluster count | 3 | 10-15 | **30-50** |
| Embedding diversity | 0.336 | 0.4-0.5 | **0.6-0.8** |
| Mean reward | 0.816 | 0.5-0.6 | 0.5-0.6 |

---

## 5. The Key Insight

**Improved reward fixes WHAT is rewarded. GFlowNet fixes HOW sampling happens.**

- Phase 0b fixes: "Don't reward garbage"
- GFlowNet fixes: "Sample all good things, not just the best"

These are **orthogonal improvements**. We need both:
1. A reward that doesn't hack → Improved Reward ✅
2. A sampler that explores all modes → GFlowNet (Phase 2)

---

## 6. GFlowNet's Value Regardless of Phase 0b Outcome

Even if GRPO-D + Improved Reward works well, GFlowNet still offers:

| GFlowNet Advantage | Why It Matters |
|-------------------|----------------|
| **Proportional sampling by design** | No diversity hyperparameter tuning needed |
| **Calibrated sampling** | P(x) ∝ R(x) - can estimate reward from frequency |
| **Better mode coverage** | Theoretically samples all modes proportionally |
| **Multi-modal exploration** | Finds isolated fitness peaks GRPO-D might miss |

---

## 7. Paper Framing

The paper can frame the comparison honestly:

> "We first identified and fixed the reward hacking issue (Phase 0b), then compared GRPO-D vs GFlowNet on a fair playing field. GFlowNet showed [X% better diversity / similar performance / etc.]"

This is actually a **stronger** paper because:
1. We identified and fixed a real problem (reward hacking)
2. We compared methods fairly
3. Whatever conclusion we reach is scientifically valid

---

## 8. Conclusion

**Good science > good-looking results.**

- If GRPO-D works with the improved reward, that's a valid finding
- If GFlowNet still outperforms, that's an even stronger result
- **Expected outcome**: GFlowNet shows 2-3× better diversity than GRPO-D + Improved Reward

The diversity limitation in GRPO-D is **architectural** (reward maximization), not just a reward problem. The improved reward is a **necessary foundation** for fair comparison, not a threat to GFlowNet's value.

---

## References

- `docs/reward-design-analysis_2025-12-24.md` - Three-perspective reward analysis
- `docs/prd-phase-0b-improved-reward.md` - Phase 0b PRD
- `docs/gflownet-master-prd.md` - Master project PRD
- `docs/phase0_decision.md` - Go/No-Go decision document

---

*End of document*
