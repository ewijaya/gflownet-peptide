# From Project 14 to GFlowNet: Advancing Peptide Candidate Diversity

*Stakeholder Briefing — December 2025*

---

## Executive Summary

Project 14 successfully developed an AI system that generates high-quality peptide candidates optimized for therapeutic efficacy. The proposed GFlowNet approach builds on this foundation to solve a remaining challenge: **generating a wider variety of promising candidates** for experimental testing, without sacrificing quality.

---

## 1. The Problem We Solved in Project 14

**Challenge:** Designing new therapeutic peptides is like searching for needles in a vast haystack. Traditional methods rely on chemists manually proposing sequences, which is slow and limited by human intuition.

**What we needed:** An AI system that could automatically generate peptide sequences predicted to have high therapeutic efficacy (as measured by our internal PEM score).

---

## 2. How We Solved It: The GRPO Approach

We developed **Group Relative Policy Optimization (GRPO)**, a reinforcement learning method that trains an AI model to generate peptides with high predicted efficacy.

**Results achieved:**
- **Mean efficacy score:** 0.86-0.89 (on a 0-1 scale)
- **Best candidates:** Reached 0.954 efficacy
- **Candidate diversity:** 0.85-0.89 (with diversity incentive)
- **Production deployment:** 3 models now available for peptide generation

**How it works (simplified):**

```
┌─────────────────────────────────────────────────────┐
│            Project 14: GRPO Approach                │
├─────────────────────────────────────────────────────┤
│                                                     │
│   AI Generator  ───►  Peptide Candidate             │
│        │                     │                      │
│        │                     ▼                      │
│        │              Efficacy Score (PEM)          │
│        │                     │                      │
│        └──── Feedback ◄──────┘                      │
│             "Generate more like the best ones"      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

The system learns by trial and feedback: generate candidates, score them, then adjust to produce more high-scoring sequences.

---

## 3. The Remaining Challenge: Candidate Diversity

**The issue:** GRPO is designed to find the *best* candidates. This naturally leads the AI to converge on a narrow set of similar sequences—all scoring high, but structurally alike.

**Why this matters for drug discovery:**

| What GRPO Tends to Produce | What Wet-Lab Teams Need |
|---------------------------|------------------------|
| 20 similar high-scoring peptides | 20 *diverse* high-scoring peptides |
| All from one "family" | Multiple structural families |
| One backup if lead fails | Several backup scaffolds |

**Current workaround:** We added a "diversity penalty" to GRPO that encourages variety. This helps, but creates a trade-off: the more we push for diversity, the more we sacrifice peak efficacy.

---

## 4. Proposed Next Step: GFlowNet

**GFlowNet** (Generative Flow Network) is a different approach that achieves diversity *naturally*, without forcing a trade-off.

**The key difference:**

| GRPO (Current) | GFlowNet (Proposed) |
|----------------|---------------------|
| "Find the best candidate" | "Sample from all good candidates" |
| Converges to similar sequences | Explores the full space of good options |
| Needs diversity penalty (workaround) | Diversity is built into the method |
| Optimizes for expected efficacy | Samples proportionally to efficacy |

**How it works (simplified):**

```
┌─────────────────────────────────────────────────────┐
│          Proposed: GFlowNet Approach                │
├─────────────────────────────────────────────────────┤
│                                                     │
│   AI Generator  ───►  Peptide Candidate             │
│        │                     │                      │
│        │                     ▼                      │
│        │              Efficacy Score                │
│        │                     │                      │
│        └──── Feedback ◄──────┘                      │
│             "Sample ALL good ones proportionally"   │
│                                                     │
│   High-scoring peptides: sampled more often         │
│   Medium-scoring peptides: sampled less often       │
│   (but still sampled, maintaining diversity)        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Analogy:** If GRPO is like "always ordering your favorite dish," GFlowNet is like "trying everything on the menu, but ordering the best dishes more frequently."

---

## 5. Expected Benefits

| Benefit | Impact |
|---------|--------|
| **More diverse candidates** | 2-3x more structural variety at equivalent quality |
| **Better experimental coverage** | Multiple backup scaffolds if lead candidate fails |
| **No diversity-efficacy trade-off** | Diversity is intrinsic, not a penalty |
| **Publishable methodology** | Novel application; no prior work on GFlowNet for therapeutic peptides |

**For wet-lab teams:** Instead of receiving 20 variations of the same scaffold, they receive 20 candidates spanning multiple structural families—all predicted to work, but offering different options if synthesis or testing reveals issues with any one family.

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GFlowNet doesn't improve on GRPO | Direct comparison study before full adoption |
| Training complexity | Build on established open-source framework (torchgfn) |
| Reward model limitations | Validate efficacy predictor independently first |
| Timeline | 2-month development, then empirical comparison |

**Validation plan:** Before recommending adoption, we will run a head-to-head comparison:
1. Generate 1,000 candidates with GRPO (current)
2. Generate 1,000 candidates with GFlowNet (proposed)
3. Compare diversity metrics while controlling for efficacy
4. Only proceed if GFlowNet shows clear improvement

---

## 7. Summary: Evolution, Not Replacement

GFlowNet is the **natural evolution** of Project 14's approach:

```
Project 14 (GRPO)              GFlowNet (Proposed)
─────────────────────────────────────────────────────
✓ High-efficacy candidates  →  ✓ High-efficacy candidates
✓ Diversity via penalty     →  ✓ Diversity built-in
✓ Production-ready          →  ○ Research prototype (2 months)
```

The goal is not to replace GRPO, but to add a complementary tool that better serves wet-lab needs when diverse candidates are the priority.

---

**Prepared by:** Computational ISDD Team
**Date:** December 22, 2025
**Related Project:** P14 (RL for De Novo Peptide Generation)
