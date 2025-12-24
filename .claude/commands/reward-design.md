---
description: Analyze reward function design from 3 expert perspectives for Deep Learning
allowed-tools: Read, Glob, WebSearch
---

# Three-Angle Perspective Analysis for Reward Function Design

Analyze the following reward function design challenge from 3 expert perspectives, then synthesize into a unified design.

## Topic

> **$ARGUMENTS**

---

## Step 0: Frame the Problem

Identify:
- **Task domain:** RL, RLHF, generative models, optimization, etc.
- **Current reward:** What signal exists (if any)?
- **Desired behavior:** What should the agent/model actually do?
- **Failure modes:** What could go wrong with naive rewards?

---

## Step 1: Detect Mode and Select Expert Trio

Based on the topic above, select the most appropriate analysis mode:

| Mode | Trigger Keywords | Expert Trio |
|------|------------------|-------------|
| **FORMULATION** | design, create, define, specify, new reward | Bengio (Causality) → Sutton (RL Foundations) → Ng (Reward Shaping) |
| **ALIGNMENT** | safe, aligned, human preference, RLHF, values | Russell (AI Safety) → Christiano (RLHF) → Bengio (GFlowNets) |
| **OPTIMIZATION** | sparse, dense, curriculum, credit assignment, exploration | Schulman (PPO/TRPO) → Bengio (Credit Assignment) → Silver (AlphaGo) |

**State your selected mode clearly before proceeding.**

---

## Step 2: Provide Each Perspective

### If FORMULATION mode:

**Perspective 1: Yoshua Bengio (Causal & Compositional Rewards)**
Thinks about causality, compositionality, and proper credit assignment. Asks:
- **Causal structure:** Does the reward capture causal effects or just correlations?
- **Compositionality:** Can the reward decompose into interpretable components?
- **GFlowNet view:** Should we reward modes/diversity rather than single optima?
- **Abstraction:** Is the reward at the right level of abstraction for learning?

Provide:
- **Key observations** (3-5 bullets from this viewpoint)
- **Unique concerns** others might miss
- **Recommended reward components** from this angle

**Perspective 2: Richard Sutton (RL Foundations)**
The voice of the Bitter Lesson. Asks:
- **Temporal structure:** Is the reward Markovian? Does it respect the MDP formalism?
- **Scalability:** Will this reward remain meaningful as the agent improves?
- **Simplicity:** Are we over-engineering? Would a simpler reward + more compute work?
- **Value function:** Can this reward be efficiently learned and propagated?

Provide:
- **Key observations** (3-5 bullets from this viewpoint)
- **Unique concerns** others might miss
- **Recommended reward components** from this angle

**Perspective 3: Andrew Ng (Reward Shaping & Practical Design)**
Pragmatic focus on making rewards actually work. Asks:
- **Shaping:** What potential-based shaping could accelerate learning without changing optimal policy?
- **Debugging:** How will you know if the reward is broken? What diagnostics?
- **Unintended optima:** What reward hacking could occur? Adversarial examples?
- **Engineering:** Is this reward numerically stable, differentiable, efficient to compute?

Provide:
- **Key observations** (3-5 bullets from this viewpoint)
- **Unique concerns** others might miss
- **Recommended reward components** from this angle

---

### If ALIGNMENT mode:

**Perspective 1: Stuart Russell (Inverse Reward Design)**
Thinks about the reward specification problem and AI safety. Asks:
- **Misspecification:** What's the gap between stated reward and true intent?
- **Assistance games:** Should the agent be uncertain about the reward?
- **Corrigibility:** Does the reward incentivize the agent to be correctable?
- **Side effects:** What negative externalities might be optimized for?

Provide:
- **Key observations** (3-5 bullets from this viewpoint)
- **Unique concerns** others might miss
- **Recommended safeguards** from this angle

**Perspective 2: Paul Christiano (RLHF & Human Feedback)**
Focuses on learning rewards from human preferences. Asks:
- **Comparison data:** Can humans reliably compare outcomes? At what granularity?
- **Reward model:** Should we learn a reward model? What architecture?
- **Distribution shift:** Will the learned reward generalize beyond training?
- **Iterated amplification:** Can we decompose complex judgments into simpler ones?

Provide:
- **Key observations** (3-5 bullets from this viewpoint)
- **Unique concerns** others might miss
- **Recommended reward learning approach** from this angle

**Perspective 3: Yoshua Bengio (GFlowNets & Diverse Solutions)**
Thinks about sampling proportional to reward rather than maximizing. Asks:
- **Diversity:** Is finding one optimum enough, or do we need mode coverage?
- **Flow matching:** Could a GFlowNet formulation be more appropriate than RL?
- **Composition:** Can rewards factor into independent sub-rewards?
- **Exploration:** Does the reward structure naturally encourage exploration?

Provide:
- **Key observations** (3-5 bullets from this viewpoint)
- **Unique concerns** others might miss
- **Recommended paradigm** from this angle

---

### If OPTIMIZATION mode:

**Perspective 1: John Schulman (Policy Optimization)**
Designer of PPO/TRPO. Focuses on making optimization work. Asks:
- **Variance:** Is the reward high-variance? Does it need baselining?
- **Density:** Sparse vs. dense? What's the credit assignment horizon?
- **Clipping:** Will extreme rewards destabilize training?
- **KL constraints:** Should we regularize toward a reference policy?

Provide:
- **Key observations** (3-5 bullets from this viewpoint)
- **Unique concerns** others might miss
- **Recommended optimization considerations** from this angle

**Perspective 2: Yoshua Bengio (Credit Assignment)**
Deep focus on the credit assignment problem. Asks:
- **Temporal credit:** How does reward information flow backward in time?
- **Structural credit:** Which components of the model should be updated?
- **Attention to reward:** Can attention mechanisms help assign credit?
- **Gradient flow:** Does the reward structure allow clean gradient signals?

Provide:
- **Key observations** (3-5 bullets from this viewpoint)
- **Unique concerns** others might miss
- **Recommended credit assignment approach** from this angle

**Perspective 3: David Silver (Strategic Rewards)**
Architect of AlphaGo/AlphaZero. Thinks about self-play and search. Asks:
- **Terminal vs. intermediate:** Should rewards be only at episode end?
- **Self-play:** Can the reward support adversarial or cooperative training?
- **Search integration:** How does this reward interact with planning/MCTS?
- **Auxiliary rewards:** What self-supervised signals could supplement the main reward?

Provide:
- **Key observations** (3-5 bullets from this viewpoint)
- **Unique concerns** others might miss
- **Recommended strategic considerations** from this angle

---

## Step 3: Synthesize Into Unified Reward Design

Merge all 3 perspectives into:

### Consensus
What all three experts agree on (2-3 points)

### Tensions
Where perspectives conflict and how to resolve them

### Proposed Reward Function

```python
# Pseudocode for the synthesized reward
def reward(state, action, next_state, info):
    # Component 1: [from Perspective 1]
    r1 = ...

    # Component 2: [from Perspective 2]
    r2 = ...

    # Component 3: [from Perspective 3]
    r3 = ...

    # Combination strategy
    return combine(r1, r2, r3)
```

### Design Rationale
Why each component exists and what failure mode it prevents

### Validation Plan
How to test that this reward produces intended behavior:
1. [Diagnostic 1]
2. [Diagnostic 2]
3. [Red-team scenario]

---

Begin with Step 0 (framing the problem), then Step 1 (detecting mode).
