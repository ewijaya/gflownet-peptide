# Phase 2 PRD: Theoretical Alignment Analysis

## Overview

This document analyzes how our Phase 2 PRD aligns with the foundational GFlowNet papers (Bengio 2021, Malkin 2022) while maintaining differentiation from Jain 2022.

---

## 1. Core GFlowNet Theory (Bengio 2021)

### Key Theoretical Requirements

| Requirement | Bengio 2021 Definition | Our PRD Status |
|-------------|------------------------|----------------|
| **Flow Network** | DAG with states S, actions A, terminal states X | ✅ Implemented via vocabulary + autoregressive generation |
| **Flow Consistency** | Σ F(s'→s) = R(s) + Σ F(s→s'') | ✅ Enforced via TB loss |
| **Policy Definition** | π(a\|s) = F(s,a) / F(s) | ✅ ForwardPolicy outputs this |
| **Sampling Property** | π(x) ∝ R(x) | ✅ Goal of training |
| **Non-negative Rewards** | R(x) > 0 required | ✅ exp/softplus transforms in reward |

### Flow Matching vs Trajectory Balance

**Bengio 2021** proposed the original **Flow Matching (FM)** objective (Eq. 12):
```
L_FM = Σ (log[ε + Σ exp F_θ(s,a)] - log[ε + R(s') + Σ exp F_θ(s',a')])²
```

**Our PRD correctly uses Trajectory Balance** (from Malkin 2022), which is superior for:
- Credit assignment over long sequences (peptides are 10-30 AA)
- Autoregressive generation (tree-structured DAG)

### Important: DAG vs Tree Structure

**Bengio 2021 Key Insight** (Proposition 1): When multiple action sequences lead to the same state (non-injective C), naive methods fail catastrophically:
```
π(x) = n(x)R(x) / Σ n(x')R(x')  ← WRONG! Biased by path count
```

**For our peptide generation**: The state space IS a tree (bijective), because:
- Each partial sequence [A, R, G] has exactly one construction path
- No two action sequences produce the same partial peptide

This means **P_B = 1** (uniform backward policy) is correct, simplifying TB loss.

---

## 2. Trajectory Balance (Malkin 2022)

### Core Equation

For trajectory τ = (s₀→s₁→...→sₙ=x):
```
Z · Π P_F(sₜ|sₜ₋₁) = F(x) · Π P_B(sₜ₋₁|sₜ)
```

**For autoregressive generation** (tree DAG), this simplifies to (Eq. 16):
```
L_TB(τ) = (log Z_θ · Π P_F(sₜ|sₜ₋₁;θ) / R(x))²
```

### Our PRD Implementation (Activity 2.5)

```python
# From PRD - this is CORRECT for autoregressive case
tb_residual = self.log_z + sum_log_pf - log_rewards - sum_log_pb
loss = (tb_residual ** 2).mean()
```

Where `sum_log_pb = 0` for uniform backward policy.

### PRD Alignment Checklist

| Malkin 2022 Requirement | PRD Implementation | Status |
|-------------------------|-------------------|--------|
| Learnable log Z | `self.log_z = nn.Parameter(...)` | ✅ Correct |
| Log-domain computation | Products → sums under log | ✅ Correct |
| Higher LR for Z | Not explicitly mentioned | ⚠️ Should add |
| P_B = 1 for trees | `log_pb = torch.zeros(...)` | ✅ Correct |
| Full trajectory sampling | `TrajectorySampler` | ✅ Correct |

### Recommended PRD Update

Add to `configs/default.yaml`:
```yaml
training:
  log_z_lr_multiplier: 10.0  # Malkin 2022 recommends higher LR for Z
```

---

## 3. Differentiation from Jain 2022

### What Jain 2022 Did

| Component | Jain 2022 | Our Project |
|-----------|-----------|-------------|
| **Backbone** | MLP with one-hot encoding | ESM-2 pretrained embeddings |
| **Policy** | MLP (2 layers, 2048 dim) | Causal Transformer (4 layers) |
| **Reward** | Single property classifier | Multi-objective composite |
| **TB Loss** | Yes (same as Malkin 2022) | Yes |
| **Off-policy data** | γ mixing from dataset | Not planned (pure on-policy) |
| **Uncertainty** | Ensembles + UCB acquisition | Not in Phase 2 |
| **Active Learning** | N rounds with oracle | Single training run |

### Key Differentiators for Publication

1. **Representation Quality**
   - Jain: One-hot → MLP embedding
   - Ours: ESM-2 pretrained → captures evolutionary/structural info

2. **Multi-Objective Reward**
   - Jain: Single AMP classifier P(AMP|x)
   - Ours: R(x) = stability^w₁ × binding^w₂ × naturalness^w₃

3. **Structural Awareness**
   - Jain: None
   - Ours: ESMFold pLDDT integration (Phase 3+)

4. **Policy Architecture**
   - Jain: Simple MLP
   - Ours: Causal Transformer with positional encoding

5. **Benchmark**
   - Jain: DBAASP AMP data, TF-Bind-8, GFP
   - Ours: FLIP stability benchmark (more rigorous, public)

### What We Should NOT Do (to avoid redundancy)

- ❌ Don't implement active learning loop (Jain's main contribution)
- ❌ Don't add uncertainty estimation with ensembles
- ❌ Don't use UCB/EI acquisition functions
- ❌ Don't mix offline data (γ parameter)

### What We SHOULD Emphasize

- ✅ ESM-2 backbone superiority
- ✅ Multi-objective therapeutic design
- ✅ Stability prediction (not just activity)
- ✅ Transformer policy for sequence modeling
- ✅ FLIP benchmark reproducibility

---

## 4. PRD Corrections & Recommendations

### 4.1 Theoretical Corrections

**Current PRD (Activity 2.5)** has minor issues:

```python
# Current - slightly confusing variable naming
sum_log_pf = log_pf_trajectory.sum(dim=-1)
sum_log_pb = log_pb_trajectory.sum(dim=-1)  # Always 0 for us
```

**Recommended clarification**:
```python
# Clearer: For autoregressive generation, P_B = 1, so log P_B = 0
# TB Loss: L = (log Z + Σ log P_F - log R)²
sum_log_pf = log_pf_trajectory.sum(dim=-1)
# sum_log_pb is zero for tree-structured DAGs (autoregressive)
```

### 4.2 Missing from PRD

1. **Temperature for reward sharpening** (both Bengio and Jain use this):
   ```python
   R_sharp = R(x) ** β  # β > 1 focuses on high-reward modes
   ```
   PRD mentions `reward.temperature` but implementation unclear.

2. **Exploration policy mixing** (Bengio 2021, Eq. 10):
   ```python
   π_explore = (1 - δ) * P_F + δ * Uniform
   ```
   PRD has `δ: Uniform Policy Coefficient 0.001` but not explicitly in sampler.

3. **Higher learning rate for log Z** (Malkin 2022 footnote 3):
   Should be ~10x higher than policy parameters.

### 4.3 Correct Theoretical Framing

Our project should be framed as:

> "We apply GFlowNet with trajectory balance loss (Malkin et al., 2022) to therapeutic peptide generation, using pretrained protein language model embeddings (ESM-2) and multi-objective reward design. Unlike Jain et al. (2022) who used simple MLP architectures with single-property rewards, we leverage modern PLM representations and composite rewards targeting stability, binding, and naturalness simultaneously."

---

## 5. Implementation Verification Checklist

### Core GFlowNet Properties (Must Have)

- [ ] π(x) ∝ R(x) at convergence
- [ ] Non-negative rewards (exp/softplus transform)
- [ ] Learnable log Z parameter
- [ ] TB loss computed on full trajectories
- [ ] P_B = 1 for autoregressive generation
- [ ] Exploration via uniform mixing or temperature

### Differentiation from Jain 2022 (Must Have)

- [ ] ESM-2 embeddings for reward model
- [ ] Transformer policy (not MLP)
- [ ] Multi-objective reward (stability + binding + naturalness)
- [ ] FLIP benchmark evaluation
- [ ] NO active learning loop
- [ ] NO uncertainty/acquisition functions

### Nice to Have (Phase 3+)

- [ ] SubTB loss for variance reduction
- [ ] Off-policy data mixing
- [ ] Temperature annealing
- [ ] ESMFold structure validation

---

## 6. Summary

**Our PRD is theoretically sound** and correctly implements:
- Trajectory Balance loss (Malkin 2022)
- Uniform backward policy for tree-structured DAG
- Learnable partition function log Z

**Key differentiators from Jain 2022**:
- ESM-2 representations (vs one-hot MLP)
- Transformer policy (vs MLP)
- Multi-objective reward (vs single classifier)
- Stability focus (vs AMP activity)

**Minor PRD updates needed**:
1. Add `log_z_lr_multiplier: 10.0` to config
2. Clarify exploration policy mixing in sampler
3. Document reward temperature β usage
4. Add explicit note that P_B = 1 for autoregressive case

---

## References

1. Bengio et al. (2021). "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation." NeurIPS.
2. Malkin et al. (2022). "Trajectory Balance: Improved Credit Assignment in GFlowNets." NeurIPS.
3. Jain et al. (2022). "Biological Sequence Design with GFlowNets." ICML.
