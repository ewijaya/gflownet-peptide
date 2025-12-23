# GFlowNet for Diverse Peptide Generation: Deep Dive and Challenge

*Collected: 2025-12-22 10:45:21*

---

## 1. Go deeper on Idea F: GFlowNet Diverse Generation

# Deep Dive: Idea F — GFlowNet for Diverse Peptide Generation

## The Core Insight

Your P14 (GRPO) optimizes for **expected reward** → the generator converges to a narrow set of high-scoring sequences. This is mode collapse by design.

**GFlowNets solve this fundamentally.** Instead of maximizing reward, they sample sequences **proportionally to reward**:

$$P(x) \propto R(x)$$

If peptide A has reward 0.9 and peptide B has reward 0.3, GFlowNet samples A three times more often than B — but it **still samples B**. This gives you a diverse pool of good candidates, not just the single best.

---

## Why This Matters for Drug Discovery

| Problem with GRPO | How GFlowNet Fixes It |
|-------------------|----------------------|
| Converges to ~5 similar sequences | Samples entire high-fitness region |
| Needs explicit diversity penalty (hacky) | Diversity is intrinsic to objective |
| Can't explore alternative scaffolds | Naturally discovers multiple modes |
| Wet-lab needs variety for testing | Provides ranked diverse candidates |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GFlowNet Peptide Generator                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐                                                │
│  │   Initial State  │  s₀ = [START]                                  │
│  │      (empty)     │                                                │
│  └────────┬─────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐    ┌─────────────────┐                        │
│  │  Forward Policy  │    │  Backward Policy │                        │
│  │    P_F(s'|s)     │◄──►│    P_B(s|s')     │                        │
│  │  (Transformer)   │    │   (Transformer)  │                        │
│  └────────┬─────────┘    └─────────────────┘                        │
│           │                                                          │
│           ▼  Add amino acid at each step                             │
│  ┌──────────────────┐                                                │
│  │ s₁ = [START, M]  │                                                │
│  │ s₂ = [START,M,K] │                                                │
│  │ ...              │                                                │
│  │ sₙ = [full seq]  │                                                │
│  └────────┬─────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐    ┌─────────────────┐                        │
│  │  Terminal State  │───►│  Reward Model   │                        │
│  │   (complete x)   │    │    R(x) ≥ 0     │                        │
│  └──────────────────┘    │  (from FLIP/    │                        │
│                          │   ProteinGym)   │                        │
│                          └─────────────────┘                        │
│                                                                      │
│  Training Objective (Trajectory Balance):                            │
│                                                                      │
│    Z · ∏ P_F(sₜ₊₁|sₜ) = R(x) · ∏ P_B(sₜ|sₜ₊₁)                       │
│                                                                      │
│  where Z is the partition function (learned)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### State Space

- **State** $s$: partial peptide sequence $[a_1, a_2, ..., a_t]$ where $a_i \in \{20 \text{ amino acids}\}$
- **Initial state** $s_0$: empty sequence (or [START] token)
- **Terminal state**: complete sequence of length $L$ (fixed or variable)
- **Actions**: append one of 20 amino acids

### Flow Matching Condition

GFlowNets learn flows $F(s)$ on states such that:

$$\sum_{s': s \to s'} F(s \to s') = \sum_{s'': s'' \to s} F(s'' \to s) + R(s) \cdot \mathbb{1}[s \text{ is terminal}]$$

This is "flow conservation" — inflow = outflow + reward at terminal.

### Trajectory Balance Objective (TB)

For a complete trajectory $\tau = (s_0 \to s_1 \to ... \to s_n = x)$:

$$\mathcal{L}_{TB} = \left( \log \frac{Z \cdot \prod_{t=0}^{n-1} P_F(s_{t+1}|s_t)}{R(x) \cdot \prod_{t=1}^{n} P_B(s_{t-1}|s_t)} \right)^2$$

where:
- $P_F(s'|s)$ = forward policy (probability of taking action)
- $P_B(s|s')$ = backward policy (probability of "undoing" action)
- $Z$ = partition function (total flow, learned as log Z)
- $R(x)$ = reward for terminal sequence $x$

### Why This Works

At convergence:
- $P_F$ samples trajectories with probability $\propto R(x)$
- High-reward sequences are sampled more often
- But all positive-reward sequences have non-zero probability
- **Diversity emerges naturally** from the proportional sampling

---

## Reward Model Design

The reward must be **non-negative** and learned from public data.

### Option 1: ProteinGym Fitness Predictor

```python
class FitnessReward(nn.Module):
    def __init__(self):
        self.esm = ESM2.from_pretrained("esm2_t33_650M")
        self.head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, sequence: str) -> float:
        # Get ESM embedding
        emb = self.esm.encode(sequence).mean(dim=0)  # [1280]

        # Predict fitness
        fitness = self.head(emb)  # scalar

        # Transform to non-negative reward
        # Option A: exp(fitness) — always positive
        # Option B: softplus(fitness) — smoother
        # Option C: max(0, fitness + offset)

        return torch.exp(fitness)  # R(x) ≥ 0
```

**Training data:** ProteinGym has 2.5M variants with fitness labels. Train regressor on ESM embeddings.

### Option 2: Multi-Objective Reward (FLIP)

```python
class MultiObjectiveReward(nn.Module):
    def __init__(self, objectives=['binding', 'stability', 'activity']):
        self.predictors = nn.ModuleDict({
            obj: FitnessPredictor() for obj in objectives
        })
        self.weights = nn.Parameter(torch.ones(len(objectives)))

    def forward(self, sequence: str) -> float:
        scores = [self.predictors[obj](sequence) for obj in self.predictors]
        # Weighted geometric mean (all must be good)
        reward = torch.prod(torch.stack(scores) ** self.weights)
        return reward
```

### Option 3: Constraint-Satisfying Reward

```python
def constrained_reward(sequence, fitness_pred, constraint_preds):
    fitness = fitness_pred(sequence)

    # Hard constraints (must satisfy)
    constraints_met = all(
        pred(sequence) > threshold
        for pred, threshold in constraint_preds
    )

    if not constraints_met:
        return 0.0  # Zero reward if any constraint violated

    return torch.exp(fitness)
```

---

## Implementation

### Forward Policy (Transformer)

```python
class ForwardPolicy(nn.Module):
    """
    P_F(a | s): probability of adding amino acid a given partial sequence s
    """
    def __init__(self, vocab_size=21, d_model=256, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=512),
            num_layers=n_layers
        )
        self.action_head = nn.Linear(d_model, vocab_size)

    def forward(self, partial_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            partial_seq: [batch, seq_len] token indices
        Returns:
            logits: [batch, vocab_size] action probabilities
        """
        x = self.embedding(partial_seq)  # [B, L, D]
        x = self.pos_enc(x)
        x = self.transformer(x)

        # Use last position for next-token prediction
        logits = self.action_head(x[:, -1, :])  # [B, vocab_size]
        return logits

    def log_prob(self, partial_seq, action):
        logits = self.forward(partial_seq)
        return F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
```

### Backward Policy

```python
class BackwardPolicy(nn.Module):
    """
    P_B(s | s'): probability of removing the last amino acid
    For linear sequence generation, this is deterministic:
    P_B(s | s') = 1 if s = s'[:-1] else 0

    But we parameterize it for flexibility (variable-length, branching)
    """
    def __init__(self, vocab_size=21, d_model=256, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8),
            num_layers=n_layers
        )
        # For simple linear generation, backward is uniform
        # P_B = 1/t where t is current length
        self.use_uniform = True

    def log_prob(self, current_seq, parent_seq):
        if self.use_uniform:
            # Uniform backward: P_B = 1 (only one parent possible)
            return torch.zeros(current_seq.shape[0], device=current_seq.device)
        else:
            # Learned backward (for more complex state spaces)
            ...
```

### Training Loop

```python
class GFlowNetTrainer:
    def __init__(self, forward_policy, backward_policy, reward_model,
                 max_len=20, lr=1e-4):
        self.P_F = forward_policy
        self.P_B = backward_policy
        self.R = reward_model
        self.max_len = max_len

        # Learnable log partition function
        self.log_Z = nn.Parameter(torch.tensor(0.0))

        self.optimizer = torch.optim.Adam(
            list(self.P_F.parameters()) +
            list(self.P_B.parameters()) +
            [self.log_Z],
            lr=lr
        )

    def sample_trajectory(self, batch_size=32):
        """Sample complete trajectories using forward policy"""
        sequences = [[START_TOKEN] for _ in range(batch_size)]
        log_pf_sum = torch.zeros(batch_size)
        log_pb_sum = torch.zeros(batch_size)

        for t in range(self.max_len):
            # Get current partial sequences as tensor
            current = self._to_tensor(sequences)

            # Sample next action from P_F
            logits = self.P_F(current)
            actions = Categorical(logits=logits).sample()
            log_pf = F.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            log_pf_sum += log_pf

            # Append actions to sequences
            for i, a in enumerate(actions.tolist()):
                sequences[i].append(a)

            # Compute backward log prob (uniform = 0 for linear generation)
            log_pb_sum += 0  # P_B = 1 for deterministic parent

        return sequences, log_pf_sum, log_pb_sum

    def trajectory_balance_loss(self, sequences, log_pf_sum, log_pb_sum):
        """Compute TB loss for a batch of trajectories"""
        # Compute rewards for terminal sequences
        rewards = torch.stack([self.R(self._decode(seq)) for seq in sequences])
        log_rewards = torch.log(rewards + 1e-8)

        # TB loss: (log Z + log P_F - log R - log P_B)^2
        loss = (self.log_Z + log_pf_sum - log_rewards - log_pb_sum) ** 2
        return loss.mean()

    def train_step(self, batch_size=32):
        self.optimizer.zero_grad()

        # Sample trajectories
        sequences, log_pf, log_pb = self.sample_trajectory(batch_size)

        # Compute loss
        loss = self.trajectory_balance_loss(sequences, log_pf, log_pb)

        # Backprop
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

### Sampling Diverse Peptides

```python
def sample_diverse_peptides(gflownet, n_samples=1000, temperature=1.0):
    """
    Sample peptides proportionally to their reward.
    Higher temperature → more uniform sampling
    Lower temperature → more focused on high reward
    """
    peptides = []
    rewards = []

    for _ in range(n_samples):
        # Sample trajectory
        seq, _, _ = gflownet.sample_trajectory(batch_size=1)
        peptide = gflownet._decode(seq[0])
        reward = gflownet.R(peptide).item()

        peptides.append(peptide)
        rewards.append(reward)

    # Return sorted by reward, but maintaining diversity
    df = pd.DataFrame({'peptide': peptides, 'reward': rewards})
    df = df.drop_duplicates(subset='peptide')
    df = df.sort_values('reward', ascending=False)

    return df
```

---

## Datasets

| Dataset | Source | Use | Size |
|---------|--------|-----|------|
| **ProteinGym** | proteingym.org | Fitness prediction training | 2.5M variants |
| **FLIP** | FLIP benchmark | Multi-property reward | 5 tasks |
| **Propedia** | propedia.org | Binding-aware reward | 19K complexes |
| **UniRef50** | UniProt | Pre-training P_F | 50M sequences |

### Data Pipeline

```python
# Step 1: Train reward model on ProteinGym
reward_model = train_fitness_predictor(
    data=load_proteingym(),
    encoder='esm2_t33_650M',
    epochs=50
)

# Step 2: Pre-train forward policy on UniRef50 peptides
forward_policy = pretrain_lm(
    data=filter_peptides(load_uniref50(), max_len=50),
    epochs=10
)

# Step 3: Train GFlowNet with reward
gflownet = GFlowNetTrainer(forward_policy, backward_policy, reward_model)
for epoch in range(1000):
    loss = gflownet.train_step(batch_size=64)
```

---

## Experiments

### Experiment 1: Diversity Comparison

**Question:** Does GFlowNet produce more diverse high-quality peptides than GRPO?

**Setup:**
- Train GFlowNet and GRPO on same reward model (ProteinGym fitness)
- Generate 1000 peptides from each
- Filter to top 10% by reward

**Metrics:**
- **Sequence diversity:** average pairwise edit distance
- **Embedding diversity:** average pairwise cosine distance in ESM space
- **Cluster count:** number of distinct clusters (UMAP + HDBSCAN)
- **Mode coverage:** % of known fitness peaks discovered

**Expected result:** GFlowNet achieves comparable reward but 2-3× higher diversity.

### Experiment 2: Reward Proportionality

**Question:** Does sampling frequency match reward magnitude?

**Setup:**
- Generate 10,000 peptides
- Bin by predicted reward (0-0.2, 0.2-0.4, ..., 0.8-1.0)
- Compare empirical frequency to theoretical (proportional to reward)

**Metrics:**
- KL divergence between empirical and theoretical distributions
- Correlation between log(frequency) and log(reward)

**Expected result:** Near-linear relationship, confirming GFlowNet is sampling correctly.

### Experiment 3: Multi-Objective Pareto Coverage

**Question:** Can GFlowNet discover the Pareto frontier for multiple objectives?

**Setup:**
- Use FLIP with 3 objectives: binding, stability, activity
- Train GFlowNet with product reward: R = binding × stability × activity
- Generate 5000 peptides

**Metrics:**
- Hypervolume of discovered Pareto front
- Number of Pareto-optimal solutions
- Spread along each objective axis

**Expected result:** GFlowNet discovers diverse Pareto frontier, not just one corner.

### Experiment 4: Transfer to Novel Targets

**Question:** Do GFlowNet-generated peptides generalize to unseen fitness landscapes?

**Setup:**
- Train on ProteinGym subset (e.g., enzymes)
- Evaluate on held-out families (e.g., binding proteins)
- Compare to GRPO and random baseline

**Metrics:**
- Mean fitness on held-out landscape
- Diversity of successful transfers

**Expected result:** GFlowNet's diversity enables better transfer than mode-collapsed GRPO.

---

## Implementation Checklist

```
[ ] Environment setup
    [ ] Install torch, transformers, esm
    [ ] Set up wandb for logging
    [ ] GPU allocation (1x A100 sufficient)

[ ] Reward model
    [ ] Download ProteinGym
    [ ] Train ESM2 → fitness regressor
    [ ] Validate on held-out proteins
    [ ] Implement non-negative transform (exp or softplus)

[ ] GFlowNet components
    [ ] Implement ForwardPolicy (Transformer)
    [ ] Implement BackwardPolicy (uniform for linear generation)
    [ ] Implement trajectory sampling
    [ ] Implement TB loss

[ ] Training
    [ ] Pre-train P_F on UniRef50 peptides (optional but helps)
    [ ] Train GFlowNet with TB loss
    [ ] Monitor: loss, log_Z, sample diversity, sample quality
    [ ] Checkpoint best models

[ ] Evaluation
    [ ] Implement diversity metrics
    [ ] Run Experiment 1 (vs GRPO)
    [ ] Run Experiment 2 (proportionality check)
    [ ] Run Experiment 3 (multi-objective)

[ ] Analysis
    [ ] Visualize sampled peptides (UMAP)
    [ ] Analyze sequence motifs in top samples
    [ ] Compare to P14 GRPO outputs
```

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Reward model is poor predictor | Medium | High | Validate fitness predictor R² > 0.7 before GFlowNet training |
| Mode collapse despite GFlowNet | Low | High | Check proportionality (Exp 2); increase exploration with temperature |
| Training instability (log_Z diverges) | Medium | Medium | Clip gradients; use SubTB instead of TB if unstable |
| Generated peptides not synthesizable | Medium | Medium | Add synthesizability filter as reward component |
| Slow sampling (Transformer overhead) | Low | Low | Batch sampling; use smaller model for production |

---

## Comparison to Your GRPO (P14)

| Aspect | GRPO (P14) | GFlowNet (Idea F) |
|--------|------------|-------------------|
| **Objective** | Maximize E[R] | Sample ∝ R |
| **Diversity** | Needs explicit penalty | Intrinsic |
| **Mode collapse** | Yes, fundamental | No, by design |
| **Training signal** | Reward difference | Trajectory balance |
| **Sample efficiency** | Good | Similar |
| **Implementation** | You have it | New implementation |
| **Interpretability** | Direct optimization | Probabilistic sampling |

**Recommendation:** GFlowNet is a **philosophical upgrade** to GRPO. It's the right tool when you need diverse candidates for wet-lab testing, not just the single predicted-best sequence.

---

## Publication Angle

**Title:** "GFlowNet-Peptide: Diverse Therapeutic Peptide Generation via Reward-Proportional Sampling"

**Story:**
1. Problem: Existing RL methods mode-collapse to narrow sequence families
2. Solution: GFlowNet samples proportionally to fitness → intrinsic diversity
3. Result: 3× more diverse candidates at equivalent fitness levels
4. Impact: Better coverage of sequence space for experimental validation

**Target venues:** NeurIPS (ML methods), ICML (generative models), Nature Machine Intelligence (application)

---

**Next steps:**
- `challenge: Idea F` — stress-test this design
- `prepare for prototype: Idea F` — sanitize for public repo

---

## 2. Challenge Idea F: Stress-test the GFlowNet approach

# Challenge: Idea F — GFlowNet for Diverse Peptide Generation

Let me stress-test this idea ruthlessly.

---

## Challenge 1: Is Diversity Actually the Problem?

### The Concern

You claim GRPO has "mode collapse" and that's bad. But is it?

- **Your P14 achieves diversity 0.85-0.89** with explicit diversity penalty (GRPO-D)
- **Mean PEM 0.86-0.89, Max PEM 0.954** — these are good numbers
- If GRPO-D already works, why add GFlowNet complexity?

### What Breaks

If GRPO-D's diversity is sufficient for wet-lab testing, then GFlowNet is:
- More complex to implement
- Harder to debug
- Solving a non-problem

### Counter-Evidence Needed

| Question | How to Verify |
|----------|---------------|
| Is 0.85 diversity enough? | Ask wet-lab: do they need more scaffold variety? |
| Does GRPO-D miss modes? | Compare cluster count: GRPO-D vs random sampling from high-fitness region |
| Is diversity penalty hacky? | Check if diversity-reward trade-off is stable across hyperparameters |

### Verdict

**Risk: MEDIUM.** You might be over-engineering. GRPO-D with diversity penalty might be "good enough."

**Mitigation:** Before building GFlowNet, quantify GRPO-D's failure cases. Show specific examples where it misses important scaffolds.

---

## Challenge 2: Reward Model Bottleneck

### The Concern

GFlowNet samples proportionally to R(x). If R(x) is wrong, you're sampling proportionally to **garbage**.

Your reward model is:
- ESM-2 embedding → MLP → fitness
- Trained on ProteinGym (mostly proteins, not therapeutic peptides)
- Never validated on your actual domain

### What Breaks

| Failure Mode | Consequence |
|--------------|-------------|
| R(x) overestimates some sequences | GFlowNet over-samples bad peptides |
| R(x) underestimates good sequences | GFlowNet misses promising candidates |
| R(x) is miscalibrated | Proportionality is meaningless |
| ProteinGym ≠ therapeutic peptides | Entire reward signal is domain-shifted |

### The Math Problem

GFlowNet guarantees $P(x) \propto R(x)$. But:

$$P(x) \propto R(x) \neq P(x) \propto R^*(x)$$

where $R^*(x)$ is the true fitness you care about.

**If R(x) has systematic bias, diversity won't save you.**

### Verdict

**Risk: HIGH.** This is the same problem as GRPO — you're optimizing a proxy. GFlowNet doesn't fix the fundamental issue of reward model quality.

**Mitigation:**
1. Validate reward model on held-out experimental data (if any public)
2. Use ensemble of reward models to estimate uncertainty
3. Consider: is GFlowNet's diversity worth it if the reward is unreliable anyway?

---

## Challenge 3: ProteinGym ≠ Peptides

### The Concern

ProteinGym contains:
- **Deep mutational scanning** of full proteins (100-500 AA)
- **Mostly enzymes and binding proteins**
- **Fitness = enzymatic activity, binding affinity, stability**

Your peptides are:
- **Short sequences** (10-30 AA)
- **Therapeutic mechanism** (LRP1 binding, regenerative effects)
- **Fitness = PEM** (your internal metric)

### Domain Gap Analysis

| Aspect | ProteinGym | Your Peptides |
|--------|------------|---------------|
| Length | 100-500 AA | 10-30 AA |
| Diversity | Natural protein families | Designed therapeutics |
| Fitness definition | Biochemical assays | Therapeutic efficacy |
| Structure | Globular, folded | Often disordered/flexible |
| Training distribution | Natural evolution | Designed sequences |

### What Breaks

A reward model trained on ProteinGym may:
- Not generalize to short peptides (different embedding behavior)
- Miss therapeutic-specific features (LRP1 binding, cell penetration)
- Overfit to protein-like features irrelevant for peptides

### Verdict

**Risk: HIGH.** Domain shift is severe. ProteinGym-trained models may be useless for therapeutic peptides.

**Mitigation:**
1. Use **FLIP benchmark** instead — it has actual peptide data
2. Fine-tune on peptide-specific data (Propedia, PDBBind peptides)
3. Validate reward model correlation with known peptide activities

---

## Challenge 4: Trajectory Balance Training is Finicky

### The Concern

GFlowNet training is known to be unstable:

- **log_Z can diverge** if reward scale is wrong
- **Credit assignment over long trajectories** (20+ steps for peptides)
- **Off-policy exploration** is tricky to balance
- **Backward policy design** affects convergence

### Known Failure Modes

| Issue | Symptom | Cause |
|-------|---------|-------|
| log_Z explosion | Loss → NaN | Reward scale too large |
| Mode collapse anyway | Low diversity despite GFlowNet | Insufficient exploration |
| Slow convergence | Loss plateaus high | Poor backward policy |
| Reward hacking | High R(x) but bad peptides | Reward model exploited |

### Comparison to GRPO

| Aspect | GRPO | GFlowNet |
|--------|------|----------|
| Training stability | Well-understood | Research-level |
| Hyperparameter sensitivity | Moderate | High |
| Debugging | Standard RL tools | Specialized |
| Community support | Large (PPO variants) | Small (Bengio lab + few groups) |

### Verdict

**Risk: MEDIUM-HIGH.** GFlowNet training requires expertise you may not have. Expect 2-3 weeks of debugging before it works.

**Mitigation:**
1. Start with **SubTB (Sub-Trajectory Balance)** — more stable than full TB
2. Use existing codebase (torchgfn) rather than implementing from scratch
3. Budget extra time for hyperparameter tuning

---

## Challenge 5: Is Proportional Sampling What You Want?

### The Concern

GFlowNet samples $P(x) \propto R(x)$. But do you actually want this?

Consider:
- Peptide A: R = 0.95, novel scaffold
- Peptide B: R = 0.90, similar to known drug
- Peptide C: R = 0.50, very novel scaffold

GFlowNet samples:
- A: 40% of time
- B: 38% of time
- C: 21% of time

But maybe you want:
- All three equally (for diversity)
- Or A and C more (for novelty)
- Or just A (for best candidate)

**Proportional sampling is one specific trade-off. Is it the right one?**

### Alternative Objectives

| Objective | What It Does | When to Use |
|-----------|--------------|-------------|
| Max E[R] | Best expected candidate | You trust your reward |
| Sample ∝ R | Proportional diversity | You want coverage |
| Sample ∝ R^β | Temperature-controlled | Tune exploration/exploitation |
| Uniform over {x: R(x) > τ} | All good candidates equally | When diversity matters most |
| Max diversity s.t. R(x) > τ | Diverse good candidates | Explicit diversity constraint |

### Verdict

**Risk: LOW-MEDIUM.** Proportional sampling is reasonable but not obviously optimal. The temperature parameter (R^β) helps, but you're still locked into a specific trade-off.

**Mitigation:** Treat GFlowNet as one tool in the toolkit. Compare to simpler approaches like "rejection sampling from GRPO with diversity filter."

---

## Challenge 6: Computational Cost

### The Concern

GFlowNet requires:
- Forward pass through reward model **for every sampled sequence**
- Trajectory sampling is sequential (not batched across steps easily)
- Training requires many trajectories per update

### Cost Comparison

| Method | Samples per Training Step | Reward Evaluations | Wall Time (estimate) |
|--------|---------------------------|--------------------|-----------------------|
| GRPO | 64 | 64 | 1× baseline |
| GFlowNet (TB) | 64 trajectories × 20 steps | 64 | 2-3× baseline |
| GFlowNet (SubTB) | 64 sub-trajectories | 64 | 1.5× baseline |

### What Breaks

If reward model is expensive (ESM-2 + MLP):
- Training time 2-3× longer than GRPO
- May not be worth the diversity gain

### Verdict

**Risk: LOW.** Compute is manageable on single GPU. But factor in extra time.

**Mitigation:** Use smaller ESM model (esm2_t6_8M) for faster reward evaluation during training. Validate with larger model only at the end.

---

## Challenge 7: Prior Art — Has This Been Done?

### What Exists

| Paper | What They Did | Gap Remaining |
|-------|---------------|---------------|
| **GFlowNet for molecules** (Bengio 2021) | Small molecule generation | Not peptides |
| **Biological sequence design GFlowNet** (2023) | DNA/RNA aptamers | Not peptides, not therapeutic |
| **Multi-objective GFlowNet** (2023) | Pareto sampling for molecules | Not applied to peptides |
| **GFlowNet for antibodies** (2024) | CDR loop generation | Antibodies ≠ peptides |

### The Novelty Claim

Your claim: "GFlowNet for therapeutic peptide generation with fitness-proportional sampling"

**Is it valid?**
- GFlowNet for sequences: exists
- GFlowNet for peptides specifically: **limited** (mostly antibodies)
- GFlowNet for therapeutic peptide optimization: **novel angle**

### What Could Scoop You

- Bengio lab is prolific — they might publish peptide GFlowNet any time
- Antibody GFlowNet papers are close — easy extension to peptides

### Verdict

**Risk: MEDIUM.** Novelty is defensible but the field is moving fast. Execute quickly.

**Mitigation:** Differentiate on the therapeutic angle + comparison to GRPO on same reward. Your unique contribution is the direct comparison showing diversity benefits.

---

## Challenge 8: Does Diversity Actually Help Wet-Lab?

### The Concern

You assume "more diverse candidates = better for drug discovery." But:

- Wet-lab can only test **10-50 peptides** per batch
- They want **high-confidence candidates**, not exploration
- Synthesizing diverse scaffolds is more expensive
- Diversity for its own sake wastes resources

### The Real Question

**Would your wet-lab collaborators prefer:**

A) 20 diverse peptides, each with 70% confidence of working

B) 20 similar peptides, each with 90% confidence of working

If B, then GFlowNet's diversity is a liability, not an asset.

### Verdict

**Risk: MEDIUM.** Diversity is good for exploration, but exploitation matters for clinical progress. GFlowNet might be the wrong tool if you're past the exploration phase.

**Mitigation:** Ask your stakeholders what they actually need. If they want reliable candidates, not diverse exploration, stick with GRPO.

---

## Summary: What Actually Might Break

| Component | Risk Level | Most Likely Failure Mode |
|-----------|------------|--------------------------|
| Problem framing (is diversity needed?) | MEDIUM | GRPO-D already sufficient |
| Reward model quality | HIGH | ProteinGym → peptide domain shift |
| ProteinGym data relevance | HIGH | Proteins ≠ short therapeutic peptides |
| GFlowNet training stability | MEDIUM-HIGH | log_Z divergence, slow convergence |
| Proportional sampling optimality | LOW-MEDIUM | May not be the right trade-off |
| Novelty claim | MEDIUM | Field moving fast, could be scooped |
| Wet-lab utility | MEDIUM | Stakeholders may prefer exploitation |

---

## Revised Recommendation

Given these challenges, here's a **de-risked approach**:

### Phase 0: Validate the Need (Before Any Implementation)

1. **Quantify GRPO-D's diversity gap**
   - Generate 1000 peptides with GRPO-D
   - Cluster them (UMAP + HDBSCAN)
   - Count distinct scaffolds
   - Compare to random sampling from high-fitness region

2. **Ask wet-lab stakeholders**
   - "Do you want more diversity or more confidence?"
   - "What's your testing budget per batch?"

**If GRPO-D diversity is sufficient OR stakeholders want exploitation → don't build GFlowNet**

### Phase 1: Fix the Reward Model First

Before GFlowNet, ensure reward model is valid:

1. Train on **FLIP** (actual peptides), not ProteinGym
2. Validate on held-out peptide fitness data
3. Check correlation with any available experimental results
4. Ensemble multiple reward models

**If reward model R² < 0.5 on peptide data → fix reward before GFlowNet**

### Phase 2: Simple Baseline First

Instead of full GFlowNet, try:

```python
# Simple diverse sampling baseline
def diverse_sample_grpo(grpo_model, n_samples=100, diversity_threshold=0.3):
    candidates = []
    while len(candidates) < n_samples:
        # Sample from GRPO
        new_peptide = grpo_model.sample()

        # Check diversity against existing
        if all(edit_distance(new_peptide, c) > diversity_threshold
               for c in candidates):
            candidates.append(new_peptide)

    return candidates
```

This is simpler, uses your existing GRPO, and achieves diversity through rejection.

**If simple baseline achieves comparable diversity → GFlowNet not needed**

### Phase 3: GFlowNet Only If Justified

If Phases 0-2 show GFlowNet is needed:

1. Use **torchgfn** library (don't implement from scratch)
2. Start with **SubTB** objective (more stable)
3. Use FLIP-trained reward model
4. Compare rigorously to GRPO-D baseline

---

## The Honest Assessment

**Is Idea F novel?** Yes, for therapeutic peptides specifically.

**Is Idea F feasible?** Yes, but requires careful implementation.

**Is Idea F necessary?** **Unclear.** GRPO-D might already solve the diversity problem.

**Should you pursue it?**
- **If** GRPO-D diversity is insufficient AND reward model is solid → Yes
- **If** GRPO-D already works → No, it's over-engineering
- **If** reward model is poor → Fix reward first, then decide

**What's the real contribution?** If you build this, the paper is "GFlowNet vs GRPO for therapeutic peptides: when does proportional sampling help?" That's a useful empirical contribution regardless of which wins.

---

**Next steps:**
- Run Phase 0 analysis on your existing GRPO-D outputs
- `go deeper: Idea E` or `Idea N` — alternatives with clearer value proposition
- `prepare for prototype: Idea F` — if you want to proceed despite risks
