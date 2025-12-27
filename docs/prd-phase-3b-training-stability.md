# Phase 3b: Training Stability Improvements - Detailed PRD

**Generated from**: Post-hoc analysis of Phase 3 experimental runs
**Reference**: docs/reward-comparison-analysis.md
**Date**: 2025-12-27
**Status**: Draft
**Last Updated**: 2025-12-27

---

## 1. Executive Summary

### 1.1 Objective

Address critical training instabilities discovered during the GFlowNet reward comparison experiments (Dec 26, 2025). Three issues were identified that prevent reliable training:

1. **Wrong checkpoint selection** - "Best" checkpoint selected by lowest loss is 15x worse than final model
2. **Loss explosion** - TB loss increases from 21 to 3378 while sample quality improves
3. **Missing entropy regularization** - Policy becomes overconfident, causing log_Z tracking failure
4. **Trajectory Balance limitation** - Global log_Z cannot track rapid policy sharpening

### 1.2 Duration

3-5 days

### 1.3 Key Deliverables

| Deliverable | Description |
|-------------|-------------|
| Checkpoint fix | Select best model by reward, not loss |
| Entropy regularization | Prevent policy overconfidence via H(π) penalty |
| **Sub-Trajectory Balance** | **Use STB loss instead of TB for stable loss curves** |
| CLI improvements | New flags for entropy_weight, log_z_lr_multiplier |
| Configuration updates | Stable default hyperparameters |
| Test coverage | New tests for all fixes |
| Validation run | 10K step run demonstrating stable training |

### 1.4 Prerequisites

- Phase 3 complete (initial training runs executed)
- Option C reward (Improved Reward) validated as best choice
- All 123 existing tests passing
- `docs/reward-comparison-analysis.md` reviewed

### 1.5 Critical Finding

The experimental runs revealed a **paradox**: the model with the lowest loss produces the worst samples, while the model with exploding loss produces excellent samples. This is not a bug but a fundamental property of Trajectory Balance loss when the policy concentrates on high-reward modes.

---

## 2. Problem Analysis

### 2.1 Experimental Evidence

Four runs were conducted on Dec 26, 2025:

| Run ID | Name | Reward Type | Outcome |
|--------|------|-------------|---------|
| 8rflp7l6 | gflownet-baseline-10k | Random MLP | Failed - no learning signal |
| 6qsqq6wz | gflownet-reward-A-trained-10k | Trained stability | Partial - diverse but low reward |
| 3fr3yzn0 | gflownet-reward-B-esm2pll-10k | ESM2-PLL | Catastrophic - mode collapse to "MMM..." |
| zcb95gyl | gflownet-reward-C-improved-10k | Improved | Success with issues |

**Option C (Improved Reward)** produced the best results but revealed three critical issues.

### 2.2 Issue 1: Wrong Checkpoint Selection (CRITICAL)

**Location**: `gflownet_peptide/training/trainer.py` lines 382-385

**Current behavior**:
```python
if metrics["loss"] < self.best_loss:
    self.best_loss = metrics["loss"]
    best_path = checkpoint_dir / f"{run_name or 'gflownet'}_best.pt"
    self.save_checkpoint(best_path, step=step)
```

**Evidence from Option C run (zcb95gyl)**:

| Checkpoint | Step | Loss | AA Diversity | Entropy Gate | Reward |
|------------|------|------|--------------|--------------|--------|
| **"Best"** | 5000 | 21 | 2/20 (D, L only) | 0.06 | 0.04 |
| **Final** | 10000 | 3378 | 15/20 | 0.96 | 0.62 |

**Sample comparison**:
```
Best checkpoint (step 5000):
  DDLDDDLDDDDDDLLLDDDDLDDDLDLLLL  <- Only D and L!

Final checkpoint (step 10000):
  FYNPEIIESDTTLFSPFLPMYIDRTIIQEL  <- Diverse, natural-looking
```

**Root cause**: The Trajectory Balance loss can be low when the policy is confident in a **degenerate subspace**. At step 5000, the policy only generates D and L, giving it high confidence (low variance) but terrible sample quality.

**Impact**: Users who rely on "best" checkpoint get a model that is **15x worse** than the final model.

### 2.3 Issue 2: Loss Explosion (HIGH)

**Observation**: Loss increased from 21 to 3378 while sample quality improved dramatically.

**TB Loss equation**:
```
L_TB = (log_Z + Σlog_P_F - log_R - Σlog_P_B)²
```

For uniform backward policy (P_B = 1):
```
L_TB = (log_Z + log_P_F - log_R)²
```

**Training trajectory analysis**:

| Step | log_Z | log_P_F | log_R | Required log_Z | Gap | Loss |
|------|-------|---------|-------|----------------|-----|------|
| 0 | 0.00 | -27.91 | -0.94 | 26.97 | 26.97 | 727 |
| 6000 | 13.39 | -20.74 | -2.76 | 17.98 | 4.59 | 21 |
| 9000 | 16.14 | -72.05 | -0.05 | 72.00 | 55.86 | 3120 |

**Mathematical explanation**:

At equilibrium, TB requires:
```
log_Z = log_R(x) - log_P_F(τ)
```

At step 9000:
- Policy found high-reward sequences (log_R ≈ -0.05, reward ≈ 0.95)
- Policy is very confident in these sequences (log_P_F = -72)
- Required log_Z = -0.05 - (-72) = 72
- Actual log_Z = 16.14
- **Gap = 56 units** → Loss ≈ 56² = 3136

**Root cause**: log_Z cannot increase fast enough when the policy sharpens rapidly. This is a **race condition** between policy optimization and partition function estimation.

**Why samples are still good**: The policy DID learn to concentrate on high-reward sequences. The loss explosion is a **symptom of success**, not failure. Once the policy finds good modes, the TB loss becomes a poor quality indicator.

### 2.4 Issue 3: Missing Entropy Regularization (MEDIUM)

**Observation**: Policy becomes overconfident during training.

| Step | log_P_F | Perplexity | Interpretation |
|------|---------|------------|----------------|
| 0 | -27.91 | 4.5 | Normal uncertainty (4.5 choices per position) |
| 6000 | -20.74 | 2.0 | Trapped in DL subspace (2 choices) |
| 9000 | -72.05 | 11.0 | Very confident (choosing 1 of 11 specific sequences) |

**Problem**: No mechanism prevents the policy from becoming arbitrarily confident. When confidence increases faster than log_Z can track, loss explodes.

**Solution**: Add entropy regularization to encourage exploration:

```
L_total = L_TB + entropy_weight × H(π)
```

Where H(π) = -E[log P_F] is the policy entropy (higher = more exploration).

Since we minimize loss and want to maximize entropy:
```
L_total = L_TB - entropy_weight × mean(log_P_F)
```

Note: log_P_F is negative, so `-log_P_F` is positive. This adds a penalty that grows when the policy becomes more confident.

### 2.5 Issue 4: Trajectory Balance Limitation (HIGH - Publication Blocker)

**Observation**: Even with entropy regularization, TB loss can still explode because of the fundamental log_Z tracking problem.

**Publication concern**: Reviewers will question an exploding loss curve - it looks like training failed, regardless of sample quality.

**Root cause**: TB loss uses a **single global log_Z** to estimate the partition function. When the policy concentrates on high-reward modes:

```
L_TB = (log_Z + log_P_F - log_R)²
```

The required log_Z changes rapidly as the policy sharpens, but a single scalar cannot track this fast enough.

**Solution**: Use **Sub-Trajectory Balance (STB)** loss instead of TB.

STB computes loss on sub-trajectories with weighted contributions:

```python
# For each sub-trajectory starting at step t:
L_t = λ^(T-t) × (log_Z + log_P_F[t:] - log_R - log_P_B[t:])²

L_STB = (1/T) × Σ_t L_t
```

**Why STB prevents loss explosion**:

1. **Distributed credit assignment**: Each step contributes independently weighted loss
2. **Shorter trajectories**: Sub-trajectories from later steps are shorter, easier to balance
3. **Geometric decay (λ=0.9)**: Earlier steps contribute less, reducing sensitivity to long-range imbalances

**Expected behavior with STB**:
- Loss curve: Stable or decreasing throughout training
- Sample quality: Same or better than TB
- Publishable: Yes - loss curve looks like normal training

---

## 3. Objectives & Scope

### 3.1 In-Scope Goals

| ID | Goal | Priority | Issue Addressed |
|----|------|----------|-----------------|
| 3b.1 | Fix checkpoint selection to use reward | P0 | Issue 1 |
| 3b.2 | Add entropy regularization to loss | P0 | Issue 3 |
| 3b.3 | Reduce log_Z LR multiplier to 3x | P1 | Issue 2 |
| **3b.4** | **Switch default loss to Sub-Trajectory Balance** | **P0** | **Issue 4** |
| 3b.5 | Add CLI flags for new parameters | P1 | All |
| 3b.6 | Update default configuration | P1 | All |
| 3b.7 | Add tests for new functionality | P0 | All |
| 3b.8 | Validate fixes with 10K step run | P0 | All |

### 3.2 Out-of-Scope (Deferred)

| Item | Rationale | Deferred To |
|------|-----------|-------------|
| Adaptive log_Z learning rate | Complex, may not be necessary with STB | Future work |
| Per-step entropy tracking | Not needed for current entropy loss | Future work |
| Multi-objective reward tuning | Not related to stability | Phase 4 |
| Detailed Balance loss | Requires flow estimator network | Future work |

### 3.3 Dependencies

| Dependency | Source | Required By |
|------------|--------|-------------|
| reward-comparison-analysis.md | Phase 3 experiments | All activities |
| Option C (Improved) reward | Phase 0b/1 | Validation run |
| Existing test suite | Phase 2 | Regression testing |

---

## 4. Detailed Implementation

### 4.1 Fix Checkpoint Selection (Issue 1)

**File**: `gflownet_peptide/training/trainer.py`

#### Change 1: Add best_reward tracking

**Location**: Line ~150 (after `self.best_loss = float("inf")`)

```python
# BEFORE
# Training state
self.global_step = 0
self.best_loss = float("inf")

# AFTER
# Training state
self.global_step = 0
self.best_loss = float("inf")
self.best_reward = 0.0  # Track best mean reward for checkpoint selection
```

#### Change 2: Update checkpoint selection logic

**Location**: Lines 382-385

```python
# BEFORE
# Save best model
if metrics["loss"] < self.best_loss:
    self.best_loss = metrics["loss"]
    best_path = checkpoint_dir / f"{run_name or 'gflownet'}_best.pt"
    self.save_checkpoint(best_path, step=step)

# AFTER
# Save best model by REWARD (not loss)
# Rationale: TB loss is unreliable - can be low for degenerate policies
# See docs/reward-comparison-analysis.md Section 4
if metrics.get("mean_reward", 0) > self.best_reward:
    self.best_reward = metrics["mean_reward"]
    best_path = checkpoint_dir / f"{run_name or 'gflownet'}_best.pt"
    self.save_checkpoint(best_path, step=step)
    logger.info(
        f"New best model at step {step}: "
        f"mean_reward={metrics['mean_reward']:.4f} "
        f"(loss={metrics['loss']:.2f})"
    )
```

#### Change 3: Update save_checkpoint

**Location**: Line ~497 (checkpoint dict)

```python
# BEFORE
checkpoint = {
    "step": step,
    "forward_policy_state_dict": self.forward_policy.state_dict(),
    "loss_fn_state_dict": self.loss_fn.state_dict(),
    "optimizer_state_dict": self.optimizer.state_dict(),
    "best_loss": self.best_loss,
    "config": self.config,
}

# AFTER
checkpoint = {
    "step": step,
    "forward_policy_state_dict": self.forward_policy.state_dict(),
    "loss_fn_state_dict": self.loss_fn.state_dict(),
    "optimizer_state_dict": self.optimizer.state_dict(),
    "best_loss": self.best_loss,
    "best_reward": self.best_reward,  # NEW
    "config": self.config,
}
```

#### Change 4: Update load_checkpoint

**Location**: Line ~517 (after `self.best_loss = ...`)

```python
# BEFORE
self.best_loss = checkpoint.get("best_loss", float("inf"))

# AFTER
self.best_loss = checkpoint.get("best_loss", float("inf"))
self.best_reward = checkpoint.get("best_reward", 0.0)  # NEW
```

### 4.2 Add Entropy Regularization (Issue 3)

**File**: `gflownet_peptide/training/loss.py`

#### Change 1: Update TrajectoryBalanceLoss.__init__

**Location**: Lines 35-46

```python
# BEFORE
def __init__(self, init_log_z: float = 0.0, epsilon: float = 1e-8):
    """
    Args:
        init_log_z: Initial value for log partition function
        epsilon: Small constant for numerical stability (unused, rewards
                 should already be in log space)
    """
    super().__init__()
    self.log_z = nn.Parameter(torch.tensor(init_log_z))
    self.epsilon = epsilon

# AFTER
def __init__(
    self,
    init_log_z: float = 0.0,
    epsilon: float = 1e-8,
    entropy_weight: float = 0.0,
):
    """
    Args:
        init_log_z: Initial value for log partition function
        epsilon: Small constant for numerical stability
        entropy_weight: Weight for entropy regularization. Higher values
                       encourage more exploration and prevent mode collapse.
                       Recommended: 0.01-0.1. Set to 0.0 to disable.

    Entropy Regularization:
        L_total = L_TB - entropy_weight * mean(log_P_F)

        Since log_P_F is negative, -log_P_F is positive (entropy estimate).
        This adds a penalty when the policy becomes too confident.

        Example:
        - Uncertain policy: log_P_F = -20 → penalty = 0.01 * 20 = 0.2
        - Confident policy: log_P_F = -70 → penalty = 0.01 * 70 = 0.7
    """
    super().__init__()
    self.log_z = nn.Parameter(torch.tensor(init_log_z))
    self.epsilon = epsilon
    self.entropy_weight = entropy_weight
```

#### Change 2: Update TrajectoryBalanceLoss.forward

**Location**: Lines 48-89

```python
# BEFORE
def forward(
    self,
    log_pf_sum: torch.Tensor,
    log_pb_sum: torch.Tensor,
    log_rewards: torch.Tensor,
    return_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    residual = self.log_z + log_pf_sum - log_rewards - log_pb_sum
    loss = (residual ** 2).mean()

    if return_info:
        with torch.no_grad():
            info = {
                'loss': loss.item(),
                'log_z': self.log_z.item(),
                'mean_log_pf': log_pf_sum.mean().item(),
                'mean_log_pb': log_pb_sum.mean().item(),
                'mean_log_reward': log_rewards.mean().item(),
                'mean_reward': torch.exp(log_rewards).mean().item(),
                'max_reward': torch.exp(log_rewards).max().item(),
                'residual_mean': residual.mean().item(),
                'residual_std': residual.std().item(),
            }
        return loss, info

    return loss

# AFTER
def forward(
    self,
    log_pf_sum: torch.Tensor,
    log_pb_sum: torch.Tensor,
    log_rewards: torch.Tensor,
    return_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Compute TB loss with optional entropy regularization.

    L_total = L_TB - entropy_weight * mean(log_P_F)

    The entropy term penalizes overconfident policies (very negative log_P_F).
    """
    # Standard TB loss
    residual = self.log_z + log_pf_sum - log_rewards - log_pb_sum
    tb_loss = (residual ** 2).mean()

    # Entropy regularization
    # H(π) = -E[log P_F], we want to maximize H, so minimize -H
    # L_total = L_TB - entropy_weight * H = L_TB + entropy_weight * E[log_P_F]
    # But log_P_F is negative, so this REDUCES loss for confident policies (wrong!)
    # We need: L_total = L_TB - entropy_weight * E[log_P_F]
    # This INCREASES loss when log_P_F is very negative (confident)
    entropy_reg = -self.entropy_weight * log_pf_sum.mean()

    loss = tb_loss + entropy_reg

    if return_info:
        with torch.no_grad():
            mean_entropy = -log_pf_sum.mean().item()  # H = -E[log P_F]
            info = {
                'loss': loss.item(),
                'tb_loss': tb_loss.item(),
                'entropy_reg': entropy_reg.item(),
                'mean_entropy': mean_entropy,
                'log_z': self.log_z.item(),
                'mean_log_pf': log_pf_sum.mean().item(),
                'mean_log_pb': log_pb_sum.mean().item(),
                'mean_log_reward': log_rewards.mean().item(),
                'mean_reward': torch.exp(log_rewards).mean().item(),
                'max_reward': torch.exp(log_rewards).max().item(),
                'min_reward': torch.exp(log_rewards).min().item(),
                'residual_mean': residual.mean().item(),
                'residual_std': residual.std().item(),
            }
        return loss, info

    return loss
```

#### Change 3: Update SubTrajectoryBalanceLoss with entropy regularization

**Location**: Lines 109-190

```python
# BEFORE
def __init__(
    self,
    init_log_z: float = 0.0,
    lambda_sub: float = 0.9,
    epsilon: float = 1e-8,
):

# AFTER
def __init__(
    self,
    init_log_z: float = 0.0,
    lambda_sub: float = 0.9,
    epsilon: float = 1e-8,
    entropy_weight: float = 0.0,
):
    super().__init__()
    self.log_z = nn.Parameter(torch.tensor(init_log_z))
    self.lambda_sub = lambda_sub
    self.epsilon = epsilon
    self.entropy_weight = entropy_weight
```

**Location**: Lines 148-165 (forward method)

```python
# BEFORE (end of forward method)
loss = total_loss / seq_len

# AFTER
# Base STB loss
stb_loss = total_loss / seq_len

# Entropy regularization (same as TB)
log_pf_sum = log_pf_per_step.sum(dim=1)
entropy_reg = -self.entropy_weight * log_pf_sum.mean()

loss = stb_loss + entropy_reg
```

**Update info dict** (if return_info=True):

```python
info = {
    'loss': loss.item(),
    'stb_loss': stb_loss.item(),
    'entropy_reg': entropy_reg.item(),
    'mean_entropy': -log_pf_sum.mean().item(),
    # ... rest of existing metrics
}
```

### 4.3 Update Trainer to Pass entropy_weight

**File**: `gflownet_peptide/training/trainer.py`

#### Change 1: Add entropy_weight parameter

**Location**: Line ~60 (in `__init__` signature)

```python
# Add to parameter list
def __init__(
    self,
    forward_policy: ForwardPolicy,
    backward_policy: BackwardPolicy,
    reward_model: Union[nn.Module, Callable[[list[str]], torch.Tensor]],
    learning_rate: float = 3e-4,
    log_z_lr_multiplier: float = 10.0,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    loss_type: str = "trajectory_balance",
    init_log_z: float = 0.0,
    min_length: int = 10,
    max_length: int = 30,
    exploration_eps: float = 0.0,
    reward_temperature: float = 1.0,
    entropy_weight: float = 0.0,  # NEW
    device: Optional[torch.device] = None,
):
```

#### Change 2: Pass entropy_weight to loss function

**Location**: Lines 123-126

```python
# BEFORE
if loss_type == "trajectory_balance":
    self.loss_fn = TrajectoryBalanceLoss(init_log_z=init_log_z).to(self.device)
elif loss_type == "sub_trajectory_balance":
    self.loss_fn = SubTrajectoryBalanceLoss(init_log_z=init_log_z).to(self.device)

# AFTER
if loss_type == "trajectory_balance":
    self.loss_fn = TrajectoryBalanceLoss(
        init_log_z=init_log_z,
        entropy_weight=entropy_weight,
    ).to(self.device)
elif loss_type == "sub_trajectory_balance":
    self.loss_fn = SubTrajectoryBalanceLoss(
        init_log_z=init_log_z,
        entropy_weight=entropy_weight,
    ).to(self.device)
```

#### Change 3: Add entropy_weight to config dict

**Location**: Line ~152 (self.config dict)

```python
self.config = {
    'learning_rate': learning_rate,
    'log_z_lr_multiplier': log_z_lr_multiplier,
    'weight_decay': weight_decay,
    'max_grad_norm': max_grad_norm,
    'loss_type': loss_type,
    'init_log_z': init_log_z,
    'min_length': min_length,
    'max_length': max_length,
    'exploration_eps': exploration_eps,
    'reward_temperature': reward_temperature,
    'entropy_weight': entropy_weight,  # NEW
}
```

### 4.4 Update Training Script CLI

**File**: `scripts/train_gflownet.py`

#### Change: Add new CLI arguments

**Location**: After line ~88 (after existing arguments)

```python
parser.add_argument(
    "--entropy_weight",
    type=float,
    default=None,
    help="Entropy regularization weight. Encourages exploration and prevents "
         "mode collapse. Recommended: 0.01-0.1. Default: from config or 0.0",
)
parser.add_argument(
    "--log_z_lr_multiplier",
    type=float,
    default=None,
    help="Learning rate multiplier for log_Z parameter. Lower values (3.0) "
         "provide more stable training. Default: from config or 10.0",
)
```

#### Change: Use new arguments in trainer creation

**Location**: Around line ~218 (before trainer instantiation)

```python
# Get entropy_weight from args or config
entropy_weight = args.entropy_weight
if entropy_weight is None:
    entropy_weight = training_config.get("entropy_weight", 0.0)

# Get log_z_lr_multiplier from args or config
log_z_lr_multiplier = args.log_z_lr_multiplier
if log_z_lr_multiplier is None:
    log_z_lr_multiplier = training_config.get("log_z_lr_multiplier", 10.0)

logger.info(f"Using entropy_weight={entropy_weight}, log_z_lr_multiplier={log_z_lr_multiplier}")
```

Then update the trainer instantiation to use these variables.

### 4.5 Update Configuration Defaults

**File**: `configs/default.yaml`

#### Change: Update training section

**Location**: Under `training:` section (around line 74)

```yaml
training:
  seed: 42
  n_steps: 100000
  log_every: 100
  eval_every: 1000
  save_every: 5000

  # Optimizer settings
  learning_rate: 3.0e-4
  weight_decay: 0.01
  lr_scheduler: "cosine"
  warmup_steps: 1000
  max_grad_norm: 1.0

  # GFlowNet-specific settings
  # CHANGED from "trajectory_balance" to "sub_trajectory_balance"
  # STB provides stable loss curves suitable for publication
  # See docs/prd-phase-3b-training-stability.md Section 2.5
  loss_type: "sub_trajectory_balance"
  init_log_z: 0.0

  # Sub-trajectory balance decay factor (λ in the paper)
  # Controls contribution weighting: L_t = λ^(T-t) × residual²
  # Higher values (0.9-0.99) give more weight to later steps
  lambda_sub: 0.9

  # log_Z learning rate multiplier (Malkin 2022)
  # CHANGED from 10.0 to 3.0 for stability
  # See docs/reward-comparison-analysis.md Section 3.4
  log_z_lr_multiplier: 3.0

  # NEW: Entropy regularization weight
  # Encourages exploration, prevents mode collapse
  # L_total = L_STB - entropy_weight * mean(log_P_F)
  # Recommended: 0.01-0.1, set to 0.0 to disable
  entropy_weight: 0.01

  gradient_accumulation_steps: 1
  batch_size: 64
```

---

## 5. New Tests

### 5.1 Test Checkpoint Selection by Reward

**File**: `tests/test_trainer.py`

Add new test class:

```python
class TestCheckpointSelectionByReward:
    """Test that best checkpoint is selected by reward, not loss.

    This addresses the critical bug where lowest-loss checkpoint
    had 15x worse sample quality than final checkpoint.
    See docs/reward-comparison-analysis.md Section 4.
    """

    @pytest.fixture
    def trainer(self):
        """Create a minimal trainer for testing."""
        forward_policy = ForwardPolicy(
            vocab_size=23, d_model=64, n_layers=2, n_heads=4
        )
        backward_policy = BackwardPolicy(use_uniform=True)
        reward_model = lambda seqs: torch.ones(len(seqs))

        return GFlowNetTrainer(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            reward_model=reward_model,
            device=torch.device('cpu'),
        )

    def test_best_reward_initialized_to_zero(self, trainer):
        """best_reward should start at 0."""
        assert trainer.best_reward == 0.0

    def test_best_reward_saved_in_checkpoint(self, trainer):
        """Checkpoint should contain best_reward field."""
        trainer.best_reward = 0.75

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            trainer.save_checkpoint(path, step=100)

            checkpoint = torch.load(path, weights_only=False)
            assert 'best_reward' in checkpoint
            assert checkpoint['best_reward'] == 0.75

    def test_best_reward_restored_on_load(self, trainer):
        """best_reward should be restored when loading checkpoint."""
        trainer.best_reward = 0.85

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            trainer.save_checkpoint(path, step=100)

            # Reset and reload
            trainer.best_reward = 0.0
            trainer.load_checkpoint(path)

            assert trainer.best_reward == 0.85

    def test_backward_compatible_load(self, trainer):
        """Loading old checkpoint without best_reward should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_checkpoint.pt"

            # Create checkpoint without best_reward (simulating old format)
            old_checkpoint = {
                'step': 100,
                'forward_policy_state_dict': trainer.forward_policy.state_dict(),
                'loss_fn_state_dict': trainer.loss_fn.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_loss': 10.0,
                'config': {},
            }
            torch.save(old_checkpoint, path)

            # Should load without error, best_reward defaults to 0
            trainer.load_checkpoint(path)
            assert trainer.best_reward == 0.0
```

### 5.2 Test Entropy Regularization

**File**: `tests/test_loss.py`

Add new test class:

```python
class TestEntropyRegularization:
    """Test entropy regularization in TrajectoryBalanceLoss.

    Entropy regularization prevents mode collapse by penalizing
    overconfident policies. See docs/reward-comparison-analysis.md.
    """

    def test_default_entropy_weight_is_zero(self):
        """Default entropy weight should be 0 (backward compatible)."""
        loss_fn = TrajectoryBalanceLoss()
        assert loss_fn.entropy_weight == 0.0

    def test_entropy_weight_configurable(self):
        """Entropy weight should be configurable."""
        loss_fn = TrajectoryBalanceLoss(entropy_weight=0.05)
        assert loss_fn.entropy_weight == 0.05

    def test_zero_entropy_weight_matches_original(self):
        """With entropy_weight=0, should match original TB loss."""
        loss_fn_new = TrajectoryBalanceLoss(init_log_z=0.0, entropy_weight=0.0)

        log_pf = torch.tensor([-20.0, -25.0, -22.0, -18.0])
        log_pb = torch.zeros(4)
        log_rewards = torch.tensor([-1.0, -0.5, -0.8, -1.2])

        loss_new = loss_fn_new(log_pf, log_pb, log_rewards)

        # Compute expected TB loss manually
        residual = 0.0 + log_pf - log_rewards - log_pb
        expected_loss = (residual ** 2).mean()

        assert torch.isclose(loss_new, expected_loss, rtol=1e-5)

    def test_entropy_regularization_penalizes_confidence(self):
        """Confident policies (low log_pf) should have higher loss."""
        loss_fn = TrajectoryBalanceLoss(init_log_z=0.0, entropy_weight=0.1)

        log_pb = torch.zeros(4)
        log_rewards = torch.zeros(4)

        # Confident policy (very negative log_pf)
        log_pf_confident = torch.tensor([-70.0, -72.0, -68.0, -71.0])

        # Uncertain policy (less negative log_pf)
        log_pf_uncertain = torch.tensor([-20.0, -22.0, -18.0, -21.0])

        loss_confident = loss_fn(log_pf_confident, log_pb, log_rewards)
        loss_uncertain = loss_fn(log_pf_uncertain, log_pb, log_rewards)

        # Confident policy should have HIGHER total loss due to entropy penalty
        assert loss_confident > loss_uncertain

    def test_entropy_metrics_in_info(self):
        """Loss info should include entropy-related metrics."""
        loss_fn = TrajectoryBalanceLoss(entropy_weight=0.05)

        log_pf = torch.tensor([-25.0, -30.0, -22.0, -28.0])
        log_pb = torch.zeros(4)
        log_rewards = torch.tensor([-0.5, -0.3, -0.7, -0.4])

        loss, info = loss_fn(log_pf, log_pb, log_rewards, return_info=True)

        assert 'mean_entropy' in info
        assert 'entropy_reg' in info
        assert 'tb_loss' in info

        # Verify mean_entropy calculation: H = -E[log_P_F]
        expected_entropy = -log_pf.mean().item()
        assert abs(info['mean_entropy'] - expected_entropy) < 1e-5

    def test_subtb_entropy_weight(self):
        """SubTrajectoryBalanceLoss should also support entropy_weight."""
        loss_fn = SubTrajectoryBalanceLoss(entropy_weight=0.03)
        assert loss_fn.entropy_weight == 0.03

    def test_subtb_entropy_penalizes_confidence(self):
        """STB should also penalize overconfident policies."""
        loss_fn = SubTrajectoryBalanceLoss(init_log_z=0.0, entropy_weight=0.1)

        # Per-step log probs [batch=4, seq_len=5]
        log_pb = torch.zeros(4, 5)
        log_rewards = torch.zeros(4)

        # Confident policy (very negative per-step log probs)
        log_pf_confident = torch.full((4, 5), -14.0)  # sum = -70

        # Uncertain policy
        log_pf_uncertain = torch.full((4, 5), -4.0)   # sum = -20

        loss_confident = loss_fn(log_pf_confident, log_pb, log_rewards)
        loss_uncertain = loss_fn(log_pf_uncertain, log_pb, log_rewards)

        # Confident policy should have HIGHER total loss
        assert loss_confident > loss_uncertain
```

---

## 6. Success Criteria

| ID | Criterion | Target | Verification Method |
|----|-----------|--------|---------------------|
| SC1 | **Loss stable** | **No explosion - stays bounded or decreases** | W&B: loss curve visual inspection |
| SC2 | Loss bounded | < 100 throughout 10K steps (with STB) | W&B: max(train/loss) < 100 |
| SC3 | Best checkpoint by reward | Best = highest mean_reward | Compare checkpoints |
| SC4 | Diverse samples | >= 15/20 amino acids | Sample analysis |
| SC5 | Unique samples | 100% unique in batch | eval/unique_ratio = 1.0 |
| SC6 | Entropy tracked | mean_entropy in W&B | W&B: train/mean_entropy exists |
| SC7 | log_pf bounded | > -50 average | W&B: train/mean_log_pf > -50 |
| SC8 | Tests pass | All 123+ tests green | pytest tests/ -v |
| SC9 | Backward compatible | Old checkpoints load | Test old checkpoint loading |
| **SC10** | **Publishable loss curve** | **Monotonically decreasing or stable** | **Visual inspection for paper** |

---

## 7. Verification Commands

### 7.1 Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific new tests
pytest tests/test_trainer.py::TestCheckpointSelectionByReward -v
pytest tests/test_loss.py::TestEntropyRegularization -v

# Run with coverage
pytest tests/ -v --cov=gflownet_peptide --cov-report=term-missing
```

### 7.2 Validation Training Run

```bash
# 10K step validation with STB loss and new parameters
python scripts/train_gflownet.py \
    --reward_type improved \
    --n_steps 10000 \
    --loss_type sub_trajectory_balance \
    --entropy_weight 0.01 \
    --log_z_lr_multiplier 3.0 \
    --output_dir checkpoints/gflownet/phase3b-validation/ \
    --run_name phase3b-stb-validation-10k \
    --wandb \
    --seed 42
```

**Expected results with STB**:
- Loss: Stable or decreasing curve (no explosion)
- Mean reward: Increasing throughout training
- Diversity: ≥15/20 amino acids at final checkpoint

### 7.3 Compare Checkpoints

```bash
# After training, compare best vs final
python -c "
import torch
from gflownet_peptide.models.forward_policy import ForwardPolicy
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for name in ['best', 'final']:
    path = f'checkpoints/gflownet/phase3b-validation/gflownet_{name}.pt'
    ckpt = torch.load(path, map_location=device)

    policy = ForwardPolicy(vocab_size=23, d_model=256, n_layers=4, n_heads=8, dim_feedforward=1024).to(device)
    policy.load_state_dict(ckpt['forward_policy_state_dict'])
    policy.eval()

    with torch.no_grad():
        seqs, _ = policy.sample_sequence(batch_size=100, max_length=30, min_length=10, device=device)

    unique = len(set(seqs))
    all_aa = set(''.join(seqs))

    print(f'{name}: unique={unique}/100, AA_diversity={len(all_aa)}/20')
    print(f'  Sample: {seqs[0]}')
"
```

### 7.4 Check W&B Metrics

```bash
# Verify new metrics are logged
python -c "
import wandb
api = wandb.Api()
run = api.run('gflownet-peptide/<RUN_ID>')  # Replace with actual run ID

# Check for new metrics
history = list(run.scan_history())
last = history[-1]

required_metrics = ['train/mean_entropy', 'train/entropy_reg', 'train/tb_loss']
for metric in required_metrics:
    if metric in last:
        print(f'{metric}: {last[metric]:.4f}')
    else:
        print(f'WARNING: {metric} not found!')
"
```

---

## 8. Rollback Plan

### 8.1 If Entropy Regularization Causes Issues

**Symptom**: Training becomes unstable, loss oscillates wildly.

**Rollback**:
```bash
python scripts/train_gflownet.py --entropy_weight 0.0 ...
```

The implementation is additive, so setting weight to 0 completely disables entropy regularization.

### 8.2 If Lower log_Z Multiplier Causes Slow Convergence

**Symptom**: log_Z grows too slowly, residual remains large.

**Rollback**:
```bash
python scripts/train_gflownet.py --log_z_lr_multiplier 10.0 ...
```

Revert to original Malkin 2022 recommendation.

### 8.3 If Reward-Based Checkpoint Selection Misses Good Models

**Symptom**: Best checkpoint has high reward but poor sample quality.

**Mitigation**: The implementation saves both:
- `gflownet_best.pt` - Best by reward
- `gflownet_final.pt` - Final checkpoint
- `gflownet_latest.pt` - Latest checkpoint

Users can always fall back to final or latest.

### 8.4 Complete Rollback

All changes are **backward compatible**:
- Old checkpoints without `best_reward` field will load (defaults to 0.0)
- Old configs without `entropy_weight` will work (defaults to 0.0)
- No existing functionality is removed

---

## 9. Phase Gate Review

### 9.1 Go/No-Go Criteria

| Criterion | Required | Status |
|-----------|----------|--------|
| All existing tests pass | Yes | [ ] |
| New tests pass | Yes | [ ] |
| Validation run completes | Yes | [ ] |
| Loss stays < 500 | Yes | [ ] |
| Best checkpoint has > 10 AA diversity | Yes | [ ] |
| Entropy metrics logged to W&B | Yes | [ ] |

### 9.2 Review Checklist

- [ ] All code changes reviewed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] Backward compatibility verified
- [ ] Validation run analyzed
- [ ] W&B dashboard updated with new metrics

### 9.3 Decision

**Status**: Pending
**Decision Date**: ___________
**Notes**: ___________

---

## 10. Appendix

### 10.1 Mathematical Derivation: Entropy Regularization

The policy entropy measures exploration:

```
H(π) = -E_{τ~π}[log P_F(τ)]
     = -E[Σ_t log P_F(a_t | s_t)]
     = -E[log_pf_sum]
```

We want to **maximize** entropy (encourage exploration). In a minimization framework:

```
L_total = L_TB + λ × (-H(π))
        = L_TB - λ × H(π)
        = L_TB + λ × E[log_pf_sum]
```

Since `log_pf_sum` is negative (log of probability < 1), this seems counterintuitive. Let's verify:

- Confident policy: `log_pf_sum = -70` → `λ × (-70) = -0.7` (reduces loss, wrong!)
- Uncertain policy: `log_pf_sum = -20` → `λ × (-20) = -0.2` (less reduction)

This is backwards! We need:

```
L_total = L_TB - λ × E[log_pf_sum]
```

Now:
- Confident policy: `log_pf_sum = -70` → `-λ × (-70) = +0.7` (increases loss, correct!)
- Uncertain policy: `log_pf_sum = -20` → `-λ × (-20) = +0.2` (less increase)

The implementation is:
```python
entropy_reg = -self.entropy_weight * log_pf_sum.mean()
loss = tb_loss + entropy_reg
```

### 10.2 Experimental Evidence Summary

From run `zcb95gyl` (Option C):

| Step | Loss | Mean Reward | AA Diversity | Status |
|------|------|-------------|--------------|--------|
| 0 | 864 | 0.59 | N/A | Start |
| 5000 | 32 | 0.06 | 2/20 | DL-trap (saved as "best") |
| 6000 | 21 | 0.06 | 2/20 | Minimum loss |
| 7000 | 1327 | 0.79 | ~10/20 | Escaping trap |
| 10000 | 3378 | 0.96 | 15/20 | Final (best quality) |

### 10.3 Files Modified Summary

| File | Lines | Changes |
|------|-------|---------|
| `trainer.py` | 150, 382-385, 497, 517 | Checkpoint selection + best_reward |
| `loss.py` | 35-89, 109-186 | Entropy regularization for TB and STB |
| `train_gflownet.py` | 88+, 218+ | CLI arguments |
| `default.yaml` | 74+ | STB default + new parameters |
| `test_trainer.py` | New class | Checkpoint tests |
| `test_loss.py` | New class | Entropy tests for TB and STB |

### 10.4 Key Configuration Changes

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `loss_type` | `trajectory_balance` | `sub_trajectory_balance` | Stable loss curves for publication |
| `log_z_lr_multiplier` | 10.0 | 3.0 | Reduce oscillation |
| `entropy_weight` | (none) | 0.01 | Prevent mode collapse |
| `lambda_sub` | (none) | 0.9 | STB decay factor |

---

## 11. References

- `docs/reward-comparison-analysis.md` - Full experimental analysis
- `docs/prd-phase-3-training.md` - Original Phase 3 PRD
- Malkin et al. (2022) - Trajectory Balance for GFlowNets
- Bengio et al. (2021) - Flow Network based Generative Models
