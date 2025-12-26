# Phase 2: GFlowNet Core Implementation - Detailed PRD

**Generated from**: docs/gflownet-master-prd.md Section 5.2
**Date**: December 26, 2025
**Status**: Draft

---

## 1. Executive Summary

- **Objective**: Implement the complete GFlowNet architecture and training loop for peptide generation, including the forward policy (Transformer), backward policy (Uniform), trajectory sampling, and trajectory balance loss computation.
- **Duration**: 2 weeks
- **Key Deliverables**:
  - ForwardPolicy Transformer module
  - BackwardPolicy (Uniform) module
  - Trajectory sampling infrastructure
  - TB and SubTB loss implementations
  - Training loop with W&B integration
  - Comprehensive unit tests (≥80% coverage)
- **Prerequisites**:
  - Phase 1 complete: Trained and validated reward model (stability predictor with R²≥0.6)
  - `ImprovedReward` and `CompositeReward` classes implemented and tested
  - Data loaders for FLIP and Propedia working
  - ESM-2 model accessible

---

## 2. Objectives & Scope

### 2.1 In-Scope Goals

1. **Forward Policy Implementation**: Causal Transformer that predicts next amino acid distribution given partial sequence
2. **Backward Policy Implementation**: Uniform backward policy (P_B = 1 for linear generation)
3. **Trajectory Sampling**: Complete trajectory generation with forward/backward log probability tracking
4. **Loss Functions**: Trajectory Balance (TB) and Sub-Trajectory Balance (SubTB) loss computation
5. **Training Loop**: Complete training infrastructure with gradient clipping, logging, and checkpointing
6. **W&B Integration**: Experiment tracking, loss curves, sample quality monitoring
7. **Unit Tests**: Comprehensive tests for all components with ≥80% coverage

### 2.2 Out-of-Scope (Deferred)

- Hyperparameter tuning (Phase 3)
- Full training runs (Phase 3)
- GRPO comparison experiments (Phase 4)
- Pre-training P_F on UniRef50 (future work)
- Structure-conditioned generation (V2)
- Advanced exploration strategies (e.g., epsilon-greedy, tempering)

### 2.3 Dependencies

| Dependency | Source | Required By |
|------------|--------|-------------|
| ESM-2 model (esm2_t12_35M or t33_650M) | `fair-esm` package | Reward model backbone |
| PyTorch ≥2.0 | pip | All modules |
| `ImprovedReward` class | Phase 1 | Training loop |
| `CompositeReward` class | Phase 1 | Ablation training |
| Stability predictor checkpoint | Phase 1 | CompositeReward |
| W&B account and API key | wandb.ai | Logging |

---

## 3. Detailed Activities

### Activity 2.1: Set Up Environment

**Description**: Configure development environment with all required dependencies.

**Steps**:
1. Update `requirements.txt` with GFlowNet-specific dependencies
2. Verify CUDA availability and PyTorch GPU support
3. Set up W&B project and entity configuration
4. Validate ESM-2 model loading

**Implementation Notes**:
- Use `torch>=2.0` for improved performance and `torch.compile` compatibility
- Pin versions for reproducibility
- Environment variables: `WANDB_API_KEY`, `HF_TOKEN`

**Verification**:
```bash
# Test imports
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import esm; model, alphabet = esm.pretrained.esm2_t12_35M_UR50D(); print('ESM-2 loaded')"
python -c "import wandb; print('W&B available')"

# Run basic tests
pytest tests/test_imports.py -v
```

**Output**: Updated `requirements.txt`, verified environment

---

### Activity 2.2: Implement ForwardPolicy (Transformer)

**Description**: Build the forward policy P_F as a causal Transformer that outputs a distribution over the next amino acid given a partial sequence.

**Steps**:
1. Define vocabulary: 20 amino acids + START (20) + STOP (21) + PAD (22)
2. Implement sinusoidal positional encoding
3. Build Transformer encoder with causal masking
4. Add action head (linear layer → softmax)
5. Implement `forward()` for logits and `log_prob()` for sampling

**Implementation Notes**:
- Use `nn.TransformerEncoder` with causal mask
- Hidden dim: 256 (configurable)
- Layers: 4 (configurable)
- Heads: 8
- Dropout: 0.1
- Output: logits over 22 tokens (20 AA + STOP + PAD; START is input-only)

**Code Template**:
```python
# gflownet_peptide/models/forward_policy.py
class ForwardPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int = 23,  # 20 AA + START + STOP + PAD
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 64
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, vocab_size)

    def forward(self, partial_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            partial_seq: [B, L] token indices
        Returns:
            logits: [B, vocab_size] next token logits
        """
        x = self.embedding(partial_seq) + self.pos_enc(partial_seq.size(1))
        mask = generate_causal_mask(partial_seq.size(1), device=partial_seq.device)
        x = self.transformer(x, mask=mask)
        logits = self.action_head(x[:, -1, :])  # Use last position
        return logits

    def log_prob(self, partial_seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log P_F(action | partial_seq)."""
        logits = self.forward(partial_seq)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
```

**Verification**:
```bash
pytest tests/test_forward_policy.py -v
# Check: output shape [B, vocab_size]
# Check: log_prob returns valid log probabilities
# Check: causal mask prevents future leakage
```

**Output**: `gflownet_peptide/models/forward_policy.py`

---

### Activity 2.3: Implement BackwardPolicy (Uniform)

**Description**: Implement the backward policy P_B. For linear autoregressive generation, P_B is deterministic (only one parent state exists).

**Steps**:
1. Create BackwardPolicy class with `log_prob()` method
2. Return log(1) = 0 for all valid parent transitions
3. Handle edge case: initial state has no parent

**Implementation Notes**:
- For linear generation: state `[A, B, C]` has unique parent `[A, B]`
- P_B = 1, so log P_B = 0
- No learnable parameters
- This simplifies the TB loss significantly

**Theoretical Justification (Malkin 2022, Bengio 2021)**:

For autoregressive sequence generation, the state space forms a **tree-structured DAG** (not a general DAG). This is because each partial sequence `[A, R, G]` has exactly one construction path—no two different action sequences can produce the same partial peptide.

From Bengio 2021 (Proposition 1): When the action space is **injective** (bijective mapping from action sequences to states), the backward policy is trivially uniform: P_B = 1.

This means for trajectory balance loss:
```
L_TB = (log Z + Σ log P_F - log R - Σ log P_B)²
     = (log Z + Σ log P_F - log R - 0)²      # Since Σ log P_B = 0
     = (log Z + Σ log P_F - log R)²
```

This simplification is **correct and intentional** for our autoregressive peptide generation.

**Code Template**:
```python
# gflownet_peptide/models/backward_policy.py
class BackwardPolicy(nn.Module):
    """Uniform backward policy for linear autoregressive generation.

    Since there's only one parent state (remove last token), P_B = 1.
    """
    def __init__(self):
        super().__init__()

    def log_prob(self, state: torch.Tensor, parent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, L] current state
            parent_state: [B, L-1] parent state
        Returns:
            log_prob: [B] always 0 (since P_B = 1)
        """
        batch_size = state.size(0)
        return torch.zeros(batch_size, device=state.device)
```

**Verification**:
```bash
pytest tests/test_backward_policy.py -v
# Check: log_prob always returns 0
# Check: handles batch dimension correctly
```

**Output**: `gflownet_peptide/models/backward_policy.py`

---

### Activity 2.4: Implement Trajectory Sampling

**Description**: Build trajectory sampler that generates complete peptide sequences from START to STOP, tracking forward and backward log probabilities.

**Steps**:
1. Create `TrajectorySampler` class
2. Implement `sample()` method: generate sequences autoregressively
3. Track log P_F at each step
4. Enforce minimum/maximum length constraints
5. Return complete trajectories with metadata

**Implementation Notes**:
- Start with START token (index 20)
- Sample until STOP token (index 21) or max_length reached
- Minimum length: prevent STOP action for first `min_length` steps
- Temperature sampling: logits / temperature before softmax
- Store full trajectory for loss computation

**Code Template**:
```python
# gflownet_peptide/training/sampler.py
@dataclass
class Trajectory:
    """A complete trajectory from START to terminal state."""
    states: torch.Tensor          # [L+1, seq_len] partial sequences at each step
    actions: torch.Tensor         # [L] actions taken
    log_pf: torch.Tensor          # [L] log P_F for each action
    log_pb: torch.Tensor          # [L] log P_B for each transition (all 0 for uniform)
    terminal_sequence: torch.Tensor  # [seq_len] final sequence (without START/STOP)
    length: int                   # sequence length

class TrajectorySampler:
    def __init__(
        self,
        forward_policy: ForwardPolicy,
        backward_policy: BackwardPolicy,
        vocab_size: int = 23,
        start_token: int = 20,
        stop_token: int = 21,
        min_length: int = 10,
        max_length: int = 30,
        temperature: float = 1.0,
        exploration_eps: float = 0.001  # δ: uniform mixing (Bengio 2021, Eq. 10)
    ):
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.stop_token = stop_token
        self.min_length = min_length
        self.max_length = max_length
        self.temperature = temperature
        self.exploration_eps = exploration_eps

    def sample_batch(self, batch_size: int, device: torch.device) -> list[Trajectory]:
        """Sample a batch of complete trajectories."""
        trajectories = []

        for _ in range(batch_size):
            traj = self._sample_single(device)
            trajectories.append(traj)

        return trajectories

    def _sample_single(self, device: torch.device) -> Trajectory:
        """Sample a single trajectory."""
        # Start with START token
        current_seq = torch.tensor([[self.start_token]], device=device)

        states = [current_seq.clone()]
        actions = []
        log_pfs = []

        for step in range(self.max_length):
            # Get action distribution
            logits = self.forward_policy(current_seq)

            # Mask STOP token if below min_length
            if step < self.min_length:
                logits[:, self.stop_token] = float('-inf')

            # Sample action with exploration mixing (Bengio 2021, Eq. 10)
            # π_explore = (1 - δ) * P_F + δ * Uniform
            probs = F.softmax(logits / self.temperature, dim=-1)
            if self.exploration_eps > 0:
                uniform = torch.ones_like(probs) / probs.size(-1)
                probs = (1 - self.exploration_eps) * probs + self.exploration_eps * uniform
            action = torch.multinomial(probs, num_samples=1)
            log_pf = F.log_softmax(logits / self.temperature, dim=-1)
            log_pf_action = log_pf.gather(-1, action).squeeze(-1)

            actions.append(action.item())
            log_pfs.append(log_pf_action.item())

            # Check for STOP
            if action.item() == self.stop_token:
                break

            # Append action to sequence
            current_seq = torch.cat([current_seq, action], dim=-1)
            states.append(current_seq.clone())

        # Extract terminal sequence (without START)
        terminal_seq = current_seq[0, 1:]  # Remove START token

        return Trajectory(
            states=states,
            actions=torch.tensor(actions, device=device),
            log_pf=torch.tensor(log_pfs, device=device),
            log_pb=torch.zeros(len(actions), device=device),  # Uniform P_B
            terminal_sequence=terminal_seq,
            length=len(terminal_seq)
        )
```

**Verification**:
```bash
pytest tests/test_sampler.py -v
# Check: sequences start with START token
# Check: sequences end with STOP or reach max_length
# Check: minimum length is enforced
# Check: log_pf values are valid log probabilities
```

**Output**: `gflownet_peptide/training/sampler.py`

---

### Activity 2.5: Implement TB Loss Computation

**Description**: Implement Trajectory Balance (TB) and Sub-Trajectory Balance (SubTB) loss functions.

**Steps**:
1. Create `TrajectoryBalanceLoss` class with learnable log_Z
2. Implement TB loss: `(log_Z + Σlog_pf - log_R - Σlog_pb)²`
3. Create `SubTrajectoryBalanceLoss` for more stable training
4. Handle batch computation efficiently

**Implementation Notes**:
- log_Z is a learnable parameter initialized to 0
- Rewards must be non-negative (use exp/softplus transform)
- Add small epsilon to rewards to avoid log(0)
- SubTB: compute loss on sub-trajectories for better credit assignment

**Code Template**:
```python
# gflownet_peptide/training/loss.py
class TrajectoryBalanceLoss(nn.Module):
    """Trajectory Balance loss for GFlowNet training.

    L_TB = (log Z + Σ log P_F(s_t|s_{t-1}) - log R(x) - Σ log P_B(s_{t-1}|s_t))²

    For uniform P_B, Σ log P_B = 0, simplifying to:
    L_TB = (log Z + Σ log P_F - log R(x))²
    """
    def __init__(self, init_log_z: float = 0.0, epsilon: float = 1e-8):
        super().__init__()
        self.log_z = nn.Parameter(torch.tensor(init_log_z))
        self.epsilon = epsilon

    def forward(
        self,
        log_pf_trajectory: torch.Tensor,  # [B, L] or list of tensors
        log_pb_trajectory: torch.Tensor,  # [B, L] or list of tensors
        rewards: torch.Tensor,            # [B] rewards for terminal states
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute TB loss for a batch of trajectories.

        Returns:
            loss: scalar loss
            info: dict with log_Z, mean_reward, etc.
        """
        # Sum log probabilities along trajectory
        sum_log_pf = log_pf_trajectory.sum(dim=-1)  # [B]
        sum_log_pb = log_pb_trajectory.sum(dim=-1)  # [B] (all zeros for uniform)

        # Compute log reward (add epsilon for numerical stability)
        log_rewards = torch.log(rewards + self.epsilon)

        # TB loss: (log_Z + sum_log_pf - log_R - sum_log_pb)²
        tb_residual = self.log_z + sum_log_pf - log_rewards - sum_log_pb
        loss = (tb_residual ** 2).mean()

        info = {
            'loss': loss.item(),
            'log_z': self.log_z.item(),
            'mean_log_pf': sum_log_pf.mean().item(),
            'mean_log_reward': log_rewards.mean().item(),
            'mean_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
        }

        return loss, info


class SubTrajectoryBalanceLoss(nn.Module):
    """Sub-Trajectory Balance loss for more stable training.

    Computes TB loss on all sub-trajectories, providing denser gradients.
    """
    def __init__(self, init_log_z: float = 0.0, epsilon: float = 1e-8):
        super().__init__()
        self.log_z = nn.Parameter(torch.tensor(init_log_z))
        self.epsilon = epsilon
        # State flow estimator for intermediate states
        self.state_flow = None  # Optional: learnable F(s) for intermediate states

    def forward(
        self,
        log_pf_trajectory: torch.Tensor,
        log_pb_trajectory: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute SubTB loss."""
        # For simplicity, use standard TB loss
        # Full SubTB requires state flow estimation
        sum_log_pf = log_pf_trajectory.sum(dim=-1)
        sum_log_pb = log_pb_trajectory.sum(dim=-1)
        log_rewards = torch.log(rewards + self.epsilon)

        tb_residual = self.log_z + sum_log_pf - log_rewards - sum_log_pb
        loss = (tb_residual ** 2).mean()

        info = {
            'loss': loss.item(),
            'log_z': self.log_z.item(),
            'mean_reward': rewards.mean().item(),
        }

        return loss, info
```

**Verification**:
```bash
pytest tests/test_loss.py -v
# Check: loss is finite and non-negative
# Check: log_Z is learnable (gradients flow)
# Check: loss decreases when P_F matches reward distribution
```

**Output**: `gflownet_peptide/training/loss.py`

---

### Activity 2.6: Implement Training Loop

**Description**: Build the complete training loop that orchestrates sampling, loss computation, and optimization.

**Steps**:
1. Create `GFlowNetTrainer` class
2. Implement training step: sample → reward → loss → update
3. Add gradient clipping for stability
4. Implement validation loop
5. Add checkpointing (latest + final)
6. Integrate W&B logging

**Implementation Notes**:
- Use AdamW optimizer with configurable learning rate
- **Important**: Use separate parameter groups with higher LR for log_Z (Malkin 2022 recommends ~10x)
- Gradient clipping: max_norm=1.0
- Checkpoint policy: overwrite `_latest.pt`, save `_final.pt` at end
- Log every N steps: loss, log_Z, sample metrics
- Evaluate on validation set periodically

**Optimizer Configuration (Malkin 2022)**:
```python
# Separate parameter groups for different learning rates
policy_params = list(forward_policy.parameters())
log_z_params = [loss_fn.log_z]

optimizer = torch.optim.AdamW([
    {'params': policy_params, 'lr': config['learning_rate']},
    {'params': log_z_params, 'lr': config['learning_rate'] * config['log_z_lr_multiplier']}
])
```

**Code Template**:
```python
# gflownet_peptide/training/trainer.py
class GFlowNetTrainer:
    def __init__(
        self,
        forward_policy: ForwardPolicy,
        backward_policy: BackwardPolicy,
        reward_model: nn.Module,
        loss_fn: TrajectoryBalanceLoss,
        optimizer: torch.optim.Optimizer,
        sampler: TrajectorySampler,
        config: dict,
        device: torch.device,
    ):
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.reward_model = reward_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.sampler = sampler
        self.config = config
        self.device = device

        self.global_step = 0
        self.best_loss = float('inf')

    def train_step(self, batch_size: int) -> dict:
        """Execute a single training step."""
        self.forward_policy.train()

        # Sample trajectories
        trajectories = self.sampler.sample_batch(batch_size, self.device)

        # Compute rewards for terminal sequences
        sequences = [t.terminal_sequence for t in trajectories]
        rewards = self._compute_rewards(sequences)

        # Stack trajectory log probs
        log_pf = torch.stack([t.log_pf.sum() for t in trajectories])
        log_pb = torch.stack([t.log_pb.sum() for t in trajectories])

        # Compute loss
        loss, info = self.loss_fn(
            log_pf.unsqueeze(1),
            log_pb.unsqueeze(1),
            rewards
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.forward_policy.parameters(),
            max_norm=self.config.get('max_grad_norm', 1.0)
        )

        self.optimizer.step()

        self.global_step += 1
        info['grad_norm'] = grad_norm.item()
        info['step'] = self.global_step

        return info

    def _compute_rewards(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        """Compute rewards for a batch of sequences."""
        with torch.no_grad():
            rewards = []
            for seq in sequences:
                # Decode sequence to string
                seq_str = self._decode_sequence(seq)
                # Compute reward
                r = self.reward_model(seq_str)
                rewards.append(r)
            return torch.tensor(rewards, device=self.device)

    def _decode_sequence(self, seq: torch.Tensor) -> str:
        """Convert token indices to amino acid string."""
        AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
        aa_list = []
        for idx in seq.tolist():
            if 0 <= idx < 20:
                aa_list.append(AA_VOCAB[idx])
        return "".join(aa_list)

    def save_checkpoint(self, path: str, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.global_step,
            'forward_policy_state_dict': self.forward_policy.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.forward_policy.load_state_dict(checkpoint['forward_policy_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['step']
```

**Verification**:
```bash
pytest tests/test_trainer.py -v
# Check: training step completes without error
# Check: loss decreases over multiple steps
# Check: checkpoint save/load works
# Check: gradient clipping is applied
```

**Output**: `gflownet_peptide/training/trainer.py`

---

### Activity 2.7: Add W&B Logging

**Description**: Integrate Weights & Biases for experiment tracking.

**Steps**:
1. Initialize W&B run with config
2. Log training metrics every step
3. Log sample sequences and rewards periodically
4. Log validation metrics
5. Save final model as artifact

**Implementation Notes**:
- Project: `gflownet-peptide`
- Entity: from environment variable or config
- Log: loss, log_Z, grad_norm, mean_reward, sample diversity
- Create sample table with sequence, reward, length

**Code Template**:
```python
# In trainer.py or separate logging.py
import wandb

def setup_wandb(config: dict, run_name: str = None):
    """Initialize W&B run."""
    wandb.init(
        project=config.get('wandb_project', 'gflownet-peptide'),
        entity=config.get('wandb_entity', None),
        name=run_name,
        config=config,
    )

def log_training_step(info: dict):
    """Log training step metrics."""
    wandb.log({
        'train/loss': info['loss'],
        'train/log_z': info['log_z'],
        'train/mean_reward': info['mean_reward'],
        'train/grad_norm': info['grad_norm'],
    }, step=info['step'])

def log_samples(sequences: list[str], rewards: list[float], step: int):
    """Log sample sequences as W&B table."""
    table = wandb.Table(columns=['sequence', 'reward', 'length'])
    for seq, r in zip(sequences, rewards):
        table.add_data(seq, r, len(seq))
    wandb.log({'samples': table}, step=step)
```

**Verification**:
```bash
# Run short training with W&B
python scripts/test_wandb_logging.py

# Check W&B dashboard for:
# - Loss curve
# - log_Z evolution
# - Sample table
```

**Output**: W&B integration in `trainer.py`

---

### Activity 2.8: Unit Tests for Each Component

**Description**: Write comprehensive unit tests achieving ≥80% coverage.

**Steps**:
1. Test ForwardPolicy: shapes, probabilities, causal masking
2. Test BackwardPolicy: uniform log_prob
3. Test TrajectorySampler: valid sequences, length constraints
4. Test TrajectoryBalanceLoss: loss computation, gradients
5. Test GFlowNetTrainer: training step, checkpointing
6. Integration test: end-to-end training loop

**Test Files**:
```
tests/
├── test_forward_policy.py
├── test_backward_policy.py
├── test_sampler.py
├── test_loss.py
├── test_trainer.py
└── test_integration.py
```

**Key Test Cases**:

```python
# tests/test_forward_policy.py
class TestForwardPolicy:
    def test_forward_shape(self):
        """Output shape should be [B, vocab_size]."""
        policy = ForwardPolicy(vocab_size=23, d_model=64, n_layers=2)
        x = torch.randint(0, 20, (4, 10))  # [B=4, L=10]
        logits = policy(x)
        assert logits.shape == (4, 23)

    def test_log_prob_valid(self):
        """log_prob should return valid log probabilities."""
        policy = ForwardPolicy(vocab_size=23, d_model=64, n_layers=2)
        x = torch.randint(0, 20, (4, 10))
        action = torch.randint(0, 21, (4,))
        log_p = policy.log_prob(x, action)
        assert log_p.shape == (4,)
        assert (log_p <= 0).all()  # Log probs are non-positive

    def test_causal_masking(self):
        """Future tokens should not affect current prediction."""
        policy = ForwardPolicy(vocab_size=23, d_model=64, n_layers=2)
        x1 = torch.tensor([[0, 1, 2, 3, 4]])
        x2 = torch.tensor([[0, 1, 2, 5, 6]])  # Different suffix

        # Predictions at position 2 should be identical
        logits1 = policy(x1[:, :3])
        logits2 = policy(x2[:, :3])
        assert torch.allclose(logits1, logits2)

# tests/test_loss.py
class TestTrajectoryBalanceLoss:
    def test_loss_finite(self):
        """Loss should be finite and non-negative."""
        loss_fn = TrajectoryBalanceLoss()
        log_pf = torch.randn(4, 5)
        log_pb = torch.zeros(4, 5)
        rewards = torch.rand(4) + 0.1

        loss, info = loss_fn(log_pf, log_pb, rewards)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_log_z_gradient(self):
        """log_Z should receive gradients."""
        loss_fn = TrajectoryBalanceLoss()
        log_pf = torch.randn(4, 5, requires_grad=True)
        log_pb = torch.zeros(4, 5)
        rewards = torch.rand(4) + 0.1

        loss, _ = loss_fn(log_pf, log_pb, rewards)
        loss.backward()

        assert loss_fn.log_z.grad is not None
```

**Verification**:
```bash
# Run all tests with coverage
pytest tests/ -v --cov=gflownet_peptide --cov-report=html

# Check coverage >= 80%
coverage report --fail-under=80
```

**Output**: Complete test suite in `tests/`

---

## 4. Technical Specifications

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GFlowNet Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Partial sequence [START, A, R, G, ...]                  │
│         (token indices)                                          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    ForwardPolicy (P_F)                      │ │
│  │                                                             │ │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │ │
│  │   │  Embedding  │───►│  Pos Enc    │───►│ Transformer │    │ │
│  │   │  [23, 256]  │    │ Sinusoidal  │    │  (Causal)   │    │ │
│  │   └─────────────┘    └─────────────┘    └──────┬──────┘    │ │
│  │                                                │            │ │
│  │                                         ┌──────▼──────┐    │ │
│  │                                         │ Action Head │    │ │
│  │                                         │ [256 → 23]  │    │ │
│  │                                         └──────┬──────┘    │ │
│  │                                                │            │ │
│  └────────────────────────────────────────────────┼────────────┘ │
│                                                   │              │
│  Output: logits [B, 23] → Softmax → P(next_token)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Code Structure

```
gflownet_peptide/
├── models/
│   ├── __init__.py
│   ├── forward_policy.py      # Activity 2.2
│   ├── backward_policy.py     # Activity 2.3
│   └── reward_model.py        # From Phase 1
├── training/
│   ├── __init__.py
│   ├── sampler.py             # Activity 2.4
│   ├── loss.py                # Activity 2.5
│   └── trainer.py             # Activity 2.6
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # Diversity, quality metrics
│   └── visualize.py           # UMAP, plots
├── data/
│   ├── __init__.py
│   ├── flip.py                # From Phase -1
│   └── propedia.py            # From Phase -1
└── configs/
    └── default.yaml           # Hyperparameters
```

### 4.3 Configuration

**`configs/default.yaml`**:
```yaml
# Model
policy:
  vocab_size: 23
  d_model: 256
  n_layers: 4
  n_heads: 8
  dropout: 0.1
  max_len: 64

# Sampling
sampling:
  min_length: 10
  max_length: 30
  temperature: 1.0
  exploration_eps: 0.001  # δ: uniform policy mixing coefficient (Bengio 2021, Eq. 10)

# Training
training:
  loss_type: "trajectory_balance"  # or "sub_trajectory_balance"
  batch_size: 64
  learning_rate: 3e-4
  log_z_lr_multiplier: 10.0  # Malkin 2022 recommends higher LR for partition function
  max_grad_norm: 1.0
  num_steps: 100000
  log_interval: 100
  eval_interval: 1000
  save_interval: 5000

# Reward
reward:
  type: "improved"  # or "composite"
  temperature: 1.0  # β: reward sharpening exponent, R_sharp = R(x)^β (β > 1 focuses on high-reward modes)

# W&B
wandb:
  project: "gflownet-peptide"
  entity: "ewijaya"
```

### 4.4 Vocabulary Specification

| Index | Token | Description |
|-------|-------|-------------|
| 0-19 | A,C,D,...,Y | 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY) |
| 20 | START | Beginning of sequence marker |
| 21 | STOP | End of sequence marker |
| 22 | PAD | Padding for batching |

---

## 5. Success Criteria

| ID | Criterion | Target | Measurement Method | Verification Command |
|----|-----------|--------|-------------------|---------------------|
| SC1 | Unit test coverage | ≥80% | pytest-cov | `pytest tests/ --cov=gflownet_peptide --cov-report=term --cov-fail-under=80` |
| SC2 | P_F forward pass | Correct shapes [B, 23] | Manual + test | `pytest tests/test_forward_policy.py::test_forward_shape -v` |
| SC3 | Trajectory sampling | Valid AA sequences | Decode and validate | `pytest tests/test_sampler.py::test_valid_sequences -v` |
| SC4 | TB loss computation | Finite, non-negative | Check NaN/Inf | `pytest tests/test_loss.py::test_loss_finite -v` |
| SC5 | Training step runs | No errors on single batch | Test script | `python -c "from scripts.test_training import test_single_step; test_single_step()"` |
| SC6 | Gradient flow | log_Z receives gradients | Gradient check | `pytest tests/test_loss.py::test_log_z_gradient -v` |
| SC7 | Min/max length | Sequences respect bounds | Validation check | `pytest tests/test_sampler.py::test_length_constraints -v` |

---

## 6. Deliverables Checklist

- [ ] `gflownet_peptide/models/forward_policy.py` - ForwardPolicy Transformer
- [ ] `gflownet_peptide/models/backward_policy.py` - BackwardPolicy (Uniform)
- [ ] `gflownet_peptide/training/sampler.py` - TrajectorySampler
- [ ] `gflownet_peptide/training/loss.py` - TrajectoryBalanceLoss, SubTrajectoryBalanceLoss
- [ ] `gflownet_peptide/training/trainer.py` - GFlowNetTrainer with W&B
- [ ] `configs/default.yaml` - Default hyperparameters
- [ ] `tests/test_forward_policy.py` - ForwardPolicy tests
- [ ] `tests/test_backward_policy.py` - BackwardPolicy tests
- [ ] `tests/test_sampler.py` - Sampler tests
- [ ] `tests/test_loss.py` - Loss function tests
- [ ] `tests/test_trainer.py` - Trainer tests
- [ ] `tests/test_integration.py` - End-to-end integration test
- [ ] All unit tests passing
- [ ] Coverage ≥80%
- [ ] All success criteria verified
- [ ] Phase gate review completed

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|------------|--------|------------|-------------|
| Causal mask implementation bug | Medium | High | Thorough testing with known inputs | Use PyTorch's built-in causal mask |
| Numerical instability in log_Z | Medium | High | Gradient clipping, smaller LR | Switch to SubTB loss |
| Memory issues with long sequences | Low | Medium | Limit max_length to 30 | Gradient checkpointing |
| Slow trajectory sampling | Medium | Medium | Vectorized batch sampling | Reduce batch size |
| W&B connection issues | Low | Low | Offline mode fallback | Local logging |

---

## 8. Phase Gate Review

### 8.1 Go/No-Go Criteria

1. All unit tests pass
2. Test coverage ≥80%
3. Single training step completes without error
4. Loss is finite and non-negative
5. Sampled sequences are valid amino acid sequences
6. Gradient flow through all components verified

### 8.2 Review Checklist

- [ ] All deliverables completed
- [ ] All success criteria met
- [ ] Code passes linting (`flake8`, `black --check`)
- [ ] Type hints present on public functions
- [ ] Documentation in docstrings
- [ ] Tests passing (`pytest tests/ -v`)
- [ ] Coverage verified (`pytest --cov-fail-under=80`)

### 8.3 Decision

**Status**: Pending
**Decision Date**: ___________
**Notes**: ___________

---

## 9. Implementation Code

This phase uses a **module-primary approach** with notebooks for testing.

### 9.1 Expected Implementation Files

| Module | Purpose | Status |
|--------|---------|--------|
| `gflownet_peptide/models/forward_policy.py` | ForwardPolicy Transformer | [ ] Not started |
| `gflownet_peptide/models/backward_policy.py` | BackwardPolicy (Uniform) | [ ] Not started |
| `gflownet_peptide/training/sampler.py` | Trajectory sampling | [ ] Not started |
| `gflownet_peptide/training/loss.py` | TB/SubTB loss | [ ] Not started |
| `gflownet_peptide/training/trainer.py` | Training loop + W&B | [ ] Not started |

| Tests | Purpose | Status |
|-------|---------|--------|
| `tests/test_forward_policy.py` | ForwardPolicy tests | [ ] Not started |
| `tests/test_backward_policy.py` | BackwardPolicy tests | [ ] Not started |
| `tests/test_sampler.py` | Sampler tests | [ ] Not started |
| `tests/test_loss.py` | Loss tests | [ ] Not started |
| `tests/test_trainer.py` | Trainer tests | [ ] Not started |
| `tests/test_integration.py` | End-to-end test | [ ] Not started |

| Notebook (testing) | Purpose | Status |
|--------------------|---------|--------|
| `notebooks/gflownet-phase-2-gflownet-core.ipynb` | Interactive testing & verification | [ ] Not started |

### 9.2 Module Requirements

1. All public functions must have type hints
2. All classes must have docstrings with Args/Returns
3. Use `@dataclass` for data containers (Trajectory, etc.)
4. Follow existing codebase patterns in `gflownet_peptide/`
5. Import order: stdlib, third-party, local

### 9.3 Testing Requirements

1. Each module has corresponding test file
2. Test both normal cases and edge cases
3. Use fixtures for reusable test components
4. Include integration tests for component interaction
5. Target ≥80% line coverage

---

## 10. Notes & References

- Master PRD: `docs/gflownet-master-prd.md`
- Phase 1 PRD (reward model): `docs/prd-phase-1-reward-model.md` (if exists)
- Reward formulation: `docs/reward_formulation.md`
- Existing models reference: `gflownet_peptide/models/`

**Key Papers**:
1. Bengio et al. (2021). "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation."
2. Malkin et al. (2022). "Trajectory Balance: Improved Credit Assignment in GFlowNets."
3. Jain et al. (2022). "Biological Sequence Design with GFlowNets."

**Implementation References**:
- torchgfn library: https://github.com/saleml/torchgfn
- GFlowNet tutorial: https://milayb.gitlab.io/gflownet-tutorial/

---

*End of Phase 2 PRD*
