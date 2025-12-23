# Models Implementation Notes

**Author:** Computational ISDD Team
**Date:** December 23, 2025
**Status:** Implementation Guide for GFlowNet Models

---

## Overview

This document details the work needed to bring `gflownet_peptide/models/` from prototype to production-ready. The core GFlowNet math is correct, but several gaps exist in robustness, testing, and production features.

---

## 1. ForwardPolicy (`forward_policy.py`)

### Current State
- Causal Transformer architecture is correctly implemented
- Action sampling with temperature works
- Log probability computation is correct

### Required Fixes

#### 1.1 Hardcoded ESM Layer Index (Critical)
**Location:** `reward_model.py:94`
```python
results = self._model(batch_tokens, repr_layers=[33], return_contacts=False)
token_embeddings = results["representations"][33]
```

**Problem:** Layer 33 is hardcoded, but only valid for `esm2_t33_650M_UR50D`. Other models have different layer counts:
- `esm2_t6_8M_UR50D` → layer 6
- `esm2_t12_35M_UR50D` → layer 12
- `esm2_t30_150M_UR50D` → layer 30
- `esm2_t36_3B_UR50D` → layer 36

**Fix:**
```python
# Add to ESMBackbone.__init__
self._repr_layers = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
    "esm2_t36_3B_UR50D": 36,
}
self.repr_layer = self._repr_layers.get(model_name, 33)

# Update forward()
results = self._model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
token_embeddings = results["representations"][self.repr_layer]
```

#### 1.2 Missing Entropy Regularization (High Priority)
**Problem:** No mechanism to encourage exploration. Policy may collapse to deterministic behavior.

**Add to ForwardPolicy:**
```python
def entropy(self, partial_seq: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of action distribution for exploration regularization.

    Args:
        partial_seq: Token indices [batch, seq_len]

    Returns:
        entropy: Entropy values [batch]
    """
    logits = self.forward(partial_seq)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy
```

**Usage in trainer:** Add entropy bonus to loss:
```python
loss = tb_loss - entropy_coef * entropy.mean()
```

#### 1.3 Missing Epsilon-Greedy Exploration
**Problem:** `sample_action` only uses temperature-based exploration.

**Add exploration modes:**
```python
def sample_action(
    self,
    partial_seq: torch.Tensor,
    temperature: float = 1.0,
    epsilon: float = 0.0,  # NEW
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample with optional epsilon-greedy exploration."""
    logits = self.forward(partial_seq)

    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    # Epsilon-greedy: with prob epsilon, sample uniformly
    if epsilon > 0 and self.training:
        uniform = torch.ones_like(probs) / probs.shape[-1]
        probs = (1 - epsilon) * probs + epsilon * uniform

    action = torch.multinomial(probs, num_samples=1).squeeze(-1)
    log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

    return action, log_prob
```

#### 1.4 Missing Temperature Annealing Support
**Problem:** No built-in support for temperature schedules during training.

**Add to ForwardPolicy:**
```python
def get_temperature(self, step: int, schedule: str = "constant", **kwargs) -> float:
    """
    Get temperature for current training step.

    Schedules:
        - "constant": Fixed temperature
        - "linear": Linear decay from temp_start to temp_end
        - "exponential": Exponential decay
    """
    if schedule == "constant":
        return kwargs.get("temperature", 1.0)
    elif schedule == "linear":
        temp_start = kwargs.get("temp_start", 2.0)
        temp_end = kwargs.get("temp_end", 0.5)
        total_steps = kwargs.get("total_steps", 100000)
        progress = min(step / total_steps, 1.0)
        return temp_start + (temp_end - temp_start) * progress
    elif schedule == "exponential":
        temp_start = kwargs.get("temp_start", 2.0)
        decay = kwargs.get("decay", 0.99999)
        temp_min = kwargs.get("temp_min", 0.5)
        return max(temp_start * (decay ** step), temp_min)
```

#### 1.5 Missing Action Masking for Invalid Tokens
**Problem:** No mechanism to mask out invalid actions (e.g., PAD token should never be sampled).

**Fix in `sample_action`:**
```python
def sample_action(self, partial_seq: torch.Tensor, ...) -> ...:
    logits = self.forward(partial_seq)

    # Mask out invalid actions (PAD should never be generated)
    # Action space is 0-19 (AA) + 20 (STOP), so this is already handled
    # But add explicit check for safety:
    if logits.shape[-1] > 21:
        logits[:, 21:] = float('-inf')

    ...
```

#### 1.6 Gradient Checkpointing for Memory Efficiency
**Problem:** Long sequences may OOM during training.

**Add:**
```python
def __init__(self, ..., use_checkpointing: bool = False):
    ...
    self.use_checkpointing = use_checkpointing

def forward(self, ...):
    ...
    if self.use_checkpointing and self.training:
        x = torch.utils.checkpoint.checkpoint(
            self.transformer, x, causal_mask, padding_mask
        )
    else:
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
```

---

## 2. BackwardPolicy (`backward_policy.py`)

### Current State
- Uniform backward policy (P_B = 1) is **mathematically correct** for linear autoregressive generation
- `LearnedBackwardPolicy` is a placeholder stub

### Required Work

#### 2.1 Complete LearnedBackwardPolicy (Low Priority)
**Note:** Only needed if we move beyond linear generation (e.g., edit-based generation).

The current stub needs:
```python
class LearnedBackwardPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int = 23,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=64, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def log_prob(
        self,
        current_seq: torch.Tensor,
        removed_token: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log P_B(removed_token | current_seq)."""
        x = self.embedding(current_seq)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Predict from last position
        logits = self.head(x[:, -1, :])
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs.gather(-1, removed_token.unsqueeze(-1)).squeeze(-1)
```

#### 2.2 Add Validation for Parent-Child Consistency
**Add to BackwardPolicy:**
```python
def validate_transition(
    self,
    current_seq: torch.Tensor,
    parent_seq: torch.Tensor,
) -> torch.Tensor:
    """
    Validate that parent is valid predecessor of current.

    For linear generation: parent = current[:-1]

    Returns:
        is_valid: Boolean tensor [batch]
    """
    # Parent should be current minus last token
    expected_parent = current_seq[:, :-1]

    # Compare (handling different lengths)
    if parent_seq.shape[1] != expected_parent.shape[1]:
        return torch.zeros(current_seq.shape[0], dtype=torch.bool, device=current_seq.device)

    return (parent_seq == expected_parent).all(dim=1)
```

---

## 3. RewardModel (`reward_model.py`)

### Current State
- ESMBackbone with lazy loading
- Three reward heads (stability, binding, naturalness)
- Multiplicative composition

### Required Fixes

#### 3.1 ESM Layer Index Bug (Critical - same as 1.1)
See section 1.1 above. This is the same bug.

#### 3.2 Device Handling in ESMBackbone (Critical)
**Problem:** ESM model device may not match input device.

**Location:** `ESMBackbone.forward()`

**Current (buggy):**
```python
batch_tokens = batch_tokens.to(next(self._model.parameters()).device)
```

**Fix:**
```python
def forward(self, sequences: list[str], device: Optional[torch.device] = None) -> torch.Tensor:
    self._load_model()

    # Determine target device
    if device is None:
        device = next(self._model.parameters()).device

    # Move model if needed (expensive, warn user)
    if next(self._model.parameters()).device != device:
        import warnings
        warnings.warn(f"Moving ESM model to {device}. Consider initializing on correct device.")
        self._model = self._model.to(device)

    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
    _, _, batch_tokens = self._batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    ...
```

#### 3.3 NaturalnessReward is a Placeholder (High Priority)
**Problem:** Current implementation uses embedding norm as "naturalness proxy" which is not meaningful.

**Location:** `NaturalnessReward.forward()` lines 274-284

**Current (placeholder):**
```python
embedding_norms = torch.norm(embeddings, dim=-1)
rewards = torch.sigmoid(embedding_norms / self.temperature)
```

**Proper implementation using pseudo-perplexity:**
```python
class NaturalnessReward(nn.Module):
    def __init__(self, esm_model: str = "esm2_t33_650M_UR50D", temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.esm_model = esm_model
        self._model = None
        self._alphabet = None
        self._batch_converter = None

    def _load_model(self):
        if self._model is None:
            import esm
            self._model, self._alphabet = esm.pretrained.load_model_and_alphabet(self.esm_model)
            self._batch_converter = self._alphabet.get_batch_converter()
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False

    def forward(self, sequences: Union[str, list[str]]) -> torch.Tensor:
        """Compute pseudo-perplexity based naturalness."""
        if isinstance(sequences, str):
            sequences = [sequences]

        self._load_model()
        device = next(self._model.parameters()).device

        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self._batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            # Get logits from ESM
            logits = self._model(batch_tokens)["logits"]  # [B, L, vocab]

            # Compute per-position log probs
            log_probs = F.log_softmax(logits, dim=-1)

            # Get log prob of actual tokens (pseudo-likelihood)
            # Shift by 1 to get predictions for each position
            target_tokens = batch_tokens[:, 1:]  # Exclude <cls>
            pred_log_probs = log_probs[:, :-1, :]  # Exclude last position

            # Gather log probs of actual tokens
            token_log_probs = pred_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

            # Mask out special tokens (padding, eos)
            mask = (target_tokens != self._alphabet.padding_idx) & (target_tokens != self._alphabet.eos_idx)

            # Average log prob (higher = more natural)
            avg_log_prob = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Convert to reward: exp(avg_log_prob / temperature)
            # Higher log_prob = higher reward
            rewards = torch.exp(avg_log_prob / self.temperature)

        return rewards
```

#### 3.4 Missing Reward Normalization/Clipping (High Priority)
**Problem:** Unbounded rewards can destabilize training.

**Add to CompositeReward:**
```python
def __init__(self, ..., reward_clip: float = 100.0, reward_min: float = 1e-6):
    ...
    self.reward_clip = reward_clip
    self.reward_min = reward_min

def forward(self, sequences, return_components=False):
    ...
    # Clip composite reward
    composite = composite.clamp(min=self.reward_min, max=self.reward_clip)
    ...
```

#### 3.5 Missing Reward Tempering (R^beta) (Medium Priority)
**Problem:** No support for reward sharpening during training.

**Add to CompositeReward:**
```python
def forward(
    self,
    sequences: Union[str, list[str]],
    return_components: bool = False,
    beta: float = 1.0,  # NEW: reward temperature
) -> ...:
    ...
    # Apply reward temperature: R^beta
    # beta > 1: sharper (focus on high rewards)
    # beta < 1: flatter (more exploration)
    composite = composite ** beta
    ...
```

#### 3.6 Caching for Repeated Sequences (Medium Priority)
**Problem:** Same sequences may be evaluated multiple times during training.

**Add LRU cache:**
```python
from functools import lru_cache

class CompositeReward(nn.Module):
    def __init__(self, ..., cache_size: int = 10000):
        ...
        self._cache_size = cache_size
        self._cache = {}

    def forward(self, sequences, ...):
        if isinstance(sequences, str):
            sequences = [sequences]

        # Check cache for known sequences
        results = {}
        uncached = []
        uncached_idx = []

        for i, seq in enumerate(sequences):
            if seq in self._cache:
                results[i] = self._cache[seq]
            else:
                uncached.append(seq)
                uncached_idx.append(i)

        # Compute uncached
        if uncached:
            new_rewards = self._compute_rewards(uncached)
            for idx, seq, reward in zip(uncached_idx, uncached, new_rewards):
                results[idx] = reward
                if len(self._cache) < self._cache_size:
                    self._cache[seq] = reward

        # Reconstruct batch order
        rewards = torch.stack([results[i] for i in range(len(sequences))])
        return rewards
```

#### 3.7 Missing Batch Size Handling for ESM (Medium Priority)
**Problem:** Large batches may OOM with ESM.

**Add chunked processing:**
```python
def forward(self, sequences, ..., max_batch_size: int = 32):
    if isinstance(sequences, str):
        sequences = [sequences]

    if len(sequences) <= max_batch_size:
        return self._forward_batch(sequences, ...)

    # Process in chunks
    all_rewards = []
    for i in range(0, len(sequences), max_batch_size):
        chunk = sequences[i:i + max_batch_size]
        rewards = self._forward_batch(chunk, ...)
        all_rewards.append(rewards)

    return torch.cat(all_rewards, dim=0)
```

---

## 4. Missing Components

### 4.1 Flow Estimator for Detailed Balance (Low Priority)
The `DetailedBalanceLoss` in `training/loss.py` requires a flow estimator network that doesn't exist.

**Create `gflownet_peptide/models/flow_estimator.py`:**
```python
"""Flow estimator for Detailed Balance loss."""

import torch
import torch.nn as nn


class FlowEstimator(nn.Module):
    """
    Estimates log F(s) for intermediate states.

    Used by DetailedBalanceLoss to enforce local flow matching.
    """

    def __init__(
        self,
        vocab_size: int = 23,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output scalar log flow
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, state_tokens: torch.Tensor) -> torch.Tensor:
        """
        Estimate log flow for states.

        Args:
            state_tokens: Token indices [batch, seq_len]

        Returns:
            log_flows: Log flow estimates [batch]
        """
        x = self.embedding(state_tokens)
        x = self.transformer(x)

        # Pool over sequence (mean pooling)
        x = x.mean(dim=1)

        log_flow = self.head(x).squeeze(-1)
        return log_flow
```

### 4.2 Replay Buffer Integration (High Priority)
**Problem:** No off-policy training support. All samples are on-policy.

**Create `gflownet_peptide/training/replay_buffer.py`:**
```python
"""Replay buffer for off-policy GFlowNet training."""

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Trajectory:
    """A complete generation trajectory."""
    sequence: str
    tokens: torch.Tensor
    log_pf_sum: float
    log_pb_sum: float
    reward: float
    log_reward: float


class ReplayBuffer:
    """
    Prioritized replay buffer for GFlowNet.

    Stores trajectories and samples proportionally to reward.
    """

    def __init__(
        self,
        capacity: int = 100000,
        priority_alpha: float = 0.6,
    ):
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, trajectory: Trajectory):
        """Add trajectory with priority based on reward."""
        priority = (trajectory.reward + 1e-6) ** self.priority_alpha
        self.buffer.append(trajectory)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> list[Trajectory]:
        """Sample batch proportionally to priorities."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        total_priority = sum(self.priorities)
        probs = [p / total_priority for p in self.priorities]

        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)
```

---

## 5. Testing Requirements

### Unit Tests Needed

| File | Test Coverage Needed |
|------|---------------------|
| `forward_policy.py` | `test_forward_shape`, `test_sampling`, `test_log_prob_sum`, `test_temperature`, `test_entropy` |
| `backward_policy.py` | `test_uniform_log_prob`, `test_transition_validation` |
| `reward_model.py` | `test_esm_loading`, `test_device_handling`, `test_composite_shape`, `test_reward_bounds` |

### Integration Tests Needed

1. **End-to-end trajectory sampling** with reward computation
2. **TB loss computation** with real trajectories
3. **ESM + reward model** on GPU with batch processing
4. **Checkpoint save/load** round-trip

---

## 6. Priority Summary

| Priority | Item | File | Effort |
|----------|------|------|--------|
| **Critical** | Fix hardcoded ESM layer | `reward_model.py` | 30 min |
| **Critical** | Fix device handling | `reward_model.py` | 30 min |
| **High** | Implement real NaturalnessReward | `reward_model.py` | 2 hrs |
| **High** | Add entropy regularization | `forward_policy.py` | 1 hr |
| **High** | Add reward clipping | `reward_model.py` | 30 min |
| **Medium** | Add epsilon-greedy | `forward_policy.py` | 1 hr |
| **Medium** | Add temperature schedules | `forward_policy.py` | 1 hr |
| **Medium** | Add reward caching | `reward_model.py` | 2 hrs |
| **Medium** | Add batch chunking | `reward_model.py` | 1 hr |
| **Low** | Complete LearnedBackwardPolicy | `backward_policy.py` | 3 hrs |
| **Low** | Create FlowEstimator | new file | 2 hrs |
| **Low** | Create ReplayBuffer | new file | 3 hrs |

---

## 7. Recommended Implementation Order

1. **Week 1:** Fix critical bugs (ESM layer, device handling)
2. **Week 1:** Implement real NaturalnessReward
3. **Week 1:** Add reward clipping and entropy regularization
4. **Week 2:** Add exploration features (epsilon-greedy, temperature schedules)
5. **Week 2:** Add reward caching and batch chunking for performance
6. **Week 3:** Write comprehensive tests
7. **Future:** FlowEstimator, ReplayBuffer, LearnedBackwardPolicy (as needed)

---

*Document generated for implementation planning. Update as work progresses.*
