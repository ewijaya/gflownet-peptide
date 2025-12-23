"""GRPO-D Trainer for peptide generation.

This module implements Group Relative Policy Optimization with Diversity
awareness (GRPO-D) for training peptide generators. It follows the user's
original implementation with:

- Group-wise advantage normalization
- Diversity-weighted reward combination
- KL divergence penalty against reference model
- Memory-efficient training with chunked processing
"""

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from .diversity import calculate_batch_diversity_stats, calculate_peptide_diversity
from ..models.grpo_policy import PolicyValueNetwork, initialize_tokenizer, tokenize_peptide

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience for GRPO buffer."""

    peptide: str
    reward: float
    diversity_score: float
    tokenized_seq: torch.Tensor
    prompt: str
    value: Optional[float] = None
    log_probs: Optional[torch.Tensor] = None


class ExperienceBuffer:
    """Experience buffer for GRPO training with diversity tracking."""

    def __init__(self, buffer_size: int):
        self.buffer: deque = deque(maxlen=buffer_size)

    def add(
        self,
        peptide: str,
        reward: float,
        diversity_score: float,
        tokenized_seq: torch.Tensor,
        prompt: str,
        value: Optional[float] = None,
        log_probs: Optional[torch.Tensor] = None,
    ):
        """Add an experience to the buffer."""
        self.buffer.append(
            Experience(
                peptide=peptide,
                reward=reward,
                diversity_score=diversity_score,
                tokenized_seq=tokenized_seq,
                prompt=prompt,
                value=value,
                log_probs=log_probs,
            )
        )

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch from the buffer."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def get_all(self) -> List[Experience]:
        """Get all experiences."""
        return list(self.buffer)

    def get_top_k(self, k: int, key: str = "reward") -> List[Experience]:
        """Get top-k experiences by specified key."""
        return sorted(
            list(self.buffer),
            key=lambda x: getattr(x, key),
            reverse=True,
        )[:k]

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


def compute_grpo_advantages(
    rewards_by_group: Dict[str, List[float]],
    diversity_by_group: Dict[str, List[float]],
    config: Dict[str, Any],
) -> Dict[str, List[float]]:
    """Compute advantages for GRPO with diversity integration.

    Combines rewards with diversity scores and normalizes within each group.

    Args:
        rewards_by_group: Dict mapping prompt to list of rewards
        diversity_by_group: Dict mapping prompt to list of diversity scores
        config: Configuration with diversity_weight

    Returns:
        Dict mapping prompt to list of advantages
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    advantages_by_group = {}
    diversity_weight = config.get("diversity_weight", 0.15)

    for prompt, rewards in rewards_by_group.items():
        if len(rewards) <= 1:
            # Can't normalize with single sample
            advantages_by_group[prompt] = [0.0] * len(rewards)
            continue

        # Get diversity scores for this group
        diversity_scores = diversity_by_group.get(prompt, [0.0] * len(rewards))

        # Combine rewards with diversity using weighted sum
        combined_rewards = []
        for r, d in zip(rewards, diversity_scores):
            combined = (1 - diversity_weight) * r + diversity_weight * d
            combined_rewards.append(combined)

        # Normalize within group
        rewards_tensor = torch.tensor(combined_rewards, dtype=torch.float, device=device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8

        # Calculate normalized advantages
        advantages = (rewards_tensor - mean_reward) / std_reward
        advantages_by_group[prompt] = advantages.tolist()

        del rewards_tensor, advantages

    torch.cuda.empty_cache()
    return advantages_by_group


def estimate_kl_divergence(
    ref_logits: torch.Tensor,
    policy_logits: torch.Tensor,
    action_tokens: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Memory-efficient KL divergence estimation using chunking.

    Args:
        ref_logits: Logits from reference model
        policy_logits: Logits from policy model
        action_tokens: The actual tokens that were chosen
        chunk_size: Number of samples to process at once

    Returns:
        Estimated KL divergence
    """
    device = action_tokens.device
    batch_size = action_tokens.shape[0]
    total_kl = 0.0

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_size_actual = end_idx - i

        ref_chunk = ref_logits[i:end_idx]
        policy_chunk = policy_logits[i:end_idx]
        action_chunk = action_tokens[i:end_idx]

        ref_probs = F.softmax(ref_chunk, dim=-1)
        policy_probs = F.softmax(policy_chunk, dim=-1)

        batch_indices = (
            torch.arange(chunk_size_actual)
            .unsqueeze(1)
            .expand(chunk_size_actual, action_chunk.shape[1])
            .to(device)
        )
        seq_indices = (
            torch.arange(action_chunk.shape[1])
            .unsqueeze(0)
            .expand(chunk_size_actual, action_chunk.shape[1])
            .to(device)
        )

        ref_action_probs = ref_probs[batch_indices, seq_indices, action_chunk]
        policy_action_probs = policy_probs[batch_indices, seq_indices, action_chunk]

        ratio = ref_action_probs / (policy_action_probs + 1e-10)
        kl_estimate_chunk = ratio - torch.log(ratio + 1e-10) - 1.0

        total_kl += kl_estimate_chunk.sum()

        del ref_probs, policy_probs, ref_action_probs, policy_action_probs
        torch.cuda.empty_cache()

    return total_kl / (batch_size * action_tokens.shape[1])


def compute_grpo_loss(
    model: PolicyValueNetwork,
    ref_model: PolicyValueNetwork,
    batch: Dict[str, Any],
    advantages_by_group: Dict[str, List[float]],
    config: Dict[str, Any],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the GRPO loss with KL penalty.

    Args:
        model: Policy model being trained
        ref_model: Reference model for KL penalty
        batch: Batch data with states, actions, prompts
        advantages_by_group: Computed advantages
        config: Configuration
        pad_token_id: Padding token ID

    Returns:
        Tuple of (total_loss, policy_loss, kl_loss)
    """
    device = next(model.parameters()).device
    states = batch["states"]
    actions = batch["actions"]
    prompts = batch["prompts"]
    group_indices = batch["group_indices"]

    # Get advantages for each sample
    advantage_list = []
    for i, prompt in enumerate(prompts):
        if prompt in advantages_by_group:
            try:
                advantage_idx = group_indices[prompt].index(i)
                advantage_list.append(advantages_by_group[prompt][advantage_idx])
            except (ValueError, IndexError):
                advantage_list.append(0.0)
        else:
            advantage_list.append(0.0)

    advantages_tensor = torch.tensor(advantage_list, dtype=torch.float, device=device)

    # Forward pass - policy model
    attention_mask = (states != pad_token_id).long()
    policy_logits, _ = model(states, attention_mask)

    # Forward pass - reference model
    with torch.no_grad():
        ref_logits, _ = ref_model(states, attention_mask)

    # Shift logits to match action tokens
    policy_logits_shifted = policy_logits[:, :-1, :]
    ref_logits_shifted = ref_logits[:, :-1, :]
    actions_trimmed = actions[:, : policy_logits_shifted.size(1)]

    # Compute policy log probabilities
    batch_size = actions_trimmed.shape[0]
    chunk_size = 4
    policy_log_probs_list = []

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)

        policy_chunk = policy_logits_shifted[i:end_idx]
        actions_chunk = actions_trimmed[i:end_idx]

        action_probs_chunk = F.softmax(policy_chunk, dim=-1)

        policy_probs_chunk = torch.gather(
            action_probs_chunk, dim=2, index=actions_chunk.unsqueeze(-1)
        ).squeeze(-1)

        policy_log_probs_chunk = torch.log(policy_probs_chunk + 1e-10)
        policy_log_probs_mean_chunk = policy_log_probs_chunk.mean(dim=1)
        policy_log_probs_list.append(policy_log_probs_mean_chunk)

        del policy_chunk, actions_chunk, action_probs_chunk, policy_probs_chunk
        torch.cuda.empty_cache()

    policy_log_probs_mean = torch.cat(policy_log_probs_list, dim=0)

    # Estimate KL divergence
    kl_div = estimate_kl_divergence(
        ref_logits_shifted, policy_logits_shifted, actions_trimmed
    )

    # Calculate losses
    beta = config.get("beta", 0.04)
    policy_loss = -(policy_log_probs_mean * advantages_tensor).mean()
    kl_loss = beta * kl_div

    total_loss = policy_loss + kl_loss

    del policy_logits, ref_logits, policy_logits_shifted, ref_logits_shifted
    torch.cuda.empty_cache()

    return total_loss, policy_loss, kl_loss


class GRPOTrainer:
    """GRPO-D Trainer for peptide generation.

    Implements Group Relative Policy Optimization with diversity awareness.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        reward_fn: callable,
        device: str = "cuda",
    ):
        """Initialize the trainer.

        Args:
            config: Training configuration
            reward_fn: Reward function that takes sequences and returns scores
            device: Device to use
        """
        self.config = config
        self.reward_fn = reward_fn
        self.device = device if torch.cuda.is_available() else "cpu"

        # Initialize models
        model_name = config.get("model_name", "littleworth/protgpt2-distilled-medium")
        hidden_dim = config.get("hidden_dim", 256)

        self.policy_model = PolicyValueNetwork(
            model_name=model_name,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Reference model (frozen copy)
        self.ref_model = PolicyValueNetwork(
            model_name=model_name,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.ref_model.load_state_dict(self.policy_model.state_dict())
        self.ref_model.eval()

        # Tokenizer
        self.tokenizer = initialize_tokenizer(model_name)

        # Optimizer
        lr = config.get("learning_rate", 3e-4)
        self.optimizer = optim.Adam(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=lr,
        )

        # Experience buffer
        buffer_size = config.get("buffer_size", 500)
        self.buffer = ExperienceBuffer(buffer_size)

        # Training state
        self.iteration = 0
        self.training_stats = {
            "iterations": [],
            "mean_reward": [],
            "max_reward": [],
            "total_loss": [],
            "policy_loss": [],
            "kl_loss": [],
            "mean_diversity": [],
        }

    def generate_and_evaluate(self) -> Tuple[List[str], List[float], List[str], List[float]]:
        """Generate peptides and evaluate them.

        Returns:
            Tuple of (peptides, rewards, prompts, diversity_scores)
        """
        config = self.config
        batch_size = config.get("batch_size", 16)
        num_generations = config.get("num_generations", 8)
        max_length = config.get("max_length", 30)
        min_length = config.get("min_length", 10)
        temperature = config.get("temperature", 1.0)
        top_p = config.get("top_p", 0.95)
        repetition_penalty = config.get("repetition_penalty", 1.0)
        valid_amino_acids = config.get("amino_acids", "ACDEFGHIKLMNPQRSTVWY")

        all_peptides = []
        all_rewards = []
        all_prompts = []

        # Generate prompts
        prompts = []
        for _ in range(batch_size):
            num_residues = random.randint(1, 3)
            prompt = "".join(random.choices(list(valid_amino_acids), k=num_residues))
            prompts.append(prompt)

        # Generate peptides for each prompt
        self.policy_model.eval()
        with torch.no_grad():
            for prompt in prompts:
                for _ in range(num_generations):
                    peptide = self.policy_model.generate_peptide(
                        tokenizer=self.tokenizer,
                        max_length=max_length,
                        min_length=min_length,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        prompt=prompt,
                    )

                    # Filter invalid characters
                    peptide = "".join(c for c in peptide if c in valid_amino_acids)

                    if len(peptide) < min_length:
                        continue

                    # Compute reward
                    reward = self.reward_fn(peptide)
                    if isinstance(reward, list):
                        reward = reward[0]

                    all_peptides.append(peptide)
                    all_rewards.append(reward)
                    all_prompts.append(prompt)

        self.policy_model.train()

        # Compute diversity scores
        diversity_scores = calculate_peptide_diversity(all_peptides, config)

        return all_peptides, all_rewards, all_prompts, diversity_scores

    def train_step(self) -> Dict[str, float]:
        """Perform one GRPO training step.

        Returns:
            Dictionary of training metrics
        """
        config = self.config
        max_length = config.get("max_length", 30)

        # Generate and evaluate peptides
        peptides, rewards, prompts, diversity_scores = self.generate_and_evaluate()

        if not peptides:
            return {"total_loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0}

        # Clear buffer and add new experiences
        self.buffer.clear()

        for peptide, reward, prompt, div_score in zip(
            peptides, rewards, prompts, diversity_scores
        ):
            tokenized = tokenize_peptide(peptide, self.tokenizer, max_length + 2)
            tokenized = tokenized.to(self.device)

            self.buffer.add(
                peptide=peptide,
                reward=reward,
                diversity_score=div_score,
                tokenized_seq=tokenized,
                prompt=prompt,
            )

        # Group rewards and diversity by prompt
        rewards_by_group: Dict[str, List[float]] = {}
        diversity_by_group: Dict[str, List[float]] = {}

        for exp in self.buffer.get_all():
            if exp.prompt not in rewards_by_group:
                rewards_by_group[exp.prompt] = []
                diversity_by_group[exp.prompt] = []
            rewards_by_group[exp.prompt].append(exp.reward)
            diversity_by_group[exp.prompt].append(exp.diversity_score)

        # Compute advantages
        advantages_by_group = compute_grpo_advantages(
            rewards_by_group, diversity_by_group, config
        )

        # Prepare batch
        experiences = self.buffer.get_all()
        states = [exp.tokenized_seq for exp in experiences]
        actions = [exp.tokenized_seq[1:] for exp in experiences]
        batch_prompts = [exp.prompt for exp in experiences]

        pad_token_id = self.tokenizer.pad_token_id
        states = pad_sequence(states, batch_first=True, padding_value=pad_token_id).to(
            self.device
        )
        actions = pad_sequence(actions, batch_first=True, padding_value=pad_token_id).to(
            self.device
        )

        # Build group indices
        group_indices: Dict[str, List[int]] = {}
        for batch_idx, prompt in enumerate(batch_prompts):
            if prompt not in group_indices:
                group_indices[prompt] = []
            group_indices[prompt].append(batch_idx)

        batch = {
            "states": states,
            "actions": actions,
            "prompts": batch_prompts,
            "group_indices": group_indices,
        }

        # Compute loss and update
        total_loss, policy_loss, kl_loss = compute_grpo_loss(
            self.policy_model,
            self.ref_model,
            batch,
            advantages_by_group,
            config,
            pad_token_id,
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        max_grad_norm = config.get("max_grad_norm", 1.0)
        nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_grad_norm)
        self.optimizer.step()

        # Update statistics
        self.iteration += 1
        self.training_stats["iterations"].append(self.iteration)
        self.training_stats["mean_reward"].append(np.mean(rewards))
        self.training_stats["max_reward"].append(np.max(rewards))
        self.training_stats["total_loss"].append(total_loss.item())
        self.training_stats["policy_loss"].append(policy_loss.item())
        self.training_stats["kl_loss"].append(kl_loss.item())
        self.training_stats["mean_diversity"].append(np.mean(diversity_scores))

        del states, actions, batch
        torch.cuda.empty_cache()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "mean_diversity": np.mean(diversity_scores),
            "num_peptides": len(peptides),
        }

    def save_checkpoint(self, path: str):
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.policy_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "config": self.config,
            "training_stats": self.training_stats,
            "top_peptides": [
                {"peptide": exp.peptide, "reward": exp.reward}
                for exp in self.buffer.get_top_k(100)
            ],
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint["iteration"]
        self.training_stats = checkpoint.get("training_stats", self.training_stats)
        logger.info(f"Checkpoint loaded from {path}")

    def get_top_peptides(self, k: int = 100) -> List[Dict[str, Any]]:
        """Get top-k peptides by reward.

        Args:
            k: Number of peptides to return

        Returns:
            List of dicts with peptide and reward
        """
        top_exp = self.buffer.get_top_k(k)
        return [{"peptide": exp.peptide, "reward": exp.reward} for exp in top_exp]
