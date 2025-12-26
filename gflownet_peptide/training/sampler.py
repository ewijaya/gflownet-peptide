"""
Trajectory Sampling for GFlowNet Training.

Samples complete trajectories from the forward policy for training.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Trajectory:
    """Container for a sampled trajectory."""

    # Sequence of states (partial sequences)
    states: list[torch.Tensor]

    # Actions taken at each step
    actions: torch.Tensor

    # Log probabilities of forward policy
    log_pf: torch.Tensor

    # Log probabilities of backward policy
    log_pb: torch.Tensor

    # Final complete sequence as string
    sequence: str

    # Reward for terminal state
    reward: Optional[torch.Tensor] = None


class TrajectorySampler:
    """
    Samples trajectories for GFlowNet training.

    A trajectory is a sequence of states (partial peptides) from the
    initial state [START] to a terminal state (complete peptide).

    Supports exploration via uniform mixing (Bengio 2021, Eq. 10):
        π_explore = (1 - ε) * P_F + ε * Uniform
    """

    def __init__(
        self,
        forward_policy: nn.Module,
        backward_policy: nn.Module,
        min_length: int = 10,
        max_length: int = 30,
        exploration_eps: float = 0.0,
    ):
        """
        Args:
            forward_policy: Forward policy P_F(a|s)
            backward_policy: Backward policy P_B(s|s')
            min_length: Minimum peptide length
            max_length: Maximum peptide length
            exploration_eps: ε for uniform mixing (Bengio 2021, Eq. 10).
                             π = (1-ε)*P_F + ε*Uniform. Default 0 (no mixing).
        """
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.min_length = min_length
        self.max_length = max_length
        self.exploration_eps = exploration_eps

        # Token indices (must match forward policy)
        self.start_idx = forward_policy.start_idx
        self.stop_idx = forward_policy.stop_idx

    @torch.no_grad()
    def sample_trajectories(
        self,
        batch_size: int,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> list[Trajectory]:
        """
        Sample complete trajectories.

        Args:
            batch_size: Number of trajectories to sample
            temperature: Sampling temperature
            device: Device to use

        Returns:
            trajectories: List of Trajectory objects
        """
        if device is None:
            device = next(self.forward_policy.parameters()).device

        # Initialize trajectories
        current_seqs = torch.full(
            (batch_size, 1), self.start_idx, dtype=torch.long, device=device
        )

        all_states = [[current_seqs[:, :1].clone()] for _ in range(batch_size)]
        all_actions = [[] for _ in range(batch_size)]
        all_log_pf = [[] for _ in range(batch_size)]
        all_log_pb = [[] for _ in range(batch_size)]

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(self.max_length):
            # Get active trajectories
            active_mask = ~finished

            if not active_mask.any():
                break

            # Sample action from forward policy
            logits = self.forward_policy(current_seqs)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Force continue if below min length
            if step < self.min_length:
                logits[:, 20] = float("-inf")  # Mask STOP token

            # Sample actions with optional exploration mixing (Bengio 2021, Eq. 10)
            probs = torch.softmax(logits, dim=-1)
            if self.exploration_eps > 0:
                uniform = torch.ones_like(probs) / probs.size(-1)
                probs = (1 - self.exploration_eps) * probs + self.exploration_eps * uniform
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Use original logits for log prob (importance sampling: log P_F, not log π_explore)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_pf = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            # Compute backward log prob (uniform = 0)
            log_pb = torch.zeros_like(log_pf)

            # Check for STOP
            is_stop = actions == 20

            # Convert action to token
            tokens = actions.clone()
            tokens[is_stop] = self.stop_idx

            # Append to sequences
            current_seqs = torch.cat([current_seqs, tokens.unsqueeze(-1)], dim=1)

            # Store trajectory data
            for i in range(batch_size):
                if not finished[i]:
                    all_states[i].append(current_seqs[i : i + 1, :].clone())
                    all_actions[i].append(actions[i].item())
                    all_log_pf[i].append(log_pf[i].item())
                    all_log_pb[i].append(log_pb[i].item())

            # Update finished
            finished = finished | is_stop

        # Convert to Trajectory objects
        trajectories = []
        for i in range(batch_size):
            # Decode sequence
            sequence = self._decode_sequence(current_seqs[i])

            trajectory = Trajectory(
                states=all_states[i],
                actions=torch.tensor(all_actions[i], device=device),
                log_pf=torch.tensor(all_log_pf[i], device=device),
                log_pb=torch.tensor(all_log_pb[i], device=device),
                sequence=sequence,
            )
            trajectories.append(trajectory)

        return trajectories

    def _decode_sequence(self, tokens: torch.Tensor) -> str:
        """Convert token indices to peptide string."""
        aa_list = "ACDEFGHIKLMNPQRSTVWY"
        peptide = []

        for idx in tokens[1:].tolist():  # Skip START
            if idx == self.stop_idx:
                break
            if idx < 20:
                peptide.append(aa_list[idx])

        return "".join(peptide)

    def sample_trajectories_with_gradients(
        self,
        batch_size: int,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Sample trajectories with gradient tracking for training.

        Returns sequences, sum of log P_F, and sum of log P_B.

        Args:
            batch_size: Number of trajectories
            temperature: Sampling temperature
            device: Device to use

        Returns:
            sequences: List of peptide strings
            log_pf_sum: Sum of log forward probs [batch]
            log_pb_sum: Sum of log backward probs [batch]
        """
        if device is None:
            device = next(self.forward_policy.parameters()).device

        # Initialize
        current_seqs = torch.full(
            (batch_size, 1), self.start_idx, dtype=torch.long, device=device
        )

        log_pf_sum = torch.zeros(batch_size, device=device)
        log_pb_sum = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(self.max_length):
            if finished.all():
                break

            # Forward policy (with gradients)
            logits = self.forward_policy(current_seqs)

            if temperature != 1.0:
                logits = logits / temperature

            if step < self.min_length:
                logits[:, 20] = float("-inf")

            # Sample action with optional exploration mixing
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                if self.exploration_eps > 0:
                    uniform = torch.ones_like(probs) / probs.size(-1)
                    probs = (1 - self.exploration_eps) * probs + self.exploration_eps * uniform
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Compute log prob WITH gradients
            log_probs = torch.log_softmax(logits, dim=-1)
            log_pf = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            # Accumulate (only for active trajectories)
            log_pf_sum = log_pf_sum + log_pf * (~finished).float()
            log_pb_sum = log_pb_sum + 0.0  # Uniform backward

            # Update sequences
            is_stop = actions == 20
            tokens = actions.clone()
            tokens[is_stop] = self.stop_idx
            current_seqs = torch.cat([current_seqs, tokens.unsqueeze(-1)], dim=1)

            finished = finished | is_stop

        # Decode sequences
        sequences = [
            self._decode_sequence(current_seqs[i]) for i in range(batch_size)
        ]

        return sequences, log_pf_sum, log_pb_sum
