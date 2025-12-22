"""
Backward Policy for GFlowNet Peptide Generation.

For linear autoregressive sequence generation, the backward policy is
deterministic: the only parent of state [a1, a2, ..., an] is [a1, a2, ..., a_{n-1}].
Thus P_B = 1 for all valid parent transitions, and log P_B = 0.
"""

import torch
import torch.nn as nn


class BackwardPolicy(nn.Module):
    """
    Backward policy P_B(s|s') for linear sequence generation.

    For autoregressive generation where actions append one token,
    the backward policy is uniform (deterministic): there is exactly
    one parent state (remove last token), so P_B = 1.
    """

    def __init__(self, use_uniform: bool = True):
        """
        Args:
            use_uniform: If True, use uniform backward policy (P_B = 1).
                        For linear generation, this is the correct choice.
        """
        super().__init__()
        self.use_uniform = use_uniform

    def log_prob(
        self,
        current_seq: torch.Tensor,
        parent_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log P_B(parent|current).

        For linear generation with uniform backward policy:
        - Only one parent exists (remove last token)
        - P_B = 1, so log P_B = 0

        Args:
            current_seq: Current state token indices [batch, seq_len]
            parent_seq: Parent state token indices [batch, seq_len - 1]

        Returns:
            log_probs: Log probabilities [batch], always 0 for uniform
        """
        batch_size = current_seq.shape[0]
        device = current_seq.device

        if self.use_uniform:
            # For linear generation, backward is deterministic
            return torch.zeros(batch_size, device=device)
        else:
            # Placeholder for learned backward policy
            raise NotImplementedError("Learned backward policy not implemented")

    def forward(
        self,
        current_seq: torch.Tensor,
        parent_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for log_prob for consistency."""
        return self.log_prob(current_seq, parent_seq)


class LearnedBackwardPolicy(nn.Module):
    """
    Learned backward policy for more complex state spaces.

    This is a placeholder for future extensions where the state space
    has multiple possible parent states (e.g., tree-structured generation,
    edit operations, etc.).
    """

    def __init__(
        self,
        vocab_size: int = 23,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def log_prob(
        self,
        current_seq: torch.Tensor,
        removed_token: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of the removed token.

        For sequence generation, this predicts which token was removed
        when transitioning from current state to parent state.

        Args:
            current_seq: Current state token indices [batch, seq_len]
            removed_token: Token that was removed [batch]

        Returns:
            log_probs: Log probabilities [batch]
        """
        # Encode current sequence
        x = self.embedding(current_seq)
        x = self.transformer(x)

        # Predict removed token from last position
        logits = self.head(x[:, -1, :])
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        return log_probs.gather(-1, removed_token.unsqueeze(-1)).squeeze(-1)
