"""
Forward Policy for GFlowNet Peptide Generation.

The forward policy P_F(a|s) gives the probability of taking action a
(appending an amino acid) given partial sequence state s.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in 'Attention is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ForwardPolicy(nn.Module):
    """
    Forward policy P_F(a|s) implemented as a causal Transformer.

    Given a partial peptide sequence, outputs probability distribution
    over next amino acid to append.
    """

    # Amino acid vocabulary
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    SPECIAL_TOKENS = ["<START>", "<STOP>", "<PAD>"]

    def __init__(
        self,
        vocab_size: int = 23,  # 20 AA + START + STOP + PAD
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_length: int = 64,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length

        # Token indices
        self.start_idx = 20
        self.stop_idx = 21
        self.pad_idx = 22

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_length, dropout)

        # Transformer encoder (causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: predicts next amino acid (20 AA + STOP)
        self.action_head = nn.Linear(d_model, 21)  # 20 AA + STOP

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        partial_seq: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute action logits for next amino acid.

        Args:
            partial_seq: Token indices [batch, seq_len]
            padding_mask: Optional padding mask [batch, seq_len]

        Returns:
            logits: Action logits [batch, 21] (20 AA + STOP)
        """
        batch_size, seq_len = partial_seq.shape
        device = partial_seq.device

        # Embed tokens
        x = self.embedding(partial_seq)  # [B, L, D]
        x = self.pos_encoding(x)

        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Transformer forward
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        # Take last position for next-token prediction
        last_hidden = x[:, -1, :]  # [B, D]

        # Project to action space
        logits = self.action_head(last_hidden)  # [B, 21]

        return logits

    def log_prob(
        self,
        partial_seq: torch.Tensor,
        action: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probability of taking action given partial sequence.

        Args:
            partial_seq: Token indices [batch, seq_len]
            action: Action indices [batch]
            padding_mask: Optional padding mask

        Returns:
            log_probs: Log probabilities [batch]
        """
        logits = self.forward(partial_seq, padding_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

    def sample_action(
        self,
        partial_seq: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample next action from policy.

        Args:
            partial_seq: Token indices [batch, seq_len]
            temperature: Sampling temperature (higher = more random)

        Returns:
            action: Sampled action indices [batch]
            log_prob: Log probabilities of sampled actions [batch]
        """
        logits = self.forward(partial_seq)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Compute log prob
        log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob

    @torch.no_grad()
    def sample_sequence(
        self,
        batch_size: int = 1,
        max_length: int = 30,
        min_length: int = 10,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> tuple[list[str], torch.Tensor]:
        """
        Sample complete peptide sequences.

        Args:
            batch_size: Number of sequences to sample
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            temperature: Sampling temperature
            device: Device to use

        Returns:
            sequences: List of peptide sequences as strings
            log_probs: Sum of log probabilities for each sequence [batch]
        """
        if device is None:
            device = next(self.parameters()).device

        # Initialize with START token
        current_seq = torch.full(
            (batch_size, 1), self.start_idx, dtype=torch.long, device=device
        )
        total_log_prob = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length):
            # Sample next action
            action, log_prob = self.sample_action(current_seq, temperature)

            # Force continue if below min length
            if step < min_length:
                action = action.clamp(max=19)  # Only allow amino acids

            # Update log prob (only for unfinished sequences)
            total_log_prob = total_log_prob + log_prob * (~finished).float()

            # Check for STOP tokens
            is_stop = action == 20  # STOP token index in action space
            finished = finished | is_stop

            # Map action to token (action 0-19 = AA, action 20 = STOP)
            token = action.clone()
            token[is_stop] = self.stop_idx

            # Append to sequence
            current_seq = torch.cat([current_seq, token.unsqueeze(-1)], dim=1)

            # Early exit if all finished
            if finished.all():
                break

        # Convert to strings
        sequences = self._decode_sequences(current_seq)

        return sequences, total_log_prob

    def _decode_sequences(self, token_indices: torch.Tensor) -> list[str]:
        """Convert token indices to peptide strings."""
        sequences = []

        for seq in token_indices:
            peptide = []
            for idx in seq[1:].tolist():  # Skip START token
                if idx == self.stop_idx or idx == self.pad_idx:
                    break
                if idx < 20:  # Valid amino acid
                    peptide.append(self.AMINO_ACIDS[idx])
            sequences.append("".join(peptide))

        return sequences

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Convert peptide string to token indices."""
        tokens = [self.start_idx]
        for aa in sequence:
            if aa in self.AMINO_ACIDS:
                tokens.append(self.AMINO_ACIDS.index(aa))
        tokens.append(self.stop_idx)
        return torch.tensor(tokens, dtype=torch.long)
