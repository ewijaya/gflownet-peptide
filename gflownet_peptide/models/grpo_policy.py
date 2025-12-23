"""GRPO Policy and Value Network for peptide generation.

This module implements the PolicyValueNetwork for Group Relative Policy
Optimization (GRPO), adapted from the user's implementation to work with
the GFlowNet peptide project.

The architecture uses a pretrained ProtGPT2-distilled model as the backbone
with frozen embeddings and a trainable value head for advantage estimation.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel


# Valid amino acids for peptide generation
VALID_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class PolicyValueNetwork(nn.Module):
    """GPT2-based policy network with value head for GRPO.

    Uses a pretrained protein language model (ProtGPT2-distilled) as backbone
    with frozen embeddings. The value head estimates state values for
    advantage computation in GRPO.
    """

    def __init__(
        self,
        model_name: str = "littleworth/protgpt2-distilled-medium",
        hidden_dim: int = 256,
        freeze_embeddings: bool = True,
    ):
        """Initialize the policy-value network.

        Args:
            model_name: HuggingFace model name for the backbone
            hidden_dim: Hidden dimension for value head
            freeze_embeddings: Whether to freeze embedding layers
        """
        super().__init__()

        self.model_name = model_name
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

        # Get model embedding dimension
        self.n_embd = self.gpt2_model.config.n_embd

        # Freeze embeddings to preserve pretrained patterns
        if freeze_embeddings:
            for param in self.gpt2_model.transformer.wte.parameters():
                param.requires_grad = False
            for param in self.gpt2_model.transformer.wpe.parameters():
                param.requires_grad = False

        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(self.n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value estimate.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple of:
                - policy_logits: [batch_size, seq_len, vocab_size]
                - value: [batch_size, 1]
        """
        if attention_mask is None:
            pad_token_id = self.gpt2_model.config.eos_token_id
            attention_mask = (input_ids != pad_token_id).long()

        outputs = self.gpt2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        policy_logits = outputs.logits

        # Get hidden states for value estimation
        last_hidden_states = outputs.hidden_states[-1]

        # Get the last non-padded position for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = torch.clamp(seq_lengths, min=0)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)

        # Extract hidden state at the last valid position
        relevant_hidden = last_hidden_states[batch_indices, seq_lengths]

        # Compute value estimate
        value = self.value_head(relevant_hidden)

        return policy_logits, value

    def get_policy(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Get policy distribution with optional temperature and nucleus sampling.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            temperature: Softmax temperature
            top_p: Nucleus sampling threshold

        Returns:
            Policy probabilities [batch_size, seq_len, vocab_size]
        """
        policy_logits, _ = self.forward(input_ids, attention_mask)
        policy_logits = policy_logits / temperature

        if top_p < 1.0:
            # Apply nucleus sampling
            sorted_logits, sorted_indices = torch.sort(policy_logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            for batch_idx in range(policy_logits.size(0)):
                for seq_idx in range(policy_logits.size(1)):
                    indices_to_remove = sorted_indices[batch_idx, seq_idx][
                        sorted_indices_to_remove[batch_idx, seq_idx]
                    ]
                    policy_logits[batch_idx, seq_idx, indices_to_remove] = float("-inf")

        policy = F.softmax(policy_logits, dim=-1)
        return policy

    def get_value(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get value estimate for states.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Value estimates [batch_size, 1]
        """
        _, value = self.forward(input_ids, attention_mask)
        return value

    def generate_peptide(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 30,
        min_length: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate a single peptide sequence.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            prompt: Optional starting prompt

        Returns:
            Generated peptide sequence
        """
        device = next(self.parameters()).device

        # Generate prompt if not provided
        if prompt is None:
            num_residues = random.randint(1, 3)
            prompt = "".join(random.choices(list(VALID_AMINO_ACIDS), k=num_residues))

        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            output_sequences = self.gpt2_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length + 5,  # Buffer for tokenization
                min_length=min_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and clean peptide
        peptide = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        peptide = peptide.strip().replace("\n", "")
        peptide = "".join(c for c in peptide if c in VALID_AMINO_ACIDS)

        # Ensure minimum length
        if len(peptide) < min_length:
            peptide = prompt + "".join(
                random.choices(list(VALID_AMINO_ACIDS), k=min_length - len(prompt))
            )

        # Truncate to max length
        if len(peptide) > max_length:
            peptide = peptide[:max_length]

        return peptide

    def generate_multiple_peptides(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int,
        max_length: int = 30,
        min_length: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """Generate multiple peptide sequences.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            num_samples: Number of peptides to generate
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens

        Returns:
            List of generated peptide sequences
        """
        peptides = []
        for _ in range(num_samples):
            peptide = self.generate_peptide(
                tokenizer=tokenizer,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            peptides.append(peptide)
        return peptides


def initialize_tokenizer(model_name: str = "littleworth/protgpt2-distilled-medium"):
    """Initialize tokenizer for the policy network.

    Args:
        model_name: HuggingFace model name

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_peptide(
    peptide: str,
    tokenizer: AutoTokenizer,
    max_length: int = 32,
) -> torch.Tensor:
    """Tokenize a peptide sequence.

    Args:
        peptide: Peptide sequence string
        tokenizer: Tokenizer to use
        max_length: Maximum token length

    Returns:
        Token IDs tensor
    """
    peptide = peptide.strip().replace("\n", "")
    encoded = tokenizer(
        peptide,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    tokens = encoded.input_ids.squeeze()
    if tokens.dim() == 0:
        tokens = tokens.unsqueeze(0)
    return tokens
