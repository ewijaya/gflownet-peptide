"""ESM-2 pseudo-likelihood reward function for peptide generation.

This module provides a reward function based on ESM-2's masked language model
pseudo-likelihood scoring. Higher scores indicate more "natural" or protein-like
sequences according to ESM-2's learned representations.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_ESM_MODEL = None
_ESM_ALPHABET = None
_ESM_BATCH_CONVERTER = None


def _load_esm_model(model_name: str = "esm2_t12_35M_UR50D", device: str = "cuda"):
    """Load ESM-2 model and cache it globally."""
    global _ESM_MODEL, _ESM_ALPHABET, _ESM_BATCH_CONVERTER

    if _ESM_MODEL is not None:
        return _ESM_MODEL, _ESM_ALPHABET, _ESM_BATCH_CONVERTER

    import esm

    logger.info(f"Loading ESM-2 model: {model_name}")

    if model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    elif model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise ValueError(f"Unknown ESM model: {model_name}")

    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    _ESM_MODEL = model
    _ESM_ALPHABET = alphabet
    _ESM_BATCH_CONVERTER = batch_converter

    logger.info(f"ESM-2 model loaded on {device}")

    return model, alphabet, batch_converter


class ESM2Reward:
    """ESM-2 based reward function using pseudo-likelihood scoring.

    This computes the pseudo-likelihood of a sequence by masking each position
    and computing the probability of the true amino acid given the context.
    The final score is the mean log-probability across all positions.
    """

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        device: str = "cuda",
        normalize: bool = True,
        temperature: float = 1.0,
    ):
        """Initialize ESM-2 reward function.

        Args:
            model_name: ESM-2 model variant to use
            device: Device to run model on
            normalize: Whether to normalize scores to [0, 1] range
            temperature: Temperature for reward sharpening (higher = sharper)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.normalize = normalize
        self.temperature = temperature

        self.model, self.alphabet, self.batch_converter = _load_esm_model(
            model_name, self.device
        )

        # Get mask token index
        self.mask_idx = self.alphabet.mask_idx

        # Running statistics for normalization
        self._score_sum = 0.0
        self._score_sq_sum = 0.0
        self._score_count = 0
        self._min_score = float("inf")
        self._max_score = float("-inf")

    def _compute_pseudo_likelihood(self, sequence: str) -> float:
        """Compute pseudo-likelihood score for a single sequence.

        For each position, mask it and compute log P(true_aa | context).
        Return the mean log-probability.
        """
        if len(sequence) < 3:
            return -10.0  # Very short sequences get low score

        # Prepare data
        data = [("seq", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # Get sequence length (excluding special tokens)
        seq_len = len(sequence)

        total_log_prob = 0.0

        with torch.no_grad():
            for pos in range(seq_len):
                # Create masked version
                masked_tokens = batch_tokens.clone()
                # Position in tokens is pos + 1 (accounting for BOS token)
                token_pos = pos + 1
                true_token = masked_tokens[0, token_pos].item()
                masked_tokens[0, token_pos] = self.mask_idx

                # Get model predictions
                results = self.model(masked_tokens)
                logits = results["logits"]

                # Get log probability of true token
                log_probs = F.log_softmax(logits[0, token_pos], dim=-1)
                total_log_prob += log_probs[true_token].item()

        # Return mean log probability
        return total_log_prob / seq_len

    def _compute_pseudo_likelihood_fast(self, sequence: str) -> float:
        """Faster pseudo-likelihood using single forward pass approximation.

        Instead of masking each position individually, we use the model's
        autoregressive-style predictions from a single forward pass.
        This is an approximation but much faster.
        """
        if len(sequence) < 3:
            return -10.0

        data = [("seq", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens)
            logits = results["logits"]

            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # For each position, get the log prob of the actual token
            # Shift by 1 to account for BOS token
            total_log_prob = 0.0
            seq_len = len(sequence)

            for pos in range(seq_len):
                token_pos = pos + 1  # Account for BOS
                true_token = batch_tokens[0, token_pos].item()
                total_log_prob += log_probs[0, token_pos, true_token].item()

        return total_log_prob / seq_len

    def __call__(
        self,
        sequences: Union[str, List[str]],
        fast: bool = True,
    ) -> Union[float, List[float]]:
        """Compute reward for one or more sequences.

        Args:
            sequences: Single sequence string or list of sequences
            fast: Use fast approximation (single forward pass)

        Returns:
            Reward score(s) - higher is better
        """
        single_input = isinstance(sequences, str)
        if single_input:
            sequences = [sequences]

        compute_fn = (
            self._compute_pseudo_likelihood_fast
            if fast
            else self._compute_pseudo_likelihood
        )

        scores = []
        for seq in sequences:
            score = compute_fn(seq)

            # Update running statistics
            self._score_sum += score
            self._score_sq_sum += score * score
            self._score_count += 1
            self._min_score = min(self._min_score, score)
            self._max_score = max(self._max_score, score)

            scores.append(score)

        if self.normalize and self._score_count > 10:
            # Normalize using observed min/max
            score_range = self._max_score - self._min_score
            if score_range > 0:
                scores = [
                    (s - self._min_score) / score_range
                    for s in scores
                ]
            else:
                scores = [0.5] * len(scores)

        # Apply temperature
        if self.temperature != 1.0:
            scores = [s ** self.temperature for s in scores]

        return scores[0] if single_input else scores

    def batch_compute(
        self,
        sequences: List[str],
        batch_size: int = 32,
        fast: bool = True,
    ) -> List[float]:
        """Compute rewards for a batch of sequences efficiently.

        Args:
            sequences: List of sequences
            batch_size: Batch size for processing
            fast: Use fast approximation

        Returns:
            List of reward scores
        """
        all_scores = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            scores = self(batch, fast=fast)
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores

    def reset_statistics(self):
        """Reset running normalization statistics."""
        self._score_sum = 0.0
        self._score_sq_sum = 0.0
        self._score_count = 0
        self._min_score = float("inf")
        self._max_score = float("-inf")


def compute_esm2_reward(
    sequences: Union[str, List[str]],
    model_name: str = "esm2_t12_35M_UR50D",
    device: str = "cuda",
    fast: bool = True,
) -> Union[float, List[float]]:
    """Convenience function to compute ESM-2 reward.

    Args:
        sequences: Sequence(s) to score
        model_name: ESM-2 model variant
        device: Device to use
        fast: Use fast approximation

    Returns:
        Reward score(s)
    """
    reward_fn = ESM2Reward(model_name=model_name, device=device, normalize=False)
    return reward_fn(sequences, fast=fast)
