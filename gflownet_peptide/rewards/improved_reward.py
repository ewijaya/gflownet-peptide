"""Improved reward function with entropy gate to prevent reward hacking.

This module addresses the ESM-2 pseudo-likelihood reward hacking issue where
repetitive sequences (e.g., QQQQQQ) receive high scores because each position
is trivially predictable from context.

The improved reward combines:
1. ESM-2 embedding naturalness (NOT pseudo-likelihood)
2. Sequence entropy gate (penalize low-complexity)
3. Length gate (penalize too-short sequences)
"""

import torch
from typing import List, Union
from collections import Counter
from math import log2
import logging

logger = logging.getLogger(__name__)


class ImprovedReward:
    """
    Improved reward function for peptide generation.

    Components:
    1. ESM-2 embedding naturalness (NOT pseudo-likelihood)
    2. Sequence entropy gate (penalize low-complexity)
    3. Length gate (penalize too-short sequences)

    This addresses the reward hacking issue where ESM-2 pseudo-likelihood
    rewards repetitive sequences like QQQQQQQQ.
    """

    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        device: str = "cuda",
        # Entropy gate parameters
        entropy_threshold: float = 0.5,
        entropy_sharpness: float = 10.0,
        # Length gate parameters
        min_length: int = 10,
        length_sharpness: float = 0.5,
        # Embedding parameters
        embedding_temperature: float = 10.0,
        # Normalization
        normalize: bool = True,
    ):
        """Initialize improved reward function.

        Args:
            model_name: ESM-2 model variant to use
            device: Device to run model on
            entropy_threshold: Minimum normalized entropy (0-1 scale)
            entropy_sharpness: Sigmoid slope for entropy gate
            min_length: Minimum peptide length before penalty
            length_sharpness: Sigmoid slope for length gate
            embedding_temperature: Sigmoid temperature for embedding norm
            normalize: Whether to normalize scores to [0, 1] range
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.normalize = normalize

        # Gate parameters
        self.entropy_threshold = entropy_threshold
        self.entropy_sharpness = entropy_sharpness
        self.min_length = min_length
        self.length_sharpness = length_sharpness
        self.embedding_temperature = embedding_temperature

        # Load ESM-2
        self._load_model()

        # Running statistics for normalization
        self._min_score = float("inf")
        self._max_score = float("-inf")
        self._score_count = 0

    def _load_model(self):
        """Load ESM-2 model."""
        import esm

        logger.info(f"Loading ESM-2 model: {self.model_name}")

        if self.model_name == "esm2_t6_8M_UR50D":
            self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.repr_layer = 6
        elif self.model_name == "esm2_t12_35M_UR50D":
            self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.repr_layer = 12
        elif self.model_name == "esm2_t33_650M_UR50D":
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.repr_layer = 33
        else:
            raise ValueError(f"Unknown ESM model: {self.model_name}")

        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        logger.info(f"ESM-2 model loaded on {self.device}")

    def _compute_entropy(self, sequence: str) -> float:
        """
        Compute normalized Shannon entropy of amino acid distribution.

        Returns value in [0, 1] where:
        - 0 = all same amino acid (e.g., QQQQQQQQ)
        - 1 = uniform distribution of all 20 amino acids
        """
        if len(sequence) == 0:
            return 0.0

        aa_counts = Counter(sequence)
        total = len(sequence)

        entropy = 0.0
        for count in aa_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * log2(p)

        # Normalize by max possible entropy (log2(20) for 20 amino acids)
        max_entropy = log2(20)
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def _compute_entropy_gate(self, sequence: str) -> float:
        """
        Compute soft entropy gate.

        Returns ~1.0 for high-entropy (diverse) sequences.
        Returns ~0.0 for low-entropy (repetitive) sequences.
        """
        entropy = self._compute_entropy(sequence)

        # Sigmoid gate: high entropy → 1.0, low entropy → 0.0
        # Using math.e approximation for consistency
        import math
        gate = 1.0 / (1.0 + math.exp(
            -self.entropy_sharpness * (entropy - self.entropy_threshold)))

        return gate

    def _compute_length_gate(self, sequence: str) -> float:
        """
        Compute soft length gate.

        Returns ~1.0 for sequences >= min_length.
        Returns ~0.0 for sequences << min_length.
        """
        import math
        length = len(sequence)
        gate = 1.0 / (1.0 + math.exp(
            -self.length_sharpness * (length - self.min_length)))

        return gate

    def _compute_embedding_naturalness(self, sequence: str) -> float:
        """
        Compute naturalness score based on ESM-2 embedding.

        Uses embedding norm as proxy for "how well ESM-2 represents this".
        Real proteins have consistent embedding norms; garbage has abnormal norms.
        """
        import math

        if len(sequence) < 3:
            return 0.0

        data = [("seq", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False
            )

            # Get embeddings (exclude BOS and EOS tokens)
            embeddings = results["representations"][self.repr_layer]
            seq_len = len(sequence)
            seq_embeddings = embeddings[0, 1:seq_len + 1, :]  # (L, d)

            # Mean pooling
            mean_embedding = seq_embeddings.mean(dim=0)  # (d,)

            # Compute norm
            emb_norm = torch.norm(mean_embedding).item()

        # Sigmoid to map to [0, 1]
        naturalness = 1.0 / (1.0 + math.exp(
            -emb_norm / self.embedding_temperature))

        return naturalness

    def __call__(
        self,
        sequences: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """
        Compute improved reward for one or more sequences.

        Args:
            sequences: Single sequence string or list of sequences

        Returns:
            Reward score(s) in [0, 1] - higher is better
        """
        single_input = isinstance(sequences, str)
        if single_input:
            sequences = [sequences]

        scores = []
        for seq in sequences:
            # Component 1: Embedding naturalness
            naturalness = self._compute_embedding_naturalness(seq)

            # Component 2: Entropy gate
            entropy_gate = self._compute_entropy_gate(seq)

            # Component 3: Length gate
            length_gate = self._compute_length_gate(seq)

            # Combine: multiplicative (all must be good)
            score = naturalness * entropy_gate * length_gate

            # Update statistics
            self._score_count += 1
            self._min_score = min(self._min_score, score)
            self._max_score = max(self._max_score, score)

            scores.append(score)

        # Normalize if enabled and enough samples seen
        if self.normalize and self._score_count > 10:
            score_range = self._max_score - self._min_score
            if score_range > 0:
                scores = [(s - self._min_score) / score_range for s in scores]
            else:
                scores = [0.5] * len(scores)

        return scores[0] if single_input else scores

    def get_components(self, sequence: str) -> dict:
        """
        Get individual reward components for debugging.

        Returns dict with each component's value.
        """
        # Temporarily disable normalization to get raw score
        old_normalize = self.normalize
        self.normalize = False

        naturalness = self._compute_embedding_naturalness(sequence)
        entropy = self._compute_entropy(sequence)
        entropy_gate = self._compute_entropy_gate(sequence)
        length_gate = self._compute_length_gate(sequence)
        total = naturalness * entropy_gate * length_gate

        self.normalize = old_normalize

        return {
            "naturalness": naturalness,
            "entropy": entropy,
            "entropy_gate": entropy_gate,
            "length_gate": length_gate,
            "total": total,
        }

    def reset_statistics(self):
        """Reset running normalization statistics."""
        self._min_score = float("inf")
        self._max_score = float("-inf")
        self._score_count = 0

    def batch_compute(
        self,
        sequences: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """Compute rewards for a batch of sequences efficiently.

        Args:
            sequences: List of sequences
            batch_size: Batch size for processing

        Returns:
            List of reward scores
        """
        all_scores = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i: i + batch_size]
            scores = self(batch)
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores
