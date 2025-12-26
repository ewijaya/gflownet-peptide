"""Composite reward function combining multiple reward components.

This module implements a composite reward function that combines:
1. Stability predictor (trained on FLIP)
2. Entropy gate (from ImprovedReward - prevents reward hacking)
3. Naturalness (ESM-2 embedding norm)

The composite reward is used for both GRPO-D and GFlowNet training,
enabling fair comparison between methods.

Architecture:
    Input: Peptide sequence x
        ↓
    ┌─────────────────────────────────────────────────┐
    │ ESM-2 Backbone (shared, frozen)                 │
    └─────────────────────┬───────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    ┌──────────┐    ┌──────────────┐   ┌──────────┐
    │Stability │    │  Entropy     │   │Naturalness│
    │ Predictor│    │   Gate       │   │ Score     │
    └────┬─────┘    └──────┬───────┘   └─────┬────┘
         │                 │                 │
         │                 │                 │
         ▼                 ▼                 ▼
      S(x)^w₁        gate ∈ [0,1]       N(x)^w₃
         │                 │                 │
         └────────────────┬┴─────────────────┘
                          │
                          ▼
              R(x) = S^w₁ × gate × N^w₃
"""

import logging
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class CompositeReward:
    """Composite reward combining stability, entropy gate, and naturalness.

    This reward function addresses Phase 0 findings:
    - Entropy gate prevents reward hacking (homopolymers, repetitive patterns)
    - Stability predictor rewards sequences with predicted high stability
    - Naturalness rewards protein-like sequences in ESM-2 embedding space

    The components are combined multiplicatively with configurable weights,
    ensuring that all components must be good for a high overall reward.

    Attributes:
        stability_predictor: Trained stability prediction model (optional)
        improved_reward: ImprovedReward for entropy gate and naturalness
        weights: Dictionary of component weights
        device: Device for computation
    """

    def __init__(
        self,
        stability_checkpoint: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        esm_model: str = "esm2_t6_8M_UR50D",
        entropy_threshold: float = 0.5,
        entropy_sharpness: float = 10.0,
        min_length: int = 10,
        device: Optional[str] = None,
    ):
        """Initialize composite reward.

        Args:
            stability_checkpoint: Path to trained stability predictor checkpoint.
                If None, only entropy gate and naturalness are used.
            weights: Dictionary of component weights. Default:
                - stability: 1.0
                - naturalness: 0.5
            esm_model: ESM-2 model name for ImprovedReward
            entropy_threshold: Minimum normalized entropy (from ImprovedReward)
            entropy_sharpness: Sigmoid slope for entropy gate
            min_length: Minimum peptide length
            device: Device for computation (auto-detected if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Default weights
        self.weights = weights or {
            'stability': 1.0,
            'naturalness': 0.5,
        }

        # Load stability predictor if checkpoint provided
        self.stability_predictor = None
        if stability_checkpoint:
            self._load_stability_predictor(stability_checkpoint)

        # Load ImprovedReward for entropy gate and naturalness
        from .improved_reward import ImprovedReward
        self.improved_reward = ImprovedReward(
            model_name=esm_model,
            device=self.device,
            entropy_threshold=entropy_threshold,
            entropy_sharpness=entropy_sharpness,
            min_length=min_length,
            normalize=False,  # We'll handle normalization ourselves
        )

        logger.info(f"CompositeReward initialized on {self.device}")
        logger.info(f"  Stability predictor: {'loaded' if self.stability_predictor else 'disabled'}")
        logger.info(f"  Weights: {self.weights}")

    def _load_stability_predictor(self, checkpoint_path: str):
        """Load trained stability predictor from checkpoint."""
        from .stability_predictor import StabilityPredictor

        logger.info(f"Loading stability predictor from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get config from checkpoint
        config = checkpoint.get('config', {})
        esm_model = config.get('esm_model', 'esm2_t6_8M_UR50D')
        hidden_dims = config.get('hidden_dims', [256, 128])
        dropout = config.get('dropout', 0.1)

        self.stability_predictor = StabilityPredictor(
            esm_model=esm_model,
            hidden_dims=hidden_dims,
            dropout=dropout,
            freeze_esm=True,
            device=self.device,
        )
        self.stability_predictor.load_state_dict(checkpoint['model_state_dict'])
        self.stability_predictor.eval()
        self.stability_predictor = self.stability_predictor.to(self.device)

        logger.info(f"Loaded stability predictor (val R²={checkpoint.get('val_r2', 'N/A')})")

    def __call__(
        self,
        sequences: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """Compute composite reward for sequences.

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            Reward score(s) in [0, 1] range. Higher is better.
        """
        single_input = isinstance(sequences, str)
        if single_input:
            sequences = [sequences]

        rewards = []
        for seq in sequences:
            # Get components from ImprovedReward
            components = self.improved_reward.get_components(seq)
            naturalness = components['naturalness']
            entropy_gate = components['entropy_gate']
            length_gate = components['length_gate']

            # Get stability score if predictor is loaded
            if self.stability_predictor is not None:
                with torch.no_grad():
                    # Stability predictor outputs normalized values, convert to [0,1]
                    raw_stability = self.stability_predictor([seq]).item()
                    # Sigmoid to map to [0, 1]
                    stability = torch.sigmoid(torch.tensor(raw_stability)).item()
            else:
                stability = 1.0  # Neutral if no predictor

            # Combine multiplicatively
            reward = (
                (stability ** self.weights.get('stability', 1.0)) *
                entropy_gate *
                length_gate *
                (naturalness ** self.weights.get('naturalness', 0.5))
            )

            rewards.append(reward)

        return rewards[0] if single_input else rewards

    def get_components(self, sequence: str) -> Dict[str, float]:
        """Get individual reward components for debugging.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary with each component's value
        """
        components = self.improved_reward.get_components(sequence)

        # Add stability if available
        if self.stability_predictor is not None:
            with torch.no_grad():
                raw_stability = self.stability_predictor([sequence]).item()
                components['stability_raw'] = raw_stability
                components['stability'] = torch.sigmoid(torch.tensor(raw_stability)).item()
        else:
            components['stability'] = 1.0

        # Compute total
        components['total'] = self([sequence])

        return components

    def batch_compute(
        self,
        sequences: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """Compute rewards for a batch of sequences efficiently.

        Args:
            sequences: List of sequences
            batch_size: Batch size for processing

        Returns:
            List of reward scores
        """
        all_rewards = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            rewards = self(batch)
            if isinstance(rewards, float):
                rewards = [rewards]
            all_rewards.extend(rewards)

        return all_rewards


def create_composite_reward(
    stability_checkpoint: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
) -> CompositeReward:
    """Factory function to create CompositeReward with default settings.

    Args:
        stability_checkpoint: Path to stability predictor checkpoint
        weights: Component weights
        device: Device for computation

    Returns:
        Configured CompositeReward instance
    """
    return CompositeReward(
        stability_checkpoint=stability_checkpoint,
        weights=weights,
        device=device,
    )
