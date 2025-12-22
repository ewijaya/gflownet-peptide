"""
GFlowNet-Peptide: Diverse Therapeutic Peptide Generation

This package implements GFlowNet for generating diverse, high-quality
therapeutic peptide candidates via reward-proportional sampling.
"""

__version__ = "0.1.0"

from gflownet_peptide.models.forward_policy import ForwardPolicy
from gflownet_peptide.models.reward_model import CompositeReward
from gflownet_peptide.training.trainer import GFlowNetTrainer

__all__ = [
    "ForwardPolicy",
    "CompositeReward",
    "GFlowNetTrainer",
]
