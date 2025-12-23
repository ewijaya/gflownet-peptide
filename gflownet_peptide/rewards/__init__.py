"""Reward functions for peptide generation."""

from .esm2_reward import ESM2Reward, compute_esm2_reward

__all__ = ["ESM2Reward", "compute_esm2_reward"]
