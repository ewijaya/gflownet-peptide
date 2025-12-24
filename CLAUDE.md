# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies (pytest, black, isort, flake8, mypy)
pip install -e ".[dev]"

# Install with visualization (matplotlib, seaborn, umap-learn, hdbscan)
pip install -e ".[viz]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_forward_policy.py -v

# Run a specific test
pytest tests/test_forward_policy.py::TestForwardPolicy::test_forward_shape -v

# Format code
black gflownet_peptide/ scripts/ tests/
isort gflownet_peptide/ scripts/ tests/

# Lint
flake8 gflownet_peptide/ scripts/ tests/
mypy gflownet_peptide/
```

## Training Pipeline

```bash
# 1. Train reward model on FLIP/Propedia data
python scripts/train_reward.py --task stability --data_path data/flip/

# 2. Train GFlowNet with trajectory balance loss
python scripts/train_gflownet.py --config configs/default.yaml

# 3. Generate peptide samples
python scripts/sample.py --checkpoint checkpoints/gflownet/best.pt --n_samples 1000

# 4. Evaluate diversity and quality
python scripts/evaluate.py --gflownet_samples samples/gflownet.csv
```

## Architecture Overview

This codebase implements GFlowNet for therapeutic peptide generation. GFlowNet samples peptides proportionally to their reward P(x) ∝ R(x), providing intrinsic diversity without explicit diversity penalties.

### Core Components

**Models** (`gflownet_peptide/models/`):
- `ForwardPolicy`: Causal Transformer that predicts next amino acid given partial sequence. Outputs distribution over 20 amino acids + STOP token. Vocabulary: 20 AA + START/STOP/PAD tokens (indices 20/21/22).
- `BackwardPolicy`: Uniform P_B=1 for linear autoregressive generation.
- `CompositeReward`: ESM-2 backbone with MLP heads for stability, binding, and naturalness. Combines as R(x) = stability^w1 × binding^w2 × naturalness^w3. Frozen during GFlowNet training.

**Training** (`gflownet_peptide/training/`):
- `TrajectoryBalanceLoss`: L_TB = (log Z + Σ log P_F - log R - Σ log P_B)². Includes learnable log partition function log_Z.
- `SubTrajectoryBalanceLoss`: Computes loss on sub-trajectories for better credit assignment.
- `TrajectorySampler`: Samples complete trajectories with forward/backward log probabilities.
- `GFlowNetTrainer`: Orchestrates sampling, loss computation, and optimization with gradient clipping.

**Evaluation** (`gflownet_peptide/evaluation/`):
- `metrics.py`: Sequence diversity (1 - mean pairwise identity), embedding diversity, cluster count, proportionality R².
- `visualize.py`: UMAP projections, reward distribution plots.

### Key Design Decisions

- ESM-2 embeddings are mean-pooled (excluding special tokens) for reward prediction
- Rewards are non-negative via exp/softplus transforms
- Temperature β controls sharpness: P(x) ∝ R(x)^β
- Minimum length enforced by clamping STOP action during early generation steps

## Configuration

Main hyperparameters in `configs/default.yaml`:
- `policy.d_model`: 256, `n_layers`: 4, `n_heads`: 8
- `training.loss_type`: "trajectory_balance" or "sub_trajectory_balance"
- `reward.temperature`: Controls reward sharpness
- `generation.min_length`/`max_length`: 10-30 amino acids

## Checkpoint Policy

Save only the latest intermediate checkpoint by overwriting to avoid disk accumulation:

- `{run_name}_latest.pt`: Overwritten at each save interval (for crash recovery)
- `{run_name}_final.pt`: Saved once at training completion (permanent)

After training completes, `_latest.pt` is redundant. Use `/clean-checkpoints` to remove it.
