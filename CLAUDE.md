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
# 1. Preprocess FLIP data
python scripts/preprocess_data.py --input data/flip/raw --output data/flip/processed

# 2. Train stability predictor (optional, for --reward_type trained)
python scripts/train_stability.py --data_dir data/flip/processed --epochs 100

# 3. Train GFlowNet with trajectory balance loss
python scripts/train_gflownet.py --config configs/default.yaml --wandb --run_name my-run

# 4. Generate peptide samples
python scripts/sample.py --checkpoint checkpoints/gflownet/best.pt --n_samples 1000

# 5. Evaluate diversity and quality
python scripts/evaluate.py --gflownet_samples samples/gflownet.csv
```

### Key Training Options

```bash
# Reward types (A/B/C options)
--reward_type composite   # Untrained MLP heads (baseline)
--reward_type esm2_pll    # ESM-2 pseudo-likelihood only
--reward_type improved    # Entropy gate + naturalness (recommended)
--reward_type trained     # Trained stability predictor (requires checkpoint)

# Training stability options
--loss_type trajectory_balance        # Standard TB loss
--loss_type sub_trajectory_balance    # More stable loss curves (default)
--entropy_weight 0.01                 # Prevent mode collapse (0.01-0.1)
--log_z_lr_multiplier 3.0             # Slower log_Z learning for stability
```

## Architecture Overview

This codebase implements GFlowNet for therapeutic peptide generation. GFlowNet samples peptides proportionally to their reward P(x) ∝ R(x), providing intrinsic diversity without explicit diversity penalties.

### Core Components

**Models** (`gflownet_peptide/models/`):
- `ForwardPolicy`: Causal Transformer that predicts next amino acid given partial sequence. Outputs distribution over 20 amino acids + STOP token. Vocabulary: 20 AA + START/STOP/PAD tokens (indices 20/21/22).
- `BackwardPolicy`: Uniform P_B=1 for linear autoregressive generation.
- `CompositeReward`: ESM-2 backbone with MLP heads for stability, binding, and naturalness. Combines as R(x) = stability^w1 × binding^w2 × naturalness^w3. Frozen during GFlowNet training.

**Rewards** (`gflownet_peptide/rewards/`):
- `StabilityPredictor`: ESM-2 backbone with MLP head trained on FLIP meltome data. Predicts thermal stability from sequence.
- `ImprovedReward`: Combines entropy gating with naturalness metrics for better reward signal.

**Training** (`gflownet_peptide/training/`):
- `TrajectoryBalanceLoss`: L_TB = (log Z + Σ log P_F - log R - Σ log P_B)². Includes learnable log partition function log_Z. Supports entropy regularization.
- `SubTrajectoryBalanceLoss`: Computes loss on sub-trajectories for better credit assignment. Provides more stable loss curves.
- `TrajectorySampler`: Samples complete trajectories with forward/backward log probabilities.
- `GFlowNetTrainer`: Orchestrates sampling, loss computation, and optimization with gradient clipping.

**Evaluation** (`gflownet_peptide/evaluation/`):
- `metrics.py`: Sequence diversity (1 - mean pairwise identity), embedding diversity, cluster count, proportionality R².
- `visualize.py`: UMAP projections, reward distribution plots.

### Scripts

| Script | Purpose |
|--------|---------|
| `train_gflownet.py` | Main GFlowNet training with W&B logging |
| `train_stability.py` | Train ESM-2 stability predictor on FLIP data |
| `preprocess_data.py` | Preprocess FLIP dataset for training |
| `validate_reward_model.py` | Validate reward model predictions |
| `sample.py` | Generate peptide samples from trained model |
| `evaluate.py` | Compute diversity and quality metrics |

### Key Design Decisions

- ESM-2 embeddings are mean-pooled (excluding special tokens) for reward prediction
- Rewards are non-negative via exp/softplus transforms
- Temperature β controls sharpness: P(x) ∝ R(x)^β
- Minimum length enforced by clamping STOP action during early generation steps
- Sub-trajectory balance (STB) loss preferred for stable training curves
- Entropy regularization prevents mode collapse (weight 0.01-0.1)

## Configuration

Main hyperparameters in `configs/default.yaml`:
- `policy.d_model`: 256, `n_layers`: 4, `n_heads`: 8
- `training.loss_type`: "sub_trajectory_balance" (default) or "trajectory_balance"
- `training.entropy_weight`: 0.01 (entropy regularization)
- `training.log_z_lr_multiplier`: 3.0 (reduced from 10.0 for stability)
- `reward.temperature`: Controls reward sharpness
- `generation.min_length`/`max_length`: 10-30 amino acids

## Checkpoint Policy

Save only the latest intermediate checkpoint by overwriting to avoid disk accumulation:

- `{run_name}_latest.pt`: Overwritten at each save interval (for crash recovery)
- `{run_name}_final.pt`: Saved once at training completion (permanent)

After training completes, `_latest.pt` is redundant. Use `/clean-checkpoints` to remove it.
