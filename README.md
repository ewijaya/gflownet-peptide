# GFlowNet-Peptide: Diverse Therapeutic Peptide Generation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![W&B](https://img.shields.io/badge/W%26B-Experiments-FFCC33?logo=weightsandbiases&logoColor=black)](https://wandb.ai/ewijaya/gflownet-peptide)

**Diverse therapeutic peptide generation via reward-proportional sampling with Generative Flow Networks.**

## Overview

This repository implements GFlowNet for generating diverse, high-quality therapeutic peptide candidates. Unlike reward-maximizing methods (PPO, GRPO) that converge to narrow sequence families, GFlowNet samples peptides **proportionally to their predicted fitness**, naturally producing diverse candidates without explicit diversity penalties.

### Key Features

- **Intrinsic diversity**: Samples P(x) ∝ R(x) — no diversity penalty needed
- **Multi-property optimization**: Composite rewards combining stability, binding, and naturalness
- **ESM-2 based rewards**: Leverages protein language model embeddings
- **Modular design**: Swap reward models, adjust temperature, compare to baselines
- **Public data only**: Trained on FLIP and Propedia benchmarks

### Experiment Tracking

Live training metrics and results are tracked on Weights & Biases:

**[View Live Dashboard →](https://wandb.ai/ewijaya/gflownet-peptide)**

## Why GFlowNet for Peptides?

| Method | Objective | Diversity | Trade-off |
|--------|-----------|-----------|-----------|
| PPO/GRPO | max E[R(x)] | Low (mode collapse) | Quality vs diversity (explicit λ) |
| GFlowNet | P(x) ∝ R(x) | High (intrinsic) | Controlled via temperature β |

**The problem**: Standard RL generators converge to ~5-20 similar peptides from one structural family.

**The solution**: GFlowNet samples from the full distribution of high-fitness sequences, providing wet-lab teams with diverse candidates spanning multiple scaffolds.

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gflownet-peptide.git
cd gflownet-peptide

# Create conda environment
conda create -n gflownet-peptide python=3.10
conda activate gflownet-peptide

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (recommended for ESM-2)
- ~16GB GPU memory for training

## Quick Start

### 1. Train Reward Model

```bash
# Train stability predictor on FLIP benchmark
python scripts/train_reward.py \
    --task stability \
    --data_path data/flip/ \
    --output_dir checkpoints/reward/

# Train binding predictor on Propedia
python scripts/train_reward.py \
    --task binding \
    --data_path data/propedia/ \
    --output_dir checkpoints/reward/
```

### 2. Train GFlowNet

```bash
# Train GFlowNet with composite reward
python scripts/train_gflownet.py \
    --config configs/default.yaml \
    --reward_checkpoint checkpoints/reward/ \
    --output_dir checkpoints/gflownet/
```

### 3. Generate Diverse Peptides

```bash
# Sample 1000 diverse peptides
python scripts/sample.py \
    --checkpoint checkpoints/gflownet/best.pt \
    --n_samples 1000 \
    --temperature 1.0 \
    --output samples/diverse_peptides.csv
```

### 4. Evaluate and Compare

```bash
# Compare GFlowNet vs GRPO baseline
python scripts/evaluate.py \
    --gflownet_samples samples/gflownet.csv \
    --grpo_samples samples/grpo.csv \
    --output results/comparison.json
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 GFlowNet Peptide Generation                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Forward Policy P_F (Transformer)                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [START] → [M] → [K] → [V] → ... → [STOP]                │   │
│  │     ↓       ↓      ↓      ↓                               │   │
│  │  Softmax over 20 amino acids + STOP at each step         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                     │
│  Terminal Sequence x                                             │
│                            ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Reward Model R(x) = stability × binding × naturalness    │   │
│  │  (ESM-2 backbone + MLP heads, frozen during training)    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                     │
│  Trajectory Balance Loss                                         │
│  L_TB = (log Z + Σ log P_F - log R(x))²                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
gflownet-peptide/
├── gflownet_peptide/
│   ├── models/
│   │   ├── forward_policy.py    # Transformer P_F
│   │   ├── backward_policy.py   # Uniform P_B (linear generation)
│   │   └── reward_model.py      # ESM-2 based composite reward
│   ├── training/
│   │   ├── sampler.py           # Trajectory sampling
│   │   ├── loss.py              # TB and SubTB losses
│   │   └── trainer.py           # Training loop
│   ├── evaluation/
│   │   ├── metrics.py           # Diversity, quality metrics
│   │   └── visualize.py         # UMAP, comparison plots
│   └── data/
│       ├── flip.py              # FLIP data loading
│       └── propedia.py          # Propedia data loading
├── configs/
│   └── default.yaml             # Default hyperparameters
├── scripts/
│   ├── train_reward.py          # Train reward model
│   ├── train_gflownet.py        # Train GFlowNet
│   ├── sample.py                # Generate peptides
│   └── evaluate.py              # Evaluation and comparison
├── tests/
│   └── test_*.py                # Unit tests
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Property | Size | Source |
|---------|----------|------|--------|
| FLIP Stability | ΔΔG (thermal stability) | 53K | [FLIP Benchmark](https://benchmark.protein.properties/) |
| FLIP GB1 | Binding fitness | 150K | [FLIP Benchmark](https://benchmark.protein.properties/) |
| Propedia | Peptide-protein binding | 19K | [Propedia](http://bioinfo.dcc.ufmg.br/propedia/) |
| ProteinGym | Multi-assay fitness | 2.5M | [ProteinGym](https://proteingym.org/) |

## Configuration

Key hyperparameters in `configs/default.yaml`:

```yaml
# Forward Policy
policy:
  d_model: 256
  n_layers: 4
  n_heads: 8
  dropout: 0.1

# Training
training:
  batch_size: 64
  learning_rate: 3e-4
  n_steps: 100000
  loss_type: "trajectory_balance"  # or "sub_trajectory_balance"

# Reward
reward:
  stability_weight: 1.0
  binding_weight: 1.0
  naturalness_weight: 0.5
  temperature: 1.0  # β in P(x) ∝ R(x)^β

# Generation
generation:
  max_length: 30
  min_length: 10
```

## Evaluation Metrics

### Diversity Metrics
- **Sequence diversity**: 1 - mean pairwise sequence identity
- **Embedding diversity**: Mean pairwise cosine distance in ESM space
- **Cluster count**: Number of HDBSCAN clusters in UMAP projection
- **Unique sequences**: Fraction of non-duplicate samples

### Quality Metrics
- **Mean reward**: Average R(x) over generated set
- **Max reward**: Highest R(x) achieved
- **Top-10% mean**: Mean of top 10% by reward
- **Validity**: Fraction of valid amino acid sequences

### Calibration Metrics
- **Proportionality R²**: Correlation between log(frequency) and log(reward)

## Results

Expected results from training on FLIP + Propedia:

| Method | Mean Reward | Diversity | Clusters | Proportionality R² |
|--------|-------------|-----------|----------|-------------------|
| Random | 0.30 | 0.95 | 50+ | N/A |
| GRPO | 0.85 | 0.45 | 5-10 | N/A |
| GRPO + diversity penalty | 0.75 | 0.70 | 15-20 | N/A |
| **GFlowNet (ours)** | 0.78 | 0.85 | 30-40 | 0.82 |

## Citation

If you use this code, please cite:

```bibtex
@article{gflownet_peptide_2025,
  title={Diverse Therapeutic Peptide Generation with GFlowNet:
         A Comparison to Reward-Maximizing Reinforcement Learning},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025}
}
```

### Key References

1. Bengio et al. (2021). "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation." NeurIPS.
2. Malkin et al. (2022). "Trajectory Balance: Improved Credit Assignment in GFlowNets." NeurIPS.
3. Jain et al. (2022). "Biological Sequence Design with GFlowNets." ICML.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This work builds on the GFlowNet framework developed by the Mila team (Bengio et al.). We thank the authors of FLIP, Propedia, and ESM-2 for making their data and models publicly available.
