# AWS Instance Recommendations for GFlowNet Peptide Generation

This document provides AWS EC2 instance recommendations for each phase of the GFlowNet project, including both minimum cost-effective options and ideal configurations.

**Reference:** [GFlowNet Master PRD](./gflownet-master-prd.md)

---

## Executive Summary

| Phase | Duration | Minimum Instance | Ideal Instance | Est. Cost (Spot) |
|-------|----------|------------------|----------------|------------------|
| -1: Data Acquisition | 1 week | t3.medium (CPU) | t3.large (CPU) | ~$5 |
| 0: Validation | 1 week | g4dn.xlarge | g5.xlarge | ~$30-50 |
| 1: Reward Model | 2 weeks | g4dn.xlarge | g5.2xlarge | ~$50-100 |
| 2: Implementation | 2 weeks | t3.medium (CPU) | t3.large (CPU) | ~$10 |
| 3: GFlowNet Training | 2 weeks | g5.xlarge | g5.2xlarge | ~$100-200 |
| 4: Evaluation | 2 weeks | g4dn.xlarge | g5.xlarge | ~$50-100 |
| 5: Documentation | 2 weeks | t3.medium (CPU) | t3.large (CPU) | ~$10 |
| **Total** | **11 weeks** | | | **~$250-475** |

---

## AWS GPU Instance Reference

### Available GPU Instances for Deep Learning

| Instance | GPU | GPU Memory | vCPUs | RAM | On-Demand | Spot | Best For |
|----------|-----|------------|-------|-----|-----------|------|----------|
| **g4dn.xlarge** | 1× T4 | 16 GB | 4 | 16 GB | $0.526/hr | $0.187/hr | Budget inference/training |
| **g4dn.2xlarge** | 1× T4 | 16 GB | 8 | 32 GB | $0.752/hr | $0.226/hr | More CPU/RAM needed |
| **g6.xlarge** | 1× L4 | 24 GB | 4 | 16 GB | $0.805/hr | $0.356/hr | Better perf than g4dn |
| **g6.2xlarge** | 1× L4 | 24 GB | 8 | 32 GB | $1.006/hr | $0.445/hr | L4 + more resources |
| **g5.xlarge** | 1× A10G | 24 GB | 4 | 16 GB | $1.006/hr | $0.429/hr | ML training/inference |
| **g5.2xlarge** | 1× A10G | 24 GB | 8 | 32 GB | $1.212/hr | $0.528/hr | Recommended for training |
| **g5.4xlarge** | 1× A10G | 24 GB | 16 | 64 GB | $1.624/hr | $0.698/hr | Large batch training |
| **p4d.24xlarge** | 8× A100 | 320 GB | 96 | 1152 GB | $21.96/hr | $7.23/hr | Large-scale distributed |

> **Note:** Prices are approximate for US East (N. Virginia). Actual prices vary by region.

---

## Phase-by-Phase Recommendations

### Phase -1: Data Acquisition & Infrastructure (1 week)

**Compute Requirements:**
- CPU-only workload
- Download ~5 GB of data (FLIP, Propedia, ESM-2 models)
- Run data validation scripts

| Option | Instance | vCPUs | RAM | Storage | On-Demand | Spot |
|--------|----------|-------|-----|---------|-----------|------|
| **Minimum** | t3.medium | 2 | 4 GB | EBS 50GB | $0.042/hr | $0.013/hr |
| **Ideal** | t3.large | 2 | 8 GB | EBS 100GB | $0.083/hr | $0.025/hr |

**Estimated Active Hours:** 10-20 hours
**Estimated Cost:** $1-5 (Spot)

**Why:**
- No GPU needed for data download and validation
- Sufficient CPU for running pytest and data preprocessing
- EBS storage for dataset persistence

---

### Phase 0: Validation (1 week)

**Compute Requirements:**
- Run existing GRPO baseline to generate 1000 peptides
- Compute diversity metrics (UMAP + HDBSCAN clustering)
- GPU Memory: 8-16 GB

| Option | Instance | GPU | GPU Mem | On-Demand | Spot |
|--------|----------|-----|---------|-----------|------|
| **Minimum** | g4dn.xlarge | 1× T4 | 16 GB | $0.526/hr | $0.187/hr |
| **Ideal** | g5.xlarge | 1× A10G | 24 GB | $1.006/hr | $0.429/hr |

**Estimated Active Hours:** 20-40 hours
**Estimated Cost:** $30-50 (Spot)

**Why:**
- g4dn.xlarge provides sufficient GPU memory for GRPO inference
- g5.xlarge offers faster inference if budget allows
- Main bottleneck is inference speed, not memory

---

### Phase 1: Reward Model Development (2 weeks)

**Compute Requirements:**
- Train ESM-2 → Stability predictor (FLIP: 53K sequences)
- Train ESM-2 → Binding predictor (Propedia: 19K sequences)
- GPU Memory: 16 GB (with ESM-2 t12_35M) or 24 GB (with t33_650M)

| Option | Instance | GPU | GPU Mem | On-Demand | Spot |
|--------|----------|-----|---------|-----------|------|
| **Minimum** | g4dn.xlarge | 1× T4 | 16 GB | $0.526/hr | $0.187/hr |
| **Cost-Effective** | g6.xlarge | 1× L4 | 24 GB | $0.805/hr | $0.356/hr |
| **Ideal** | g5.2xlarge | 1× A10G | 24 GB | $1.212/hr | $0.528/hr |

**Estimated Active Hours:** 40-80 hours
**Estimated Cost:** $50-100 (Spot)

**Why:**
- **Minimum (g4dn.xlarge):** Works with ESM-2 t12_35M (35M params, ~3-4 GB). Requires smaller batch sizes.
- **Cost-Effective (g6.xlarge):** L4 GPU provides better training performance than T4, 24 GB allows larger ESM-2 models.
- **Ideal (g5.2xlarge):** A10G has best tensor core performance for training, 32 GB RAM allows larger data loading.

**Configuration Tips:**
- Use ESM-2 t12_35M_UR50D (480-dim embeddings) on 16 GB GPUs
- Use ESM-2 t33_650M_UR50D (1280-dim embeddings) on 24 GB GPUs
- Batch size 32 is default; reduce to 16 if OOM

---

### Phase 2: GFlowNet Core Implementation (2 weeks)

**Compute Requirements:**
- Code development and unit testing
- Minimal GPU needed (quick forward pass tests)
- CPU-intensive: code writing, testing

| Option | Instance | vCPUs | RAM | Storage | On-Demand | Spot |
|--------|----------|-------|-----|---------|-----------|------|
| **Minimum** | t3.medium | 2 | 4 GB | EBS 50GB | $0.042/hr | $0.013/hr |
| **Ideal** | t3.large | 2 | 8 GB | EBS 100GB | $0.083/hr | $0.025/hr |

**For GPU testing (occasional):**
| Option | Instance | GPU | GPU Mem | On-Demand | Spot |
|--------|----------|-----|---------|-----------|------|
| **Testing** | g4dn.xlarge | 1× T4 | 16 GB | $0.526/hr | $0.187/hr |

**Estimated Active Hours:** 40-80 hours (CPU), 5-10 hours (GPU)
**Estimated Cost:** $5-15 (Spot)

**Why:**
- Most work is code development, not compute
- Spin up GPU instance only for quick integration tests
- Use spot instances for testing (acceptable interruption)

---

### Phase 3: GFlowNet Training & Hyperparameter Tuning (2 weeks)

**Compute Requirements:**
- Train GFlowNet to convergence (100K steps)
- Hyperparameter sweeps (learning rate, batch size, model size)
- GPU Memory: 16 GB minimum, 24 GB recommended
- Training time: 8-12 hours per run on A100, 16-24 hours on A10G

| Option | Instance | GPU | GPU Mem | On-Demand | Spot |
|--------|----------|-----|---------|-----------|------|
| **Minimum** | g5.xlarge | 1× A10G | 24 GB | $1.006/hr | $0.429/hr |
| **Ideal** | g5.2xlarge | 1× A10G | 24 GB | $1.212/hr | $0.528/hr |
| **Fast Sweeps** | g5.4xlarge | 1× A10G | 24 GB | $1.624/hr | $0.698/hr |

**Estimated Active Hours:** 100-200 hours
**Estimated Cost:** $100-200 (Spot)

**Why:**
- **Minimum (g5.xlarge):** A10G required for efficient transformer training. T4 too slow.
- **Ideal (g5.2xlarge):** 32 GB RAM enables larger trajectory batches, faster data loading with more workers.
- **Fast Sweeps (g5.4xlarge):** 64 GB RAM for aggressive hyperparameter sweeping with large batch accumulation.

**Configuration Tips:**
- Default batch size: 64 trajectories
- Use mixed precision (AMP) for ~30-40% memory savings
- Enable gradient checkpointing for longer sequences
- Consider checkpointing every 5000 steps (~110-220 MB per checkpoint)

**Not Recommended:**
- g4dn instances: T4 too slow for transformer training
- p4d.24xlarge: Overkill for single-GPU training

---

### Phase 4: Evaluation & GRPO Comparison (2 weeks)

**Compute Requirements:**
- Generate 1000 peptides from GFlowNet + 1000 from GRPO
- Compute all metrics (diversity, quality, proportionality)
- ESM embeddings for clustering analysis
- GPU Memory: 8-16 GB

| Option | Instance | GPU | GPU Mem | On-Demand | Spot |
|--------|----------|-----|---------|-----------|------|
| **Minimum** | g4dn.xlarge | 1× T4 | 16 GB | $0.526/hr | $0.187/hr |
| **Ideal** | g5.xlarge | 1× A10G | 24 GB | $1.006/hr | $0.429/hr |

**Estimated Active Hours:** 40-80 hours
**Estimated Cost:** $50-100 (Spot)

**Why:**
- Evaluation is inference-only (no training)
- g4dn.xlarge sufficient for sample generation and ESM embedding
- g5.xlarge faster for large-scale embedding computation

**Breakdown:**
- Sample generation: ~30 min each (GFlowNet + GRPO)
- Reward evaluation: ~30 min
- ESM embedding (2000 seqs): ~10-15 min
- UMAP + HDBSCAN: ~5 min
- Statistical tests: ~5 min

---

### Phase 5: Documentation & Publication (2 weeks)

**Compute Requirements:**
- CPU-only workload
- Paper writing, figure generation
- Code cleanup and documentation

| Option | Instance | vCPUs | RAM | Storage | On-Demand | Spot |
|--------|----------|-------|-----|---------|-----------|------|
| **Minimum** | t3.medium | 2 | 4 GB | EBS 50GB | $0.042/hr | $0.013/hr |
| **Ideal** | t3.large | 2 | 8 GB | EBS 100GB | $0.083/hr | $0.025/hr |

**Estimated Active Hours:** 40-80 hours
**Estimated Cost:** $5-10 (Spot)

**Why:**
- No GPU needed for documentation
- Minimal compute for figure generation
- Use local development where possible

---

## Cost Optimization Strategies

### 1. Use Spot Instances

Spot instances offer **60-90% savings** over On-Demand:

| Instance | On-Demand | Spot | Savings |
|----------|-----------|------|---------|
| g4dn.xlarge | $0.526/hr | $0.187/hr | 64% |
| g5.xlarge | $1.006/hr | $0.429/hr | 57% |
| g5.2xlarge | $1.212/hr | $0.528/hr | 56% |

**Best Practices:**
- Use spot for training with checkpointing (resume on interruption)
- Use on-demand for critical final runs only
- Set up spot fleet for automatic capacity management

### 2. Right-Size Instances

| Phase | Don't Use | Use Instead | Why |
|-------|-----------|-------------|-----|
| Phase -1 | GPU instances | t3.medium | CPU-only workload |
| Phase 0 | g5.2xlarge | g4dn.xlarge | Inference doesn't need A10G |
| Phase 1 | p4d.24xlarge | g5.2xlarge | Single GPU sufficient |
| Phase 3 | g4dn.xlarge | g5.xlarge | T4 too slow for training |

### 3. Use Reserved Instances for Long Projects

For 3+ month projects:

| Instance | On-Demand | 1-Year Reserved | 3-Year Reserved |
|----------|-----------|-----------------|-----------------|
| g5.xlarge | $1.006/hr | $0.634/hr | $0.435/hr |
| g5.2xlarge | $1.212/hr | $0.764/hr | $0.524/hr |

### 4. Regional Pricing Differences

Prices vary by region. US East (N. Virginia) is typically cheapest:

| Region | Relative Cost |
|--------|---------------|
| us-east-1 (N. Virginia) | Baseline |
| us-west-2 (Oregon) | +0-5% |
| eu-west-1 (Ireland) | +5-10% |
| ap-northeast-1 (Tokyo) | +15-20% |

---

## Storage Recommendations

### EBS Volume Configuration

| Phase | Recommended Size | Type | IOPS |
|-------|------------------|------|------|
| All Phases | 100 GB | gp3 | 3000 |

**Cost:** ~$8/month for 100 GB gp3

### Storage Breakdown

| Component | Size |
|-----------|------|
| FLIP datasets (raw) | ~500 MB |
| Propedia dataset (raw) | ~200 MB |
| ESM-2 t12_35M model | ~500 MB |
| ESM-2 t33_650M model | ~2.5 GB |
| Processed/cached data | ~1 GB |
| Checkpoints (100K steps) | ~4 GB |
| Generated samples | ~100 MB |
| **Total Required** | **~10 GB** |
| **Recommended** | **100 GB** (headroom for experiments) |

---

## Total Project Cost Estimate

### Scenario 1: Minimum Budget (Spot Instances)

| Phase | Instance | Hours | Cost |
|-------|----------|-------|------|
| -1 | t3.medium | 20 | $0.26 |
| 0 | g4dn.xlarge | 30 | $5.61 |
| 1 | g4dn.xlarge | 60 | $11.22 |
| 2 | t3.medium | 60 | $0.78 |
| 3 | g5.xlarge | 150 | $64.35 |
| 4 | g4dn.xlarge | 60 | $11.22 |
| 5 | t3.medium | 60 | $0.78 |
| Storage | 100 GB gp3 | 3 months | $24.00 |
| **Total** | | | **~$120** |

### Scenario 2: Ideal Setup (Spot Instances)

| Phase | Instance | Hours | Cost |
|-------|----------|-------|------|
| -1 | t3.large | 20 | $0.50 |
| 0 | g5.xlarge | 40 | $17.16 |
| 1 | g5.2xlarge | 80 | $42.24 |
| 2 | t3.large | 80 | $2.00 |
| 3 | g5.2xlarge | 200 | $105.60 |
| 4 | g5.xlarge | 80 | $34.32 |
| 5 | t3.large | 80 | $2.00 |
| Storage | 100 GB gp3 | 3 months | $24.00 |
| **Total** | | | **~$230** |

### Scenario 3: On-Demand (No Interruptions)

| Phase | Instance | Hours | Cost |
|-------|----------|-------|------|
| -1 | t3.large | 20 | $1.66 |
| 0 | g5.xlarge | 40 | $40.24 |
| 1 | g5.2xlarge | 80 | $96.96 |
| 2 | t3.large | 80 | $6.64 |
| 3 | g5.2xlarge | 200 | $242.40 |
| 4 | g5.xlarge | 80 | $80.48 |
| 5 | t3.large | 80 | $6.64 |
| Storage | 100 GB gp3 | 3 months | $24.00 |
| **Total** | | | **~$500** |

---

## Quick Reference: Instance Selection Decision Tree

```
Is GPU needed?
├── No → t3.medium or t3.large
└── Yes
    └── Training or Inference?
        ├── Inference only → g4dn.xlarge (budget) or g5.xlarge (faster)
        └── Training
            └── Model size?
                ├── ESM-2 t12_35M (35M params) → g4dn.xlarge (tight) or g5.xlarge
                └── ESM-2 t33_650M (650M params) → g5.xlarge or g5.2xlarge
```

---

## Sources

- [AWS EC2 G5 Instances](https://aws.amazon.com/ec2/instance-types/g5/)
- [AWS EC2 G6 Instances](https://aws.amazon.com/ec2/instance-types/g6/)
- [AWS EC2 P4d Instances](https://aws.amazon.com/ec2/instance-types/p4/)
- [AWS EC2 On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
- [Vantage EC2 Instance Pricing](https://instances.vantage.sh/)

---

*Document generated: December 2024*
*Prices subject to change. Verify current pricing on AWS website.*
