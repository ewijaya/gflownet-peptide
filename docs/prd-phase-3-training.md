# Phase 3: Training & Hyperparameter Tuning - Detailed PRD

**Generated from**: docs/gflownet-master-prd.md Section 5.3
**Date**: 2025-12-26
**Status**: Draft
**Last Updated**: 2025-12-26 (sanity check verified)

---

## 1. Executive Summary

- **Objective**: Train the GFlowNet model to convergence using Trajectory Balance loss, tune hyperparameters for optimal diversity-quality trade-off, and generate a large peptide dataset for downstream evaluation.

- **Duration**: 2 weeks

- **Key Deliverables**:
  - Trained GFlowNet model (best checkpoint)
  - W&B experiment dashboard with training curves
  - Hyperparameter sweep results
  - 10,000 generated peptides from final model
  - Training configuration for reproducibility

- **Prerequisites**:
  - Phase 2 complete (GFlowNet core implementation)
  - All 123 tests passing
  - `ImprovedReward` functional (Phase 0b/1)
  - GPU available (A100 recommended, T4/V100 acceptable)

- **Verified**:
  - ✅ Sanity check (100 steps) completed successfully
  - W&B run: https://wandb.ai/ewijaya/gflownet-peptide/runs/5ej9lfs9

- **Bug Fixes Applied** (2025-12-26):
  - Fixed `ESMBackbone` device handling for lazy loading (now properly moves to CUDA)
  - Fixed `RewardHead` sigmoid transform (was returning raw values instead of sigmoid)

---

## 2. Objectives & Scope

### 2.1 In-Scope Goals

| ID | Goal | Priority |
|----|------|----------|
| 3.1 | Run initial training with default hyperparameters | P0 |
| 3.2 | Monitor training stability (loss, log_Z) via W&B | P0 |
| 3.3 | Conduct hyperparameter sweep | P0 |
| 3.4 | Address any training instabilities | P0 |
| 3.5 | Train final model to convergence | P0 |
| 3.6 | Generate 10,000 peptides from trained model | P0 |
| 3.7 | Document best hyperparameters | P1 |

### 2.2 Out-of-Scope (Deferred)

| Item | Deferred To |
|------|-------------|
| GRPO comparison | Phase 4 |
| Statistical analysis | Phase 4 |
| Multi-objective reward tuning | Future work |
| Distributed training | Not required for this scale |

### 2.3 Dependencies

| Dependency | Source | Required By |
|------------|--------|-------------|
| ForwardPolicy, BackwardPolicy | Phase 2 | Activity 3.1 |
| TrajectoryBalanceLoss | Phase 2 | Activity 3.1 |
| GFlowNetTrainer | Phase 2 | Activity 3.1 |
| ImprovedReward | Phase 0b/1 | Activity 3.1 |
| W&B account configured | Environment | Activity 3.2 |
| GPU (A100/V100/T4) | Infrastructure | All activities |

---

## 3. Detailed Activities

### Activity 3.1: Initial Training Run (Default Hyperparameters)

**Description**: Run the first GFlowNet training with default configuration to establish a baseline and verify the training pipeline works end-to-end.

**Steps**:

1. Verify environment and dependencies:
   ```bash
   # Check GPU availability
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

   # Check W&B login
   wandb login --verify
   ```

2. Run short sanity check (100 steps):
   ```bash
   python scripts/train_gflownet.py \
     --config configs/default.yaml \
     --n_steps 100 \
     --batch_size 32 \
     --output_dir checkpoints/gflownet/sanity_check/ \
     --wandb
   ```

3. Run baseline training (10K steps):
   ```bash
   python scripts/train_gflownet.py \
     --config configs/default.yaml \
     --n_steps 10000 \
     --output_dir checkpoints/gflownet/baseline/ \
     --wandb
   ```

**Implementation Notes**:
- Default config uses: LR=3e-4, batch_size=64, d_model=256, n_layers=4
- log_Z LR multiplier is 10x (Malkin 2022 recommendation)
- Exploration epsilon is 0.001 for uniform mixing
- Uses `ImprovedReward` by default (not CompositeReward)

**Verification**:
```bash
# Check training completed
ls -la checkpoints/gflownet/baseline/

# Verify checkpoint loadable
python -c "
import torch
ckpt = torch.load('checkpoints/gflownet/baseline/gflownet_latest.pt')
print('Checkpoint keys:', ckpt.keys())
print('Step:', ckpt.get('step', 'N/A'))
"
```

**Output**: Baseline model checkpoint, initial W&B run

---

### Activity 3.2: Monitor Training Metrics

**Description**: Set up comprehensive monitoring via W&B to track training health and catch issues early.

**Key Metrics to Monitor**:

| Metric | Expected Behavior | Warning Sign |
|--------|-------------------|--------------|
| `train/loss` | Decreasing, plateaus | Diverging, oscillating |
| `train/log_z` | Stabilizes around true log Z | Diverging (→∞ or →-∞) |
| `train/mean_log_pf` | Stable, slightly increasing | Collapsing to 0 |
| `train/mean_log_reward` | Increasing | Stuck at low value |
| `eval/mean_reward` | Increasing | Decreasing after initial rise |
| `eval/diversity` | High and stable | Collapsing to 0 |

**W&B Dashboard Setup**:

1. Create custom panels:
   - Loss curves (train/loss, log scale)
   - Partition function (train/log_z)
   - Sample quality (eval/mean_reward, eval/max_reward)
   - Sample diversity (eval/diversity)

2. Set up alerts:
   - Alert if loss > 100 (divergence)
   - Alert if log_z > 50 or < -50 (instability)

**Verification**:
```bash
# Check W&B run exists
wandb sync --dry-run

# View runs
wandb runs list --project gflownet-peptide
```

**Output**: W&B dashboard with training curves

---

### Activity 3.3: Hyperparameter Sweep

**Description**: Systematic search over key hyperparameters to find optimal configuration.

**Hyperparameter Grid** (from Master PRD):

| Hyperparameter | Values | Rationale |
|----------------|--------|-----------|
| Learning rate | 1e-4, **3e-4**, 1e-3 | Standard range for Transformers |
| Batch size | 32, **64**, 128 | Memory vs variance trade-off |
| P_F layers | **4**, 6 | Depth vs speed |
| P_F hidden dim | **256**, 512 | Capacity vs speed |
| Max sequence length | 20, **30**, 40 | Peptide length range |
| Training steps | 10K, **50K**, 100K | Convergence point |

**Bold** = default values

**Sweep Configuration** (`configs/sweep.yaml`):

```yaml
program: scripts/train_gflownet.py
method: bayes
metric:
  name: eval/diversity_quality_ratio
  goal: maximize
parameters:
  learning_rate:
    values: [1e-4, 3e-4, 1e-3]
  batch_size:
    values: [32, 64, 128]
  n_layers:
    values: [4, 6]
  d_model:
    values: [256, 512]
```

**Steps**:

1. Create sweep:
   ```bash
   wandb sweep configs/sweep.yaml --project gflownet-peptide
   ```

2. Run sweep agents (can run multiple in parallel):
   ```bash
   wandb agent ewijaya/gflownet-peptide/<sweep_id>
   ```

3. Analyze results:
   ```bash
   # Best run analysis in W&B UI
   # Or export results:
   wandb api runs ewijaya/gflownet-peptide --filter='sweep=<sweep_id>'
   ```

**Alternative: Manual Grid Search**:

If W&B sweeps are problematic, run manually:

```bash
# Learning rate sweep
for lr in 1e-4 3e-4 1e-3; do
  python scripts/train_gflownet.py \
    --config configs/default.yaml \
    --n_steps 10000 \
    --output_dir checkpoints/gflownet/sweep_lr_${lr}/ \
    --wandb
done
```

**Output**: Sweep results, best hyperparameter configuration

---

### Activity 3.4: Address Training Instabilities

**Description**: Diagnose and fix any training issues discovered during initial runs.

**Common Issues and Fixes**:

| Issue | Symptom | Fix |
|-------|---------|-----|
| Loss divergence | loss → ∞ | Reduce LR, increase grad clip |
| log_Z divergence | log_z → ±∞ | Reduce log_Z LR multiplier |
| Mode collapse | diversity → 0 | Increase exploration_eps |
| Reward hacking | high reward, low diversity | Check reward function |
| Gradient explosion | NaN in loss | Enable gradient clipping |
| Slow convergence | loss plateaus early | Increase model capacity |

**Diagnostic Commands**:

```bash
# Check for NaN in checkpoints
python -c "
import torch
ckpt = torch.load('checkpoints/gflownet/baseline/gflownet_latest.pt')
for k, v in ckpt['forward_policy_state_dict'].items():
    if torch.isnan(v).any():
        print(f'NaN found in {k}')
"

# Analyze training curves
python -c "
import wandb
api = wandb.Api()
run = api.run('ewijaya/gflownet-peptide/<run_id>')
history = run.history()
print('Loss range:', history['train/loss'].min(), '-', history['train/loss'].max())
print('log_Z range:', history['train/log_z'].min(), '-', history['train/log_z'].max())
"
```

**Verification**: Training runs to completion without divergence

**Output**: Fixes applied, stable training configuration

---

### Activity 3.5: Train to Convergence

**Description**: Train the final model with best hyperparameters until convergence.

**Steps**:

1. Update config with best hyperparameters:
   ```bash
   # Edit configs/best.yaml based on sweep results
   ```

2. Run full training:
   ```bash
   python scripts/train_gflownet.py \
     --config configs/best.yaml \
     --n_steps 100000 \
     --output_dir checkpoints/gflownet/final/ \
     --wandb
   ```

3. Monitor until convergence:
   - Loss plateaus for >5000 steps
   - log_Z stabilizes
   - eval/diversity stable

**Convergence Criteria**:

| Metric | Convergence Definition |
|--------|------------------------|
| Loss | < 5% change over last 5000 steps |
| log_Z | < 0.1 change over last 5000 steps |
| Mean reward | Plateaued |

**Early Stopping** (if implemented):
- Patience: 10 evaluations without improvement
- Improvement threshold: 1% in diversity-quality metric

**Verification**:
```bash
# Verify final checkpoint exists
ls -la checkpoints/gflownet/final/gflownet_final.pt

# Check training completed
grep "Training complete" logs/train_gflownet.log
```

**Output**: Final trained model checkpoint

---

### Activity 3.6: Generate Peptide Dataset

**Description**: Generate a large dataset of peptides from the trained model for evaluation.

**Steps**:

1. Generate 10,000 peptides:
   ```bash
   python scripts/sample.py \
     --checkpoint checkpoints/gflownet/final/gflownet_final.pt \
     --n_samples 10000 \
     --output samples/gflownet_final.csv \
     --temperature 1.0
   ```

2. Quick validation:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('samples/gflownet_final.csv')
   print(f'Total samples: {len(df)}')
   print(f'Unique samples: {df[\"sequence\"].nunique()}')
   print(f'Mean length: {df[\"sequence\"].str.len().mean():.1f}')
   print(f'Mean reward: {df[\"reward\"].mean():.4f}')
   "
   ```

3. Compute basic statistics:
   ```bash
   python scripts/evaluate.py \
     --gflownet_samples samples/gflownet_final.csv \
     --output outputs/phase3_eval.json
   ```

**Output**: `samples/gflownet_final.csv` with 10,000 peptides

---

## 4. Technical Specifications

### 4.1 Architecture

The training loop follows this flow:

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  for step in range(n_steps):                           │
│    1. Sample batch of trajectories from P_F            │
│       - Start with [START] token                       │
│       - Sample next AA until STOP or max_length        │
│       - Collect log P_F for each action                │
│                                                         │
│    2. Compute rewards for terminal sequences           │
│       - R(x) = ImprovedReward(sequence)               │
│       - Apply temperature: R^β                         │
│                                                         │
│    3. Compute Trajectory Balance loss                  │
│       - L = (log Z + Σlog P_F - log R - Σlog P_B)²    │
│                                                         │
│    4. Backpropagate and update                         │
│       - Policy params: LR = 3e-4                       │
│       - log_Z: LR = 3e-3 (10x)                        │
│       - Gradient clipping: max_norm = 1.0              │
│                                                         │
│    5. Log metrics to W&B                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Code Structure

**Files to Use/Modify**:

| File | Purpose | Status |
|------|---------|--------|
| `scripts/train_gflownet.py` | Main training script | ✅ Exists |
| `scripts/sample.py` | Peptide generation | ✅ Exists |
| `scripts/evaluate.py` | Basic evaluation | ✅ Exists |
| `configs/default.yaml` | Default config | ✅ Exists |
| `configs/best.yaml` | Best config (from sweep) | ⏳ Create |
| `configs/sweep.yaml` | W&B sweep config | ⏳ Create |

**New Outputs**:

| Directory | Contents |
|-----------|----------|
| `checkpoints/gflownet/baseline/` | Baseline model |
| `checkpoints/gflownet/sweep_*/` | Sweep runs |
| `checkpoints/gflownet/final/` | Final model |
| `samples/` | Generated peptides |
| `outputs/` | Evaluation results |

### 4.3 Configuration

**Default Training Config** (`configs/default.yaml` excerpt):

```yaml
training:
  learning_rate: 3.0e-4
  weight_decay: 0.01
  lr_scheduler: "cosine"
  warmup_steps: 1000
  log_z_lr_multiplier: 10.0
  batch_size: 64
  n_steps: 100000
  max_grad_norm: 1.0
  loss_type: "trajectory_balance"

generation:
  min_length: 10
  max_length: 30
  exploration_eps: 0.001
```

**Environment Variables**:

```bash
# Required (add to ~/.zshrc)
export WANDB_API_KEY="your-api-key"
export HF_TOKEN="your-hf-token"  # For ESM-2 if needed

# Optional
export CUDA_VISIBLE_DEVICES=0
```

### 4.4 Reward Model Selection

**For Phase 3 Training**:

Use `ImprovedReward` (not CompositeReward) as specified in the Master PRD:

```python
from gflownet_peptide.rewards.improved_reward import ImprovedReward

reward_model = ImprovedReward(
    esm_model="esm2_t12_35M_UR50D",  # Faster model for training
    entropy_threshold=2.5,
    min_length=10,
    device="cuda"
)
```

**Rationale**: Using the same reward for both GFlowNet and GRPO-D ensures fair comparison in Phase 4.

---

## 5. Success Criteria

| ID | Criterion | Target | Measurement Method | Verification Command |
|----|-----------|--------|-------------------|---------------------|
| SC1 | Training loss | Converged | Loss curve plateau | `wandb: train/loss slope < 0.01` |
| SC2 | log_Z stability | No divergence | log_Z bounded | `abs(log_z) < 50` |
| SC3 | Sample validity | ≥99% valid AA | Validation check | `python scripts/validate_peptides.py` |
| SC4 | Sample diversity | > random init | Diversity metric | `eval/diversity > 0.5` |
| SC5 | Sample quality | Mean R > 0.5 | Reward evaluation | `eval/mean_reward > 0.5` |
| SC6 | Generated samples | 10,000 peptides | File count | `wc -l samples/gflownet_final.csv` |

---

## 6. Deliverables Checklist

- [x] Sanity check training (100 steps) completed ✅
- [ ] Baseline training (10K steps) completed
- [ ] W&B dashboard configured with key metrics
- [ ] Hyperparameter sweep completed (≥9 configurations)
- [ ] Best hyperparameters documented in `configs/best.yaml`
- [ ] Training instabilities addressed (if any)
- [ ] Final model trained to convergence
- [ ] 10,000 peptides generated and saved
- [ ] All success criteria verified
- [ ] Phase gate review completed

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|------------|--------|------------|-------------|
| Training divergence | Medium | High | Gradient clipping, LR reduction | Restart with lower LR |
| Mode collapse | Medium | High | Exploration epsilon, higher temp | Increase exploration_eps |
| Slow convergence | Medium | Medium | Larger batch, higher LR | More training steps |
| GPU OOM | Low | Medium | Reduce batch size, gradient accum | Use gradient checkpointing |
| log_Z instability | Low | High | Lower log_Z LR multiplier | Fix at 5x instead of 10x |
| Poor sample quality | Low | High | Check reward model | Debug reward function |

---

## 8. Phase Gate Review

### 8.1 Go/No-Go Criteria

| Criterion | Required | Status |
|-----------|----------|--------|
| Training converged without divergence | Yes | [ ] |
| log_Z stable (not diverging) | Yes | [ ] |
| ≥99% valid sequences | Yes | [ ] |
| Mean reward > 0.5 | Yes | [ ] |
| Diversity > random baseline | Yes | [ ] |
| 10,000 samples generated | Yes | [ ] |

### 8.2 Review Checklist

- [ ] All deliverables completed
- [ ] All success criteria met
- [ ] Training logs reviewed for anomalies
- [ ] Best hyperparameters documented
- [ ] Sample quality spot-checked manually
- [ ] Checkpoint verified loadable

### 8.3 Decision

**Status**: Pending
**Decision Date**: ___________
**Notes**: ___________

---

## 9. Implementation

### 9.1 Format Selection

Phase 3 uses **Python scripts** as the primary format due to:
- Long-running training (hours)
- Need for checkpointing and resume
- W&B integration
- Hyperparameter sweeps

Notebooks are used for analysis only.

### 9.2 Implementation Files

**Scripts** (primary):

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/train_gflownet.py` | Main training | ✅ Exists |
| `scripts/sample.py` | Peptide generation | ✅ Exists |
| `scripts/evaluate.py` | Basic evaluation | ✅ Exists |

**Configs**:

| Config | Purpose | Status |
|--------|---------|--------|
| `configs/default.yaml` | Default hyperparameters | ✅ Exists |
| `configs/best.yaml` | Best hyperparameters | ⏳ Create after sweep |
| `configs/sweep.yaml` | W&B sweep definition | ⏳ Create |

**Notebooks** (analysis only):

| Notebook | Purpose | Status |
|----------|---------|--------|
| `notebooks/phase3-training-analysis.ipynb` | Analyze training curves | ⏳ Create |

### 9.3 Quick Start Commands

```bash
# 1. Sanity check (2-5 min) - interactive
python scripts/train_gflownet.py --n_steps 100 --wandb

# 2. Baseline training (30-60 min) - interactive
python scripts/train_gflownet.py --n_steps 10000 --wandb

# 3. Full training (4-8 hours) - use nohup + auto-shutdown
nohup python scripts/train_gflownet.py \
  --config configs/default.yaml \
  --n_steps 100000 \
  --output_dir checkpoints/gflownet/final/ \
  --wandb \
  > logs/train_gflownet_full.log 2>&1 && /home/ubuntu/bin/stopinstance &

# 4. Generate samples
python scripts/sample.py \
  --checkpoint checkpoints/gflownet/gflownet_final.pt \
  --n_samples 10000 \
  --output samples/gflownet_final.csv

# 5. Quick evaluation
python scripts/evaluate.py --gflownet_samples samples/gflownet_final.csv
```

### 9.4 Long-Running Training Commands

For training runs longer than 30 minutes, use `nohup` with auto-shutdown to save costs:

```bash
# Create logs directory
mkdir -p logs

# Baseline training (10K steps, ~30-60 min)
nohup python scripts/train_gflownet.py \
  --config configs/default.yaml \
  --n_steps 10000 \
  --output_dir checkpoints/gflownet/baseline/ \
  --wandb \
  > logs/train_baseline.log 2>&1 && /home/ubuntu/bin/stopinstance &

# Full training (100K steps, ~4-8 hours)
nohup python scripts/train_gflownet.py \
  --config configs/default.yaml \
  --n_steps 100000 \
  --output_dir checkpoints/gflownet/final/ \
  --wandb \
  > logs/train_final.log 2>&1 && /home/ubuntu/bin/stopinstance &

# Monitor progress (from another terminal or after reconnect)
tail -f logs/train_final.log

# Check if still running
ps aux | grep train_gflownet
```

**Notes**:
- `nohup` keeps the process running after SSH disconnect
- `&& /home/ubuntu/bin/stopinstance` shuts down the instance after training completes
- Logs are saved to `logs/` directory
- W&B syncs progress in real-time, viewable at https://wandb.ai/ewijaya/gflownet-peptide
- Checkpoints are saved periodically (every 5000 steps by default)

### 9.5 W&B Configuration

```yaml
# In configs/default.yaml
logging:
  use_wandb: true
  wandb_project: "gflownet-peptide"
  wandb_entity: "ewijaya"
```

---

## 10. Estimated Timeline

| Day | Activity | Duration |
|-----|----------|----------|
| 1 | Sanity check + baseline training | 2-4 hours |
| 2-3 | Hyperparameter sweep (9 configs) | 1-2 days |
| 4 | Analyze sweep, create best.yaml | 2-4 hours |
| 5-7 | Full training with best config | 1-3 days |
| 8 | Generate samples, basic evaluation | 4-8 hours |
| 9-10 | Buffer for issues/refinement | 1-2 days |

**Total**: ~2 weeks (as specified in Master PRD)

---

## 11. Notes & References

- Master PRD: `docs/gflownet-master-prd.md` Section 5.3
- Phase 2 PRD: `docs/prd-phase-2-gflownet-core.md`
- Default config: `configs/default.yaml`
- Malkin et al. 2022: Trajectory Balance paper (log_Z learning rate)
- Bengio et al. 2021: GFlowNet paper (exploration)
