---
description: Check W&B training status and interpret results
---

# W&B Training Status

Check the latest W&B training run status and provide an interpreted summary.

## Step 1: Check if Training is Running

```bash
! ps aux | grep -E "train.*\.py" | grep -v grep | head -3
```

## Step 2: Check W&B Run Exists

```bash
! ls -la wandb/latest-run 2>/dev/null || echo "No W&B runs found in wandb/"
```

## Step 3: Read Run Metadata

```bash
! cat wandb/latest-run/files/wandb-metadata.json 2>/dev/null
```

## Step 4: Get Latest Training Logs

```bash
! tail -80 wandb/latest-run/files/output.log 2>/dev/null
```

## Step 5: Interpret Results

Based on the output above, provide a summary with:

### 1. Run Status
- **Status**: Running / Completed / Failed / Not Found
- **Run Name**: (from metadata or log)
- **Started**: (from metadata startedAt field)
- **Config**: (key params like reward_type, total_iterations)

### 2. Progress
- **Iteration**: X / total
- **Progress**: X%
- **ETA**: (from tqdm progress bar, format: HH:MM:SS)
- **Speed**: X.XX s/it

### 3. Latest Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Loss | X.XXXX | decreasing | |
| Mean Reward | X.XXXX | 0.5-0.8 | |
| Max Reward | X.XXXX | <1.0 | |
| Diversity | X.XXXX | >0.4 | |

### 4. Top Peptides (Latest)
List the 5 most recent top peptides with their rewards.

### 5. Health Assessment

Evaluate against Phase 0b success criteria:

- **Diversity**: Is it above 0.4 target?
- **Reward Hacking**: Any signs? (repetitive patterns like QQQQ, NNNN, or repeating motifs)
- **Top Peptides Quality**: Do they look like real proteins? (varied AA composition, no obvious repeats)
- **Training Stability**: Is loss decreasing? Any NaN or spikes?

### 6. Links
- **W&B Dashboard**: https://wandb.ai/ewijaya/gflownet-peptide/runs/{run_id}
- **Local Logs**: `tail -f logs/train_grpo_improved.log`

## Health Indicators

| Indicator | Good | Warning | Bad |
|-----------|------|---------|-----|
| Diversity | >0.5 | 0.4-0.5 | <0.4 |
| Mean Reward | 0.5-0.8 | 0.8-0.9 | >0.95 (hacking) |
| Top Peptides | Varied AAs | Some repeats | Homopolymers |
| Loss | Decreasing | Flat | Increasing/NaN |
