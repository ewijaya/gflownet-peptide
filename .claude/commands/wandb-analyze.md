---
description: Analyze W&B training runs and interpret results
argument-hint: "[N | ID...] - number of latest runs OR specific run IDs"
---

# W&B Run Analysis

Use maximum ultrathink. Take all the time you need. It's much better to do too much research and thinking than not enough.

Analyze W&B training runs. Pass a number N for latest N runs, or one or more run IDs.

**Argument**: $ARGUMENTS

## Step 1: Check if Training is Running

```bash
! ps aux | grep -E "train.*\.py" | grep -v grep | head -3
```

## Step 2: Fetch and Analyze Runs

```bash
! python3 -c "
import wandb
import sys

arg = '$ARGUMENTS'.strip()
api = wandb.Api()

if not arg:
    # Default: get latest run
    runs = api.runs('gflownet-peptide')[:1]
elif arg.isdigit():
    # Single integer N: get latest N runs
    n = int(arg)
    runs = api.runs('gflownet-peptide')[:n]
else:
    # One or more run IDs
    run_ids = arg.split()
    runs = [api.run(f'gflownet-peptide/{rid}') for rid in run_ids]

if not runs:
    print('No runs found in gflownet-peptide')
    sys.exit(0)

print(f'Analyzing {len(runs)} latest run(s):\n')

for i, run in enumerate(runs):
    print(f'## Run {i+1}: {run.name}')
    print(f'- **ID**: {run.id}')
    print(f'- **State**: {run.state}')
    print(f'- **Created**: {run.created_at}')
    print(f'- **URL**: https://wandb.ai/ewijaya/gflownet-peptide/runs/{run.id}')

    # Config
    config = run.config
    if config:
        print(f'- **Config**:')
        for key in ['reward_type', 'n_steps', 'esm_model', 'seed']:
            if key in config:
                print(f'  - {key}: {config[key]}')

    # Summary metrics
    summary = run.summary
    if summary:
        print(f'- **Metrics**:')
        for key in ['loss', 'mean_reward', 'max_reward', 'diversity', 'log_z']:
            if key in summary:
                print(f'  - {key}: {summary[key]:.4f}')

    print()

# Comparison table if N > 1
if len(runs) > 1:
    print('## Comparison Table')
    print('| Run | State | Diversity | Mean Reward | Loss |')
    print('|-----|-------|-----------|-------------|------|')
    for run in runs:
        s = run.summary
        div = f\"{s.get('diversity', 'N/A'):.4f}\" if 'diversity' in s else 'N/A'
        mr = f\"{s.get('mean_reward', 'N/A'):.4f}\" if 'mean_reward' in s else 'N/A'
        loss = f\"{s.get('loss', 'N/A'):.4f}\" if 'loss' in s else 'N/A'
        print(f'| {run.name[:30]} | {run.state} | {div} | {mr} | {loss} |')
"
```

## Step 3: Interpret Results

Based on the output above, provide a summary with:

### 1. Run Status
- **Status**: Running / Completed / Failed
- **Progress**: (if running, from metrics or logs)

### 2. Health Assessment

Evaluate against success criteria:

| Indicator | Good | Warning | Bad |
|-----------|------|---------|-----|
| Diversity | >0.5 | 0.4-0.5 | <0.4 |
| Mean Reward | 0.5-0.8 | 0.8-0.9 | >0.95 (hacking) |
| Loss | Decreasing | Flat | Increasing/NaN |

### 3. Recommendations
- Any signs of reward hacking?
- Training stability issues?
- Suggested next steps?
