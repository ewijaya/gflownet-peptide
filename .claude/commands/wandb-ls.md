---
description: List all W&B runs in the gflownet-peptide project
---

# List W&B Runs

List all runs in the gflownet-peptide project using the W&B API.

## Step 1: Fetch and Display Runs

```bash
python3 -c "
import wandb
api = wandb.Api()
runs = api.runs('gflownet-peptide')
print(f'Found {len(runs)} runs in gflownet-peptide:\n')
print(f'{\"ID\":<10} {\"Name\":<45} {\"State\":<10} {\"Created\"}')
print('-' * 90)
for run in runs:
    print(f'{run.id:<10} {run.name:<45} {run.state:<10} {run.created_at}')
"
```

## Step 2: Summary

Provide a brief summary:
- Total number of runs
- How many are running vs finished vs failed
- Any runs that might need attention (crashed, long-running, etc.)
