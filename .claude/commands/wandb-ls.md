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
print('| ID       | Name                                         | State    | Created    |')
print('|----------|----------------------------------------------|----------|------------|')
for run in runs:
    date = run.created_at[:10]
    print(f'| {run.id:<8} | {run.name:<44} | {run.state:<8} | {date} |')
"
```

## Step 2: Summary

Provide a brief summary:
- Total number of runs
- How many are running vs finished vs failed
- Any runs that might need attention (crashed, long-running, etc.)
