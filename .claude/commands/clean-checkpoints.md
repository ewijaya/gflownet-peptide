---
description: Remove intermediate checkpoints from completed training runs
---

# Clean Intermediate Checkpoints

Remove intermediate checkpoint files (`_iter*.pt`) from completed training runs while preserving final checkpoints and in-progress runs.

## Step 1: Scan Checkpoints

```bash
! find checkpoints -name "*.pt" -exec ls -lh {} \; 2>/dev/null | sort
```

## Step 2: Identify Completed vs In-Progress Runs

Group checkpoint files by run prefix (everything before `_iter` or `_final`).

A run is **completed** if it has a `_final.pt` file.
A run is **in-progress** if it only has `_iter*.pt` files (no `_final.pt`).

## Step 3: Analyze and Report

For each checkpoint directory, report:

### Completed Runs (safe to clean)
- Run prefix
- Final checkpoint: kept (with size)
- Intermediate checkpoints: will be deleted (with sizes)

### In-Progress Runs (skip)
- Run prefix
- Intermediate checkpoints: preserved (no _final.pt exists yet)

### Summary
- Total files to delete
- Total space to be freed

## Step 4: Delete Intermediate Checkpoints

Only delete `_iter*.pt` files from runs that have a corresponding `_final.pt` file.

```bash
# Example deletion command (generate based on analysis):
# rm checkpoints/grpo/{prefix}_iter200.pt checkpoints/grpo/{prefix}_iter400.pt ...
```

## Step 5: Verify

```bash
! find checkpoints -name "*.pt" -exec ls -lh {} \; 2>/dev/null | sort
! du -sh checkpoints/*/
```

## Safety Rules

1. **Never delete `_final.pt` files** - these are the completed training results
2. **Never delete from in-progress runs** - if no `_final.pt` exists, keep all intermediates
3. **Always show what will be deleted before deleting**
