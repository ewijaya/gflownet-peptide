---
description: Generate a timestamped TODO file summarizing the session
argument-hint: "[optional notes to include]"
---

# Session TODO Generator

Generate a phase-aware TODO markdown file for the current project, summarizing the session's work, next steps, and observations.

## Instructions

### Step 1: Detect Project Context

Run these commands to gather context:

! basename $(git rev-parse --show-toplevel 2>/dev/null) || basename $(pwd)
! git remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//' || echo ""
! ls -1 doc/prd-phase-*.md 2>/dev/null | head -5
! ls -1t doc/TODO-*.md 2>/dev/null | head -1
! grep -r "wandb_entity\|wandb_project" configs/*.yaml 2>/dev/null | head -2

### Step 2: Check Running Processes

! ps aux | grep -E "train.*\.py|python.*train" | grep -v grep | head -5
! ls -lt logs/*.log 2>/dev/null | head -3

### Step 3: Generate Timestamp

Generate the filename using JST timezone:
- Format: `doc/TODO-YYYY-MM-DD-HHMM.md`
- Use current date/time in JST (Asia/Tokyo)

### Step 4: Analyze Session Context

From the current conversation, identify:
1. **Files created or modified** this session
2. **Key decisions** made
3. **Observations and insights** (problems, solutions, learnings)
4. **Next steps** discussed or implied
5. **Running processes** (training jobs, background tasks)

### Step 5: Generate TODO Content

Create the TODO file with this structure:

```markdown
# {Project Name} - Session TODO

**Created**: {YYYY-MM-DD HH:MM} JST
**Last Updated**: {YYYY-MM-DD HH:MM} JST
**Previous TODO**: [{previous_todo_filename}](./{previous_todo_filename}) (if exists)

---

## Current Phase: {Phase N - Name}

{Detected from doc/prd-phase-*.md files. Include brief phase objective.}

---

## In Progress

{List any running processes with monitoring commands}

- [ ] **{Task Name}** (started {timestamp})
  - Monitor: {W&B URL if applicable}
  - Logs: `tail -f logs/{logfile}.log`

---

## Completed This Session

{Bullet list of what was accomplished}

- {Task 1}
- {Task 2}

---

## Next Steps

{Checklist of next actions}

- [ ] {Next step 1}
- [ ] {Next step 2}

---

## Observations

{Key insights, warnings, or notes from the session}

- {Observation 1}
- {Observation 2}

---

## Custom Notes

{Include if $ARGUMENTS was provided, otherwise omit this section}

---

## Files Modified

| File | Action |
|------|--------|
| `{filepath}` | {Created/Modified/Deleted} |
```

### Step 6: Write and Report

1. Write the content to `doc/TODO-{YYYY-MM-DD}-{HHMM}.md`
2. Report to user:
   - File path created
   - Summary of key sections

**Custom notes from user**: $ARGUMENTS
