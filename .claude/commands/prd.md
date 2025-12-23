---
description: Generate detailed phase-specific PRD from master gflownet-master-prd.md
argument-hint: "<phase_number> (-1, 0, 1, 2, 3, 4, or 5)"
---

# Phase PRD Generator

Generate a comprehensive, detailed PRD for a specific implementation phase of the GFlowNet peptide project.

## Phase Reference

| Phase | Name | Section |
|-------|------|---------|
| -1 | Data Acquisition & Infrastructure | 5.-1 |
| 0 | Validation (Is GFlowNet Needed?) | 5.0 |
| 1 | Reward Model Development | 5.1 |
| 2 | GFlowNet Core Implementation | 5.2 |
| 3 | Training & Hyperparameter Tuning | 5.3 |
| 4 | Evaluation & GRPO Comparison | 5.4 |
| 5 | Documentation & Publication | 5.5 |

## Instructions

**Input**: $ARGUMENTS (the phase number: -1, 0, 1, 2, 3, 4, or 5)

### Step 1: Validate Phase Number

Parse the phase number from arguments. Valid values are: -1, 0, 1, 2, 3, 4, 5.
If invalid, inform the user and list valid options.

### Step 2: Read Master PRD

Read the master PRD file at `docs/gflownet-master-prd.md` and extract the relevant phase section (Section 5.X where X is the phase number).

### Step 3: Check for Existing Files

Check if a file already exists with the pattern `docs/prd-phase-{N}-*.md`. If it does:
- Find the highest version number (v2, v3, etc.)
- Create a new file with the next version number

**Naming convention**:
- First file: `docs/prd-phase-{N}-{name}.md`
- Subsequent: `docs/prd-phase-{N}-{name}-v2.md`, `docs/prd-phase-{N}-{name}-v3.md`, etc.

**Phase name slugs** (used for both PRD and notebook filenames):
- Phase -1: `data-acquisition`
- Phase 0: `validation`
- Phase 1: `reward-model`
- Phase 2: `gflownet-core`
- Phase 3: `training`
- Phase 4: `evaluation`
- Phase 5: `documentation`

**Implementation notebooks**: `notebooks/gflownet-phase-{N}-{slug}.ipynb`

### Step 4: Generate Detailed Phase PRD

Create a comprehensive markdown document with the following structure. Expand significantly on the master PRD content - add implementation details, code examples, verification commands, and actionable checklists.

```markdown
# Phase {N}: {Full Name} - Detailed PRD

**Generated from**: docs/gflownet-master-prd.md Section 5.{N}
**Date**: {today's date}
**Status**: Draft

---

## 1. Executive Summary

- **Objective**: {one paragraph describing what this phase accomplishes}
- **Duration**: {from master PRD}
- **Key Deliverables**: {bullet list}
- **Prerequisites**: {what must be complete before starting}

---

## 2. Objectives & Scope

### 2.1 In-Scope Goals
{Detailed list of what this phase will accomplish}

### 2.2 Out-of-Scope (Deferred)
{What is explicitly NOT part of this phase}

### 2.3 Dependencies
| Dependency | Source | Required By |
|------------|--------|-------------|
{List dependencies from other phases, external data, etc.}

---

## 3. Detailed Activities

{For each activity in the master PRD, expand into a detailed subsection}

### Activity {N}.{X}: {Activity Name}

**Description**: {detailed description}

**Steps**:
1. {Step 1 with specific commands or actions}
2. {Step 2}
...

**Implementation Notes**:
- {Technical details}
- {Code patterns to follow}
- {Potential pitfalls}

**Verification**:
```bash
# Commands to verify this activity is complete
{verification commands}
```

**Output**: {What artifact is produced}

---

## 4. Technical Specifications

### 4.1 Architecture
{Relevant architecture details for this phase}

### 4.2 Code Structure
{Files to create/modify, directory structure}

### 4.3 Configuration
{Config files, environment variables, parameters}

### 4.4 Code Examples
{Provide starter code or templates where applicable}

---

## 5. Success Criteria

| ID | Criterion | Target | Measurement Method | Verification Command |
|----|-----------|--------|-------------------|---------------------|
{Expand each criterion from master PRD with specific verification}

---

## 6. Deliverables Checklist

- [ ] {Deliverable 1}
- [ ] {Deliverable 2}
...
- [ ] All success criteria verified
- [ ] Phase gate review completed

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|------------|--------|------------|-------------|
{Risks specific to this phase}

---

## 8. Phase Gate Review

### 8.1 Go/No-Go Criteria
{Specific criteria that must be met to proceed}

### 8.2 Review Checklist
- [ ] All deliverables completed
- [ ] All success criteria met
- [ ] Documentation updated
- [ ] Code reviewed (if applicable)
- [ ] Tests passing (if applicable)

### 8.3 Decision
**Status**: {Pending | Go | No-Go}
**Decision Date**: ___________
**Notes**: ___________

---

## 9. Implementation Code

This phase uses a **hybrid approach**: notebooks for exploration/analysis, Python scripts for training.

### 9.1 When to Use Notebooks vs Scripts

| Use Case | Format | Location |
|----------|--------|----------|
| Data exploration & validation | Notebook | `notebooks/` |
| Visualization & analysis | Notebook | `notebooks/` |
| Prototyping new components | Notebook | `notebooks/` |
| Long-running training (>10 min) | Python script | `scripts/` |
| Hyperparameter sweeps | Python script | `scripts/` |
| Production/reusable code | Python module | `gflownet_peptide/` |

### 9.2 Phase-Specific Guidance

| Phase | Primary Format | Rationale |
|-------|---------------|-----------|
| -1 Data Acquisition | Notebooks | Interactive data validation |
| 0 Validation | **Scripts** + Notebooks | GRPO-D training is long-running; notebook for analysis |
| 1 Reward Model | **Scripts** + Notebooks | Training is long-running; notebook for evaluation |
| 2 GFlowNet Core | **Modules** + Notebooks | Production code; notebook for testing |
| 3 Training | **Scripts** | Multi-hour runs, hyperparameter sweeps |
| 4 Evaluation | Notebooks | Analysis, comparisons, figures |
| 5 Documentation | Notebooks | Final figures and analysis |

### 9.3 Expected Implementation Files

{List the specific files needed based on the phase - adjust format per phase guidance above}

**For notebook-primary phases (-1, 0, 4, 5):**
| Notebook | Purpose | Status |
|----------|---------|--------|
| `notebooks/gflownet-phase-{N}-{activity}.ipynb` | {description} | [ ] Not started |

**For script-primary phases (1, 3):**
| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/{script_name}.py` | {description} | [ ] Not started |

| Notebook (analysis) | Purpose | Status |
|---------------------|---------|--------|
| `notebooks/gflownet-phase-{N}-analysis.ipynb` | Validation & visualization | [ ] Not started |

**For module-primary phases (2):**
| Module | Purpose | Status |
|--------|---------|--------|
| `gflownet_peptide/{module}/` | {description} | [ ] Not started |

| Tests | Purpose | Status |
|-------|---------|--------|
| `tests/test_{module}.py` | Unit tests | [ ] Not started |

### 9.4 Notebook Requirements (when using notebooks)

1. Organize with clear numbered sections using markdown headers
2. Include a minimal, self-sufficient description before each code cell
3. Configure all plots to display inline AND save to `outputs/` directory
4. Ensure fully executable from top to bottom (no manual interventions)
5. Include verification cells at the end of each major section

### 9.5 Script Requirements (when using scripts)

1. Use `argparse` or config files for all parameters
2. Implement checkpointing for runs >30 minutes
3. Log to wandb or similar for experiment tracking
4. Include `--dry-run` flag for testing
5. Print clear progress updates
6. Example invocation in docstring or README

**Required Environment Variables** (set in `~/.zshrc` or shell config):
- `WANDB_API_KEY`: W&B API key for experiment tracking
- `HF_TOKEN`: Hugging Face token for model downloads

**W&B Configuration** (in config YAML or CLI args):
```yaml
wandb_project: "gflownet-peptide"
wandb_entity: "ewijaya"
```

---

## 10. Notes & References

- Master PRD: docs/gflownet-master-prd.md
- {Other relevant documents}
- {External references}
```

### Step 5: Write the File

Write the generated content to the appropriate file path in `docs/`.

Report to the user:
- The file path created
- A brief summary of what was generated
- Next steps (review the file, start implementation, etc.)

$ARGUMENTS
