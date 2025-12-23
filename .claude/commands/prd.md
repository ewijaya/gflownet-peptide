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

Read the master PRD file at `doc/gflownet-master-prd.md` and extract the relevant phase section (Section 5.X where X is the phase number).

### Step 3: Check for Existing Files

Check if a file already exists with the pattern `doc/prd-phase-{N}-*.md`. If it does:
- Find the highest version number (v2, v3, etc.)
- Create a new file with the next version number

**Naming convention**:
- First file: `doc/prd-phase-{N}-{name}.md`
- Subsequent: `doc/prd-phase-{N}-{name}-v2.md`, `doc/prd-phase-{N}-{name}-v3.md`, etc.

**Phase name slugs**:
- Phase -1: `data-acquisition`
- Phase 0: `validation`
- Phase 1: `reward-model`
- Phase 2: `gflownet-core`
- Phase 3: `training`
- Phase 4: `evaluation`
- Phase 5: `documentation`

### Step 4: Generate Detailed Phase PRD

Create a comprehensive markdown document with the following structure. Expand significantly on the master PRD content - add implementation details, code examples, verification commands, and actionable checklists.

```markdown
# Phase {N}: {Full Name} - Detailed PRD

**Generated from**: doc/gflownet-master-prd.md Section 5.{N}
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

## 9. Notes & References

- Master PRD: doc/gflownet-master-prd.md
- {Other relevant documents}
- {External references}
```

### Step 5: Write the File

Write the generated content to the appropriate file path in `doc/`.

Report to the user:
- The file path created
- A brief summary of what was generated
- Next steps (review the file, start implementation, etc.)

$ARGUMENTS
