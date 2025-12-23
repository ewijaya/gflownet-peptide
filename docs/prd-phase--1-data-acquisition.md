# Phase -1: Data Acquisition & Infrastructure - Detailed PRD

**Generated from**: docs/gflownet-master-prd.md Section 5.-1
**Date**: 2025-12-23
**Status**: Draft

---

## 1. Executive Summary

- **Objective**: Download, validate, and prepare all datasets and infrastructure required for the GFlowNet peptide generation project. This foundational phase ensures all data dependencies are resolved and data loading pipelines are functional before any model development begins.
- **Duration**: 1 week (prerequisite phase)
- **Key Deliverables**:
  - FLIP Stability dataset downloaded and validated
  - Propedia binding dataset downloaded and validated
  - ESM-2 model verified and functional
  - Data loader modules implemented and tested
  - Data validation script operational
- **Prerequisites**: None (this is the first phase)

---

## 2. Objectives & Scope

### 2.1 In-Scope Goals
- Download FLIP Stability dataset (≥50K sequences) for stability reward model training
- Download Propedia dataset (≥15K sequences) for binding reward model training
- Optionally download FLIP GB1 dataset for alternative binding task
- Create standardized directory structure for data storage
- Implement Python data loaders for each dataset
- Validate all sequences contain only canonical amino acids
- Verify ESM-2 pretrained models load correctly
- Create comprehensive data validation script
- Write unit tests for data loading pipeline

### 2.2 Out-of-Scope (Deferred)
- Training any models (Phase 1+)
- Data augmentation strategies (Phase 1)
- ProteinGym benchmark download (optional, Phase 4)
- UniRef50 download for P_F pretraining (optional, Phase 2)
- Proprietary dataset integration (future)

### 2.3 Dependencies
| Dependency | Source | Required By |
|------------|--------|-------------|
| Internet access | Network | All downloads |
| Python 3.9+ | System | Data loaders |
| pip/conda | System | Package installation |
| ~5 GB disk space | Storage | Dataset storage |
| `fair-esm` package | PyPI | ESM-2 verification |
| `pandas`, `numpy` | PyPI | Data processing |
| `pytest` | PyPI | Testing |

---

## 3. Detailed Activities

### Activity -1.1: Set Up Data Directory Structure

**Description**: Create a standardized directory tree for storing raw and processed datasets. This ensures consistent paths across all scripts and phases.

**Steps**:
1. Create the main data directory structure
2. Add .gitkeep files to preserve empty directories in git
3. Create a README documenting the expected contents

**Implementation Notes**:
- Use lowercase, hyphen-separated names for consistency
- Keep raw and processed data separate for reproducibility
- Add the data directory to .gitignore (except .gitkeep files)

**Verification**:
```bash
# Verify directory structure exists
ls -la data/
ls -la data/flip/
ls -la data/propedia/
ls -la data/processed/
```

**Output**: Directory tree structure

**Expected Structure**:
```
data/
├── README.md           # Documents data sources and formats
├── flip/
│   ├── stability/      # FLIP stability task data
│   └── gb1/            # FLIP GB1 task data (optional)
├── propedia/           # Propedia binding data
└── processed/          # Preprocessed/cached data
```

---

### Activity -1.2: Download FLIP Stability Dataset

**Description**: Download the FLIP benchmark stability dataset, which contains ~53K protein sequences with experimentally measured stability values (ΔΔG scores).

**Steps**:
1. Navigate to the FLIP benchmark website
2. Download the stability task CSV file
3. Verify file integrity and expected row count
4. Inspect data format and column names

**Implementation Notes**:
- The FLIP benchmark is hosted at https://benchmark.protein.properties/
- Primary download method: direct wget from the benchmark website
- Alternative: use the `flip-benchmark` Python package
- Expected columns: `sequence`, `fitness` (or similar)
- Data may need column renaming for consistency

**Verification**:
```bash
# Check file exists and has expected size
ls -lh data/flip/stability/
wc -l data/flip/stability/*.csv

# Preview first few rows
head -n 5 data/flip/stability/stability.csv
```

**Output**: `data/flip/stability/stability.csv` with ≥50K sequences

**Download Commands**:
```bash
# Option 1: Direct download from FLIP data server (preferred)
mkdir -p data/flip/stability
wget http://data.bioembeddings.com/public/FLIP/meltome/splits/mixed_split.csv \
    -O data/flip/stability/stability.csv

# Option 2: Clone the FLIP repository and use provided splits
git clone https://github.com/J-SNACKKB/FLIP.git /tmp/flip
cp /tmp/flip/splits/meltome/* data/flip/stability/

# Note: The FLIP benchmark website (benchmark.protein.properties) redirects
# to a Google Sites page that may require authentication. Use the direct
# data server URLs above instead.
```

**Expected Format**:
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| sequence | string | Amino acid sequence (uppercase) | `MKFLILFLPFAS` |
| fitness | float | ΔΔG stability score | `-1.23` |

---

### Activity -1.3: Download FLIP GB1 Dataset (Optional)

**Description**: Download the FLIP GB1 binding fitness dataset as an alternative/supplementary binding task. Contains ~150K sequence variants.

**Steps**:
1. Download GB1 task CSV from FLIP benchmark
2. Verify file integrity
3. Document format differences from stability task

**Implementation Notes**:
- GB1 is a protein binding domain with fitness measured by binding assay
- Larger dataset (150K) but simpler binding task
- Can serve as validation or alternative to Propedia

**Verification**:
```bash
ls -lh data/flip/gb1/
wc -l data/flip/gb1/*.csv
```

**Output**: `data/flip/gb1/gb1.csv` (optional)

**Download Commands**:
```bash
mkdir -p data/flip/gb1
wget http://data.bioembeddings.com/public/FLIP/gb1/splits/one_vs_rest.csv \
    -O data/flip/gb1/gb1.csv

# Alternative: all GB1 splits
wget -r -np -nH --cut-dirs=4 http://data.bioembeddings.com/public/FLIP/gb1/splits/ \
    -P data/flip/gb1/
```

---

### Activity -1.4: Download Propedia Binding Dataset

**Description**: Download the Propedia protein-peptide interaction database containing ~19K peptide-protein binding complexes with affinity measurements.

**Steps**:
1. Access Propedia download portal
2. Download the peptide binding dataset
3. Extract relevant columns (peptide sequence, binding affinity)
4. Verify data format and sequence validity

**Implementation Notes**:
- Propedia is hosted at http://bioinfo.dcc.ufmg.br/propedia/
- May require navigating the website for download link
- Data may be in JSON or CSV format
- Filter for entries with experimentally measured binding affinity

**Verification**:
```bash
ls -lh data/propedia/
wc -l data/propedia/*.csv

# Check for required columns
head -n 1 data/propedia/propedia.csv
```

**Output**: `data/propedia/propedia.csv` with ≥15K sequences

**Download Commands**:
```bash
mkdir -p data/propedia

# Navigate to the Propedia website and download manually:
# https://bioinfo.dcc.ufmg.br/propedia/
#
# Note: Propedia v2.3 (2023) contains ~49,300 peptide-protein complexes.
# The download may require navigating through the website interface.
# Look for "Download" or "Data" section on the website.
#
# Expected download: CSV or JSON file with peptide sequences and binding data
```

**Expected Format**:
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| peptide_sequence | string | Peptide amino acid sequence | `RKKRRQRRR` |
| binding_affinity | float | Binding strength (pKd or similar) | `7.5` |
| target_protein | string | Target protein ID (optional) | `P12345` |

---

### Activity -1.5: Verify ESM-2 Model Download

**Description**: Ensure ESM-2 pretrained models can be downloaded and loaded via the `fair-esm` package. ESM-2 provides protein embeddings for the reward model.

**Steps**:
1. Install the `fair-esm` package
2. Attempt to load ESM-2 model (auto-downloads on first use)
3. Run a forward pass on a test sequence
4. Verify output shape and embedding extraction

**Implementation Notes**:
- ESM-2 models auto-download to `~/.cache/torch/hub/checkpoints/`
- `esm2_t12_35M_UR50D` is the smallest model (~500MB) - good for testing
- `esm2_t33_650M_UR50D` is recommended for production (~2.5GB)
- Forward pass should return token-level representations

**Verification**:
```bash
# Test ESM-2 loading and forward pass
python -c "
import esm
import torch

# Load model (will download if not cached)
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Test sequence
data = [('test', 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL')]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Forward pass
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12])
    embeddings = results['representations'][12]

print(f'Embedding shape: {embeddings.shape}')
print('ESM-2 verification: PASSED')
"
```

**Output**: ESM-2 model loads successfully and produces embeddings

---

### Activity -1.6: Implement FLIP Data Loader

**Description**: Create a Python module for loading FLIP datasets with proper preprocessing, filtering, and train/val/test splitting.

**Steps**:
1. Create `gflownet_peptide/data/flip.py`
2. Implement `load_flip_stability()` function
3. Implement `load_flip_gb1()` function (optional)
4. Add length filtering (10-50 AA)
5. Add sequence validation (canonical AA only)
6. Add train/val/test splitting (80/10/10)

**Implementation Notes**:
- Use pandas for CSV reading
- Validate sequences contain only `ACDEFGHIKLMNPQRSTVWY`
- Remove sequences with non-standard amino acids (X, U, B, Z, etc.)
- Normalize fitness values (zero-mean, unit-variance)
- Return tuple of (sequences: List[str], labels: np.ndarray)

**Verification**:
```bash
# Test the data loader
python -c "
from gflownet_peptide.data import load_flip_stability
sequences, labels = load_flip_stability('data/flip/stability/')
print(f'Loaded {len(sequences)} sequences')
print(f'Label range: [{labels.min():.2f}, {labels.max():.2f}]')
print(f'Sample sequence: {sequences[0][:30]}...')
"
```

**Output**: Working `gflownet_peptide/data/flip.py` module

**Code Template**:
```python
"""FLIP benchmark data loading utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

CANONICAL_AA = set('ACDEFGHIKLMNPQRSTVWY')


def validate_sequence(seq: str) -> bool:
    """Check if sequence contains only canonical amino acids."""
    return all(aa in CANONICAL_AA for aa in seq.upper())


def load_flip_stability(
    data_path: str,
    min_length: int = 10,
    max_length: int = 50,
    normalize: bool = True,
    split: Optional[str] = None,
    seed: int = 42
) -> Tuple[List[str], np.ndarray]:
    """
    Load FLIP stability task dataset.

    Args:
        data_path: Path to FLIP stability directory or CSV file
        min_length: Minimum sequence length to include
        max_length: Maximum sequence length to include
        normalize: Whether to normalize labels to zero-mean, unit-variance
        split: Optional split ('train', 'val', 'test', or None for all)
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (sequences, labels) where sequences is a list of strings
        and labels is a numpy array of fitness values.
    """
    # Implementation here
    pass


def load_flip_gb1(
    data_path: str,
    min_length: int = 10,
    max_length: int = 50,
    normalize: bool = True,
    split: Optional[str] = None,
    seed: int = 42
) -> Tuple[List[str], np.ndarray]:
    """Load FLIP GB1 binding task dataset."""
    # Implementation here
    pass
```

---

### Activity -1.7: Implement Propedia Data Loader

**Description**: Create a Python module for loading Propedia binding data with proper preprocessing.

**Steps**:
1. Create `gflownet_peptide/data/propedia.py`
2. Implement `load_propedia()` function
3. Handle different possible data formats (CSV/JSON)
4. Extract peptide sequences and binding affinities
5. Apply same filtering as FLIP (length, canonical AA)

**Implementation Notes**:
- Propedia may have different column names than FLIP
- Binding affinity may be pKd, IC50, or other units
- May need to convert/normalize affinity values
- Filter for peptides only (not full proteins)

**Verification**:
```bash
# Test the data loader
python -c "
from gflownet_peptide.data import load_propedia
sequences, labels = load_propedia('data/propedia/')
print(f'Loaded {len(sequences)} sequences')
print(f'Label range: [{labels.min():.2f}, {labels.max():.2f}]')
print(f'Sample sequence: {sequences[0]}')
"
```

**Output**: Working `gflownet_peptide/data/propedia.py` module

**Code Template**:
```python
"""Propedia protein-peptide binding data loading utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

from .flip import CANONICAL_AA, validate_sequence


def load_propedia(
    data_path: str,
    min_length: int = 10,
    max_length: int = 50,
    normalize: bool = True,
    split: Optional[str] = None,
    seed: int = 42
) -> Tuple[List[str], np.ndarray]:
    """
    Load Propedia protein-peptide binding dataset.

    Args:
        data_path: Path to Propedia directory or CSV file
        min_length: Minimum peptide length to include
        max_length: Maximum peptide length to include
        normalize: Whether to normalize labels
        split: Optional split ('train', 'val', 'test', or None for all)
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (sequences, labels) where sequences is a list of peptide
        strings and labels is a numpy array of binding affinities.
    """
    # Implementation here
    pass
```

---

### Activity -1.8: Create Data Validation Script

**Description**: Create a comprehensive script that validates all downloaded data, checks for integrity issues, and reports statistics.

**Steps**:
1. Create `scripts/validate_data.py`
2. Check for required files
3. Validate sequence formats
4. Report statistics (counts, length distributions, label ranges)
5. Flag any issues found

**Implementation Notes**:
- Script should be runnable standalone
- Use exit codes: 0 for success, 1 for warnings, 2 for errors
- Generate both console output and optional JSON report
- Check for common issues: empty files, wrong columns, invalid characters

**Verification**:
```bash
# Run the validation script
python scripts/validate_data.py --verbose

# Check exit code
echo "Exit code: $?"
```

**Output**: `scripts/validate_data.py` that validates all data

**Script Template**:
```python
#!/usr/bin/env python
"""Validate downloaded datasets for the GFlowNet peptide project."""

import argparse
import json
import sys
from pathlib import Path

from gflownet_peptide.data import load_flip_stability, load_propedia


def validate_flip_stability(data_path: str) -> dict:
    """Validate FLIP stability dataset."""
    results = {'name': 'FLIP Stability', 'status': 'unknown', 'issues': []}
    # Implementation here
    return results


def validate_propedia(data_path: str) -> dict:
    """Validate Propedia dataset."""
    results = {'name': 'Propedia', 'status': 'unknown', 'issues': []}
    # Implementation here
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate project datasets')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--json', help='Output JSON report to file')
    args = parser.parse_args()

    # Run validations
    # Implementation here


if __name__ == '__main__':
    main()
```

---

### Activity -1.9: Run Full Data Pipeline Test

**Description**: Create and run comprehensive unit tests for the entire data loading pipeline to ensure everything works end-to-end.

**Steps**:
1. Create `tests/test_data_loading.py`
2. Write tests for each data loader function
3. Test edge cases (empty sequences, boundary lengths)
4. Run full test suite with pytest
5. Ensure all tests pass

**Implementation Notes**:
- Use pytest fixtures for data paths
- Mock data for fast testing (don't require full dataset)
- Test both success and error cases
- Aim for ≥80% coverage of data loading code

**Verification**:
```bash
# Run all data loading tests
pytest tests/test_data_loading.py -v

# Run with coverage
pytest tests/test_data_loading.py -v --cov=gflownet_peptide.data
```

**Output**: Passing test suite in `tests/test_data_loading.py`

**Test Template**:
```python
"""Tests for data loading modules."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from gflownet_peptide.data import load_flip_stability, load_propedia
from gflownet_peptide.data.flip import validate_sequence, CANONICAL_AA


class TestSequenceValidation:
    def test_valid_sequence(self):
        assert validate_sequence('ACDEFGHIKLMNPQRSTVWY')

    def test_invalid_sequence_with_x(self):
        assert not validate_sequence('ACXDEF')

    def test_invalid_sequence_with_numbers(self):
        assert not validate_sequence('AC123')


class TestFlipStability:
    @pytest.fixture
    def mock_flip_data(self, tmp_path):
        """Create mock FLIP data for testing."""
        csv_content = 'sequence,fitness\n'
        csv_content += 'ACDEFGHIKLMNPQRSTVWY,1.0\n'
        csv_content += 'MKTVRQERLKSIV,-0.5\n'

        data_file = tmp_path / 'stability.csv'
        data_file.write_text(csv_content)
        return tmp_path

    def test_load_returns_correct_types(self, mock_flip_data):
        sequences, labels = load_flip_stability(str(mock_flip_data))
        assert isinstance(sequences, list)
        assert isinstance(labels, np.ndarray)

    def test_length_filtering(self, mock_flip_data):
        sequences, labels = load_flip_stability(
            str(mock_flip_data),
            min_length=15,
            max_length=25
        )
        for seq in sequences:
            assert 15 <= len(seq) <= 25


class TestPropedia:
    # Similar tests for Propedia
    pass
```

---

## 4. Technical Specifications

### 4.1 Architecture

This phase focuses on data infrastructure, not model architecture. The key architectural decisions are:

1. **Directory Layout**: Flat structure under `data/` with subdirectories per dataset
2. **Data Loading Interface**: Consistent function signatures returning `(List[str], np.ndarray)`
3. **Validation Strategy**: Eager validation at load time with clear error messages

### 4.2 Code Structure

**Files to create/modify:**

```
gflownet_peptide/
├── data/
│   ├── __init__.py          # Export public API
│   ├── flip.py              # FLIP data loaders
│   └── propedia.py          # Propedia data loader
├── tests/
│   └── test_data_loading.py # Unit tests
scripts/
└── validate_data.py         # Validation script
data/
├── README.md                # Data documentation
├── flip/
│   ├── stability/
│   └── gb1/
├── propedia/
└── processed/
```

### 4.3 Configuration

**Environment Variables (optional)**:
```bash
GFLOWNET_DATA_DIR=/path/to/data  # Override default data directory
ESM_CACHE_DIR=/path/to/cache     # Override ESM model cache location
```

**Data Paths (defaults)**:
```python
DATA_ROOT = Path('data')
FLIP_STABILITY_PATH = DATA_ROOT / 'flip' / 'stability'
FLIP_GB1_PATH = DATA_ROOT / 'flip' / 'gb1'
PROPEDIA_PATH = DATA_ROOT / 'propedia'
PROCESSED_PATH = DATA_ROOT / 'processed'
```

### 4.4 Code Examples

**Loading data for training:**
```python
from gflownet_peptide.data import load_flip_stability, load_propedia

# Load training data
train_seqs, train_labels = load_flip_stability(
    'data/flip/stability/',
    split='train',
    min_length=10,
    max_length=50,
    normalize=True
)

# Load validation data
val_seqs, val_labels = load_flip_stability(
    'data/flip/stability/',
    split='val',
    min_length=10,
    max_length=50,
    normalize=True
)

print(f"Training samples: {len(train_seqs)}")
print(f"Validation samples: {len(val_seqs)}")
```

**Verifying ESM-2:**
```python
import esm
import torch

def verify_esm2():
    """Verify ESM-2 installation and functionality."""
    # Load model
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # Test sequence
    test_seq = "MKFLILFLPFASMGKLL"
    data = [("test", test_seq)]
    _, _, batch_tokens = batch_converter(data)

    # Forward pass
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])

    embeddings = results['representations'][12]
    assert embeddings.shape == (1, len(test_seq) + 2, 480)  # +2 for BOS/EOS

    print("ESM-2 verification passed!")
    return True

if __name__ == "__main__":
    verify_esm2()
```

---

## 5. Success Criteria

| ID | Criterion | Target | Measurement Method | Verification Command |
|----|-----------|--------|-------------------|---------------------|
| SC-1 | FLIP Stability downloaded | ≥50,000 sequences | Line count minus header | `wc -l data/flip/stability/stability.csv` |
| SC-2 | Propedia downloaded | ≥15,000 sequences | Line count minus header | `wc -l data/propedia/propedia.csv` |
| SC-3 | Sequences valid | 100% canonical AA | Validation script | `python scripts/validate_data.py` |
| SC-4 | Data loaders importable | No import errors | Python import test | `python -c "from gflownet_peptide.data import *"` |
| SC-5 | ESM-2 loads successfully | Forward pass works | Python test script | `python -c "import esm; esm.pretrained.esm2_t12_35M_UR50D()"` |
| SC-6 | Unit tests pass | 100% pass rate | pytest exit code | `pytest tests/test_data_loading.py -v` |
| SC-7 | Test coverage | ≥80% coverage | pytest-cov report | `pytest tests/test_data_loading.py --cov=gflownet_peptide.data` |

---

## 6. Deliverables Checklist

- [ ] Directory structure created (`data/flip/`, `data/propedia/`, `data/processed/`)
- [ ] FLIP Stability dataset downloaded (≥50K sequences)
- [ ] Propedia dataset downloaded (≥15K sequences)
- [ ] FLIP GB1 dataset downloaded (optional)
- [ ] `gflownet_peptide/data/__init__.py` created with exports
- [ ] `gflownet_peptide/data/flip.py` implemented
- [ ] `gflownet_peptide/data/propedia.py` implemented
- [ ] `scripts/validate_data.py` created
- [ ] `tests/test_data_loading.py` created
- [ ] ESM-2 model verified (forward pass succeeds)
- [ ] `data/README.md` documenting data sources
- [ ] All unit tests passing
- [ ] Data validation script runs successfully
- [ ] All success criteria verified
- [ ] Phase gate review completed

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|------------|--------|------------|-------------|
| FLIP website unavailable | Low | High | Cache downloaded data; use alternative mirror | Use flip-benchmark package |
| Propedia download requires registration | Medium | Medium | Check website requirements early | Contact maintainers; use alternative binding data |
| Data format changed from documented | Medium | Medium | Inspect files immediately after download | Adapt data loaders to actual format |
| ESM-2 download slow/fails | Low | Medium | Use cached models if available | Use smaller model (t6_8M) for testing |
| Disk space insufficient | Low | Medium | Check space before downloads | Use smaller model; compress raw data |
| Non-canonical amino acids in data | High | Low | Filter at load time | Document excluded sequences |

---

## 8. Phase Gate Review

### 8.1 Go/No-Go Criteria

**Must have (Go requires ALL):**
1. FLIP Stability dataset has ≥50,000 valid sequences
2. Propedia dataset has ≥15,000 valid sequences
3. All data loaders import and execute without errors
4. ESM-2 forward pass succeeds
5. All unit tests pass

**Nice to have (Not required for Go):**
1. FLIP GB1 dataset downloaded
2. ≥90% test coverage
3. Automated download scripts

### 8.2 Review Checklist
- [ ] All deliverables completed
- [ ] All success criteria met (SC-1 through SC-7)
- [ ] Documentation updated (`data/README.md`)
- [ ] Code reviewed
- [ ] Tests passing
- [ ] No critical issues outstanding

### 8.3 Decision
**Status**: Pending
**Decision Date**: ___________
**Notes**: ___________

---

## 9. Implementation Notebooks

All implementation code for this phase MUST be written in Jupyter notebooks (`.ipynb`).

**Notebook naming convention**: `gflownet-phase--1-{activity-slug}.ipynb`

**Notebook location**: `notebooks/` directory

**Expected notebooks for this phase**:

| Notebook | Purpose | Status |
|----------|---------|--------|
| `gflownet-phase--1-data-acquisition.ipynb` | Download and explore datasets, implement data loaders | [ ] Not started |

**Notebook requirements** (MUST follow):
1. Organize the notebook with clear numbered sections using markdown headers
2. Include a minimal, self-sufficient description before each code cell explaining what that code block does
3. Configure all plots to display inline within the notebook AND save as external files (to `outputs/` directory)
4. Ensure the notebook is fully executable from top to bottom (no manual interventions needed)
5. Include verification cells at the end of each major section

**Suggested notebook structure**:
```markdown
# Phase -1: Data Acquisition & Infrastructure

## 1. Setup
- Import libraries
- Define paths and constants

## 2. Directory Structure
- Create data directories

## 3. FLIP Stability Dataset
### 3.1 Download
### 3.2 Explore and Validate
### 3.3 Implement Data Loader
### 3.4 Verification

## 4. Propedia Dataset
### 4.1 Download
### 4.2 Explore and Validate
### 4.3 Implement Data Loader
### 4.4 Verification

## 5. ESM-2 Verification
- Load model
- Test forward pass

## 6. Final Validation
- Run all validation checks
- Summary statistics
```

---

## 10. Notes & References

- Master PRD: docs/gflownet-master-prd.md
- **FLIP Benchmark**:
  - GitHub Repository: https://github.com/J-SNACKKB/FLIP
  - Raw Data Download: http://data.bioembeddings.com/public/FLIP/
  - FASTA Format: http://data.bioembeddings.com/public/FLIP/fasta/
  - Paper (NeurIPS 2021): https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/2b44928ae11fb9384c4cf38708677c48-Abstract-round2.html
  - Papers with Code: https://paperswithcode.com/dataset/flip
- **Propedia Database** (v2.3, 2023):
  - Website: https://bioinfo.dcc.ufmg.br/propedia/
  - Paper: https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2023.1103103/full
  - Note: Contains ~49,300 peptide-protein complexes
- **ESM-2** (archived Aug 2024, still usable):
  - GitHub Repository: https://github.com/facebookresearch/esm
  - PyPI Package: https://pypi.org/project/fair-esm/

**Amino Acid Reference**:
- 20 standard amino acids: `ACDEFGHIKLMNPQRSTVWY`
- Special tokens for GFlowNet: START (20), STOP (21), PAD (22)

**Storage Requirements Summary**:
| Component | Size |
|-----------|------|
| FLIP datasets (raw) | ~500 MB |
| Propedia dataset (raw) | ~200 MB |
| ESM-2 t12_35M model | ~500 MB |
| ESM-2 t33_650M model | ~2.5 GB |
| Processed/cached data | ~1 GB |
| **Total** | ~5 GB |
