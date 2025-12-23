# Data Directory

This directory contains all datasets for the GFlowNet peptide generation project.

## Directory Structure

```
data/
├── README.md           # This file
├── flip/
│   ├── stability/      # FLIP stability task data (meltome)
│   │   └── stability.csv
│   └── gb1/            # FLIP GB1 binding task data (optional)
│       └── gb1.csv
├── propedia/           # Propedia protein-peptide binding data
│   └── propedia.csv
└── processed/          # Preprocessed/cached data
```

## Dataset Descriptions

### FLIP Stability (Meltome)

- **Source**: [FLIP Benchmark](https://github.com/J-SNACKKB/FLIP)
- **Download URL**: https://github.com/J-SNACKKB/FLIP/raw/main/splits/meltome/splits.zip
- **Size**: ~28,000 protein sequences (median length 413 AA)
- **Format**: CSV with columns `sequence`, `target` (melting temperature in °C), `set`, `validation`
- **Use**: Training stability reward model
- **Note**: This dataset contains full proteins, not peptides. For peptide generation,
  the reward model trained on this data will be applied to peptide-length sequences.

### FLIP GB1 (Optional)

- **Source**: [FLIP Benchmark](https://github.com/J-SNACKKB/FLIP)
- **Download URL**: http://data.bioembeddings.com/public/FLIP/gb1/splits/one_vs_rest.csv
- **Size**: ~150,000 sequence variants
- **Format**: CSV with columns `sequence`, `target` (binding fitness)
- **Use**: Alternative/supplementary binding task

### Propedia/PepBDB

- **Source**: [PepBDB Database](http://huanglab.phys.hust.edu.cn/pepbdb/)
- **Version**: 2020-03-18
- **Size**: ~13,300 peptide-protein complexes (~4,800 unique peptides 10-50 AA)
- **Format**: CSV with columns `pdb_id`, `sequence`, `length`
- **Use**: Training binding reward model (peptides are verified binders)
- **Note**: Since all peptides are structural binders, we use binary labels (1.0).
  For binding affinity regression, consider using FLIP GB1 instead.

## Data Loading

Use the provided data loaders:

```python
from gflownet_peptide.data import load_flip_stability, load_propedia

# Load FLIP stability data
sequences, labels = load_flip_stability('data/flip/stability/')

# Load Propedia binding data
sequences, labels = load_propedia('data/propedia/')
```

## Validation

Run the validation script to verify all data:

```bash
python scripts/validate_data.py --verbose
```

## Storage Requirements

| Component | Size |
|-----------|------|
| FLIP datasets (raw) | ~500 MB |
| Propedia dataset (raw) | ~200 MB |
| Processed/cached data | ~1 GB |
| **Total** | ~2 GB |
