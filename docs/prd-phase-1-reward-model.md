# Phase 1: Reward Model Development - Detailed PRD

**Generated from**: docs/gflownet-master-prd.md Section 5.1
**Date**: 2025-12-24
**Status**: Ready for Implementation
**Last Updated**: 2025-12-24

---

## 0. Phase 0 Decision Context

**Phase 0 Outcome**: Both GRPO-D Vanilla (Phase 0a) and GRPO-D Improved (Phase 0b) establish **two baselines** for GFlowNet comparison

### Two GRPO-D Versions

| Version | Reward Function | Key Results |
|---------|-----------------|-------------|
| **GRPO-D Vanilla (0a)** | ESM-2 pseudo-likelihood only | 97% homopolymers, diversity 0.336 |
| **GRPO-D Improved (0b)** | Entropy gate + embedding naturalness | 10.9% homopolymers, diversity 0.945 |

### Three-Way Comparison Strategy

| Comparison | What It Demonstrates |
|------------|---------------------|
| **GFlowNet vs GRPO-D Vanilla** | GFlowNet robustness to reward hacking |
| **GFlowNet vs GRPO-D Improved** | GFlowNet diversity advantage even with optimized reward |
| **GRPO-D Improved vs Vanilla** | Importance of reward design |

**Remaining Characteristics to Quantify** (baseline metrics for GFlowNet comparison):
1. **AA bias**: W (3.9x), M (2.7x), H (2.2x), Y (2.1x) over-represented vs natural proteins
2. **Motif over-representation**: QRP (143x), TLG, PYP appear far more than expected
3. **Dipeptide repeats**: 45% of sequences have patterns like ICIC, FYFY

**Phase 1 Dual Purpose**:
1. Train reward model on FLIP/Propedia for improved peptide quality scoring
2. **Establish GRPO-D baseline metrics** for BOTH versions via biological validation (ESMFold, AA composition)

**Project Direction**: **GO for GFlowNet**
- GFlowNet implementation will proceed (Phase 2)
- Both Phase 0a AND Phase 0b results establish baselines for three-way comparison
- This enables the primary publication goal: *"First empirical comparison of GFlowNet vs GRPO for therapeutic peptide generation"*

---

## 1. Executive Summary

- **Objective**: Train a public reward model on FLIP/Propedia datasets AND establish comprehensive baseline metrics for GRPO-D generated peptides. This reward model will be used for both GRPO-D and GFlowNet, enabling fair comparison. The biological validation establishes the benchmark that GFlowNet must improve upon.

- **Duration**: 2 weeks

- **Key Deliverables**:
  - ESM-2 based stability predictor trained on FLIP dataset
  - ESM-2 based binding predictor trained on Propedia dataset
  - Composite reward function combining stability, binding, and naturalness
  - Validation metrics showing R² ≥ 0.5 on held-out test sets
  - Integration with existing `ImprovedReward` entropy gate from Phase 0b
  - **GRPO-D baseline report** for Phase 0b peptides (ESMFold pLDDT, AA composition, structural metrics)
  - **Baseline metrics document** defining targets for GFlowNet to beat

- **Prerequisites**:
  - Phase 0 completed (GRPO-D baseline established)
  - Phase -1 data infrastructure ready
  - FLIP and Propedia datasets downloaded and validated
  - Phase 0b peptides available for baseline analysis

---

## 2. Objectives & Scope

### 2.1 In-Scope Goals

1. **Download and preprocess FLIP Stability dataset**
   - Filter to peptide lengths 10-50 AA
   - Validate sequence characters (canonical 20 AA only)
   - Split into train/val/test (80/10/10)

2. **Download and preprocess Propedia binding dataset**
   - Extract peptide sequences and binding affinities
   - Filter to relevant peptide length range
   - Handle missing/invalid data

3. **Train ESM-2 → Stability predictor**
   - Use ESM-2 t6_8M or t12_35M as backbone (frozen or fine-tuned)
   - MLP head for regression
   - Target: R² ≥ 0.6 on FLIP test set

4. **Train ESM-2 → Binding predictor**
   - Same architecture as stability predictor
   - Target: R² ≥ 0.5 on Propedia test set

5. **Implement composite reward function**
   - Combine stability, binding, and naturalness scores
   - Multiplicative combination with configurable weights
   - Non-negative output via exp/softplus transform

6. **Integrate with entropy gate**
   - Combine trained predictors with Phase 0b entropy gate
   - Ensure repetitive sequences still receive low rewards

### 2.2 Out-of-Scope (Deferred)

| Item | Deferred To | Rationale |
|------|-------------|-----------|
| ~~Structure-based rewards (pLDDT)~~ | **Now in scope** | Needed for biological validation |
| Proprietary PEM integration | Post-publication | Publication requires public data only |
| Multi-task learning | Future work | Keep initial implementation simple |
| Hyperparameter optimization for reward | Phase 3 | Focus on baseline functionality first |

### 2.3 NEW: GRPO-D Baseline Characterization (Both Versions)

To enable fair three-way GFlowNet comparison, this phase establishes comprehensive baselines for BOTH GRPO-D versions:

**Phase 0a (Vanilla) Baseline**:
1. Use existing metrics from `outputs/grpo_metrics.json`
2. Document homopolymer dominance and low diversity as "worst case" baseline

**Phase 0b (Improved) Baseline**:
1. **Run ESMFold on Phase 0b peptides** to get pLDDT scores (structural plausibility)
2. **Compare AA composition** to known therapeutic peptides (biological relevance)
3. **Analyze motif frequency** vs natural peptide databases (novelty vs artificiality)
4. **Document baseline metrics** that GFlowNet must improve upon in Phase 4

**Three-Way Comparison Table** (to populate):

| Metric | GRPO-D Vanilla | GRPO-D Improved | GFlowNet | Winner |
|--------|----------------|-----------------|----------|--------|
| Homopolymer rate | 97% | 10.9% | TBD | TBD |
| Sequence entropy | ~0.30 | 0.83 | TBD | TBD |
| Embedding diversity | 0.336 | 0.093 | TBD | TBD |
| Cluster count | 3 | 15 | TBD | TBD |
| Mean pLDDT | TBD | TBD | TBD | TBD |
| AA KL divergence | TBD | TBD | TBD | TBD |

### 2.4 Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| FLIP Stability dataset | benchmark.protein.properties | To download |
| Propedia dataset | bioinfo.dcc.ufmg.br/propedia | To download |
| ESM-2 model | fair-esm package | Available |
| ESMFold | fair-esm package | Available |
| ImprovedReward (entropy gate) | Phase 0b | ✅ Complete |
| Phase 0b peptides | `results/grpo/20251224_*_peptides.csv` | ✅ Available |
| Data loading infrastructure | Phase -1 | Partially complete |

---

## 3. Detailed Activities

### Activity 1.1: Download FLIP Stability Dataset

**Description**: Download the FLIP benchmark stability task dataset containing ~53K protein sequences with stability measurements.

**Steps**:
1. Navigate to https://benchmark.protein.properties/
2. Download stability task CSV file
3. Save to `data/flip/stability/`
4. Verify file integrity

**Implementation**:
```bash
# Create directory
mkdir -p data/flip/stability

# Download FLIP stability dataset
# Option 1: Direct download (if available)
wget -O data/flip/stability/stability.csv \
  "https://benchmark.protein.properties/downloads/stability.csv"

# Option 2: Using FLIP package
pip install flip-benchmark
python -c "
from flip.benchmark import load_dataset
data = load_dataset('stability')
data.to_csv('data/flip/stability/stability.csv', index=False)
"
```

**Verification**:
```bash
# Check file exists and has expected size
wc -l data/flip/stability/stability.csv
# Expected: ~53,000 lines

# Check column names
head -1 data/flip/stability/stability.csv
# Expected: sequence,fitness (or similar)
```

**Output**: `data/flip/stability/stability.csv`

---

### Activity 1.2: Download Propedia Dataset

**Description**: Download Propedia peptide-protein interaction database containing binding affinity data.

**Steps**:
1. Navigate to http://bioinfo.dcc.ufmg.br/propedia/
2. Download peptide binding data
3. Save to `data/propedia/`
4. Extract peptide sequences and binding affinities

**Implementation**:
```bash
# Create directory
mkdir -p data/propedia

# Download Propedia (check current URL)
wget -O data/propedia/propedia_raw.csv \
  "http://bioinfo.dcc.ufmg.br/propedia/download/propedia_v2.csv"
```

**Verification**:
```bash
wc -l data/propedia/propedia_raw.csv
# Expected: ~19,000 lines

head -1 data/propedia/propedia_raw.csv
```

**Output**: `data/propedia/propedia_raw.csv`

---

### Activity 1.3: Preprocess Datasets

**Description**: Filter, validate, and split datasets for model training.

**Steps**:
1. Filter sequences to length 10-50 AA
2. Validate all characters are canonical amino acids
3. Remove duplicates
4. Normalize labels (zero-mean, unit-variance)
5. Split into train/val/test (80/10/10)

**Implementation**:
```python
# scripts/preprocess_data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')

def validate_sequence(seq):
    """Check if sequence contains only valid amino acids."""
    return all(aa in VALID_AAS for aa in seq.upper())

def preprocess_flip_stability(input_path, output_dir):
    """Preprocess FLIP stability dataset."""
    df = pd.read_csv(input_path)

    # Rename columns if needed
    if 'sequence' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'sequence', df.columns[1]: 'fitness'})

    # Filter by length
    df = df[df['sequence'].str.len().between(10, 50)]

    # Validate sequences
    df = df[df['sequence'].apply(validate_sequence)]

    # Remove duplicates
    df = df.drop_duplicates(subset='sequence')

    # Normalize labels
    mean_fitness = df['fitness'].mean()
    std_fitness = df['fitness'].std()
    df['fitness_normalized'] = (df['fitness'] - mean_fitness) / std_fitness

    # Split
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Save
    train.to_csv(f'{output_dir}/train.csv', index=False)
    val.to_csv(f'{output_dir}/val.csv', index=False)
    test.to_csv(f'{output_dir}/test.csv', index=False)

    # Save normalization params
    np.save(f'{output_dir}/norm_params.npy', [mean_fitness, std_fitness])

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

if __name__ == '__main__':
    preprocess_flip_stability(
        'data/flip/stability/stability.csv',
        'data/flip/stability'
    )
```

**Verification**:
```bash
python scripts/preprocess_data.py
# Expected output: Train: ~42K, Val: ~5K, Test: ~5K

# Verify splits
wc -l data/flip/stability/train.csv data/flip/stability/val.csv data/flip/stability/test.csv
```

**Output**: `data/flip/stability/{train,val,test}.csv`, `data/flip/stability/norm_params.npy`

---

### Activity 1.4: Train ESM-2 → Stability Predictor

**Description**: Train a stability predictor using ESM-2 embeddings as input.

**Architecture**:
```
Input: Peptide sequence
    ↓
ESM-2 (frozen): Extract mean-pooled embeddings
    ↓
MLP: 320 → 256 → 128 → 1
    ↓
Output: Stability score
```

**Implementation**:
```python
# gflownet_peptide/rewards/stability_predictor.py

import torch
import torch.nn as nn
import esm

class StabilityPredictor(nn.Module):
    """ESM-2 based stability predictor."""

    def __init__(
        self,
        esm_model: str = "esm2_t6_8M_UR50D",
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        freeze_esm: bool = True,
    ):
        super().__init__()

        # Load ESM-2
        if esm_model == "esm2_t6_8M_UR50D":
            self.esm, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.embed_dim = 320
            self.repr_layer = 6
        elif esm_model == "esm2_t12_35M_UR50D":
            self.esm, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.embed_dim = 480
            self.repr_layer = 12
        else:
            raise ValueError(f"Unknown ESM model: {esm_model}")

        self.batch_converter = self.alphabet.get_batch_converter()

        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False

        # MLP head
        layers = []
        in_dim = self.embed_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.head = nn.Sequential(*layers)

    def get_embeddings(self, sequences: list[str]) -> torch.Tensor:
        """Extract ESM-2 embeddings for sequences."""
        data = [(f"seq{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(next(self.esm.parameters()).device)

        with torch.no_grad():
            results = self.esm(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False
            )

        # Mean pool (excluding BOS/EOS)
        embeddings = []
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            emb = results["representations"][self.repr_layer][i, 1:seq_len+1, :]
            embeddings.append(emb.mean(dim=0))

        return torch.stack(embeddings)

    def forward(self, sequences: list[str]) -> torch.Tensor:
        """Predict stability for sequences."""
        embeddings = self.get_embeddings(sequences)
        return self.head(embeddings).squeeze(-1)
```

**Training Script**:
```python
# scripts/train_stability.py

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from gflownet_peptide.rewards.stability_predictor import StabilityPredictor
import wandb

class StabilityDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.sequences = df['sequence'].tolist()
        self.labels = torch.tensor(df['fitness_normalized'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return list(sequences), torch.stack(labels)

def train_stability_predictor(config):
    wandb.init(project="gflownet-peptide", name="stability_predictor")

    # Data
    train_ds = StabilityDataset('data/flip/stability/train.csv')
    val_ds = StabilityDataset('data/flip/stability/val.csv')

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'],
                            collate_fn=collate_fn)

    # Model
    model = StabilityPredictor(
        esm_model=config['esm_model'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
    ).cuda()

    optimizer = torch.optim.Adam(model.head.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()

    best_val_r2 = -float('inf')

    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            labels = labels.cuda()

            optimizer.zero_grad()
            preds = model(sequences)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for sequences, labels in val_loader:
                preds = model(sequences)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.tolist())

        # Compute R²
        val_preds = torch.tensor(val_preds)
        val_labels = torch.tensor(val_labels)
        ss_res = ((val_labels - val_preds) ** 2).sum()
        ss_tot = ((val_labels - val_labels.mean()) ** 2).sum()
        val_r2 = 1 - ss_res / ss_tot

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'val_r2': val_r2.item(),
        })

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), 'checkpoints/stability_predictor_best.pt')
            print(f"Epoch {epoch}: New best R² = {val_r2:.4f}")

    return best_val_r2

if __name__ == '__main__':
    config = {
        'esm_model': 'esm2_t6_8M_UR50D',
        'hidden_dims': [256, 128],
        'dropout': 0.1,
        'lr': 1e-3,
        'batch_size': 32,
        'epochs': 50,
    }
    train_stability_predictor(config)
```

**Verification**:
```bash
# Train model
python scripts/train_stability.py

# Check final R²
# Target: R² ≥ 0.6
```

**Output**: `checkpoints/stability_predictor_best.pt`

---

### Activity 1.5: Train ESM-2 → Binding Predictor

**Description**: Train a binding predictor using ESM-2 embeddings on Propedia data.

**Implementation**: Same architecture as stability predictor, different dataset.

```python
# scripts/train_binding.py
# Similar to train_stability.py but using Propedia data
```

**Verification**:
```bash
python scripts/train_binding.py
# Target: R² ≥ 0.5
```

**Output**: `checkpoints/binding_predictor_best.pt`

---

### Activity 1.6: Implement Composite Reward

**Description**: Combine stability, binding, and naturalness into a single reward function.

**Implementation**:
```python
# gflownet_peptide/rewards/composite_reward.py

import torch
import torch.nn as nn
from typing import List, Union
from .stability_predictor import StabilityPredictor
from .binding_predictor import BindingPredictor
from .improved_reward import ImprovedReward

class CompositeReward(nn.Module):
    """
    Composite reward combining:
    - Stability (FLIP-trained)
    - Binding (Propedia-trained)
    - Naturalness (ESM-2 embedding norm)
    - Entropy gate (from Phase 0b)
    """

    def __init__(
        self,
        stability_checkpoint: str = None,
        binding_checkpoint: str = None,
        weights: dict = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Default weights
        self.weights = weights or {
            'stability': 1.0,
            'binding': 1.0,
            'naturalness': 0.5,
        }

        # Load predictors
        self.stability = StabilityPredictor().to(device)
        if stability_checkpoint:
            self.stability.load_state_dict(torch.load(stability_checkpoint))
        self.stability.eval()

        self.binding = BindingPredictor().to(device)
        if binding_checkpoint:
            self.binding.load_state_dict(torch.load(binding_checkpoint))
        self.binding.eval()

        # Naturalness + entropy gate from Phase 0b
        self.naturalness = ImprovedReward(device=device)

    def forward(
        self,
        sequences: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """Compute composite reward."""
        single_input = isinstance(sequences, str)
        if single_input:
            sequences = [sequences]

        with torch.no_grad():
            # Get individual scores
            stability_scores = torch.sigmoid(self.stability(sequences))
            binding_scores = torch.sigmoid(self.binding(sequences))
            naturalness_scores = torch.tensor(
                [self.naturalness(seq) for seq in sequences],
                device=self.device
            )

        # Combine multiplicatively with weights
        rewards = (
            stability_scores ** self.weights['stability'] *
            binding_scores ** self.weights['binding'] *
            naturalness_scores ** self.weights['naturalness']
        )

        rewards = rewards.cpu().tolist()
        return rewards[0] if single_input else rewards

    def get_components(self, sequence: str) -> dict:
        """Get individual reward components for debugging."""
        with torch.no_grad():
            stability = torch.sigmoid(self.stability([sequence])).item()
            binding = torch.sigmoid(self.binding([sequence])).item()
            naturalness = self.naturalness(sequence)

        return {
            'stability': stability,
            'binding': binding,
            'naturalness': naturalness,
            'total': self([sequence]),
        }
```

**Verification**:
```python
# Test composite reward
from gflownet_peptide.rewards import CompositeReward

reward = CompositeReward(
    stability_checkpoint='checkpoints/stability_predictor_best.pt',
    binding_checkpoint='checkpoints/binding_predictor_best.pt',
)

# Test on example sequences
good_peptide = "MKTLLILAVVALACARSSAQAANPF"
bad_peptide = "QQQQQQQQQQQQQQQQQQQQQQQQ"

print(f"Good peptide: {reward(good_peptide)}")
print(f"Bad peptide: {reward(bad_peptide)}")
print(f"Components: {reward.get_components(good_peptide)}")

assert reward(good_peptide) > reward(bad_peptide), "Good peptide should score higher"
```

**Output**: `gflownet_peptide/rewards/composite_reward.py`

---

### Activity 1.7: Biological Validation of Phase 0b Peptides

**Description**: Validate that Phase 0b generated peptides are biologically plausible using structure prediction and comparison to known peptides.

**This activity addresses the Phase 0 remaining concerns:**
- AA bias (W, M, H, Y over-represented)
- Motif over-representation (QRP, TLG, PYP)
- Dipeptide repeats (45% rate)

**Steps**:

1. **Run ESMFold on Phase 0b peptides**
```python
# scripts/validate_peptides_structure.py

import torch
import esm
import pandas as pd

def predict_structure(sequences):
    """Run ESMFold to get pLDDT scores."""
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    results = []
    for seq in sequences:
        with torch.no_grad():
            output = model.infer_pdb(seq)
            # Extract pLDDT from output
            plddt = extract_plddt(output)
            results.append({
                'sequence': seq,
                'mean_plddt': plddt.mean(),
                'min_plddt': plddt.min(),
            })
    return pd.DataFrame(results)

# Load Phase 0b peptides
peptides = pd.read_csv('results/grpo/20251224_093311_grpod_it1000_dw0.15_beta0.04_peptides.csv')
structure_results = predict_structure(peptides['peptide'].tolist())
```

2. **Compare AA composition to known therapeutic peptides**
```python
# Compare to APD3 antimicrobial peptide database frequencies
apd3_frequencies = {
    'L': 0.089, 'G': 0.087, 'K': 0.082, 'A': 0.078, 'I': 0.061,
    'C': 0.054, 'V': 0.052, 'R': 0.049, 'S': 0.047, 'F': 0.044,
    # ... etc
}

# Compute KL divergence between generated and natural
kl_div = compute_kl_divergence(generated_aa_freq, apd3_frequencies)
```

3. **Check if over-represented motifs appear in natural peptides**
```python
# Search UniProt/APD3 for QRP, TLG, PYP motifs
# If these motifs are common in natural peptides, the bias is acceptable
```

**Validation Criteria**:

| Criterion | Target | Interpretation |
|-----------|--------|----------------|
| Mean pLDDT | >70 | Peptides likely fold into stable structures |
| pLDDT >50 rate | >80% | Most peptides are structurally plausible |
| AA KL divergence | <0.5 | AA distribution similar to natural peptides |
| Motifs in UniProt | >10% match | Over-represented motifs are biologically relevant |

**Baseline Interpretation** (for GFlowNet comparison in Phase 4):

| GRPO-D Baseline Result | Interpretation | GFlowNet Target |
|------------------------|----------------|-----------------|
| Mean pLDDT = X | Structural quality baseline | GFlowNet should achieve ≥X |
| AA KL divergence = Y | Naturalness baseline | GFlowNet should achieve ≤Y |
| Motif artificiality = Z% | Pattern quality baseline | GFlowNet should achieve ≤Z% |

**Note**: These metrics establish the baseline. GFlowNet proceeds regardless of GRPO-D quality - the goal is to demonstrate improvement, not to gate GFlowNet.

**Output**:
- `outputs/grpod_baseline_metrics.json` - Quantitative baseline for Phase 4 comparison
- `outputs/grpod_structure_report.md` - Detailed structural analysis

---

### Activity 1.8: Validate Reward Model on Test Set

**Description**: Final validation of all reward components on held-out test data.

**Implementation**:
```python
# scripts/validate_reward_model.py

import torch
import pandas as pd
from sklearn.metrics import r2_score
from gflownet_peptide.rewards import StabilityPredictor, BindingPredictor, CompositeReward

def validate_predictor(model, test_csv, label_col='fitness_normalized'):
    """Compute R² on test set."""
    df = pd.read_csv(test_csv)
    sequences = df['sequence'].tolist()
    labels = df[label_col].values

    model.eval()
    with torch.no_grad():
        preds = model(sequences).cpu().numpy()

    r2 = r2_score(labels, preds)
    return r2

def main():
    print("=== Reward Model Validation ===\n")

    # Stability
    stability = StabilityPredictor().cuda()
    stability.load_state_dict(torch.load('checkpoints/stability_predictor_best.pt'))
    r2_stability = validate_predictor(stability, 'data/flip/stability/test.csv')
    print(f"Stability R²: {r2_stability:.4f} (target: ≥0.6)")

    # Binding
    binding = BindingPredictor().cuda()
    binding.load_state_dict(torch.load('checkpoints/binding_predictor_best.pt'))
    r2_binding = validate_predictor(binding, 'data/propedia/test.csv', 'binding_affinity')
    print(f"Binding R²: {r2_binding:.4f} (target: ≥0.5)")

    # Composite reward sanity check
    reward = CompositeReward(
        stability_checkpoint='checkpoints/stability_predictor_best.pt',
        binding_checkpoint='checkpoints/binding_predictor_best.pt',
    )

    test_seqs = [
        "MKTLLILAVVALACARSSAQAANPF",  # Real peptide
        "GIGKFLHSAKKFGKAFVGEIMNS",     # Antimicrobial
        "QQQQQQQQQQQQQQQQQQQQQQQQ",    # Homopolymer
        "ACDEFGHIKLMNPQRSTVWY",        # All different AAs
    ]

    print("\nComposite reward sanity check:")
    for seq in test_seqs:
        r = reward(seq)
        print(f"  {seq[:20]}... : {r:.4f}")

    # Check non-negativity
    rewards = [reward(seq) for seq in test_seqs]
    assert all(r >= 0 for r in rewards), "All rewards must be non-negative"
    print("\n✓ All rewards are non-negative")

    # Check spread
    print(f"Reward range: {min(rewards):.4f} - {max(rewards):.4f}")
    print(f"Reward std: {torch.tensor(rewards).std():.4f} (target: >0.1)")

if __name__ == '__main__':
    main()
```

**Verification**:
```bash
python scripts/validate_reward_model.py
```

**Output**: Validation report showing R² scores and sanity checks

---

### Activity 1.8: Implement Non-Negative Transform

**Description**: Ensure all reward outputs are non-negative via appropriate transforms.

**Implementation**: Already included in composite reward via sigmoid for predictors.

Alternative transforms if needed:
```python
# Option 1: Softplus (smooth, always positive)
reward = torch.nn.functional.softplus(raw_score)

# Option 2: Exponential (ensures positivity, can be large)
reward = torch.exp(raw_score / temperature)

# Option 3: Sigmoid (bounded [0, 1])
reward = torch.sigmoid(raw_score)
```

---

## 4. Technical Specifications

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPOSITE REWARD MODEL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Peptide sequence x                                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     ESM-2 Backbone                       │   │
│  │                  (Frozen, shared)                        │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
│            ┌────────────┼────────────┐                         │
│            ▼            ▼            ▼                         │
│      ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│      │ Stability │ │ Binding  │ │Naturalness│                   │
│      │   Head    │ │   Head   │ │ + Entropy │                   │
│      │   MLP     │ │   MLP    │ │   Gate    │                   │
│      └─────┬─────┘ └─────┬────┘ └─────┬─────┘                   │
│            │             │            │                         │
│            ▼             ▼            ▼                         │
│         S(x)^w₁      B(x)^w₂      N(x)^w₃                      │
│            │             │            │                         │
│            └─────────────┼────────────┘                         │
│                          ▼                                      │
│                    R(x) = ∏ᵢ Cᵢ(x)^wᵢ                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Code Structure

**New Files to Create:**
```
gflownet_peptide/
├── rewards/
│   ├── __init__.py              # Update exports
│   ├── stability_predictor.py   # NEW
│   ├── binding_predictor.py     # NEW
│   ├── composite_reward.py      # NEW
│   ├── improved_reward.py       # From Phase 0b
│   └── esm2_reward.py           # From Phase 0a

scripts/
├── preprocess_data.py           # NEW
├── train_stability.py           # NEW
├── train_binding.py             # NEW
├── validate_reward_model.py     # NEW

configs/
├── reward_model.yaml            # NEW
```

### 4.3 Configuration

**`configs/reward_model.yaml`**:
```yaml
# ESM-2 backbone
esm_model: "esm2_t6_8M_UR50D"
freeze_esm: true

# MLP heads
hidden_dims: [256, 128]
dropout: 0.1

# Training
learning_rate: 1.0e-3
batch_size: 32
epochs: 50
early_stopping_patience: 10

# Composite weights
weights:
  stability: 1.0
  binding: 1.0
  naturalness: 0.5

# Data paths
flip_stability_path: "data/flip/stability"
propedia_path: "data/propedia"

# Checkpoints
checkpoint_dir: "checkpoints/reward_models"
```

### 4.4 Data Format Specifications

**FLIP Stability (preprocessed)**:
| Column | Type | Description |
|--------|------|-------------|
| sequence | string | Amino acid sequence (10-50 AA) |
| fitness | float | Raw stability score |
| fitness_normalized | float | Z-score normalized |

**Propedia (preprocessed)**:
| Column | Type | Description |
|--------|------|-------------|
| peptide_sequence | string | Peptide sequence |
| binding_affinity | float | Binding strength (pKd) |
| binding_normalized | float | Z-score normalized |

---

## 5. Success Criteria

### 5.1 Reward Model Criteria

| ID | Criterion | Target | Measurement Method | Verification Command |
|----|-----------|--------|-------------------|---------------------|
| SC1.1 | FLIP data downloaded | ≥50K sequences | Line count | `wc -l data/flip/stability/stability.csv` |
| SC1.2 | Propedia data downloaded | ≥15K sequences | Line count | `wc -l data/propedia/propedia_raw.csv` |
| SC1.3 | Data preprocessing complete | Train/val/test splits exist | File existence | `ls data/flip/stability/*.csv` |
| SC1.4 | Stability predictor R² | ≥0.6 | sklearn r2_score | `python scripts/validate_reward_model.py` |
| SC1.5 | Binding predictor R² | ≥0.5 | sklearn r2_score | `python scripts/validate_reward_model.py` |
| SC1.6 | Reward non-negativity | 100% ≥ 0 | min(R) check | Validation script |
| SC1.7 | Reward spread | std(R) > 0.1 | Standard deviation | Validation script |
| SC1.8 | Inference speed | <100ms/seq | Timing benchmark | Timing test |
| SC1.9 | Entropy gate integration | Homopolymers score <0.1 | Direct test | `reward("QQQQQQQQQQQQQQ")` |

### 5.2 GRPO-D Baseline Metrics (Phase 0b Peptides)

These metrics establish the baseline that GFlowNet must improve upon in Phase 4.

| ID | Metric | Measurement | Purpose |
|----|--------|-------------|---------|
| SC1.10 | Mean pLDDT | ESMFold | Structural quality baseline |
| SC1.11 | pLDDT >50 rate | ESMFold | Structural plausibility rate |
| SC1.12 | AA KL divergence | vs APD3/UniProt | Naturalness baseline |
| SC1.13 | Motif artificiality | Database search | Pattern quality baseline |
| SC1.14 | Diversity (from Phase 0b) | 0.5447 | Diversity baseline for GFlowNet to beat |
| SC1.15 | Sequence entropy | 0.83 | Complexity baseline |

### 5.3 GFlowNet Comparison Targets (Three-Way for Phase 4)

GFlowNet success will be measured against BOTH GRPO-D baselines:

**Comparison A: GFlowNet vs GRPO-D Vanilla (Robustness)**

| Metric | GRPO-D Vanilla | GFlowNet Target | Success Criterion |
|--------|----------------|-----------------|-------------------|
| Homopolymer rate | 97% | <20% | Robust to reward hacking |
| Sequence entropy | ~0.30 | >0.6 | No mode collapse |
| Embedding diversity | 0.336 | >0.336 | Better mode coverage |

**Comparison B: GFlowNet vs GRPO-D Improved (Diversity)**

| Metric | GRPO-D Improved | GFlowNet Target | Success Criterion |
|--------|-----------------|-----------------|-------------------|
| Sequence diversity | 0.945 | ≥0.945 | At least as diverse |
| Cluster count | 15 (K-means) | ≥45 | ≥3× more modes |
| Mean pLDDT | TBD (Phase 1) | ≥ GRPO-D | No quality loss |
| AA KL divergence | TBD (Phase 1) | ≤ GRPO-D | No naturalness loss |
| Proportionality R² | N/A | ≥0.8 | GFlowNet-specific |

**Publication Claims**:
1. "GFlowNet is robust to reward hacking" (vs Vanilla)
2. "GFlowNet achieves Nx more modes at equivalent quality" (vs Improved)

**Note**: GFlowNet proceeds to Phase 2 regardless of GRPO-D baseline quality. The baselines inform comparison, not gating.

---

## 6. Deliverables Checklist

### Data
- [ ] FLIP Stability dataset downloaded
- [ ] Propedia dataset downloaded
- [ ] Preprocessing scripts complete
- [ ] Train/val/test splits created

### Models
- [ ] Stability predictor implemented
- [ ] Stability predictor trained (R² ≥ 0.6)
- [ ] Binding predictor implemented
- [ ] Binding predictor trained (R² ≥ 0.5)
- [ ] Composite reward implemented
- [ ] Entropy gate integrated

### Reward Validation
- [ ] Test set validation complete
- [ ] Non-negativity verified
- [ ] Spread verified (std > 0.1)
- [ ] Inference speed verified (<100ms)

### GRPO-D Baseline Characterization (Both Versions)

**Phase 0a (Vanilla) Baseline**:
- [ ] Existing metrics documented from `outputs/grpo_metrics.json`
- [ ] Homopolymer samples preserved for comparison

**Phase 0b (Improved) Baseline**:
- [ ] ESMFold structure prediction complete
- [ ] pLDDT scores computed for all 128 peptides
- [ ] AA composition compared to APD3/UniProt
- [ ] Motif frequency analyzed vs natural databases
- [ ] Baseline metrics document generated (`outputs/grpod_baseline_metrics.json`)

**Three-Way Comparison Table**:
- [ ] All metrics populated for both GRPO-D versions
- [ ] Ready for GFlowNet results in Phase 4

### Phase 2 Preparation
- [ ] GRPO-D baseline documented for GFlowNet comparison
- [ ] Reward model ready for GFlowNet training
- [ ] Success criteria defined for GFlowNet vs GRPO-D comparison

### Documentation
- [ ] Code documented with docstrings
- [ ] Config file created
- [ ] README updated with reward model usage

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|------------|--------|------------|-------------|
| FLIP download fails | Low | High | Try multiple download methods | Use alternative stability dataset |
| Propedia format changed | Medium | Medium | Check documentation | Parse manually |
| Stability R² < 0.6 | Medium | High | Tune hyperparameters; try larger ESM | Use simpler proxy (embedding norm) |
| Binding R² < 0.5 | High | Medium | Expected for peptide binding | Weight binding lower in composite |
| Overfitting | Medium | Medium | Use dropout, early stopping | Increase regularization |
| Inference too slow | Low | Low | Use smaller ESM model | Batch inference |
| **ESMFold shows low pLDDT** | Medium | High | May indicate reward hacking | GO for GFlowNet |
| **AA bias is biologically invalid** | Medium | Medium | Compare to multiple databases | Adjust reward entropy threshold |
| **Motifs are artificial** | Medium | Medium | Check if motifs appear in UniProt | May need motif penalty in reward |

---

## 8. Phase Gate Review

### 8.1 Reward Model Criteria

| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| Stability predictor R² ≥ 0.6 | Yes | TBD | ⏳ |
| Binding predictor R² ≥ 0.5 | Yes | TBD | ⏳ |
| Reward non-negative | Yes | TBD | ⏳ |
| Inference speed <100ms | Yes | TBD | ⏳ |

### 8.2 GRPO-D Baseline Metrics (for Phase 4 Comparison)

| Metric | Value | Status | Purpose |
|--------|-------|--------|---------|
| Mean pLDDT | TBD | ⏳ | Structural quality baseline |
| pLDDT >50 rate | TBD | ⏳ | Structural plausibility baseline |
| AA KL divergence | TBD | ⏳ | Naturalness baseline |
| Diversity | 0.5447 | ✅ Known | From Phase 0b |
| Sequence entropy | 0.83 | ✅ Known | From Phase 0b |

### 8.3 Review Checklist
- [ ] All reward model deliverables completed
- [ ] All reward model success criteria met
- [ ] GRPO-D baseline characterization complete
- [ ] Baseline metrics documented for Phase 4
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Tests passing

### 8.4 Phase 1 Exit → Phase 2 Entry
**Reward Model Status**: Pending
**Baseline Characterization Status**: Pending
**Phase 2 (GFlowNet) Status**: **GO** (confirmed)
**Decision Date**: ___________
**Notes**: GFlowNet implementation proceeds. Phase 1 baseline metrics will inform Phase 4 comparison.

---

## 9. Implementation Code

### 9.1 Phase 1 Uses Script-Primary Format

This phase primarily involves **training**, which is long-running and benefits from scripts rather than notebooks.

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/preprocess_data.py` | Data preprocessing | [ ] Not started |
| `scripts/train_stability.py` | Train stability predictor | [ ] Not started |
| `scripts/train_binding.py` | Train binding predictor | [ ] Not started |
| `scripts/validate_reward_model.py` | Final reward validation | [ ] Not started |
| `scripts/validate_peptides_structure.py` | **ESMFold biological validation** | [ ] Not started |

| Notebook (analysis) | Purpose | Status |
|---------------------|---------|--------|
| `notebooks/gflownet-phase-1-reward-model.ipynb` | Reward model analysis | [ ] Not started |
| `notebooks/gflownet-phase-1-biological-validation.ipynb` | **Phase 0b peptide validation** | [ ] Not started |

### 9.2 Module Structure

| Module | Purpose | Status |
|--------|---------|--------|
| `gflownet_peptide/rewards/stability_predictor.py` | Stability predictor class | [ ] Not started |
| `gflownet_peptide/rewards/binding_predictor.py` | Binding predictor class | [ ] Not started |
| `gflownet_peptide/rewards/composite_reward.py` | Composite reward class | [ ] Not started |

### 9.3 Required Environment Variables

```bash
export WANDB_API_KEY="your-wandb-key"
export HF_TOKEN="your-huggingface-token"  # For ESM-2 if needed
```

### 9.4 W&B Configuration

```yaml
wandb_project: "gflownet-peptide"
wandb_entity: "ewijaya"
```

---

## 10. Notes & References

### Internal Documents
- Master PRD: `docs/gflownet-master-prd.md`
- Phase 0b PRD: `docs/prd-phase-0b-improved-reward.md`
- Reward formulation: `docs/reward_formulation.md`
- **Phase 0 decision: `docs/phase0_decision.md`** (CONDITIONAL NO-GO)
- Phase 0b TODO: `docs/TODO-2025-12-24-2333.md`

### External References
- [FLIP Benchmark](https://benchmark.protein.properties/)
- [Propedia Database](http://bioinfo.dcc.ufmg.br/propedia/)
- [ESM-2 Repository](https://github.com/facebookresearch/esm)
- [APD3 Antimicrobial Peptide Database](https://aps.unmc.edu/)

### Key Considerations from Phase 0

1. **Entropy gate is critical**: Without it, ESM-2 based rewards encourage repetitive sequences
2. **Multiplicative combination**: All components must be good for high reward
3. **Sigmoid outputs**: Keep individual scores in [0, 1] range before combining

### GRPO-D Baseline Characteristics (To Quantify in Phase 1)

From `docs/phase0_decision.md`, these characteristics define the GRPO-D baseline:

1. **AA bias**: W (3.9x), M (2.7x), H (2.2x), Y (2.1x) over-represented
   - Quantify via comparison to APD3/UniProt frequencies
   - Establishes naturalness baseline for GFlowNet to match or beat

2. **Motif over-representation**: QRP (143x), TLG, PYP, ICI
   - Search UniProt for these motifs
   - Establishes pattern quality baseline

3. **Dipeptide repeats**: 45% have ICIC, FYFY, etc.
   - Validate via ESMFold pLDDT scores
   - Establishes structural quality baseline

### Project Flow

```
Phase 1 Complete
       │
       ▼
┌─────────────────────────────────────┐
│   GRPO-D Baseline Established       │
│  (pLDDT, AA composition, motifs)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Phase 2: GFlowNet Implementation  │
│   (Uses same reward model)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Phase 3: GFlowNet Training        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Phase 4: Three-Way Comparison     │
│   GFlowNet vs Vanilla vs Improved   │
└──────────────┬──────────────────────┘
               │
               ▼
       Publication:
       - "GFlowNet robust to reward hacking (vs Vanilla)"
       - "GFlowNet achieves Nx modes (vs Improved)"
```

---

*End of Phase 1 PRD*
