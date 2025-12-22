#!/usr/bin/env python3
"""
Train reward model on FLIP or Propedia data.

This script trains an ESM-2 based reward model for peptide fitness prediction.

Usage:
    python scripts/train_reward.py --task stability --data_path data/flip/
    python scripts/train_reward.py --task binding --data_path data/propedia/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train reward model")

    parser.add_argument(
        "--task",
        type=str,
        default="stability",
        choices=["stability", "binding", "gb1"],
        help="Task to train on",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/reward/",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--esm_model",
        type=str,
        default="esm2_t12_35M_UR50D",
        help="ESM-2 model to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for MLP head",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


class SimpleRewardModel(nn.Module):
    """Simple reward model: ESM-2 + MLP head."""

    def __init__(
        self,
        esm_model: str = "esm2_t12_35M_UR50D",
        hidden_dim: int = 256,
        freeze_esm: bool = True,
    ):
        super().__init__()

        self.esm_model_name = esm_model
        self.freeze_esm = freeze_esm
        self.hidden_dim = hidden_dim

        # Lazy loading of ESM
        self._esm = None
        self._alphabet = None
        self._batch_converter = None

        # ESM embedding dimensions
        self._embed_dims = {
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
        }
        embed_dim = self._embed_dims.get(esm_model, 480)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def _load_esm(self):
        if self._esm is None:
            import esm
            self._esm, self._alphabet = esm.pretrained.load_model_and_alphabet(
                self.esm_model_name
            )
            self._batch_converter = self._alphabet.get_batch_converter()

            if self.freeze_esm:
                self._esm.eval()
                for param in self._esm.parameters():
                    param.requires_grad = False

    def forward(self, sequences: list[str]) -> torch.Tensor:
        self._load_esm()

        device = next(self.head.parameters()).device

        # Prepare batch
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self._batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Get ESM representations
        with torch.set_grad_enabled(not self.freeze_esm):
            # Use appropriate layer based on model
            if "t33" in self.esm_model_name:
                repr_layer = 33
            elif "t30" in self.esm_model_name:
                repr_layer = 30
            elif "t12" in self.esm_model_name:
                repr_layer = 12
            else:
                repr_layer = 6

            results = self._esm(
                batch_tokens,
                repr_layers=[repr_layer],
                return_contacts=False,
            )
            token_embeddings = results["representations"][repr_layer]

        # Mean pool (excluding special tokens)
        sequence_embeddings = token_embeddings[:, 1:-1, :].mean(dim=1)

        # Predict
        return self.head(sequence_embeddings).squeeze(-1)


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return sequences, labels


class SequenceDataset(torch.utils.data.Dataset):
    """Simple dataset for sequences and labels."""

    def __init__(self, sequences: list[str], labels: np.ndarray):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for sequences, labels in tqdm(dataloader, desc="Training"):
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    n_batches = 0

    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Evaluating"):
            labels = labels.to(device)
            predictions = model(sequences)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            n_batches += 1

    # Compute R²
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    ss_res = np.sum((all_labels - all_preds) ** 2)
    ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return total_loss / n_batches, r2


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading {args.task} data from {args.data_path}")

    if args.task == "stability":
        from gflownet_peptide.data.flip import load_flip_stability
        sequences, labels = load_flip_stability(args.data_path)
    elif args.task == "gb1":
        from gflownet_peptide.data.flip import load_flip_gb1
        sequences, labels = load_flip_gb1(args.data_path)
    elif args.task == "binding":
        from gflownet_peptide.data.propedia import load_propedia
        sequences, labels = load_propedia(args.data_path)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    logger.info(f"Loaded {len(sequences)} sequences")

    # Normalize labels
    labels = (labels - np.mean(labels)) / (np.std(labels) + 1e-8)

    # Create dataset and split
    dataset = SequenceDataset(sequences, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Create model
    model = SimpleRewardModel(
        esm_model=args.esm_model,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Optimizer and loss
    optimizer = AdamW(model.head.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_r2 = -float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_r2 = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_r2={val_r2:.4f}"
        )

        # Save best model
        if val_r2 > best_r2:
            best_r2 = val_r2
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_r2": val_r2,
                "config": {
                    "esm_model": args.esm_model,
                    "hidden_dim": args.hidden_dim,
                    "task": args.task,
                },
            }
            torch.save(checkpoint, output_dir / f"{args.task}_best.pt")
            logger.info(f"Saved best model with R² = {best_r2:.4f}")

    logger.info(f"Training complete. Best R² = {best_r2:.4f}")


if __name__ == "__main__":
    main()
