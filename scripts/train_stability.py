#!/usr/bin/env python3
"""Train ESM-2 based stability predictor on FLIP dataset.

This script trains a stability predictor using ESM-2 embeddings as input
to an MLP regression head. The model predicts normalized stability scores.

Usage:
    python scripts/train_stability.py
    python scripts/train_stability.py --epochs 100 --batch_size 16
    python scripts/train_stability.py --esm_model esm2_t12_35M_UR50D
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StabilityDataset(Dataset):
    """Dataset for stability prediction training."""

    def __init__(self, csv_path: str, target_col: str = 'target_normalized'):
        """Initialize dataset.

        Args:
            csv_path: Path to preprocessed CSV file
            target_col: Column name for target values
        """
        df = pd.read_csv(csv_path)
        self.sequences = df['sequence'].tolist()
        self.labels = torch.tensor(df[target_col].values, dtype=torch.float32)
        logger.info(f"Loaded {len(self.sequences)} sequences from {csv_path}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    sequences, labels = zip(*batch)
    return list(sequences), torch.stack(list(labels))


def compute_r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute R² score."""
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    # Keep ESM frozen
    model.esm.eval()

    total_loss = 0
    num_batches = 0

    for sequences, labels in dataloader:
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(sequences)
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            predictions = model(sequences)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    predictions = torch.tensor(all_predictions)
    labels = torch.tensor(all_labels)

    mse = ((predictions - labels) ** 2).mean().item()
    r2 = compute_r2(predictions, labels)

    return {'mse': mse, 'r2': r2}


def train_stability_predictor(config: dict):
    """Main training function.

    Args:
        config: Training configuration dictionary

    Returns:
        Best validation R² achieved
    """
    # Setup logging
    try:
        import wandb
        wandb.init(
            project=config.get('wandb_project', 'gflownet-peptide'),
            name=config.get('run_name', 'stability_predictor'),
            config=config,
        )
        use_wandb = True
    except Exception as e:
        logger.warning(f"W&B initialization failed: {e}. Continuing without W&B.")
        use_wandb = False

    device = config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    data_dir = Path(config['data_dir'])
    train_ds = StabilityDataset(data_dir / 'train.csv')
    val_ds = StabilityDataset(data_dir / 'val.csv')
    test_ds = StabilityDataset(data_dir / 'test.csv')

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # ESM requires main process
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
    )

    # Create model
    from gflownet_peptide.rewards.stability_predictor import StabilityPredictor

    model = StabilityPredictor(
        esm_model=config['esm_model'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        freeze_esm=True,
    ).to(device)

    # Only optimize head parameters
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.0),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    criterion = torch.nn.MSELoss()

    # Training loop
    best_val_r2 = -float('inf')
    patience_counter = 0
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training for {config['epochs']} epochs")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # Log metrics
        log_msg = (f"Epoch {epoch+1}/{config['epochs']}: "
                   f"train_loss={train_loss:.4f}, "
                   f"val_mse={val_metrics['mse']:.4f}, "
                   f"val_r2={val_metrics['r2']:.4f}, "
                   f"time={epoch_time:.1f}s")
        logger.info(log_msg)

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_mse': val_metrics['mse'],
                'val_r2': val_metrics['r2'],
                'learning_rate': optimizer.param_groups[0]['lr'],
            })

        # Scheduler step
        scheduler.step(val_metrics['r2'])

        # Save best model
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            patience_counter = 0

            checkpoint_path = checkpoint_dir / 'stability_predictor_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': best_val_r2,
                'config': config,
            }, checkpoint_path)
            logger.info(f"New best model saved: R²={best_val_r2:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.get('early_stopping_patience', 15):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Final evaluation on test set
    logger.info("\n" + "="*50)
    logger.info("Final Evaluation on Test Set")
    logger.info("="*50)

    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'stability_predictor_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test MSE: {test_metrics['mse']:.4f}")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Target: R² ≥ 0.6")
    logger.info(f"Status: {'✅ PASSED' if test_metrics['r2'] >= 0.6 else '⚠️ BELOW TARGET'}")

    if use_wandb:
        wandb.log({
            'test_mse': test_metrics['mse'],
            'test_r2': test_metrics['r2'],
            'best_val_r2': best_val_r2,
        })
        wandb.finish()

    # Save final results
    results = {
        'best_val_r2': best_val_r2,
        'test_mse': test_metrics['mse'],
        'test_r2': test_metrics['r2'],
        'epochs_trained': epoch + 1,
        'config': config,
    }
    with open(checkpoint_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return best_val_r2


def main():
    parser = argparse.ArgumentParser(description="Train stability predictor")
    parser.add_argument('--data_dir', type=str, default='data/processed/flip_stability',
                        help='Directory with preprocessed FLIP data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/reward_models',
                        help='Directory to save checkpoints')
    parser.add_argument('--esm_model', type=str, default='esm2_t6_8M_UR50D',
                        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t33_650M_UR50D'],
                        help='ESM-2 model to use')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden dimensions for MLP head')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--wandb_project', type=str, default='gflownet-peptide',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default='stability_predictor',
                        help='W&B run name')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detected if not specified)')

    args = parser.parse_args()

    config = {
        'data_dir': args.data_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'esm_model': args.esm_model,
        'hidden_dims': args.hidden_dims,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'wandb_project': args.wandb_project,
        'run_name': args.run_name,
        'device': args.device,
    }

    print("="*60)
    print("Phase 1: Training Stability Predictor")
    print("="*60)
    print(f"ESM Model: {config['esm_model']}")
    print(f"Hidden dims: {config['hidden_dims']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print("="*60)

    train_stability_predictor(config)


if __name__ == '__main__':
    main()
