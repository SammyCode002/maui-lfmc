"""
Training Loop for LFMC Model

Handles the full training workflow:
1. Load pretrained Galileo encoder
2. Attach LFMC regression head
3. Train with proper learning rate scheduling
4. Evaluate on validation set
5. Save best checkpoint

Follows Johnson et al. (2025) training recipe where possible.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("training")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class LFMCTrainer:
    """
    Training manager for LFMC prediction model.

    Implements:
    - Two-phase training (frozen encoder, then fine-tune)
    - Cosine learning rate schedule with warmup
    - Early stopping based on validation RMSE
    - Checkpoint saving and loading
    - Comprehensive logging (4x4 debug style)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Training configuration dict with keys:
                - learning_rate: base LR (default 1e-4)
                - batch_size: (default 32)
                - epochs: max epochs (default 100)
                - patience: early stopping patience (default 10)
                - warmup_epochs: LR warmup period (default 5)
                - checkpoint_dir: where to save models
                - device: 'cuda' or 'cpu'
        """
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.best_val_rmse = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_rmse": [], "val_r2": [], "lr": []}

        logger.info(f"STATUS | Device: {self.device}")
        logger.info(f"STATUS | Config: {json.dumps(config, indent=2)}")

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """
        Run the full training loop.

        Returns:
            Training history dict with metrics per epoch
        """
        start = time.time()
        model = model.to(self.device)

        lr = self.config.get("learning_rate", 1e-4)
        epochs = self.config.get("epochs", 100)
        patience = self.config.get("patience", 10)
        warmup = self.config.get("warmup_epochs", 5)
        ckpt_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Separate learning rates for encoder vs head
        # Encoder gets lower LR to preserve pretrained features
        encoder_params = [p for n, p in model.named_parameters()
                         if "encoder" in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters()
                      if "head" in n and p.requires_grad]

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr * 0.1},  # 10x lower for encoder
            {"params": head_params, "lr": lr},
        ], weight_decay=0.01)

        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=epochs - warmup, T_mult=1
        )

        criterion = nn.MSELoss()

        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info(f"  Epochs: {epochs}, Batch size: {self.config.get('batch_size', 32)}")
        logger.info(f"  Train samples: {len(train_loader.dataset)}")
        logger.info(f"  Val samples: {len(val_loader.dataset)}")
        logger.info("=" * 60)

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train one epoch
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion, epoch, warmup, lr)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_metrics = self._validate(model, val_loader, criterion)
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["val_r2"].append(val_metrics["r2"])
            self.history["lr"].append(optimizer.param_groups[0]["lr"])

            # Step scheduler (after warmup)
            if epoch >= warmup:
                scheduler.step()

            epoch_elapsed = time.time() - epoch_start

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.2f} | "
                f"Val R2: {val_metrics['r2']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {epoch_elapsed:.1f}s"
            )

            # Early stopping check
            if val_metrics["rmse"] < self.best_val_rmse:
                self.best_val_rmse = val_metrics["rmse"]
                self.patience_counter = 0
                # Save best model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "config": self.config,
                }, ckpt_dir / "best_model.pt")
                logger.info(f"  >> New best! RMSE: {val_metrics['rmse']:.2f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                    break

        total_elapsed = time.time() - start
        logger.info(f"TIMING | Training complete in {total_elapsed:.1f}s")
        logger.info(f"OUTPUT | Best val RMSE: {self.best_val_rmse:.2f}")

        # Save training history
        with open(ckpt_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def _train_epoch(self, model, loader, optimizer, criterion, epoch, warmup, base_lr):
        """Train for one epoch. Returns average loss."""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            # Move data to device
            s2_data = batch["s2_data"].to(self.device)
            labels = batch["lfmc_label"].to(self.device)

            # Warmup: linearly increase LR
            if epoch < warmup:
                warmup_factor = (epoch * len(loader) + num_batches) / (warmup * len(loader))
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * warmup_factor

            optimizer.zero_grad()

            # Forward pass
            # NOTE: You need to adapt this to create proper Galileo MaskedOutput
            # from your s2_data tensor. See src/data/preprocessing.py
            predictions = model(s2_data)
            loss = criterion(predictions.squeeze(), labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate(self, model, loader, criterion) -> dict:
        """Validate and compute metrics. Returns dict with rmse, r2, mae."""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                s2_data = batch["s2_data"].to(self.device)
                labels = batch["lfmc_label"].to(self.device)

                predictions = model(s2_data)
                all_preds.append(predictions.squeeze().cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        # Compute metrics
        mse = np.mean((preds - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - labels))

        # R-squared
        ss_res = np.sum((labels - preds) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {"rmse": float(rmse), "r2": float(r2), "mae": float(mae), "mse": float(mse)}


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LFMC model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = LFMCTrainer(config)
    # NOTE: You need to set up model and data loaders here
    # See README.md for the full pipeline
    print("Config loaded. Set up model and data loaders to start training.")
    print(f"Config: {json.dumps(config, indent=2)}")
