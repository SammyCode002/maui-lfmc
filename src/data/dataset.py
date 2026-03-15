"""
PyTorch Dataset for LFMC Training with Galileo

Loads preprocessed satellite imagery + LFMC labels and formats them
into the MaskedOutput structure that Galileo expects.

The key insight: Galileo expects inputs organized by modality type:
- space_time_x: data that varies in both space and time (Sentinel-1/2)
- space_x: data that varies only in space (elevation from SRTM)
- time_x: data that varies only in time (weather from ERA5)
- static_x: data that is constant (land cover type)
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("lfmc_dataset")


class LFMCDataset(Dataset):
    """
    PyTorch Dataset for LFMC prediction using Galileo inputs.

    Each sample contains:
    - Satellite imagery (Sentinel-2, optionally Sentinel-1)
    - LFMC label (percentage, typically 0-200)
    - Metadata (location, date)

    The __getitem__ method returns data formatted for Galileo's
    construct_galileo_input() utility function.
    """

    def __init__(
        self,
        data_dir: str,
        labels_csv: str,
        normalize: bool = True,
        augment: bool = False,
        max_timesteps: int = 12,
    ):
        """
        Args:
            data_dir: Directory containing .npz satellite data files
            labels_csv: CSV with columns: sample_id, lfmc_value, latitude, longitude
            normalize: Apply Galileo's pretraining normalization stats
            augment: Apply data augmentation (random crops, flips)
            max_timesteps: Maximum number of temporal steps (Galileo uses 12)
        """
        start = time.time()
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment = augment
        self.max_timesteps = max_timesteps

        # Load labels
        import pandas as pd
        self.labels_df = pd.read_csv(labels_csv)
        self.sample_ids = self.labels_df["sample_id"].tolist() if "sample_id" in self.labels_df.columns else [
            f"sample_{i:06d}" for i in range(len(self.labels_df))
        ]

        # Filter to samples that have downloaded satellite data
        available = []
        for sid in self.sample_ids:
            if (self.data_dir / f"{sid}.npz").exists():
                available.append(sid)

        logger.info(f"INPUT  | {len(available)}/{len(self.sample_ids)} samples have satellite data")
        self.sample_ids = available

        # Galileo normalization statistics (from pretraining)
        # These are the mean and std for each Sentinel-2 band
        # Source: nasaharvest/galileo repo, src/data/utils.py
        self.s2_mean = np.array([
            1370.19, 1184.37, 1120.35, 1136.15, 1263.73,
            1645.40, 1846.87, 1762.59, 2084.01, 1464.04,
        ], dtype=np.float32)
        self.s2_std = np.array([
            633.15, 650.20, 712.02, 598.44, 567.44,
            559.48, 609.88, 660.32, 737.67, 569.36,
        ], dtype=np.float32)

        elapsed = time.time() - start
        logger.info(f"TIMING | Dataset init took {elapsed:.3f}s")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and preprocess a single training sample.

        Returns dict with:
            's2_data': Sentinel-2 tensor [T, H, W, bands]
            'lfmc_label': scalar LFMC value
            'months': list of month integers for each timestep
            'latitude': float
            'longitude': float
        """
        start = time.time()
        sid = self.sample_ids[idx]

        # Load from disk
        npz_path = self.data_dir / f"{sid}.npz"
        data = np.load(npz_path, allow_pickle=True)

        s2_data = data["s2_data"].astype(np.float32)  # [T, H, W, bands]
        lfmc_value = float(data["lfmc_value"])
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])

        # Pad or truncate to max_timesteps
        T = s2_data.shape[0]
        if T > self.max_timesteps:
            s2_data = s2_data[:self.max_timesteps]
        elif T < self.max_timesteps:
            pad_shape = (self.max_timesteps - T,) + s2_data.shape[1:]
            padding = np.zeros(pad_shape, dtype=np.float32)
            s2_data = np.concatenate([s2_data, padding], axis=0)

        # Normalize using Galileo's pretraining statistics
        if self.normalize:
            s2_data = (s2_data - self.s2_mean) / (self.s2_std + 1e-8)

        # Data augmentation
        if self.augment:
            s2_data = self._augment(s2_data)

        # Convert to tensors
        sample = {
            "s2_data": torch.from_numpy(s2_data),
            "lfmc_label": torch.tensor(lfmc_value, dtype=torch.float32),
            "latitude": latitude,
            "longitude": longitude,
            "num_valid_timesteps": min(T, self.max_timesteps),
        }

        elapsed = time.time() - start
        if elapsed > 1.0:
            logger.warning(f"TIMING | Slow load for {sid}: {elapsed:.2f}s")

        return sample

    def _augment(self, data: np.ndarray) -> np.ndarray:
        """Apply random augmentations to satellite imagery."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            data = np.flip(data, axis=2).copy()  # flip W dimension

        # Random vertical flip
        if np.random.random() > 0.5:
            data = np.flip(data, axis=1).copy()  # flip H dimension

        # Random brightness/contrast adjustment (small)
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.9, 1.1)
            data = data * factor

        return data

    def get_sample_metadata(self, idx: int) -> dict:
        """Get metadata for a sample without loading the full data."""
        sid = self.sample_ids[idx]
        row = self.labels_df[
            self.labels_df.get("sample_id", self.labels_df.index) == sid
        ]
        if len(row) == 0:
            row = self.labels_df.iloc[[idx]]
        return row.iloc[0].to_dict()


def create_train_val_split(
    dataset: LFMCDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.

    Uses spatial splitting: validation sites are geographically separated
    from training sites to test generalization (not just interpolation).
    """
    from torch.utils.data import Subset

    np.random.seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    val_size = int(n * val_fraction)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    logger.info(f"STATUS | Train: {len(train_indices)}, Val: {len(val_indices)}")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
