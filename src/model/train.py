"""
Fine-tune Galileo-Tiny on Globe-LFMC data using the AllenAI LFMC pipeline.

This script wraps the allenai/lfmc finetune_and_evaluate function and adds:
- Automatic Galileo-Tiny weight loading from the nasaharvest/galileo repo
- Configurable data paths for our project structure
- Results logging

Usage:
    # Train CONUS baseline (reproduces Johnson et al. 2025)
    python -m src.model.train \\
        --galileo-dir path/to/allenai-lfmc/submodules/galileo \\
        --data-dir data/tifs/ \\
        --h5py-dir data/h5pys/ \\
        --labels data/labels/lfmc_data_conus.csv \\
        --output checkpoints/conus/

Prerequisites:
    1. Download TIFs: python -m src.data.download_tifs --labels ... --output data/tifs/
    2. Galileo + allenai/lfmc installed (see requirements.txt)
"""

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_galileo_encoder(galileo_config_dir: Path, load_weights: bool = True):
    """Load Galileo-Tiny encoder from the galileo-data config directory."""
    import json

    import torch
    from galileo.data.config import CONFIG_FILENAME, ENCODER_FILENAME
    from galileo.galileo import Encoder
    from galileo.utils import device

    config_path = galileo_config_dir / "models" / "nano" / CONFIG_FILENAME
    encoder_path = galileo_config_dir / "models" / "nano" / ENCODER_FILENAME

    if not config_path.exists():
        raise FileNotFoundError(
            f"Galileo config not found at {config_path}.\n"
            "Download from: https://huggingface.co/nasaharvest/galileo\n"
            "Or clone: git clone https://huggingface.co/nasaharvest/galileo galileo-data"
        )

    with config_path.open("r") as f:
        config = json.load(f)
    encoder_config = config["model"]["encoder"]
    encoder = Encoder(**encoder_config)

    if load_weights:
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Galileo encoder weights not found at {encoder_path}.\n"
                "Download from HuggingFace: nasaharvest/galileo"
            )
        state_dict = torch.load(encoder_path, map_location=device)
        for key in list(state_dict.keys()):
            state_dict[key.replace(".backbone", "")] = state_dict.pop(key)
        encoder.load_state_dict(state_dict)
        logger.info("Loaded Galileo-Tiny weights from %s", encoder_path)
    else:
        logger.info("Initialized Galileo-Tiny with random weights (baseline comparison)")

    return encoder


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune Galileo-Tiny on Globe-LFMC data (reproduces Johnson et al. 2025)"
    )
    parser.add_argument(
        "--galileo-config-dir",
        type=Path,
        required=True,
        help="Path to galileo config dir containing models/tiny/config.json and encoder.pt"
             " (e.g. path/to/allenai-lfmc/submodules/galileo-data)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded TIF files"
    )
    parser.add_argument(
        "--h5py-dir",
        type=Path,
        required=True,
        help="Directory for H5PY cache files (will be created)"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to lfmc_data_conus.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for model checkpoint and results"
    )
    parser.add_argument(
        "--output-hw",
        type=int,
        default=32,
        help="Spatial patch size (default: 32 = 320m at 10m resolution)"
    )
    parser.add_argument(
        "--output-timesteps",
        type=int,
        default=12,
        help="Number of monthly timesteps (default: 12)"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use random weights instead of pretrained (for baseline comparison)"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.h5py_dir.mkdir(parents=True, exist_ok=True)

    # The allenai/lfmc LFMCDataset reads labels from a hardcoded relative path.
    # We override this by setting the LABELS_PATH before import.
    os.environ["LFMC_LABELS_PATH"] = str(args.labels.resolve())

    # Patch the labels path in the lfmc package before importing
    import lfmc.core.const as const
    const.LABELS_PATH = args.labels.resolve()

    from galileo.data.config import NORMALIZATION_DICT_FILENAME
    from galileo.data.dataset import Dataset, Normalizer
    from lfmc.core.eval import finetune_and_evaluate
    from lfmc.core.splits import DEFAULT_TEST_FOLDS, DEFAULT_VALIDATION_FOLDS

    # Load normalization values from galileo-data
    norm_path = args.galileo_config_dir / NORMALIZATION_DICT_FILENAME
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Normalization dict not found at {norm_path}.\n"
            "This file should be in the galileo-data submodule."
        )
    normalizing_dicts = Dataset.load_normalization_values(norm_path)
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dicts)
    logger.info("Loaded normalization values from %s", norm_path)

    # Load Galileo-Tiny encoder
    encoder = load_galileo_encoder(
        args.galileo_config_dir,
        load_weights=not args.no_pretrained
    )

    logger.info("Starting fine-tuning...")
    logger.info("  Data dir: %s", args.data_dir)
    logger.info("  H5PY dir: %s", args.h5py_dir)
    logger.info("  Labels: %s", args.labels)
    logger.info("  Output: %s", args.output)
    logger.info("  Pretrained: %s", not args.no_pretrained)

    results, df = finetune_and_evaluate(
        normalizer=normalizer,
        pretrained_model=encoder,
        data_folder=args.data_dir,
        h5py_folder=args.h5py_dir,
        output_folder=args.output,
        output_hw=args.output_hw,
        output_timesteps=args.output_timesteps,
        validation_folds=DEFAULT_VALIDATION_FOLDS,
        test_folds=DEFAULT_TEST_FOLDS,
    )

    logger.info("Results:\n%s", json.dumps(results, indent=2))

    results_path = args.output / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    df_path = args.output / "results.csv"
    df.to_csv(df_path, index=False)

    overall = results.get("all", {})
    logger.info(
        "FINAL: RMSE=%.2f  MAE=%.2f  R²=%.3f  (paper targets: RMSE≈18.9, R²≈0.72)",
        overall.get("rmse", 0),
        overall.get("mae", 0),
        overall.get("r2_score", 0),
    )


if __name__ == "__main__":
    main()
