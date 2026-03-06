# Mapping Wildfire Risk in Maui County Using Live Fuel Moisture Content

## Project Overview

This project creates high-resolution maps of Live Fuel Moisture Content (LFMC)
for Maui County, Hawaii, using the Galileo foundation model fine-tuned on satellite
remote sensing data. LFMC measures the water content in living vegetation as a
percentage, and is a critical factor in wildfire risk assessment.

The 2023 Lahaina wildfire demonstrated the devastating impact of dry vegetation
conditions. This project provides monthly LFMC maps that could support wildfire
risk assessment and prevention efforts.

## Key References

| Resource | Link |
|----------|------|
| Johnson et al. (2025) paper | https://arxiv.org/abs/2506.20132 |
| Galileo model (Tseng et al. 2025) | https://arxiv.org/abs/2502.09356 |
| Galileo code + weights | https://github.com/nasaharvest/galileo |
| AllenAI OlmoEarth LFMC pipeline | https://github.com/allenai/olmoearth_projects |
| Globe-LFMC 2.0 dataset | https://doi.org/10.1038/s41597-024-03159-6 |
| Globe-LFMC 2.0 data (figshare) | https://doi.org/10.6084/m9.figshare.24312164 |
| Galileo weights (HuggingFace) | https://huggingface.co/nasaharvest/galileo |

## Architecture

The pipeline has three stages:

**Data Pipeline:** Globe-LFMC 2.0 labels are filtered to Hawaii. Sentinel-2
optical imagery and Sentinel-1 SAR data are downloaded for each sample location,
producing 12-month temporal stacks formatted as Galileo input tensors.

**Model Pipeline:** A pretrained Galileo-Tiny encoder (5.3M params, Vision
Transformer) is loaded from HuggingFace. An LFMC regression head is attached
and the model is fine-tuned first on CONUS data (validation), then on Hawaii
data (or used via transfer learning if Hawaii samples are sparse).

**Output Pipeline:** For each month, satellite imagery covering all of Maui
County is tiled into overlapping patches, run through the model, and stitched
into a single GeoTIFF at 10m resolution.

## Setup

    git clone https://github.com/SammyCode002/maui-lfmc.git
    cd maui-lfmc
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    # Download Galileo model weights
    pip install huggingface_hub
    huggingface-cli download nasaharvest/galileo --include "models/**" --local-dir data/galileo

## Quick Start

    # Step 1: Filter Globe-LFMC data to Hawaii
    python -m src.data.globe_lfmc_filter --input data/raw/Globe-LFMC-2.0.xlsx --output data/processed/ --region maui

    # Step 2: Download satellite imagery
    python -m src.data.sentinel_download --sites data/processed/globe_lfmc_conus_west.csv --output data/satellite/

    # Step 3: Train on CONUS data (validation)
    python -m src.model.training --config configs/conus_finetune.yaml

    # Step 4: Fine-tune for Maui
    python -m src.model.training --config configs/maui_finetune.yaml

    # Step 5: Generate LFMC maps
    python -m src.inference.map_generator --checkpoint checkpoints/maui/best_model.pt --year 2023 --month 8

## Timeline (8-10 weeks)

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Literature review + architecture | This repo, architecture doc |
| 3 | Data preparation | Filtered Globe-LFMC, satellite downloads |
| 4-5 | CONUS validation | Reproduced Johnson et al. baseline |
| 6-7 | Maui fine-tuning | Trained Maui-specific model |
| 8-10 | Map generation + analysis | Monthly LFMC maps 2023-2026 |
