# Architecture: LFMC Mapping for Maui County

## What is LFMC?

Live Fuel Moisture Content (LFMC) measures how much water is in living vegetation.
Expressed as: LFMC = ((fresh_weight - dry_weight) / dry_weight) * 100

Values range from 0 percent (dry, fire risk) to 180+ percent (moist, safe).
Below 80 percent is generally considered critical for wildfire risk.

## The Galileo Foundation Model

Galileo (Tseng et al., 2025) is a Vision Transformer pretrained on diverse
remote sensing data via self-supervised masked modeling.

Galileo-Tiny properties:
- 5.3M parameters (fine-tunable on a single GPU)
- Processes 10 remote sensing modalities across space and time
- Input: 12 timesteps of satellite data per prediction
- Output: learned embeddings for regression head

Input modalities:
- Sentinel-2 L2A: 10 optical bands (10m resolution)
- Sentinel-1 RTC: SAR backscatter (VV, VH)
- SRTM: Elevation (static)
- ERA5: Weather variables

## Data Pipeline

Globe-LFMC 2.0: 280K+ field measurements, 2000+ sites, 15 countries.
Filter to Hawaii/Maui by bounding box. If sparse, use CONUS for transfer.

## Model Pipeline

Phase 1 (CONUS validation): Fine-tune Galileo-Tiny on CONUS Globe-LFMC data.
Phase 2 (Maui): Either fine-tune further on Hawaii data or transfer directly.

Training: AdamW, cosine LR schedule, MSE loss, early stopping.

## Inference

Tile Maui County into overlapping patches, predict, stitch into GeoTIFF.
Maui bounds: 20.5-21.1N, 156.7-155.9W. CRS: EPSG:32604 (UTM Zone 4N).

## OlmoEarth vs Galileo

Johnson et al. uses Galileo (nasaharvest/galileo).
AllenAI repo uses OlmoEarth (allenai/olmoearth_projects), a newer related model.
OlmoEarth has a pre-trained LFMC checkpoint for direct inference.
This codebase targets Galileo per the project description.
