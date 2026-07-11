# Road Detection

Quick road-segmentation project using a small U-Net in PyTorch.

## What it does
- Trains a binary segmentation model to detect roads from satellite images.
- Uses BCEWithLogits + Dice loss.
- Tracks IoU/F1 during training.
- Saves:
	- `checkpoints/best_model.pth`
	- `checkpoints/last_model.pth`

## Project files
- `train.py`: training loop + early stopping + sample visualization.
- `inference.py`: loads best checkpoint and visualizes predictions.
- `dataloader.py`: dataset loader for image/mask pairs.
- `u_net.py`: U-Net model.
- `main.py`: quick dataset sanity check.

## Dataset structure
Expected naming pattern inside data folders:
- Image: `*_sat.jpg`
- Mask: `*_mask.png`

Recommended layout:
```text
data/
	train/
		xxx_sat.jpg
		xxx_mask.png
	valid/
		yyy_sat.jpg
		yyy_mask.png
```

Notes:
- Training currently reads from `data/train`.
- Inference currently reads images from `data/valid`.

## Install
Create and activate an environment, then install dependencies:

```bash
pip install torch torchvision opencv-python matplotlib
```

## Train
```bash
python train.py
```

## Run inference
```bash
python inference.py
```

## Quick checks
- Dataset sample:
```bash
python main.py
```

## Current limitations
- `RoadDataset` in `dataloader.py` expects masks in `__getitem__`, so inference may need a small loader adjustment for image-only mode.
- Dataset is currently capped to first 300 images (`self.images = self.images[:300]`).

---
If you want, I can also make this README more polished with examples, metrics table, and troubleshooting.

