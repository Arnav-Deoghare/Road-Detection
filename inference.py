import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from u_net import UNet
from dataloader import RoadDataset

# -------------------------
# Config
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/best_model.pth"
BATCH_SIZE = 4

# -------------------------
# Load Dataset (Use VAL, not TRAIN)
# -------------------------
dataset = RoadDataset("data/valid",mask_dir=None)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Load Model
# -------------------------
model = UNet().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded model trained up to epoch {checkpoint['epoch']}")

# -------------------------
# Inference Loop
# -------------------------
with torch.no_grad():
    for images in loader:
        images = images.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Visualize first image of batch
        img = images[0].cpu().permute(1, 2, 0)
        prob = probs[0, 0].cpu()
        pred = preds[0, 0].cpu()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(prob)
        plt.title("Probability")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        break  # remove this to run full dataset