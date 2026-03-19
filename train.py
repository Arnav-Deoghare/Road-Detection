import torch
from torch.utils.data import DataLoader
from dataloader import RoadDataset
from u_net import UNet
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)

BEST_PATH = "checkpoints/best_model.pth"
LAST_PATH = "checkpoints/last_model.pth"

# -------------------------
# Metrics
# -------------------------
def iou_score(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()
    inter = (pred * target).sum((1,2,3))
    union = pred.sum((1,2,3)) + target.sum((1,2,3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def f1_score(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()
    tp = (pred * target).sum((1,2,3))
    fp = (pred * (1 - target)).sum((1,2,3))
    fn = ((1 - pred) * target).sum((1,2,3))
    return ((2*tp + eps) / (2*tp + fp + fn + eps)).mean().item()

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum((1,2,3))
    union = pred.sum((1,2,3)) + target.sum((1,2,3))
    return 1 - ((2*inter + eps) / (union + eps)).mean()

# -------------------------
# Data + Model
# -------------------------
dataset = RoadDataset("data/train", "data/train")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet().to(device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# Early Stopping Settings
# -------------------------
best_iou = 0.0
patience = 20
min_delta = 1e-4
epochs_no_improve = 0
max_epochs = 500

# -------------------------
# Training
# -------------------------
for epoch in range(max_epochs):

    model.train()
    total_loss = total_iou = total_f1 = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks) + dice_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += iou_score(preds.detach(), masks)
        total_f1 += f1_score(preds.detach(), masks)

    n = len(loader)
    epoch_loss = total_loss / n
    epoch_iou = total_iou / n
    epoch_f1 = total_f1 / n

    print(f"Epoch {epoch+1} | Loss {epoch_loss:.4f} | IoU {epoch_iou:.4f} | F1 {epoch_f1:.4f}")

    # ---- Early Stopping Check ----
    if epoch_iou > best_iou + min_delta:
        best_iou = epoch_iou
        epochs_no_improve = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iou": best_iou,
        }, BEST_PATH)

        print(f"   Saved best model (IoU={best_iou:.4f})")

    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

# -------------------------
# Save last model
# -------------------------
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, LAST_PATH)

print("Saved last model")

# -------------------------
# Quick Visualization
# -------------------------
model.eval()
img, mask = dataset[0]

with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device)).cpu()

print(f"Sample IoU: {iou_score(pred, mask.unsqueeze(0)):.4f}")

plt.subplot(1,3,1)
plt.imshow(img.permute(1,2,0))
plt.title("Image")

plt.subplot(1,3,2)
plt.imshow(mask[0], cmap="gray")
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow((torch.sigmoid(pred)[0,0] > 0.5).float(), cmap="gray")
plt.title("Prediction")

plt.show()