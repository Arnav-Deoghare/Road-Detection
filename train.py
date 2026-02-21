import torch
from torch.utils.data import DataLoader
from dataloader import RoadDataset
from u_net import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = RoadDataset("data/train", "data/train")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet().to(device)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(3):
    total_loss = 0

    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} Batch {batch_idx}/{len(loader)} Loss {loss.item():.4f}")

    print(f"Epoch {epoch} Average Loss {total_loss/len(loader):.4f}")
import matplotlib.pyplot as plt

model.eval()

img, mask = dataset[0]
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device))[0].cpu()

plt.subplot(1,3,1)
plt.imshow(img.permute(1,2,0))
plt.title("Image")

plt.subplot(1,3,2)
plt.imshow(mask[0], cmap="gray")
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow(pred[0], cmap="gray")
plt.title("Prediction")

plt.show()
