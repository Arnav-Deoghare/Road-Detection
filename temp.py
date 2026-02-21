from dataloader import RoadDataset

ds = RoadDataset("data/train","data/train")

print("Dataset length:", len(ds))

img, mask = ds[0]

print(img.shape)
print(mask.shape)
print(img.min(), img.max())
print(mask.min(), mask.max())
