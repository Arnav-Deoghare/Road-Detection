from dataloader import RoadDataset

dataset = RoadDataset(
    "C:/Users/arnav/Documents/Desktop/machine learning/data/train",
    "C:/Users/arnav/Documents/Desktop/machine learning/data/train"
)

img, mask = dataset[0]

print(img.shape)
print(mask.shape)
