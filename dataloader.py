import os
import cv2
import torch
from torch.utils.data import Dataset

class RoadDataset(Dataset):
    def __init__(self,image_dir,mask_dir,size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.images = [f for f in os.listdir(image_dir) if f.endswith("_sat.jpg")]
        self.images = self.images[:300]

    def __len__(self):
        return len(self.images)     
    
    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir,
            img_name.replace("_sat.jpg", "_mask.png")
        )

        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.size, self.size))
        image = image / 255.0
        image = torch.tensor(image).permute(2,0,1).float()

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.size, self.size))
        mask = mask / 255.0
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask