import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsImageFolder(Dataset):
    """Wrapper around torchvision's ImageFolder that uses Albumentations.
    Supply an ImageFolder dataset and an albumentations transform.
    """

    def __init__(self, image_folder, transform=None):
        super().__init__()
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        path, label = self.image_folder.imgs[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            image = self.transform(image=img)["image"]

            # image shape: [1, H, W]  [3, H, W]
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)

            return image, label
        return img, label


def get_transforms(input_size):
    train_tf = A.Compose([
        A.RandomResizedCrop(
            size=(input_size, input_size),
            scale=(0.8, 1.0)
        ),
        A.Affine(
            rotate=(-10, 10),
            translate_percent=(0.02, 0.02),
            scale=(0.9, 1.1),
            p=0.7
        ),
        A.ElasticTransform(
            alpha=20,
            sigma=5,
            p=0.3
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            p=0.2
        ),

        # ---- FORCE GRAYSCALE (1 CHANNEL) ----
        A.ToGray(p=1.0),

        A.Normalize(
            mean=(0.5,),
            std=(0.5,)
        ),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(input_size, input_size),

        A.ToGray(p=1.0),

        A.Normalize(
            mean=(0.5,),
            std=(0.5,)
        ),
        ToTensorV2(),
    ])

    return train_tf, val_tf

