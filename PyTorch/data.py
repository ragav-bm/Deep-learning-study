import pandas as pd
import torchvision
from monai.transforms import ToPIL
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from torchvision.transforms import ToPILImage, ToTensor, Normalize, RandomRotation,RandomHorizontalFlip,RandomVerticalFlip,RandomResizedCrop,ColorJitter
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data : pd.DataFrame, mode: str):
        self.mode = mode
        self.data = data
        self.transform = torchvision.transforms.Compose([ToPILImage(),
                                                         # RandomRotation(degrees=60),
                                                         # RandomHorizontalFlip(p=0.5),
                                                         # RandomVerticalFlip(p=0.5),
                                                         ToTensor(), Normalize(mean=train_mean, std=train_std)])
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("Data shape:", self.data.shape)

        l1 = self.data.iloc[idx, 1]
        l2 = self.data.iloc[idx, 2]

        img_p =self.data.iloc[idx, 0]


        image = imread(img_p)
        image = gray2rgb(image)
        image =self.transform(image)
        label =torch.tensor([l1,l2],dtype=torch.float)
        return image, label


