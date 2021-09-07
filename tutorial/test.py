import torch as nn
from torch.utils.data import dataset, dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import open as read_file
import os
import pandas as pd
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

testing_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

'''
class custom_image(dataset):
    def __init__(self,annotation_file, img_dir, transform = None, target_transform = None):
        self.image_label = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_label)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_label.iloc[idx, 0])
        image = read_file(img_path)
        label = self.image_label.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

'''
training_dataloader = dataloader(training_data, batchsize=64, shuffle=True)
testing_dataloader = dataloader(testing_data, batchsize=64, shuffle=True)
print(training_data.__len__())
print(testing_data)