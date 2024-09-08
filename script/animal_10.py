import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor,Resize,Compose
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class Animal10(Dataset):
    def __init__(self,root, train, transform):
        self.categories = ["butterfly","cat","chicken","cow","dog","elephant","horse","sheep","spider","squirrel"]
        self.root = root
        if train:
            data_paths = os.path.join(root,"train")
        else:
            data_paths = os.path.join(root,"test")
        self.image = []
        self.label = []
        for idx, categori in enumerate(self.categories):
            categories_path = os.path.join(data_paths,categori)
            for image in os.listdir(categories_path):
                img_path = os.path.join(categories_path,image)
                self.image.append(img_path)
                self.label.append(idx)
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        # img = cv2.imread(self.image[item],1)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.open(self.image[item]).convert("RGB") # convert từ ảnh đen trắng , transparent -> rgb
        if self.transform:
            img = self.transform(img)
        label = self.label[item]
        return img,label
