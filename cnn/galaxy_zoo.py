import os
import torch
from torch.utils.data import TensorDataset, Dataset
from torch import from_numpy
# from torchvision import VisionDataset
import pandas as pd
from PIL import Image

# *****************************************************************************
# Citation: https://github.com/jimsiak/kaggle-galaxies-pytorch/blob/master/GalaxiesDataset.py
class DatasetGalaxyZoo(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.classes_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # TODO - replace this with full data length
        # return len(self.classes_frame)
        max_len = 16
        print(f'DatasetGalaxyZoo capping dataset length at {max_len}')
        return min(len(self.classes_frame), max_len)

    def __getitem__(self, idx):
        img_id = self.classes_frame.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, str(img_id)) + ".jpg"
        image = Image.open(img_name)
        labels = self.classes_frame.iloc[idx, 1:].values
        if self.transform:
            sample = {'image': self.transform(image), 'labels': labels, 'id': img_id}
        else:
            sample = {'image': image, 'labels': labels, 'id': img_id}
        # unpack image, labels and img_id; this pipeline wants only image and labels
        image = sample['image']
        labels = sample['labels']
        # convert labels to a PyTorch tensor
        labels = torch.FloatTensor(labels)
        if (idx==0):
            print(f'image.shape={image.shape}')
            print(f'labels.shape={labels.shape}')
        return image, labels


