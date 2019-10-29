import os
import torch
from torch.utils.data import TensorDataset, Dataset
from torch import from_numpy
# from torchvision import VisionDataset
import pandas as pd
from PIL import Image

# *****************************************************************************
# Citation: https://github.com/jimsiak/kaggle-galaxies-pytorch/blob/master/GalaxiesDataset.py
class DatasetGalaxyZoo_v1(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.classes_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.classes_frame)

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
        print(f'image.shape={image.shape}')
        print(f'labels.shape={labels.shape}')
        return image, labels


# # *****************************************************************************
# class DatasetGalaxyZoo(VisionDataset):
    # """Dataset for Galaxy Zoo 
    # Args:
        # root (string): Root directory of dataset where directory
            # ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        # train (bool, optional): If True, creates dataset from training set, otherwise
            # creates from test set.
        # transform (callable, optional): A function/transform that takes in an PIL image
            # and returns a transformed version. E.g, ``transforms.RandomCrop``
        # target_transform (callable, optional): A function/transform that takes in the
            # target and transforms it.
        # download (bool, optional): If true, downloads the dataset from the internet and
            # puts it in root directory. If dataset is already downloaded, it is not
            # downloaded again.

    # """
    # # base_folder = 'cifar-10-batches-py'
    # def __init__(self, root, csv_file, train=True, transform=None, target_transform=None):

        # super(DatasetGalaxyZoo, self).__init__(root, transform=transform, target_transform=target_transform)

        # self.train = train  # training set or test set
        # self.classes_frame = pd.read_csv(csv_file)

        # # if self.train:
            # # downloaded_list = self.train_list
        # # else:
            # # downloaded_list = self.test_list

        # # self.data = []
        # # self.targets = []

        # # # now load the picked numpy arrays
        # # for file_name, checksum in downloaded_list:
            # # file_path = os.path.join(self.root, self.base_folder, file_name)
            # # with open(file_path, 'rb') as f:
                # # if sys.version_info[0] == 2:
                    # # entry = pickle.load(f)
                # # else:
                    # # entry = pickle.load(f, encoding='latin1')
                # # self.data.append(entry['data'])
                # # if 'labels' in entry:
                    # # self.targets.extend(entry['labels'])
                # # else:
                    # # self.targets.extend(entry['fine_labels'])

        # # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # # self._load_meta()

    # def __getitem__(self, index):
        # """
        # Args:
            # index (int): Index

        # Returns:
            # tuple: (image, target) where target is index of the target class.
        # """
        # # img, target = self.data[index], self.targets[index]

        # img_id = self.classes_frame.iloc[idx, 0]
        # img_name = os.path.join(self.root, str(img_id)) + ".jpg"
        # image = Image.open(img_name)
        # labels = self.classes_frame.iloc[idx, 1:].values

        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img)

        # if self.transform is not None:
            # img = self.transform(img)

        # if self.target_transform is not None:
            # target = self.target_transform(target)

        # return img, target


    # def __len__(self):
        # return len(self.data)

    # def _check_integrity(self):
        # root = self.root
        # for fentry in (self.train_list + self.test_list):
            # filename, md5 = fentry[0], fentry[1]
            # fpath = os.path.join(root, self.base_folder, filename)
            # if not check_integrity(fpath, md5):
                # return False
        # return True

    # def extra_repr(self):
        # return "Split: {}".format("Train" if self.train is True else "Test")


