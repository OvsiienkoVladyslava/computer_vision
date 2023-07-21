import torch
from torch.utils.data import Dataset
from torch import Tensor
from utils import unpickle
from configs import DataPaths

"""
CIFAR-10 dataset loading, check configs.py for proper paths.
Although we can load CIFAR 10 directly from torchvision datasets I decided to implement custom dataset loading.
"""

class DatasetFromPickle(Dataset):
    """
    Form torch.Dataset from pickle images of downloaded CIFAR-10 data.
    """
    images: Tensor
    int_labels: list

    def __init__(self, paths: list | str):
        if type(paths) is list:
            (self.images, self.int_labels) = self.get_pickle_files_data(paths)
        else:
            (self.images, self.int_labels) = self.get_pickle_file_data(paths)

    def __len__(self):
        return len(self.int_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.int_labels[idx]

        return image, label

    @staticmethod
    def get_pickle_file_data(path: str) -> (Tensor, list):
        """
        read 1 pickle file and get  with torch.uint8 type, labels
        :param path:
        :return:
        """
        byte_data = unpickle(path)
        images = torch.from_numpy(byte_data[b'data'])
        int_labels = byte_data[b'labels']
        return images, int_labels

    @staticmethod
    def get_pickle_files_data(paths: list) -> (Tensor, list):
        """
            read 1 pickle file and get  with torch.uint8 type, labels
            :param paths:
            :return:
        """
        all_images = torch.tensor([], dtype=torch.uint8)
        all_labels = []
        for path in paths:
            images, int_labels = DatasetFromPickle.get_pickle_file_data(path)
            all_images = torch.cat((all_images, images), 0)
            all_labels += int_labels
        return all_images, all_labels


class DatasetCIFAR:
    """
    All paths considered in config.py.
    """
    label_coding: list
    train: Dataset
    validation: Dataset
    test: Dataset

    def __init__(self):
        # load data
        self.label_coding = self.get_labels_coding()
        self.test = DatasetFromPickle(DataPaths.test_batch)
        self.train = DatasetFromPickle(DataPaths.train_batches)
        self.validation = DatasetFromPickle(DataPaths.validation_batch)

    def get_labels_coding(self):
        meta_data = unpickle(DataPaths.batch_meta)
        labels_coding = [name.decode('utf-8') for name in meta_data[b'label_names']]
        return labels_coding