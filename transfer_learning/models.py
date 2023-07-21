from abc import ABC
from torch.utils.data import Dataset
from torchvision import models
import torch

"""
AlexNet, Inception, VGG, ResNet
"""

class ClassificationPipeline(ABC):
    pass

class AlexNet:
    def __init__(self):
        #load and initialize model
        self.model = models.alexnet(pretrained=True)

    def evaluate(self, data:Dataset):
        pass

    def train(self):
        pass