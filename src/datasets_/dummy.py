import torch
from utils import Modality
from .base_datasets import BaseDataset
from registries import register_dataset

@register_dataset(name='Dummy')
class DummyDataset(BaseDataset):
    def __init__(self, size=50000, dim=20):
        """
        Args:
            size (int): Number of data points in the dataset.
            dim (int): Dimensionality of each data point.
        """
        self.size = size
        self.dim = dim

    def load(self):
        pass

    def __len__(self):
        return self.size
    
    @property
    def modality(self) -> Modality:
        return Modality.DUMMY

    def __getitem__(self, index):
        out = dict()
        x = torch.randn(self.dim)
        y = torch.sin(x)
        out['x'] = x
        out['target'] = y
        return out
