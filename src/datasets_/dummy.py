"""
This module defines a dummy dataset for testing purposes.
"""
import torch
from utils import Modality
from .base_datasets import BaseDataset
from registries import register_dataset
from typing import Dict

@register_dataset(name='Dummy')
class DummyDataset(BaseDataset):
    def __init__(self, size=30000, dim=20):
        """Dummy dataset for testing purposes. Data is randomly generated.

        Args:
            size (int, optional): The number of samples in the dataset. Defaults to 30000.
            dim (int, optional): The dimension of each datapoint. Defaults to 20.
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

    def __getitem__(self, index:int) -> Dict[str, torch.Tensor]:
        out = dict()
        x = torch.randn(self.dim)
        y = torch.sin(x)
        out['x'] = x
        out['target'] = y
        return out
