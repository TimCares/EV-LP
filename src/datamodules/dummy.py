"""
This module defines the data module for the dummy dataset. This is used for testing purposes.
"""
from .unimodal_datamodules import BaseDataModule
from datasets_ import DummyDataset
import os
from registries import register_datamodule

@register_datamodule(name='Dummy')
class DummyDataModule(BaseDataModule):
    def __init__(
        self,
        data_path:os.PathLike,
        size:int=30000,
        dim:int=20,
        *args,
        **kwargs,
    ):
        """The data module for the dummy dataset.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            size (int, optional): The number of samples in the dataset. Defaults to 30000.
            dim (int, optional): The dimension of each datapoint. Defaults to 20.
        """        
        super().__init__(data_path=data_path, *args, **kwargs)
        self.size = size
        self.dim = dim

    def set_train_dataset(self):
        self.train_dataset =  DummyDataset(size=self.size, dim=self.dim)

    def set_val_dataset(self):
        self.val_dataset =  DummyDataset(size=self.size//10, dim=self.dim)

    def set_test_dataset(self):
        self.test_dataset =  DummyDataset(size=self.size//10, dim=self.dim)
