from typing import List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader 
from torch.utils.data.dataset import ConcatDataset
from datasets_ import data_utils

class MultiDataModule(LightningDataModule):
    def __init__(
        self,
        datamodules: List[LightningDataModule],
        batch_size:int,
        num_workers:int,
        shuffle:bool=True,
        drop_last:bool=True,
    ):
        super().__init__()
        self.datamodules = datamodules
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        for datamodule in self.datamodules:
            datamodule.prepare_data()

    def setup(self, stage=None):
        train_datasets = []
        val_datasets = []
        test_datasets = []
        for datamodule in self.datamodules:
            datamodule.setup(stage)
            if stage == 'fit' or stage is None:
                if hasattr(datamodule, 'train_dataset'):
                    train_datasets.append(datamodule.train_dataset)
                if hasattr(datamodule, 'val_dataset'):
                    val_datasets.append(datamodule.val_dataset)
            if stage == 'test' or stage is None:
                if hasattr(datamodule, 'test_dataset'):
                    test_datasets.append(datamodule.test_dataset)

        if train_datasets:
            self.train_dataset = ConcatDataset(train_datasets)
        if val_datasets:
            self.val_dataset = ConcatDataset(val_datasets)
        if test_datasets:
            self.test_dataset = ConcatDataset(test_datasets)
    
    def train_dataloader(self):
        assert self.train_dataset is not None, "Train dataset is not set."
        return DataLoader(self.train_dataset,
                          collate_fn=self.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset,
                          collate_fn=self.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=False,
                          drop_last=False,)
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset,
                          collate_fn=self.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=False,
                          drop_last=False,)
    
    def collater(self, samples):
        return data_utils.collater(samples)

    def teardown(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            del self.train_dataset
            del self.val_dataset
        if stage == 'test' or stage is None:
            del self.test_dataset
