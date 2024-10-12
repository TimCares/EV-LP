from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, List
from datasets_ import ImageNetDataset, MaskedLMDataset
from datasets_ import DATASET_REGISTRY
from functools import partial
from utils import Modality
import os
from transformers import BertTokenizer

class BaseDataModule(LightningDataModule):
    def __init__(self,
                 data_path:str,
                 batch_size:int,
                 num_workers:int,
                 shuffle:bool=True,
                 drop_last:bool=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def set_train_dataset(self):
        raise NotImplementedError("set train dataset")

    def set_val_dataset(self):
        pass # optional: not all datasets have a validation set

    def set_test_dataset(self):
        pass # optional: not all datasets have a test set

    @property
    def modality(self) -> Modality:
        raise NotImplementedError

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()
        if not hasattr(self, 'val_dataset'):
            self.set_val_dataset()
        if not hasattr(self, 'test_dataset'):
            self.set_test_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=self.train_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,)

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
            return None
        return DataLoader(self.val_dataset,
                          collate_fn=self.val_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=False,
                          drop_last=False,)

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            return None
        return DataLoader(self.test_dataset,
                          collate_fn=self.test_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=False,
                          drop_last=False,)
    
    def teardown(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            if hasattr(self, 'train_dataset'):
                del self.train_dataset
            if hasattr(self, 'val_dataset'):
                del self.val_dataset
        if stage == 'test' or stage is None:
            if hasattr(self, 'test_dataset'):
                del self.test_dataset


class MaskedLMDataModule(BaseDataModule):
    def __init__(
        self,
        name: str,
        data_path: str,
        text_file: os.PathLike,
        block_size: int=512,
        mask_prob: float=0.0,
        *args,
        **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.name = name
        self.train_text_file = text_file + '.train'
        self.val_text_file = text_file + '.val'
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()
        if not hasattr(self, 'val_dataset'):
            self.set_val_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = MaskedLMDataset(name=self.name, data_path=self.data_path, split='train', text_file=self.train_text_file,
                                             tokenizer=self.tokenizer,
                                             block_size=self.block_size, mask_prob=self.mask_prob)
        
    def set_val_dataset(self):
        self.val_dataset = MaskedLMDataset(name=self.name, data_path=self.data_path, split='val', text_file=self.val_text_file,
                                           tokenizer=self.tokenizer,
                                           block_size=self.block_size, mask_prob=self.mask_prob)


class CIFARDataModule(BaseDataModule):
    def __init__(self, 
                 data_path:str,
                 type:str="cifar10",
                 aa="rand-m9-mstd0.5-inc1",
                 reprob=0.25,
                 remode="pixel",
                 recount=1,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        assert type in ['cifar10', 'cifar100'], "Cifar dataset type must be in ['cifar10', 'cifar100']."
        self.type = type
        self.aa = aa
        self.reprob = reprob
        self.remode = remode
        self.recount = recount

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()
        if not hasattr(self, 'val_dataset'):
            self.set_val_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = DATASET_REGISTRY[self.type](data_path=self.data_path, split='train',
                                                         aa=self.aa, reprob=self.reprob, remode=self.remode, recount=self.recount)

    def set_val_dataset(self):
        self.val_dataset = DATASET_REGISTRY[self.type](data_path=self.data_path, split='test',
                                                        aa=self.aa, reprob=self.reprob, remode=self.remode, recount=self.recount)


class ImageNetDataModule(BaseDataModule):
    def __init__(
            self,
            data_path:str,
            pretraining,
            color_jitter=None,
            aa="rand-m9-mstd0.5-inc1",
            reprob=0.25,
            remode="pixel",
            recount=1,
            beit_transforms:bool=False,
            crop_scale:Tuple[float, float]=(0.08, 1.0),
            *args,
            **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.pretraining = pretraining
        self.color_jitter = color_jitter
        self.aa = aa
        self.reprob = reprob
        self.remode = remode
        self.recount = recount
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()
        if not hasattr(self, 'val_dataset'):
            self.set_val_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = ImageNetDataset(data_path=self.data_path, 
                                             split='train',
                                             pretraining=self.pretraining,
                                             color_jitter=self.color_jitter,
                                             aa=self.aa,
                                             reprob=self.reprob,
                                             remode=self.remode,
                                             recount=self.recount,
                                             beit_transforms=self.beit_transforms,
                                             crop_scale=self.crop_scale,
                                             )

    def set_val_dataset(self):
        self.val_dataset = ImageNetDataset(data_path=self.data_path,
                                           split='val',
                                           pretraining=self.pretraining,
                                           )


UNIMODAL_DATAMODULE_REGISTRY = {
    'masked_lm': MaskedLMDataModule,
    'cifar10': partial(CIFARDataModule, type='cifar10'),
    'cifar100': partial(CIFARDataModule, type='cifar100'),
    'imagenet': ImageNetDataModule,
}
