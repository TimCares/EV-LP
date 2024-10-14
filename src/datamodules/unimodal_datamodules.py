"""
This module contains the data modules for the unimodal datasets ImageNet, CIFAR-10, CIFAR-100, and the generic MaskedLM dataset.
"""
from pytorch_lightning import LightningDataModule
from torch import nn
from torch.utils.data import DataLoader
from typing import Union, Dict, Any, List
from datasets_ import ImageNetDataset, MaskedLMDataset
from registries import register_datamodule, DATASET_REGISTRY
from utils import Modality
import os
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_path:os.PathLike,
        batch_size:int,
        num_workers:int,
        shuffle:bool=True,
        drop_last:bool=True,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """Base class for all data modules.

        Args:
            data_path (os.PathLike): Path to the data.
            batch_size (int): The batch size.
            num_workers (int): Number of workers for the data loader.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            drop_last (bool, optional): Whether to drop the last batch if it is < batch_size. Defaults to True.
        """        
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def set_train_dataset(self) -> None:
        """
        Sets the training dataset. As this will call the constructor of the train dataset in the subtype,
        it automatically generates the index, and therefore the data, of the training dataset.
        """        
        raise NotImplementedError("set train dataset")

    def set_val_dataset(self) -> None:
        """
        Sets the validation dataset. As this will call the constructor of the val dataset in the subtype,
        it automatically generates the index, and therefore the data, of the val dataset.
        """        
        pass # optional: not all datasets have a validation set

    def set_test_dataset(self) -> None:
        """
        Sets the test dataset. As this will call the constructor of the test dataset in the subtype,
        it automatically generates the index, and therefore the data, of the test dataset.
        """        
        pass # optional: not all datasets have a test set

    @property
    def modality(self) -> Modality:
        raise NotImplementedError

    def prepare_data(self) -> None:
        """
        Prepares the data, meaning the data is generated. This can include downloading the data, creating the index, etc.
        What exactly happens is speficic to the underlying dataset.
        """        
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()
        if not hasattr(self, 'val_dataset'):
            self.set_val_dataset()
        if not hasattr(self, 'test_dataset'):
            self.set_test_dataset()

    def setup(self, stage:str=None) -> None:
        """Loads the index of all datasets that are used in the respective stage.
        After this, the dataloaders can be used -> train_dataloader, val_dataloader, test_dataloader.

        Args:
            stage (str, optional): For which stage the data should be used. Usually "fit" for train and val dataset,
                and "test" for the test dataset. If None, then train, val, and test dataset are loaded. Defaults to None.
        """        
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """        
        return DataLoader(self.train_dataset,
                          collate_fn=self.train_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,)

    def val_dataloader(self) -> Union[DataLoader, None]:
        """Returns the validation dataloader. Shuffling is always disabled to ensure reproducibility.

        Returns:
            Union[DataLoader, None]: The validation dataloader. If the dataset does not have a validation set, None is returned.
        """        
        if not hasattr(self, 'val_dataset'):
            return None
        return DataLoader(self.val_dataset,
                          collate_fn=self.val_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=False,
                          drop_last=False,)

    def test_dataloader(self) -> Union[DataLoader, None]:
        """Returns the test dataloader. Shuffling is always disabled to ensure reproducibility.

        Returns:
            Union[DataLoader, None]: The test dataloader. If the dataset does not have a test set, None is returned.
        """        
        if not hasattr(self, 'test_dataset'):
            return None
        return DataLoader(self.test_dataset,
                          collate_fn=self.test_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=False,
                          drop_last=False,)
    
    def teardown(self, stage:str=None) -> None:
        """Destroys the datasets to free up memory. How much memory is freed up depends on the index of the dataset,
        which is usually the driving factor of the memory consumption, as it is loaded into memory when the dataset is loaded.

        Args:
            stage (str, optional): For which stage the dataset should be destroyed. Usually "fit" for train and val dataset,
                and "test" for the test dataset. If None, then train, val, and test dataset are destroyed. Defaults to None.
        """        
        if stage == 'fit' or stage is None:
            if hasattr(self, 'train_dataset'):
                del self.train_dataset
            if hasattr(self, 'val_dataset'):
                del self.val_dataset
        if stage == 'test' or stage is None:
            if hasattr(self, 'test_dataset'):
                del self.test_dataset


@register_datamodule(name='MaskedLM')
class MaskedLMDataModule(BaseDataModule):
    def __init__(
        self,
        data_path:os.PathLike,
        name:str,
        text_file:os.PathLike,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        mask_prob:float=0.0,
        block_size:int=512,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """Data module for the MaskedLM dataset.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            name (str): The name of the dataset, used for saving the index files.
            text_file (os.PathLike): The path to the text file containing the raw text data. Should have the postfix '.<split>'.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data. Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default in the MaskedLMDataset:
                BertTokenizer.from_pretrained("bert-base-uncased").
            mask_prob (float, optional): The probability of masking a token in the text data. Defaults to 0.0, so no masking is done.
            block_size (int, optional): How many tokens should be in one block. One block is equal to one training example, i.e.
                a single text sequence. This is the maximum sequence length. The text file will be sliced into chunks
                of <block_size> tokens. Defaults to 512.
            *args (List[Any]): Additional arguments for the BaseDataModule.
            **kwargs (Dict[str, Any): Additional keyword arguments for the BaseDataModule. Usually arguments for the DataLoader.
        """        
        super().__init__(data_path, *args, **kwargs)
        self.name = name
        self.train_text_file = text_file + '.train'
        self.val_text_file = text_file + '.val'
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.block_size = block_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = MaskedLMDataset(
            name=self.name,
            data_path=self.data_path,
            split='train',
            text_file=self.train_text_file,
            tokenizer=self.tokenizer,
            mask_prob=self.mask_prob,
            block_size=self.block_size, 
        )
        
    def set_val_dataset(self):
        self.val_dataset = MaskedLMDataset(
            name=self.name,
            data_path=self.data_path,
            split='val',
            text_file=self.val_text_file,
            tokenizer=self.tokenizer,
            mask_prob=self.mask_prob,
            block_size=self.block_size,
        )


class CIFARDataModule(BaseDataModule):
    def __init__(self, 
        data_path:os.PathLike,
        type:str,
        train_transform:nn.Module=None,
        eval_transform:nn.Module=None,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """Data module for the CIFAR-10 and CIFAR-100 datasets.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            type (str): Whether to use CIFAR-10 or CIFAR-100. One of ["cifar10", "cifar100"].
            train_transform (nn.Module, optional): PyTorch transforms to apply to training image data.
                If None, no augmentation will be applied. Defaults to None.
            eval_transform (nn.Module, optional): PyTorch transforms to apply to validation/test image data.
                If None, no augmentation will be applied. Defaults to None.
            *args (List[Any]): Additional arguments for the BaseDataModule.
            **kwargs (Dict[str, Any): Additional keyword arguments for the BaseDataModule. Usually arguments for the DataLoader.
        """        
        super().__init__(data_path, *args, **kwargs)
        assert type in ['cifar10', 'cifar100'], "Cifar dataset type must be in ['cifar10', 'cifar100']."
        self.type = type
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = DATASET_REGISTRY[self.type](
            data_path=self.data_path,
            split='train',
            transform=self.train_transform,
        )

    def set_val_dataset(self):
        self.val_dataset = DATASET_REGISTRY[self.type](
            data_path=self.data_path,
            split='test', # we use the test split as validation set -> CIFAR-10/100 do not have a validation set
            transform=self.eval_transform,
        )


@register_datamodule(name='ImageNet')
class ImageNetDataModule(BaseDataModule):
    def __init__(
        self,
        data_path:os.PathLike,
        train_transforms:Dict[str, nn.Module]=None,
        eval_transforms:Dict[str, nn.Module]=None,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """_summary_

        Args:
            data_path (os.PathLike): The path where the data is stored.
            train_transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to training image data.
                If None, no augmentation will be applied. Defaults to None.
            eval_transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to validation/test image data.
                If None, no augmentation will be applied. Defaults to None.
        """        
        super().__init__(data_path, *args, **kwargs)
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms

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
        self.train_dataset = ImageNetDataset(
            data_path=self.data_path, 
            split='train',
            transforms=self.train_transforms,
        )

    def set_val_dataset(self):
        self.val_dataset = ImageNetDataset(
            data_path=self.data_path,
            split='val',
            transforms=self.eval_transforms,
        )


@register_datamodule(name="CIFAR-10")
def cifar_10(*args, **kwargs):
    return CIFARDataModule(*args, type="cifar10", **kwargs)

@register_datamodule(name="CIFAR-100")
def cifar_100(*args, **kwargs):
    return CIFARDataModule(*args, type="cifar100", **kwargs)
