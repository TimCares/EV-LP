"""
This module contains the implementation of the ImageNet, CIFAR-10, and CIFAR-100 datasets.
"""
from torch import nn
import os
import logging
from typing import *
from .data_utils import write_data_into_jsonl
from torchvision.datasets import CIFAR10, CIFAR100
from .base_datasets import ImageMixin, BaseDataset
from utils import Modality
from .imagenet_classes import IMAGENET2012_CLASSES
from registries import register_dataset

logger = logging.getLogger(__name__)

@register_dataset(name='ImageNet')
class ImageNetDataset(ImageMixin, BaseDataset):
    def __init__(
            self,
            data_path:os.PathLike,
            split:str,
            transforms:Dict[str, nn.Module]=None,
    ):
        """The ImageNet-1K dataset.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            split (str): The split of the data. One of 'train', 'val', or 'test'.
            transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to image data.
                If None, no transforms/augmentations will by applied. Defaults to None.
        """
        super().__init__(
            data_path=data_path, 
            split=split,
            transforms=transforms,)

        # maps each synset to a class index
        self.classes = {synset: i for i, synset in enumerate(IMAGENET2012_CLASSES.keys())}

        if not self.index_exists():
            self.create_index()

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE
    
    @property
    def data_dir(self) -> str:
        return 'imagenet'
    
    def get_index_files(self) -> Tuple[str]:
        """Returns a tuple of strings, where each string is the name of an index file containing the data
        for the split of the dataset.

        Returns:
            Tuple[str]: Tuple of strings, where each string is the name of an index file containing the data.
        """        
        return (f'imagenet.{self.split}.jsonl',)

    def __getitem__(self, index:int):
        item = self.items[index]
        data = {
            'id': index,
            'target': item['target']
        }
        img_path = item["image_path"]
        image_dict = self.get_image(img_path)
        data.update(image_dict)
        return data
    
    def create_index(self):
        items = []
        for file in os.listdir(self.path_to_split):
            if self.split != 'test':
                root, _ = os.path.splitext(file)
                _, synset_id = os.path.basename(root).rsplit("_", 1)
            else:
                synset_id = -1
            items.append({
                'image_path': os.path.join(self.path_to_split, file),
                'target': self.classes[synset_id],
            })

        write_data_into_jsonl(items, os.path.join(self.path_to_data, f'imagenet.{self.split}.jsonl'))


# we do not use the ImageMixin here, as we only ever apply a single transform to the images,
# and most of the functionality is already implemented in the torchvision datasets for CIFAR-10 and CIFAR-100
# this allows for a more concise implementation
# (as a side note: we sometimes need multiple transforms to be applied to the images, if we use multiple image models during training,
#  or also want, next to the augmented image, the original image. In this case, the ImageMixin is useful.)
class CIFARDataset(BaseDataset):
    def __init__(
        self, 
        data_path:os.PathLike,
        split:str,
        type:str,
        transform:nn.Module=None,
    ):
        """Implements the CIFAR-10 and CIFAR-100 datasets.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            split (str): The split of the data. One of 'train', 'val', or 'test'.
            type (str): Whether to use CIFAR-10 or CIFAR-100. One of ["cifar10", "cifar100"].
            transforms (nn.Module, optional): PyTorch transforms to apply to image data.
                If None, no transforms/augmentations will by applied. Defaults to None.

        Raises:
            ValueError: If an unknown dataset type is provided, i.e. not "cifar10" or "cifar100".
        """        
        super().__init__(
            data_path=data_path,
            split=split,)
        self.type = type
        self.transform = transform

        if self.type == "cifar10":
            CIFAR10(self.data_path, train=self.split == "train", download=True)
        elif self.type == "cifar100":
            CIFAR100(self.data_path, train=self.split == "train", download=True)
        else:
            raise ValueError(f'CIFARDataset: Unknown dataset type: {self.type}, available options: ["cifar10", "cifar100"].')
        
    @property
    def modality(self) -> Modality:
        return Modality.IMAGE

    def load(self):
        if self.type == "cifar10":
            self.items = CIFAR10(self.data_path, train=self.split == "train", transform=self.transform)
        else:
            self.items = CIFAR100(self.data_path, train=self.split == "train", transform=self.transform)

    def __getitem__(self, index:int):
        item = self.items[index]
        return {"image": item[0], "target": item[1], "id": index}

@register_dataset(name="CIFAR-10")
def cifar_10(*args, **kwargs):
    return CIFARDataset(*args, type="cifar10", **kwargs)

@register_dataset(name="CIFAR-100")
def cifar_100(*args, **kwargs):
    return CIFARDataset(*args, type="cifar100", **kwargs)
