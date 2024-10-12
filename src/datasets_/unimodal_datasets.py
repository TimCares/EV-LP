import os
import logging
import json
from typing import *
from functools import partial
from .data_utils import write_data_into_jsonl
from torchvision.datasets import CIFAR10, CIFAR100
from .base_datasets import ImageDataset
from utils import Modality
from .imagenet_classes import IMAGENET2012_CLASSES

logger = logging.getLogger(__name__)

class ImageNetDataset(ImageDataset):
    def __init__(
            self,
            data_path:str,
            split,
            pretraining,
            color_jitter=None,
            aa="rand-m9-mstd0.5-inc1",
            reprob=0.25,
            remode="pixel",
            recount=1,
            beit_transforms:bool=False,
            crop_scale:Tuple[float, float]=(0.08, 1.0),
    ):
        super().__init__(
            data_path=data_path, 
            split=split,
            pretraining=pretraining,
            color_jitter=color_jitter,
            aa=aa,
            reprob=reprob,
            remode=remode,
            recount=recount,
            beit_transforms=beit_transforms,
            crop_scale=crop_scale,)
        self.path_to_data = os.path.join(self.data_path, 'imagenet')
        if not os.path.exists(self.path_to_data):
            raise FileNotFoundError(f"Directory {self.path_to_data} does not exists, "
                                    "please create it and add the correponding files from HuggingFace: "
                                    f"https://huggingface.co/datasets/imagenet-1k")
        
        self.path_to_split = os.path.join(self.path_to_data, self.split)
        os.makedirs(self.path_to_split, exist_ok=True)

        self.classes = {synset: i for i, synset in enumerate(IMAGENET2012_CLASSES.keys())}

        if not os.path.exists(os.path.join(self.path_to_data, f'imagenet.{self.split}.jsonl')):
            self._make_imagenet_dataset_index()


    def load(self):
        items = []
        with open(os.path.join(self.path_to_data, f'imagenet.{self.split}.jsonl'), 'r', encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                items.append(data)
            self.log(f"Loaded {len(items)} {self.split} examples.")
        self.items = items

    def __getitem__(self, index):
        item = self.items[index]
        image = self._get_image(image_path=item['image_path'])
        data = {
            'image': image,
            'id': index,
            'target': item['target']
        }
        return data
    
    def _make_imagenet_dataset_index(self):
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

    
class CIFARDataset(ImageDataset):
    def __init__(self, 
                 data_path:str,
                 split:str,
                 type:str="cifar10",
                 aa="rand-m9-mstd0.5-inc1",
                 reprob=0.25,
                 remode="pixel",
                 recount=1,
                 ):
        super().__init__(
            data_path=data_path, 
            split=split,
            pretraining=False,
            aa=aa,
            reprob=reprob,
            remode=remode,
            recount=recount,)
        self.type = type

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

    def __getitem__(self, index):
        item = self.items[index]
        return {"image": item[0], "target": item[1]}


UNIMODAL_DATASET_REGISTRY = {
    "imagenet": ImageNetDataset,
    "cifar10": partial(CIFARDataset, type='cifar10'),
    "cifar100": partial(CIFARDataset, type='cifar100'),
}
