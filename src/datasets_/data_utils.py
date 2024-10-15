"""
This module provides utility functions for data processing and transformation.
"""
from torchvision.transforms import v2 as transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import logging
import json
from typing import Tuple, List, Dict, Union, Iterable, Any
import PIL
from timm.data.transforms import RandomResizedCropAndInterpolation
import numpy as np
import os

logger = logging.getLogger(__name__)

def write_data_into_jsonl(items:List[Dict[str, Any]], jsonl_file:os.PathLike) -> None:
    """Write a list of dictionaries into a jsonl file. Each dictionary
    can be thought of as a single item in the dataset, and will be written as one json object in one line.

    Args:
        items (List[Dict[str, Any]]): The items/dictionaries to save.
        jsonl_file (os.PathLike): Path to the jsonl file.
    """    
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    logger.info("Write %s with %d items !" % (jsonl_file, len(items)))

def get_transform(
    pretraining:bool,
    train:bool,
    size:int=224,
    color_jitter:float=0.4,
    aa="rand-m9-mstd0.5-inc1",
    reprob:float=0.25,
    remode:str="pixel",
    recount:int=1,
    crop_scale:Tuple[float, float]=(0.08, 1.0)
) -> transforms.Compose:
    """Creates a set of data transformations/augmentations for image pretraining or finetuning.

    Args:
        pretraining (bool): Whether the transformations should be used for pretraining (True) or finetuning (False).
            If finetuning, then the augmentations will be stronger, and contain methods like rand augment.
        train (bool): Whether the transformations should be used on the training set.
            Setting this to False excludes any augmentations, and will only resize the image and normalize it.
        size (int, optional): Which size the images should have. Defaults to 224.
        color_jitter (float, optional): The amount of color jitter to apply to each channel. The same amount
            is applied to all channels. Defaults to 0.4.
        aa (str, optional): Auto Augment configuration as a string, see https://timm.fast.ai/AutoAugment for more.
            Defaults to "rand-m9-mstd0.5-inc1".
        reprob (float, optional): Random erasing probability, see https://timm.fast.ai/RandomErase for more.
            Defaults to 0.25.
        remode (str, optional): Random erasing mode, see https://timm.fast.ai/RandomErase for more.
            Defaults to "pixel".
        recount (int, optional): How many regions to erase, see https://timm.fast.ai/RandomErase for more.
            Defaults to 1.
        crop_scale (Tuple[float, float], optional): The range in which the image should be cropped randomly. Expressed
            as a fraction: (<min_frac>, <max_frac>). Defaults to (0.08, 1.0), meaning between 8% and 100% of the image size.

    Returns:
        transforms.Compose: The set of transformations to be applied to an image.
    """    
    
    if pretraining:
        return get_transform_pretraining(
            train=train,
            size=size,
            color_jitter=color_jitter,
            crop_scale=crop_scale
        )
    else:
        return get_transform_finetuning(
            train=train,
            size=size,
            color_jitter=color_jitter,
            aa=aa,
            reprob=reprob,
            remode=remode,
            recount=recount)


def get_transform_pretraining(
    train:bool=True,
    size:int=224,
    color_jitter:float=0.4,
    crop_scale:Tuple[float, float]=(0.08, 1.0)
) -> transforms.Compose:
    """Creates a set of data transformations/augmentations for image pretraining.

    Args:
        train (bool, optional): Whether the transformations should be used on the training set.
            Setting this to False excludes augmentations like cropping, color jitter etc. Defaults to True.
        size (int, optional): Which size the images should have. Defaults to 224.
        color_jitter (float, optional): The amount of color jitter to apply to each channel. The same amount
            is applied to all channels. Defaults to 0.4.
        crop_scale (Tuple[float, float], optional): The range in which the image should be cropped randomly. Expressed
            as a fraction: (<min_frac>, <max_frac>). Defaults to (0.08, 1.0), meaning between 8% and 100% of the image size.

    Returns:
        transforms.Compose: The set of transformations to be applied to an image.
    """    

    # encouraged by torchvision docs for efficiency:
    transform_prepare = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
            ]
    )

    if train:
        transform_train = transforms.Compose([
                transforms.ColorJitter(color_jitter, color_jitter, color_jitter),
                transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolation(
                    size=(size, size),
                    scale=crop_scale,
                    interpolation="bicubic",
                ),
        ])
    else:
        transform_train = transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC)
    
    final_transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
    )

    return transforms.Compose([transform_prepare, transform_train, final_transform])

def get_transform_finetuning(
    train:bool,
    size:int=224,
    color_jitter:float=0.4,
    aa:str="rand-m9-mstd0.5-inc1",
    reprob:float=0.25,
    remode:str="pixel",
    recount:int=1
) -> transforms.Compose:
    """Creates a set of data transformations/augmentations for image finetuning.
    Suitable for supervised learning in general.

    Args:
        train (bool, optional): Whether the transformations should be used on the training set.
            Setting this to False excludes augmentations like cropping, color jitter etc. Defaults to True.
        size (int, optional): Which size the images should have. Defaults to 224.
        color_jitter (float, optional): The amount of color jitter to apply to each channel. The same amount
            is applied to all channels. Defaults to 0.4.
        aa (str, optional): Auto Augment configuration as a string, see https://timm.fast.ai/AutoAugment for more.
            Defaults to "rand-m9-mstd0.5-inc1".
        reprob (float, optional): Random erasing probability, see https://timm.fast.ai/RandomErase for more.
            Defaults to 0.25.
        remode (str, optional): Random erasing mode, see https://timm.fast.ai/RandomErase for more.
            Defaults to "pixel".
        recount (int, optional): How many regions to erase, see https://timm.fast.ai/RandomErase for more.
            Defaults to 1.

    Returns:
        transforms.Compose: The set of transformations to be applied to an image.
    """
    # train transform
    if train:
        transform = create_transform(
            input_size=size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation="bicubic",
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        return transform

    # eval transform
    t = [
        transforms.ToImage(),
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),
        transforms.CenterCrop(size),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    return transforms.Compose(t)

def create_transforms_from_config(cfg_list:List[Dict[str, Union[str, int, float]]], train:bool=True) -> List[transforms.Compose]:
    """Creates a set of data transformations/augmentations from a list of transform configurations.
    Allowed keys are "_name" (str, mandatory), and any other key that is a valid argument for the function
    `get_transform` of this module. 
    
    Args:
        cfg_list (List[Dict[str, Union[str, int, float]]]): The list of transforms to create. Each dictionary
            contains the name (key "_name") and parameters for a single transform pipeline (Compose).
        train (bool, optional): Whether the transformations should be used on the training set. Defaults to True.

    Returns:
        List[transforms.Compose]: The named of transformations given by the configuration list. One key wiht a
            Compose object per item in cfg_list.
    """
    transforms_list:Dict[str, transforms.Compose] = []
    for cfg in cfg_list:
        assert "_name" in cfg, "Each transform configuration must have a '_name' key."
        assert isinstance(cfg["_name"], str), "The '_name' key must be a string."
        name = cfg.pop("_name")
        cfg["train"] = train
        transforms_list[name] = get_transform(**cfg)
    return transforms_list

def collater(samples:List[Dict[str, Union[torch.Tensor, np.ndarray, Iterable]]]) -> Dict[str, torch.Tensor]:
    """Batches a set of items, where each item is a dictionary of tensors, numpy arrays, or iterable objects.
    The result is a dictionary of tensors, with the same keys as the input items.
    All items are expected to have the exact same(!) keys, and the values associated
    with each key are stacked in a new tensor.

    Args:
        samples (List[Dict[str, Union[torch.Tensor, np.ndarray, Iterable]]]): A list of items to batch.
            Each item must be a dictionary.

    Returns:
        Dict[str, torch.Tensor]: The batched items.
    """    
    batch_tensors = {}
    for tensor_key in samples[0]: # iterate over all keys
        if isinstance(samples[0][tensor_key], torch.Tensor):
            batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in samples])
        elif isinstance(samples[0][tensor_key], np.ndarray):
            batch_tensors[tensor_key] = torch.from_numpy(np.stack([d[tensor_key] for d in samples]))
        else:
            batch_tensors[tensor_key] = torch.tensor([d[tensor_key] for d in samples], dtype=torch.long)

    return batch_tensors
