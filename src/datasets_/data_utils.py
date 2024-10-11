from torchvision.transforms import v2 as transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import logging
import json
from typing import Tuple
import PIL
from timm.data.transforms import RandomResizedCropAndInterpolation

logger = logging.getLogger(__name__)

def write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    logger.info("Write %s with %d items !" % (jsonl_file, len(items)))

def get_transforms(
        pretraining,
        train,
        size:int=224,
        color_jitter=None,
        aa="rand-m9-mstd0.5-inc1",
        reprob=0.25,
        remode="pixel",
        recount=1,
        beit_transforms:bool=False,
        crop_scale:Tuple[float, float]=(0.08, 1.0)):
    
    if pretraining:
        return get_transforms_pretraining(
            train=train,
            size=size,
            beit_transforms=beit_transforms,
            color_jitter=color_jitter,
            crop_scale=crop_scale
        )
    else:
        return get_transforms_finetuning(
            train=train,
            size=size,
            color_jitter=color_jitter,
            aa=aa,
            reprob=reprob,
            remode=remode,
            recount=recount)


def get_transforms_pretraining(
    train:bool=True,
    size:int=224,
    beit_transforms:bool=False,
    color_jitter=None,
    crop_scale:Tuple[float, float]=(0.08, 1.0)):

    transform_prepare = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
            ]
    )

    if not train:
        transform_train = transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC)
    elif beit_transforms:
        beit_transform_list = []
        # beit_transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4))
        beit_transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolation(
                    size=(size, size),
                    scale=crop_scale,
                    interpolation="bicubic",
                ),
            ]
        )
        transform_train = transforms.Compose(beit_transform_list)
    else:
        transform_train_list = [
            transforms.RandomResizedCrop(
                size=(size, size), scale=crop_scale, interpolation=3
            ),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
        ]
        if color_jitter is not None:
            transform_train_list.append(
                transforms.ColorJitter(color_jitter, color_jitter, color_jitter)
            )
        transform_train = transforms.Compose(transform_train_list)
    
    final_transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
    )

    return transforms.Compose([transform_prepare, transform_train, final_transform])

def get_transforms_finetuning(
        train,
        size:int=224,
        color_jitter=None,
        aa="rand-m9-mstd0.5-inc1",
        reprob=0.25,
        remode="pixel",
        recount=1):

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation="bicubic",
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
            mean=mean,
            std=std,
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
        transforms.Normalize(mean, std)
    ]
    return transforms.Compose(t)
