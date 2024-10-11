import os
import logging
import torch
import json
from typing import *
import numpy as np
from collections import namedtuple
from .data_utils import get_transforms
import random
from data2vec_fairseq.data.modality import Modality
from utils import pad_text_sequence
from torchvision.datasets.folder import default_loader
from transformers import BertTokenizer
from beit2.datasets import DataAugmentationForBEiT

logger = logging.getLogger(__name__)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path:str, split:str):
        self.data_path = data_path
        self.split = split

        self.tokenizer:BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

    def load(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.items)
    
    def tokenize_text(self, text:str):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
    
    @property
    def modality(self) -> Modality:
        raise NotImplementedError
    
    # adjusted from transformers.data.data_collator.DataCollatorForWholeWordMask
    def whole_word_mask(self, input_tokens: List[str], mlm_probability=0.15):
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = max(1, int(round(len(input_tokens) * mlm_probability)))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def collater(self, samples):
        batch_tensors = {}
        for tensor_key in samples[0]:
            if isinstance(samples[0][tensor_key], torch.Tensor):
                batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in samples])
            elif isinstance(samples[0][tensor_key], np.ndarray):
                batch_tensors[tensor_key] = torch.from_numpy(np.stack([d[tensor_key] for d in samples]))
            else:
                batch_tensors[tensor_key] = torch.tensor([d[tensor_key] for d in samples], dtype=torch.long)

        batch_tensors['modality'] = self.modality
        return batch_tensors
    
    def log(self, msg:str):
        logger.info(f"[{self.__class__.__name__}]: {msg}")

class ImageDataset(BaseDataset):
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
            crop_scale:Tuple[float, float]=(0.08, 1.0),):
        super().__init__(data_path=data_path,
                         split=split,)
        self.pretraining = pretraining
        self.color_jitter = color_jitter
        self.aa = aa
        self.reprob = reprob
        self.remode = remode
        self.recount = recount

        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale

        self.loader = default_loader

        self.transform = get_transforms(
            pretraining=self.pretraining,
            train=self.split=="train",
            color_jitter=self.color_jitter,
            aa=self.aa,
            reprob=self.reprob,
            remode=self.remode,
            recount=self.recount,
            beit_transforms=self.beit_transforms,
            crop_scale=self.crop_scale,
        )

    def _get_image(self, image_path: str):
        image = self.loader(image_path)
        return self.transform(image)

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE
    

class BaseImageText(ImageDataset):
    def __init__(
        self,
        data_path,
        split,
        num_max_bpe_tokens,
        color_jitter=None,
        beit_transforms=False,
        crop_scale=(0.6, 1.0),
        text_token_mask_prob=0.0,
    ):
        super().__init__(
            data_path=data_path, 
            split=split,
            pretraining=True,
            color_jitter=color_jitter,
            beit_transforms=beit_transforms,
            crop_scale=crop_scale,)
        
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.text_token_mask_prob = text_token_mask_prob
        self.path_to_data = None

        BeitTransformsArgs = namedtuple('BeitTransformsArgs', ['imagenet_default_mean_and_std', 'input_size',
                                        'second_input_size', 'min_crop_scale', 'train_interpolation',
                                        'second_interpolation',],)
        transforms_args = BeitTransformsArgs(imagenet_default_mean_and_std=True, input_size=224, second_input_size=None,
                                             min_crop_scale=0.9, train_interpolation='bicubic', second_interpolation='bicubic')
        
        self.transform_teacher = DataAugmentationForBEiT(transforms_args)
        
    @property
    def modality(self) -> Modality:
        return Modality.VL
        
    def index_exists(self, dataset_path):
        for index_file in self.get_index_files():
            if not os.path.exists(os.path.join(dataset_path, index_file)):
                return False
        self.log(f"Data already exists under: {dataset_path}")
        return True

    def load(self):
        index_files = self.get_index_files()
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(self.path_to_data, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                self.log("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items

    def get_index_files(self):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image = self.loader(image_path)
        return self.transform(image), self.transform_teacher(image)

    def _get_text_segment(self, text_segment, max_len=None):
        assert isinstance(text_segment, list)
        tokens = text_segment
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        if self.text_token_mask_prob > 0.0:
            mask_labels = self.whole_word_mask(tokens, mlm_probability=self.text_token_mask_prob)
            mask_labels = [0] + mask_labels + [0] * (max_len - len(mask_labels) - 1)
        else:
            mask_labels = None
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=max_len,
                                                          pad_idx=self.pad_token_id, bos_idx=self.cls_token_id,
                                                          eos_idx=self.sep_token_id)
        

        return language_tokens, padding_mask, mask_labels

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img, img_teacher = self._get_image(img_path)
        data["image"] = img
        data["image_teacher"] = img_teacher
        data["id"] = item["id"]

        text_segment = item["text"]
        language_tokens, padding_mask, mask_labels = self._get_text_segment(text_segment)
        data["text"] = language_tokens
        data["padding_mask"] = padding_mask
        if self.text_token_mask_prob > 0.0:
            data["mask_labels"] = mask_labels

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data
    
    def collater(self, samples):
        batch_tensors = super().collater(samples)
        if self.text_token_mask_prob > 0.0:
            batch_tensors['targets'] = batch_tensors['text'].clone()
            batch_tensors['masked_text'] = batch_tensors['text'].clone()
            batch_tensors['masked_text'][batch_tensors['mask_labels'].bool()] = self.mask_token_id
            batch_tensors['targets'][~batch_tensors['mask_labels'].bool()] = -100

            batch_tensors['mlm_padding_mask'] = batch_tensors['padding_mask'].long() | batch_tensors['mask_labels'].long()
        return batch_tensors
