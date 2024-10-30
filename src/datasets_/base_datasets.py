"""
This module contains the base classes and mixins for datasets.
"""
import os
import logging
import torch
import json
from typing import *
from torch import nn
from .data_utils import collater as default_collater
from .mixins import ImageMixin, TextMixin
import numpy as np
from utils import Modality
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path:os.PathLike,
        split:str,
    ):
        """Base class for all datasets.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            split (str): The split of the data. One of 'train', 'val', or 'test'.
        """
        assert split in ["train", "val", "test"], f"Split must be one of 'train', 'val', or 'test', found {split}"
        self.data_path = data_path
        self.split = split

        self.path_to_data = os.path.join(self.data_path, self.data_dir)
        self.path_to_split = os.path.join(self.path_to_data, self.split)
        # self.path_to_split is deepest directory, so makedirs will create all parent directories
        # if they do not exist, e.g. self.path_to_data
        # therefore, just this one call is sufficient
        os.makedirs(self.path_to_split, exist_ok=True)

    def get_index_files(self) -> Tuple[str]:
        """Returns a tuple of strings, where each string is the name of an index file containing the data
        for the split of the dataset.

        Returns:
            Tuple[str]: Tuple of strings, where each string is the name of an index file containing the data.
        """        
        raise NotImplementedError()
    
    def index_exists(self) -> bool:
        """Check if the index files for the current split exist in the data path.

        Returns:
            bool: True if all index files of the current split exist, False otherwise.
        """        
        for index_file in self.get_index_files():
            if not os.path.exists(os.path.join(self.path_to_data, index_file)):
                return False
        self.log(f"Data already exists under: {self.path_to_data}")
        return True
    
    def create_index(self) -> None:
        """
        Create the index files for the current split.
        It should write the index files to self.path_to_data.
        """        
        raise NotImplementedError

    def load(self) -> None:
        """
        Load the data of the current split from the index files into memory. After calling this method, the
        dataset is ready to be used.
        """        
        index_files = self.get_index_files()
        items = []
        self.index_files = index_files

        offset = 0
        # we can have multiple index files for a split
        # items are appended to the list in the order they appear in the index files
        for _index_file in index_files:
            index_file = os.path.join(self.path_to_data, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                self.log("Load %d examples from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
    
    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """        
        return len(self.items)
    
    @property
    def data_dir(self) -> str:
        """
        Name of the directory in self.data_path where the data is stored.
        """        
        raise NotImplementedError
    
    @property
    def modality(self) -> Modality:
        """
        The modality of the dataset.
        """        
        raise NotImplementedError

    def collater(self, samples:List[Dict[str, torch.Tensor|np.ndarray|Iterable]]) -> Dict[str, torch.Tensor]:
        """Batches a set of items, where each item is a dictionary of tensors, numpy arrays, or iterable objects.
        All items should have the exact same keys.

        Args:
            samples (List[Dict[str, torch.Tensor|np.ndarray|Iterable]]): The items to batch. Each item must be a dictionary.

        Returns:
            Dict[str, torch.Tensor]: The batched items. The result is a dictionary of tensors, with the same keys as the input items.
        """        
        return default_collater(samples)
    
    def log(self, msg:str) -> None:
        """Utility method to log messages with the class name. Helps identify where the log message is coming from.

        Args:
            msg (str): The message to log.
        """        
        logger.info(f"[{self.__class__.__name__}]: {msg}")    

class ImageTextDataset(ImageMixin, TextMixin, BaseDataset):
    def __init__(
        self,
        data_path:os.PathLike,
        split:str,
        transforms:Dict[str, nn.Module]=None,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=64,
        text_token_mask_prob:float=0.0,
    ):
        """The base class for image-text datasets.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            split (str): The split of the data. One of 'train', 'val', or 'test'.
            transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to image data.
                get_image will return a dictionary with as many image tensors as there are keys in this dictionary.
                Keys in the dictionary returned get_image containing image data will be named "<key>_image".
                If None, no transforms/augmentations will by applied. Defaults to None.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data.
                Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default:
                BertTokenizer.from_pretrained("bert-base-uncased")
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 64.
            mlm_probability (float, optional): The probability of masking a token in the text data. Defaults to 0.0, so no masking is done.
        """        
        super().__init__(
            data_path=data_path, 
            split=split,
            transforms=transforms,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mlm_probability=text_token_mask_prob,
        )
        
    @property
    def modality(self) -> Modality:
        return Modality.VL

    def __getitem__(self, index:int) -> Dict[str, Any]:
        data = dict()

        item = self.items[index]
        data["id"] = item["id"]

        img_path = item["image_path"]
        image_dict = self.get_image(img_path)
        data.update(image_dict) # add all image tensors to the data dictionary

        text = item["text"]
        # always do whole word masking for text, as this will force the model to rely more on image patches for context
        # actually only relevant if mlm_probability > 0.0
        text_dict = self.get_text(text, whole_word_mask=True)
        data.update(text_dict) # add all text tensors to the data dictionary

        return data
    
    def collater(self, samples):
        batch_tensors = super().collater(samples)
        if self.mlm_probability > 0.0:
            batch_tensors['targets'] = batch_tensors['text'].clone()
            batch_tensors['masked_text'] = batch_tensors['text'].clone()
            batch_tensors['masked_text'][batch_tensors['mask_labels'].bool()] = self.tokenizer.mask_token_id
            batch_tensors['targets'][~batch_tensors['mask_labels'].bool()] = -100 # -100 is default ignore_index in cross-entropy loss

            batch_tensors['mlm_padding_mask'] = batch_tensors['padding_mask'].long() | batch_tensors['mask_labels'].long()
        return batch_tensors
