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
import numpy as np
from utils import Modality
from utils import pad_text_sequence
from torchvision.datasets.folder import default_loader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BertTokenizer
import random

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

class ImageMixin:
    def __init__(
        self,
        transforms:Dict[str, nn.Module]=None,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """A Mixin designed to add image support to a dataset.

        Args:
            transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to image data.
                get_image will return a dictionary with as many image tensors as there are keys in this dictionary.
                Keys in the dictionary returned get_image containing image data will be named "<key>_image".
                If None, no transforms/augmentations will by applied. Defaults to None.
            *args (List[Any]): Positional arguments passed to other Mixins, if present. Defaults to None.
            **kwargs (Dict[str, Any]): Keyword arguments passed to other Mixins, if present. Defaults to None.
        """        
        super().__init__(*args, **kwargs)
        if transforms is None:
            transforms = {
                "": nn.Identity(), # resulting key in the dictionary returned by get_image will be "image"
            }
        self.transforms = transforms
        self.loader = default_loader


    def get_image(self, image_path:str) -> Dict[str, torch.Tensor]:
        out_dict = dict()
        image = self.loader(image_path)
        for transform_name, transform in self.transforms.items():
            if transform_name != "":
                transform_name = f"{transform_name}_"
            out_dict[transform_name + "image"] = transform(image)
        return out_dict
    
class TextMixin:
    def __init__(
        self,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=512,
        mlm_probability:float=0.0,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """A Mixin designed to add text/tokenization and text masking support to a dataset.

        Args:
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data. Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default:
                BertTokenizer.from_pretrained("bert-base-uncased").
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 512.
            mlm_probability (float, optional): The probability of masking a token in the text data. Defaults to 0.0, so no masking is done.
            *args (List[Any]): Positional arguments passed to other Mixins, if present. Defaults to None.
            **kwargs (Dict[str, Any]): Keyword arguments passed to other Mixins, if present. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.tokenizer_path = ".tokenizers"
        self.tokenizer:PreTrainedTokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if tokenizer is None else tokenizer
        self.tokenizer.save_pretrained(self.tokenizer_path)
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability

    def get_text(self, text:Union[str, List[str]], whole_word_mask:bool=True) -> Dict[str, List[int]]:
        """Tokenizes the text, masks tokens (optional), and pads it to the maximum sequence length.

        Args:
            text (Union[str, List[str]]): The text. Either already tokenized as a list of subwords, or the raw text as a string.
            whole_word_mask (bool, optional): Whether to mask whole words or individual tokens. Ignored if mlm_probability == 0. Defaults to True.

        Returns:
            Dict[str, List[int]]: A dictionary containing the tokenized text, the padding mask, and the mask labels.
        """
        tokens = self.tokenize_text(text)
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one token!")
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2] # truncate the tokens to the maximum sequence length, -2 because of [CLS] and [SEP]
        mask_labels = self.mask_sequence(input_tokens=tokens, whole_word_mask=whole_word_mask)

        # pad the mask labels to the maximum sequence length
        # first token will be [CLS], that is why we prepend [0]
        # after last token, there will be a [SEP] and the rest will be padded, both padded tokens and [SEP] will never be masked,
        # that is why we append as many 0 as needed until the maximum sequence length is reached
        mask_labels = [0] + mask_labels + [0] * (self.max_seq_len - len(mask_labels) - 1)

        language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.max_seq_len,
                                                          pad_idx=self.tokenizer.pad_token_id, bos_idx=self.tokenizer.cls_token_id,
                                                          eos_idx=self.tokenizer.sep_token_id)
        return {
            "text": language_tokens,
            "padding_mask": padding_mask,
            "mask_labels": mask_labels,
        }

    def tokenize_text(self, text:Union[str, List[str]]) -> Union[int, List[int]]:
        """Wrapper for easy tokenization of text data. Converts text to tokens and then to token ids using the
        (HuggingFace) tokenizer provided during initialization. Tokens are not truncated or padded, and do not
        include special tokens like [CLS] or [SEP].

        Args:
            text (Union[str, List[str]]): The text. Either already tokenized as a list of subwords, or the raw text as a string.

        Returns:
            Union[int, List[int]]: Token ids of the text, without special tokens.
        """
        if isinstance(text, list):
            return self.tokenizer.convert_tokens_to_ids(text)
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text.strip())) # else
    
    def mask_sequence(self, input_tokens: List[int], whole_word_mask:bool=True) -> List[int]:
        """Mask tokens in the input sequence either completely independent for each other,
        or one whole words, so subwords together. Special tokens [CLS] and [SEP] are not masked.

        Args:
            input_tokens (List[int]): The input sequence of tokens, already tokenized and converted to token ids.
            whole_word_mask (bool, optional): Whether to mask whole words, setting to True is encouraged. Defaults to True.

        Returns:
            List[int]: The mask indicator for each token in the input sequence. 1 is mask, 0 is not masked.
        """        
        if self.mlm_probability == 0.0: # if no masking is required, return a list of zeros because no tokens are masked
            return [0] * len(input_tokens)
        if whole_word_mask:
            # sequence is already tokenized and converted to token ids, to detect which token is a subword,
            # we need to convert the tokens ids back to tokens (tokens that are subwords start with "##")
            input_tokens:List[str] = self.tokenizer.convert_ids_to_tokens(input_tokens)
            return self._whole_word_mask(input_tokens)
        return self._token_mask(input_tokens)

    def _token_mask(self, input_tokens: List[int]) -> List[int]:
        """Mask tokens in the input sequence with the given probability. Special tokens [CLS] and [SEP] are not masked.
        Sequence is expected to not be padded.

        Args:
            input_tokens (List[int]): The input sequence of tokens, already tokenized and converted to token ids.

        Returns:
            List[int]: The mask indicator for each token in the input sequence. 1 is mask, 0 is not masked.
        """        
        input_tokens = torch.Tensor(input_tokens)
        special_tokens_mask = input_tokens == self.tokenizer.sep_token_id or input_tokens == self.tokenizer.cls_token_id
        probabilities = torch.full(input_tokens.shape, self.mlm_probability)
        probabilities.masked_fill_(special_tokens_mask, value=0.0)
        return torch.bernoulli(probabilities).tolist()
    
    # adjusted from transformers.data.data_collator.DataCollatorForWholeWordMask
    def _whole_word_mask(self, input_tokens: List[str]) -> List[int]:
        """Performs whole word masking on a sequence.
        If we have a word that is split into multiple subwords e.g. "bicycle" -> ["bi", "##cycle"],
        and the first subword "bi" is masked, then the second subword "##cycle" will also be masked.
        Special tokens [CLS] and [SEP] are not masked.
        Sequence is expected to not be padded.

        Args:
            input_tokens (List[int]): The input sequence of tokens, has to be the raw tokens, not token ids.
                Subwords must start with "##".

        Returns:
            List[int]: The mask indicator for each token in the input sequence. 1 is mask, 0 is not masked.
        """         
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = max(1, int(round(len(input_tokens) * self.mlm_probability)))
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
