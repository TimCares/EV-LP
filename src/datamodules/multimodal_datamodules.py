"""
This module contains the data modules for the multimodal datasets COCO Captions, Flickr30K, and Conceptual Captions.
"""
import os
from typing import Dict, Union, List, Any
from torch import nn
from .unimodal_datamodules import BaseDataModule
from datasets_ import COCOCaptions, Flickr30K, ConceptualCaptions
from registries import register_datamodule
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from utils import Modality

@register_datamodule(name='COCOCaptions')
class COCOCaptionsDataModule(BaseDataModule):
    def __init__(self,
        data_path:os.PathLike,
        train_transforms:Dict[str, nn.Module]=None,
        eval_transforms:Dict[str, nn.Module]=None,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=64,
        text_token_mask_prob:float=0.0,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """Data module for the COCO Captions dataset.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            train_transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to training image data.
                If None, no augmentation will be applied. Defaults to None.
            eval_transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to validation/test image data.
                If None, no augmentation will be applied. Defaults to None.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data.
                Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default in the COCOCaptions dataset:
                BertTokenizer.from_pretrained("bert-base-uncased")
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 64.
            text_token_mask_prob (float, optional): The probability of masking a token in the captions. Defaults to 0.0, so no masking is done.
            *args (List[Any]): Additional arguments for the BaseDataModule.
            **kwargs (Dict[str, Any): Additional keyword arguments for the BaseDataModule. Usually arguments for the DataLoader.
        """        
        super().__init__(data_path, *args, **kwargs)
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_token_mask_prob = text_token_mask_prob

    @property
    def modality(self) -> Modality:
        return Modality.VL

    def set_train_dataset(self):
        self.train_dataset = COCOCaptions(
            data_path=self.data_path,
            split='train',
            transforms=self.train_transforms,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            text_token_mask_prob=self.text_token_mask_prob,
        )

    def set_val_dataset(self):
        self.val_dataset = COCOCaptions(
            data_path=self.data_path,
            split='val',
            transforms=self.eval_transforms,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            text_token_mask_prob=self.text_token_mask_prob,
        )

    def set_test_dataset(self):
        self.test_dataset = COCOCaptions(
            data_path=self.data_path,
            split='test',
            transforms=self.eval_transforms,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            text_token_mask_prob=self.text_token_mask_prob,
        )
        

class ConceptualCaptionsDataModule(BaseDataModule):
    def __init__(self,
        data_path:os.PathLike,
        transforms:Dict[str, nn.Module]=None,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=64,
        text_token_mask_prob:float=0.0,
        type:str="cc3m",
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """Data module for the Conceptual Captions dataset. Only supports the 'train' split.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to training image data.
                If None, no augmentation will be applied. Defaults to None.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data.
                Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default in the Conceptual Captions dataset:
                BertTokenizer.from_pretrained("bert-base-uncased")
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 64.
            text_token_mask_prob (float, optional): The probability of masking a token in the captions. Defaults to 0.0, so no masking is done.
            type (str, optional): The type of the Conceptual Captions dataset. One of 'cc3m' or 'cc12m'. Defaults to 'cc3m'.
            *args (List[Any]): Additional arguments for the BaseDataModule.
            **kwargs (Dict[str, Any): Additional keyword arguments for the BaseDataModule. Usually arguments for the DataLoader.
        """     
        super().__init__(data_path, *args, **kwargs)
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_token_mask_prob = text_token_mask_prob
        self.type = type

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = ConceptualCaptions(
            type=self.type,
            data_path=self.data_path,
            split='train',
            transforms=self.transforms,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            text_token_mask_prob=self.text_token_mask_prob,
        )


@register_datamodule(name='Flickr30K')
class Flickr30KDataModule(BaseDataModule):
    def __init__(self,
        data_path:os.PathLike,
        train_transforms:Dict[str, nn.Module]=None,
        eval_transforms:Dict[str, nn.Module]=None,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=64,
        text_token_mask_prob:float=0.0,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """Data module for the Flickr30K dataset.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            train_transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to training image data.
                If None, no augmentation will be applied. Defaults to None.
            eval_transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to validation/test image data.
                If None, no augmentation will be applied. Defaults to None.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data.
                Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default in the Flickr30K dataset:
                BertTokenizer.from_pretrained("bert-base-uncased")
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 64.
            text_token_mask_prob (float, optional): The probability of masking a token in the captions. Defaults to 0.0, so no masking is done.
            *args (List[Any]): Additional arguments for the BaseDataModule.
            **kwargs (Dict[str, Any): Additional keyword arguments for the BaseDataModule. Usually arguments for the DataLoader.
        """     
        super().__init__(data_path, *args, **kwargs)
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_token_mask_prob = text_token_mask_prob

    def set_train_dataset(self):
        self.train_dataset = Flickr30K(
            data_path=self.data_path,
            split='train',
            transforms=self.train_transforms,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            text_token_mask_prob=self.text_token_mask_prob,
        )

    def set_val_dataset(self):
        self.val_dataset = Flickr30K(
            data_path=self.data_path,
            split='val',
            transforms=self.eval_transforms,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            text_token_mask_prob=self.text_token_mask_prob,
        )

    def set_test_dataset(self):
        self.test_dataset = Flickr30K(
            data_path=self.data_path,
            split='test',
            transforms=self.eval_transforms,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            text_token_mask_prob=self.text_token_mask_prob,
        )


@register_datamodule(name="ConceptualCaptions3m")
def conceptual_captions_cc3m(*args, **kwargs):
    return ConceptualCaptionsDataModule(*args, type="cc3m", **kwargs)

@register_datamodule(name="ConceptualCaptions12m")
def conceptual_captions_cc12m(*args, **kwargs):
    return ConceptualCaptionsDataModule(*args, type="cc12m", **kwargs)
