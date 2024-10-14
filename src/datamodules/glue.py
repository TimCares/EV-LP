"""
This module defines an universal data module for the GLUE datasets (tasks), from which specific data modules for each GLUE task are derived.
"""
from .unimodal_datamodules import BaseDataModule
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Any, Dict, Union
from functools import partial
import os
from utils import Modality
from torch.utils.data import DataLoader
from registries import register_datamodule, DATASET_REGISTRY

class GLUEDataModule(BaseDataModule):
    def __init__(
        self,
        data_path:os.PathLike,
        dataset:str,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=512,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """Data module for the GLUE datasets (tasks).

        Args:
            data_path (os.PathLike): The path where the data is stored.
            dataset (str): Name of the GLUE dataset to use. One of 'cola_glue', 'mnli_glue',
                'mrpc_glue', 'qnli_glue', 'qqp_glue', 'rte_glue', 'sst2_glue', 'stsb_glue', 'wnli_glue'.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data. Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default:
                BertTokenizer.from_pretrained("bert-base-uncased").
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 512.
            *args (List[Any]): Additional arguments for the BaseDataModule.
            **kwargs (Dict[str, Any): Additional keyword arguments for the BaseDataModule. Usually arguments for the DataLoader.
        """        
        super().__init__(data_path, *args, **kwargs)
        self.dataset = dataset
        self.val_split_name = 'dev'
        if self.dataset == 'mrpc_glue':
            self.val_split_name = 'test' # MRPC only has a test split (with labels), no dev split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @property
    def modality(self) -> Modality:
        return Modality.TEXT
    
    def setup(self, stage:str=None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            if self.dataset == 'mnli_glue':
                self.val_dataset_1.load()
                self.val_dataset_2.load()
            else:
                self.val_dataset.load()
    
    def set_train_dataset(self):
        self.train_dataset = DATASET_REGISTRY[self.dataset](
            data_path=self.data_path, 
            split='train',
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len
        )

    def set_val_dataset(self):
        if self.dataset == 'mnli_glue':
            self.val_dataset_1 = DATASET_REGISTRY[self.dataset](
                data_path=self.data_path, 
                split='dev_matched', 
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len
            )
            self.val_dataset_2 = DATASET_REGISTRY[self.dataset](
                data_path=self.data_path, 
                split='dev_mismatched',
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len
            )
        else:
            self.val_dataset = DATASET_REGISTRY[self.dataset](
                data_path=self.data_path, 
                split=self.val_split_name,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len
            )

    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            return DataLoader(self.val_dataset,
                            collate_fn=self.val_dataset.collater,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            sampler=None,
                            shuffle=False,
                            drop_last=False,)
        else:
            # for MNLI, we have two validation datasets
            # 1 -> matched, 2 -> mismatched
            val_dataloders = [
                DataLoader(self.val_dataset_1,
                    collate_fn=self.val_dataset_1.collater,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    sampler=None,
                    shuffle=False,
                    drop_last=False,),
                DataLoader(self.val_dataset_2,
                    collate_fn=self.val_dataset_2.collater,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    sampler=None,
                    shuffle=False,
                    drop_last=False,)
            ]
            return val_dataloders


for ds_key in DATASET_REGISTRY:
    if 'glue' in ds_key: # means it is a GLUE dataset, e.g. 'cola_glue', 'mnli_glue', etc.
        # generate and register one datamodule for each GLUE dataset/task
        dataset_func = partial(GLUEDataModule, dataset=ds_key)
        register_datamodule(name=ds_key)(dataset_func)
