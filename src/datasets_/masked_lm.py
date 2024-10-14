"""
This module defines a memory efficient dataset for training models on raw unstructured text data.
The code is adapted from Meta's fairseq library: https://github.com/facebookresearch/fairseq/blob/main/fairseq/tasks/masked_lm.py
"""
import os
import mmap
import numpy as np
import multiprocessing
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from typing import Union, Tuple, Dict, Any, List
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .base_datasets import BaseDataset, TextMixin
from utils import Modality
import subprocess
import tempfile
try:
    import pyarrow.plasma as plasma
except ImportError:
    plasma = None
from registries import register_dataset

def init_worker(tokenizer_path:os.PathLike) -> None:
    """Initialize the tokenizer in the worker process. Each worker process will have its own tokenizer instance.

    Args:
        tokenizer_path (os.PathLike): The path to the tokenizer to be used.
    """    
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def tokenize_line(line:str) -> List[int]:
    """Tokenize a single line of text using the global tokenizer instance.

    Args:
        line (str): The text to be tokenized.

    Returns:
        List[int]: The tokenized text as a list of token ids.
    """    
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line.strip()))


@register_dataset(name='MaskedLM')
class MaskedLMDataset(TextMixin, BaseDataset):
    def __init__(
        self,
        name:str,
        data_path:os.PathLike,
        split:str,
        text_file:os.PathLike,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        mask_prob:float=0.0,
        block_size:int=512,
    ):
        """A universal dataset for raw unstructured text data that supports masked language modeling.

        Args:
            name (str): The name of the dataset, used for saving the index files.
            data_path (os.PathLike): The path where the data is stored.
            split (str): The split of the data. One of 'train', 'val', or 'test'.
            text_file (os.PathLike): The path to the text file containing the raw text data. Should have the postfix '.<split>'.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data. Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default:
                BertTokenizer.from_pretrained("bert-base-uncased").
            mask_prob (float, optional): The probability of masking a token in the text data. Defaults to 0.0, so no masking is done.
            block_size (int, optional): How many tokens should be in one block. One block is equal to one training example, i.e.
                a single text sequence. This is the maximum sequence length. The text file will be sliced into chunks
                of <block_size> tokens. Defaults to 512.
        """        
        super().__init__(
            data_path=data_path,
            split=split,
            tokenizer=tokenizer,
            max_seq_len=block_size,
            mlm_probability=mask_prob,
        )
        self.name = name
        self.text_file = text_file
        # Subtract 2 for CLS and SEP tokens -> each slice of the text data will be block_size-2 tokens long
        # if then then add 2 tokens for CLS and SEP tokens, the total length will be block_size
        self.block_size = block_size - 2
        self.token_file = os.path.join(self.data_path, f'mlm_{self.name}_{self.split}.bin')
        self.index_file = os.path.join(self.data_path, f'mlm_{self.name}_{self.split}.idx')
        self.index_entry_size = 16  # Each index entry has two int64 (offset and length)

        if not self.index_exists():
            self.preprocess()

    @property
    def data_dir(self) -> str:
        """
        Name of the directory in self.data_path where the data is stored.
        """        
        raise self.name

    def __getstate__(self):
        """
        Exclude non-picklable objects from the pickled state.
        This is necessary for multiprocessing environments, as is the case when using pytorch dataloaders.
        """
        state = self.__dict__.copy()
        state.pop('fp', None)  # Remove the file pointer
        state.pop('mmap_file', None)  # Remove the mmap object
        return state

    def __setstate__(self, state):
        """
        Reinitialize the non-picklable objects after unpickling.
        Each worker process will call this method when it is created, and therefore has its own file pointer and mmap object.
        """
        self.__dict__.update(state)
        # Reopen the file and recreate the mmap object
        self.fp = open(self.token_file, 'rb')
        self.mmap_file = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

    def load(self) -> None:
        self.build_sequences()

    def preprocess(self) -> None:
        """
        Tokenize the text file and store tokenized data into a binary mmap file.
        Create an index file that stores the offset and length of each tokenized sequence/line.
        """
        n_unk_tokens = 0
        n_total_tokens = 0
        # Calculate total lines for progress bar
        total_lines = sum(1 for _ in open(self.text_file, 'r', encoding='utf-8'))
        with open(self.text_file, 'r', encoding='utf-8') as f_in, \
            open(self.token_file, 'wb') as f_out, \
            open(self.index_file, 'wb') as f_idx:
            
            offset = 0
            batch_size = 10000 # works well for large files

            # Initialize multiprocessing pool
            pool = multiprocessing.Pool(initializer=init_worker, initargs=(self.tokenizer_path,))
            pbar = tqdm(total=total_lines, desc="Tokenizing")

            lines = []
            for line in f_in:
                lines.append(line)
                if len(lines) >= batch_size: # collect lines until batch_size is reached
                    # Tokenize the batch in parallel
                    tokenized_lines = pool.map(tokenize_line, lines)
                    # Write tokenized data to file and update index
                    for tokens in tokenized_lines:
                        length = len(tokens)
                        tokens = np.array(tokens, dtype=np.int32)
                        n_unk_tokens += (tokens == self.tokenizer.unk_token_id).sum() # count unknown tokens
                        n_total_tokens += length
                        tokens.tofile(f_out)
                        # Write offset (current position) and length as int64 to index file
                        f_idx.write(np.array([offset, length], dtype=np.int64).tobytes())
                        # Tokens(!) are stored in int32, so each token is 4 bytes
                        # -> advance offset by "number of tokens in the current line" * 4 bytes
                        offset += length * 4
                        pbar.update(1) # one line processed, update progress bar
                    lines = [] # clear the batch of lines

            # Process remaining lines
            if lines:
                tokenized_lines = pool.map(tokenize_line, lines)
                for tokens in tokenized_lines:
                    length = len(tokens)
                    tokens = np.array(tokens, dtype=np.int32)
                    n_unk_tokens += (tokens == self.tokenizer.unk_token_id).sum()
                    n_total_tokens += length
                    tokens.tofile(f_out)
                    f_idx.write(np.array([offset, length], dtype=np.int64).tobytes())
                    offset += length * 4
                    pbar.update(1)

            pbar.close()
            pool.close()
            pool.join()

        self.log(f'Preprocessing complete. Processed {n_total_tokens} tokens, '
                 f'found {n_unk_tokens}({n_unk_tokens/n_total_tokens*100:.05f}%) unknown tokens.')

    def build_sequences(self) -> None:
        """
        Slice the tokenized data into blocks of a specific size.
        This means potentially concatenating multiple lines of the tokenized text file to form a sequence of tokens (if 
        they together have not more than block_size tokens).
        This allows for less padding, more efficient memory usage and richer context for the model.
        """
        # Memory-map the index file
        self.index_fp = open(self.index_file, 'rb')
        self.index_mmap = mmap.mmap(self.index_fp.fileno(), 0, access=mmap.ACCESS_READ)

        items = []
        current_offset = 0
        current_length = 0

        num_lines = self.get_num_lines()

        for idx in tqdm(range(num_lines), desc="Building sequences"):
            offset, length = self.get_index_entry(idx) # get position of one line in the tokenized text file
            if current_length == 0:
                current_offset = offset

            # Handle lines longer than block_size
            if length >= self.block_size:
                if current_length > 0:
                    # if we encounter a line that is longer than block_size, and we are currently in the middle of a chunk,
                    # then we need to stop the current chunk
                    items.append([current_offset, current_length])

                # Split the line into chunks of block_size
                num_splits = (length + self.block_size - 1) // self.block_size
                for i in range(num_splits):
                    split_offset = offset + i * self.block_size * 4  # 4 bytes per int32 token
                    split_length = min(self.block_size, length - i * self.block_size)
                    items.append([split_offset, split_length])
                
                current_length = 0 # reset current_length, that means we start a new chunk
            else:
                if current_length + length <= self.block_size:
                    # if there is still space in the current chunk, add the current line to it
                    current_length += length
                else:
                    # if there is no space in the current chunk, save it as a sequence
                    items.append([current_offset, current_length])
                    # ... and start a new chunk with the current line (that is too long for the previous chunk)
                    current_offset = offset
                    current_length = length

        # Add any remaining tokens as the last sequence
        if current_length > 0:
            items.append([current_offset, current_length])

        self._items = PlasmaArray(
            np.array(items, dtype=np.int64)
        ) # wrap the items in a PlasmaArray to store them in shared memory -> more efficient memory usage in multiprocessing environments

        # Close the index mmap and file since it's no longer needed
        self.index_mmap.close()
        self.index_fp.close()
        del self.index_mmap
        del self.index_fp

    @property
    def items(self):
        return self._items.array

    def get_num_lines(self) -> int:
        """Calculate the number of lines in the index file.

        Returns:
            int: The number of lines in the index file.
        """
        index_file_size = os.path.getsize(self.index_file) # get the size of the index file in bytes
        num_lines = index_file_size // self.index_entry_size  # Each index entry has two int64 (offset [8 byte] and length [8 byte]) -> 16 bytes
        return num_lines

    def get_index_entry(self, idx:int) -> Tuple[int, int]:
        """Retrieve an index entry (offset and length) from the mmap index file.

        Args:
            idx (int): The current index in the index file. Can also be seen as the line number of the lines in the text file.

        Returns:
            Tuple[int, int]: The offset, at which position in the tokenized text file the sentence is located,
                and length of the sequence (in bytes).
        """
        start = idx * self.index_entry_size
        end = start + self.index_entry_size
        data = self.index_mmap[start:end]
        offset, length = np.frombuffer(data, dtype=np.int64)
        return offset, length

    def __getitem__(self, idx:int) -> Dict[str, Any]:
        # Ensure that the mmap file is initialized -> not the case if not using multiprocessing
        if not hasattr(self, 'mmap_file'):
            self.fp = open(self.token_file, 'rb')
            self.mmap_file = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

        result_dict = dict()
        # each items stores the position and length of one block/sequence in the tokenized text file
        offset, length = self.items[idx]
        # sequence length is in number of tokens, each token is 4 bytes (length * 4 = length in bytes)
        tokens = np.frombuffer(self.mmap_file[offset:offset + length * 4], dtype=np.int32).tolist()

        if self.mlm_probability > 0.0:
            mask_labels = self.mask_sequence(input_tokens=tokens, whole_word_mask=True)
            mask_labels = [0] + mask_labels + [0] * (self.max_seq_len - length - 1)
            result_dict["mask_labels"] = mask_labels

        tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        num_tokens = length + 2 # Added CLS and SEP tokens
        padding_mask = [0] * num_tokens + [1] * (self.max_seq_len - num_tokens)
        language_tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - num_tokens)

        result_dict["text"] = language_tokens
        result_dict["padding_mask"] = padding_mask
        result_dict["id"] = idx

        return result_dict
    
    @property
    def modality(self) -> Modality:
        return Modality.TEXT

    def get_index_files(self) -> Tuple[str]:
        """Returns a tuple of strings, where each string is the name of an index file containing the data
        for the split of the dataset.

        Returns:
            Tuple[str]: Tuple of strings, where each string is the name of an index file containing the data.
        """        
        return (self.token_file, self.index_file)
    
    def collater(self, samples:List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_tensors = super().collater(samples)
        if self.mlm_probability > 0.0:
            batch_tensors['targets'] = batch_tensors['text'].clone()
            batch_tensors['targets'][~batch_tensors['mask_labels'].bool()] = -100
            batch_tensors['text'][batch_tensors['mask_labels'].bool()] = self.tokenizer.mask_token_id
        return batch_tensors


# copied from fairseq/data/plasma_utils.py:

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
class PlasmaArray:
    """
    Wrapper around numpy arrays that automatically moves the data to shared
    memory upon serialization. This is particularly helpful when passing numpy
    arrays through multiprocessing, so that data is not unnecessarily
    duplicated or pickled.
    """

    def __init__(self, array):
        super().__init__()
        self.array = array
        self.disable = array.nbytes < 134217728  # disable for arrays <128MB
        self.object_id = None
        self.path = None

        # variables with underscores shouldn't be pickled
        self._client = None
        self._server = None
        self._server_tmp = None
        self._plasma = None

    @property
    def plasma(self):
        if self._plasma is None and not self.disable:
            self._plasma = plasma
        return self._plasma

    def start_server(self):
        if self.plasma is None or self._server is not None:
            return
        assert self.object_id is None
        assert self.path is None
        self._server_tmp = tempfile.NamedTemporaryFile()
        self.path = self._server_tmp.name
        self._server = subprocess.Popen(
            ["plasma_store", "-m", str(int(1.05 * self.array.nbytes)), "-s", self.path]
        )

    @property
    def client(self):
        if self._client is None:
            assert self.path is not None
            self._client = self.plasma.connect(self.path, num_retries=200)
        return self._client

    def __getstate__(self):
        """Called on pickle load"""
        if self.plasma is None:
            return self.__dict__
        if self.object_id is None:
            self.start_server()
            self.object_id = self.client.put(self.array)
        state = self.__dict__.copy()
        del state["array"]
        state["_client"] = None
        state["_server"] = None
        state["_server_tmp"] = None
        state["_plasma"] = None
        return state

    def __setstate__(self, state):
        """Called on pickle save"""
        self.__dict__.update(state)
        if self.plasma is None:
            return
        self.array = self.client.get(self.object_id)

    def __del__(self):
        if self._server is not None:
            self._server.kill()
            self._server = None
            self._server_tmp.close()
            self._server_tmp = None
