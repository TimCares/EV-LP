"""
This module contains the implementation of the COCO Captions, Flickr30k, and Conceptual Captions 3M/12M datasets.
"""
import os
from typing import Dict, Union, Any, List
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from torch import nn
import json
import pandas as pd
from datasets_.data_utils import write_data_into_jsonl
from tqdm import tqdm
import os
import json
import zipfile
from torchvision.datasets.utils import download_url
from .base_datasets import ImageTextDataset
from registries import register_dataset

@register_dataset(name='COCOCaptions')
class COCOCaptions(ImageTextDataset):
    def __init__(
        self,
        data_path:os.PathLike,
        split:str,
        transforms:Dict[str, nn.Module]=None,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=64,
        text_token_mask_prob:float=0.0,
    ):
        """The COCO Captions dataset.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            split (str): The split of the data. One of 'train', 'val', or 'test'.
            transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to image data.
                If None, no augmentation will be applied. Defaults to None.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data.
                Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default:
                BertTokenizer.from_pretrained("bert-base-uncased")
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 64.
            text_token_mask_prob (float, optional): The probability of masking a token in the captions. Defaults to 0.0, so no masking is done.
        """        
        super().__init__(
            data_path=data_path,
            split=split,
            transforms=transforms,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            text_token_mask_prob=text_token_mask_prob,
        )
        if self.index_exists():
            return

        data_already_downloaded = os.path.exists(os.path.join(self.path_to_data, "train2014")) and \
            os.path.exists(os.path.join(self.path_to_data, "val2014")) and \
            os.path.exists(os.path.join(self.path_to_data, "dataset_coco.json"))
        
        if not data_already_downloaded:
            self.log("Downloading COCO dataset...")
            urls = ["http://images.cocodataset.org/zips/train2014.zip",
                    "http://images.cocodataset.org/zips/val2014.zip",
                    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"]

            for url in urls:
                download_url(url=url, root=self.path_to_data)
                filepath = os.path.join(self.path_to_data, os.path.basename(url))
                with zipfile.ZipFile(filepath, 'r') as zip:
                    zip.extractall(self.path_to_data)
                os.remove(filepath)
            os.remove(os.path.join(self.path_to_data, 'dataset_flickr8k.json'))
            os.remove(os.path.join(self.path_to_data, 'dataset_flickr30k.json'))
        else:
            self.log("COCO dataset already downloaded!")

        self.create_index()

    def get_index_files(self):
        return (f"coco.{self.split}.jsonl", )
    
    @property
    def data_dir(self) -> str:
        """
        Name of the directory in self.data_path where the data is stored.
        """        
        raise "coco"
    
    def create_index(self):
        if self.split == "train":
            karpathy_split = ("train", "restval")
        elif self.split == "val":
            karpathy_split = ("val", )
        elif self.split == "test":
            karpathy_split = ("test", )
        else:
            raise RuntimeError("split %s is not found!" % self.split)
        
        coco_karpathy_split_json_file = os.path.join(self.path_to_data, "dataset_coco.json")
        items = []
        image_counter = set()
        self.log("Read %s" % coco_karpathy_split_json_file)
        with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
            data = json.loads(reader.read())
            for item in data["images"]:
                if item["split"] in karpathy_split:
                    image_path = os.path.join(self.path_to_data, item["filepath"], item["filename"])
                    items += self._encode_all(item, image_path)
                    if image_path not in image_counter:
                        image_counter.add(image_path)
        self.log("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
            (len(image_counter), len(items), self.split))
        index_file = os.path.join(self.path_to_data, self.get_index_files()[0])
        write_data_into_jsonl(items, index_file)

    def _encode_all(self, item:Dict[str, Any], image_path:os.PathLike) -> List[Dict[str, Any]]:
        """Creates one training example for each caption (usually 5) of one image.

        Args:
            item (Dict[str, Any]): The item representing one image and its captions.
            image_path (os.PathLike): The path to the image.

        Returns:
            List[Dict[str, Any]]: The training examples for the image.
        """        
        return [
            {
                "image_path": image_path, # image is the same for all captions of one image
                "text": self.tokenizer.tokenize(sent["raw"]),
                "id": item["cocoid"], # image is always the same for all captions, and therefore the id is the same
            }
            for sent in item["sentences"]
        ]


@register_dataset(name='Flickr30K')
class Flickr30K(ImageTextDataset):
    def __init__(
        self,
        data_path:os.PathLike,
        split:str,
        transforms:Dict[str, nn.Module]=None,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=64,
        text_token_mask_prob:float=0.0,
    ):
        """The COCO Captions dataset.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            split (str): The split of the data. One of 'train', 'val', or 'test'.
            transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to image data.
                If None, no augmentation will be applied. Defaults to None.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data.
                Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default:
                BertTokenizer.from_pretrained("bert-base-uncased")
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 64.
            text_token_mask_prob (float, optional): The probability of masking a token in the captions. Defaults to 0.0, so no masking is done.
        """        
        super().__init__(
            data_path=data_path,
            split=split,
            transforms=transforms,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            text_token_mask_prob=text_token_mask_prob,
        )
        if self.index_exists():
            return
        
        url="https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
        download_url(url=url, root=self.path_to_data)
        filepath = os.path.join(self.path_to_data, os.path.basename(url))
        with zipfile.ZipFile(filepath, 'r') as zip:
            zip.extractall(self.path_to_data)

        os.remove(os.path.join(self.path_to_data, 'dataset_flickr8k.json'))
        os.remove(os.path.join(self.path_to_data, 'dataset_coco.json'))
        
        self.create_index()

    def get_index_files(self):
        return (f"flickr30k.{self.split}.jsonl", )
        
    @property
    def data_dir(self) -> str:
        """
        Name of the directory in self.data_path where the data is stored.
        """        
        raise "flickr30k"

    def create_index(self):

        with open(os.path.join(self.path_to_data, "dataset_flickr30k.json"), "r") as reader:
            captions = json.loads(reader.read())

        captions = captions["images"]
        index = []

        all_images = set()

        for each_item in captions:
            image_path = os.path.join(self.path_to_data, "flickr30k-images", each_item["filename"])

            if each_item["split"] != self.split:
                continue

            assert os.path.exists(image_path), f"Image {image_path} not found!"

            for text_segment in each_item["sentences"]: 
                index.append({
                    "image_path": image_path, 
                    "text": self.tokenizer.tokenize(text_segment["raw"]), 
                    "id": len(all_images),
                })

            assert each_item["filename"] not in all_images
            all_images.add(each_item["filename"])

        self.log(f"{len(all_images)} images and {len(index)} image-text pairs!")
        write_data_into_jsonl(index, os.path.join(self.path_to_data, f"flickr30k.{self.split}.jsonl"))

class ConceptualCaptions(ImageTextDataset):
    def __init__(
        self,
        data_path:os.PathLike,
        split:str,
        transforms:Dict[str, nn.Module]=None,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=64,
        text_token_mask_prob:float=0.0,
        type:str="cc3m",
    ):
        """The COCO Captions dataset.

        Args:
            data_path (os.PathLike): The path where the data is stored.
            split (str): The split of the data. One of 'train', 'val', or 'test'.
            transforms (Dict[str, nn.Module], optional): A list of named PyTorch transforms to apply to image data.
                If None, no augmentation will be applied. Defaults to None.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data.
                Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default:
                BertTokenizer.from_pretrained("bert-base-uncased")
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 64.
            text_token_mask_prob (float, optional): The probability of masking a token in the captions. Defaults to 0.0, so no masking is done.
            type (str, optional): The type of the Conceptual Captions dataset. One of 'cc3m' or 'cc12m'. Defaults to 'cc3m'.
        """        
        super().__init__(
            data_path=data_path,
            split=split,
            transforms=transforms,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            text_token_mask_prob=text_token_mask_prob,
        )
        assert type in ["cc3m", "cc12m"]
        self.type = type

        if not self.index_exists():
            self.make_conceptual_captions_dataset_index()

    def get_index_files(self):
        return (f"conceptual_captions_{self.type[2:]}.jsonl", ) # only for pretraining, so no splits
    
    @property
    def data_dir(self) -> str:
        """
        Name of the directory in self.data_path where the data is stored.
        """        
        raise f"conceptual_captions_{self.type[2:]}"

    def make_conceptual_captions_dataset_index(self):
        img_path = os.path.join(self.path_to_data, "images")
        assert os.path.exists(img_path), f"Images not found at {img_path}!"

        items = []
        if self.type == "cc3m":
            index_name = "Train-GCC-training.tsv"
            col_names = ['caption', 'image_url']
        else:
            index_name = "cc12m.tsv"
            col_names = ['image_url', 'caption']
        index_path = os.path.join(self.data_path, index_name) 
        index = pd.read_csv(index_path, sep='\t', header=None).reset_index(drop=True)
        index.columns = col_names
        
        for img in tqdm(os.listdir(img_path), desc="Making index"):
            idx = int(os.path.splitext(img)[0])
            items.append({
                'image_path': os.path.join(img_path, img),
                'text': self.tokenizer.tokenize(index.at[idx, 'caption'].strip()),
                'id': idx,
            })
        self.log(f"Collected {len(items)} image-text pairs!")
        write_data_into_jsonl(items, os.path.join(self.path_to_data, self.get_index_files()[0]))


@register_dataset(name="ConceptualCaptions3m")
def conceptual_captions_cc3m(*args, **kwargs):
    return ConceptualCaptions(*args, type="cc3m", **kwargs)

@register_dataset(name="ConceptualCaptions12m")
def conceptual_captions_cc12m(*args, **kwargs):
    return ConceptualCaptions(*args, type="cc12m", **kwargs)
