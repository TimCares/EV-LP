import os
import json
import pandas as pd
from datasets_.data_utils import write_data_into_jsonl
from tqdm import tqdm
import os
import json
import zipfile
from torchvision.datasets.utils import download_url
from .base_datasets import BaseImageText
from registries import register_dataset

@register_dataset(name='COCOCaptions')
class COCOCaptions(BaseImageText):
    def __init__(
        self,
        data_path,
        split,
        num_max_bpe_tokens=64,
        task="captioning", # TODO
        color_jitter=None,
        beit_transforms=False,
        crop_scale=(0.6, 1.0),
        text_token_mask_prob=0.0,
    ):
        assert task in ["captioning", "retrieval"]
        self.task = task
        if self.task == "retrieval": # yields no augmentation, as retrieval is zero-shot (testing)
            color_jitter = None
            beit_transforms = False
            crop_scale = (1.0, 1.0)
        super().__init__(
            data_path=data_path,
            split=split,
            num_max_bpe_tokens=num_max_bpe_tokens,
            color_jitter=color_jitter,
            beit_transforms=beit_transforms,
            crop_scale=crop_scale,
            text_token_mask_prob=text_token_mask_prob
        )

        self.path_to_data = os.path.join(self.data_path, "coco")        

        os.makedirs(self.path_to_data, exist_ok=True)
        if self.index_exists(dataset_path=self.path_to_data):
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

        self._make_coco_karpathy_dataset_index()

    def get_index_files(self):
        return (f"coco_{self.task}.{self.split}.jsonl", )
    
    def _make_coco_karpathy_dataset_index(self):
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
        self.log("Task is: %s" % self.task)
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

    def _encode_all(self, item, image_path):
        return [
            {
                "image_path": image_path,
                "text": self.tokenizer.tokenize(sent["raw"]),
                "id": item["cocoid"],
            }
            for sent in item["sentences"]
        ]


@register_dataset(name='Flickr30k')
class Flickr30Dataset(BaseImageText):
    def __init__(self,
                 data_path,
                 split,
                 num_max_bpe_tokens,
                 color_jitter,
                 beit_transforms,
                 crop_scale,
                 text_token_mask_prob=0.0
                 ):
        super().__init__(
            data_path=data_path,
            split=split,
            num_max_bpe_tokens=num_max_bpe_tokens,
            color_jitter=color_jitter,
            beit_transforms=beit_transforms,
            crop_scale=crop_scale,
            text_token_mask_prob=text_token_mask_prob
        )

        self.path_to_data = os.path.join(self.data_path, "flickr30k")

        os.makedirs(self.path_to_data, exist_ok=True)
        if self.index_exists(dataset_path=self.path_to_data):
            return
        
        url="https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
        download_url(url=url, root=self.path_to_data)
        filepath = os.path.join(self.path_to_data, os.path.basename(url))
        with zipfile.ZipFile(filepath, 'r') as zip:
            zip.extractall(self.path_to_data)

        os.remove(os.path.join(self.path_to_data, 'dataset_flickr8k.json'))
        os.remove(os.path.join(self.path_to_data, 'dataset_coco.json'))
        
        self.make_flickr30k_dataset_index()

    def get_index_files(self):
        if self.split == "train":
            return (f"flickr30k.train.jsonl", )
        elif self.split == "val":
            return (f"flickr30k.val.jsonl", )
        elif self.split == "test":
            return (f"flickr30k.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % self.split)
        
    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["id"] = self.items[index]["id"]
        return data

    def make_flickr30k_dataset_index(self):

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

class ConceptualCaptions(BaseImageText):
    def __init__(
        self,
        type,
        data_path,
        split,
        num_max_bpe_tokens=64,
        color_jitter=None,
        beit_transforms=False,
        crop_scale=(0.6, 1.0),
        text_token_mask_prob=0.0,
    ):
        super().__init__(
            data_path=data_path,
            split=split,
            num_max_bpe_tokens=num_max_bpe_tokens,
            color_jitter=color_jitter,
            beit_transforms=beit_transforms,
            crop_scale=crop_scale,
            text_token_mask_prob=text_token_mask_prob
        )
        self.type = type
        assert type in ["cc3m", "cc12m"]
        self.path_to_data = os.path.join(self.data_path, f"conceptual_captions_{self.type[2:]}")
        self.img_path = os.path.join(self.path_to_data, "images")
        os.makedirs(self.path_to_data, exist_ok=True)
        os.makedirs(self.img_path, exist_ok=True)
        if self.index_exists(dataset_path=self.path_to_data):
            return
        
        self.make_conceptual_captions_dataset_index()

    def get_index_files(self):
        return (f"conceptual_captions_{self.type[2:]}.jsonl", ) # only for pretraining, so no splits

    def make_conceptual_captions_dataset_index(self):
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
        
        for img in tqdm(os.listdir(self.img_path), desc="Making index"):
            idx = int(os.path.splitext(img)[0])
            items.append({
                'image_path': os.path.join(self.img_path, img),
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
