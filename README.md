# Efficient Vision-Language Pretraining (EV-LP)
This repository is the official Pytorch implementation of
"Leveraging Pretrained Unimodal Models for Efficient Vision-Language Pretraining".
See [here](docs/thesis.pdf) for details.

## Data
Most of the data, including all GLUE tasks and COCO, are downloaded automatically before training starts.
Some however, need to be downloaded using separate scripts or actions.

### Conceptual Captions
To generate CC3M and CC12M, one must scrape the images from the web.

To download CC3M data, please first download the index from
[https://ai.google.com/research/ConceptualCaptions/download](https://ai.google.com/research/ConceptualCaptions/download).
Under the header `Downloads`, click on the button `Training split`. This should download the file `Train-GCC-training.tsv`
(other splits are not supported).
Then run:

```shell
python src/datasets_/download/download_cc3m_images.py --data_path <path_to_store_all_data> --path_to_index <path_to_Train-GCC-training.tsv>
```

This will create a directory `conceptual_captions_3m` in \<path_to_store_all_data\>, which will be used during pretraining.

To download CC12M data, it is sufficient to run:

```shell
python src/datasets_/download/download_cc12m_images.py --data_path <path_to_store_all_data>
```

The index will be downloaded automatically.

Note that not all images in both indices will be downloaded successfully, as they are scraped from various websites
and could be taken down at any time by the respective owners.

### Flickr30K
Flickr30K is not a public dataset, and you have to fill out a form to request access. The form can be found [here](https://forms.illinois.edu/sec/229675).
Upon receiving access via email, download the archive `flickr30k-images.tar.gz` and unpack it to get the folder `flickr30k-images` with the images.

Then, under \<path_to_store_all_data\>, create a folder `flickr30k` (this is the dataset folder,
like `conceptual_captions_3m` for CC3M), and copy the image folder `flickr30k-images`
into this directory. Everything else will be handled by the code!

### ImageNet
The ImageNet-1K dataset is downloaded from HuggingFace. This requires a HuggingFace account!

First, run:

```shell
huggingface-cli login
```

To authenticate, use an access token created on the HuggingFace website.

To download the data, run:

```shell
HF_HUB_ENABLE_HF_TRANSFER=1 python src/datasets_/download/download_imagenet.py --data_path <path_to_store_all_data>
```

`HF_HUB_ENABLE_HF_TRANSFER=1` will allow for a faster download if using high bandwidth. Do not be confused if the download is
complete and the script appears to hang, this is because the downloaded tar.gz files need to be extracted, which can take quite long.

Again, the rest is handled by the code.

## Models
**Note:** Due to the amount of downstream tasks and running costs we do not provide pretrained weights yet.
### Pretrain

To pretrain S-SMKE, first download the weights for Data2Vec2Image by clicking [here](https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_imagenet.pt),
and the weights of BEiTv2 by clicking [here](https://github.com/addf400/files/releases/download/BEiT-v2/beitv2_base_patch16_224_pt1k.pth).
Next, create a folder `models`in the directory \<path_to_store_all_data\> and copy both downloaded state dicts into the `models` folder.
The downloaded files should be named `base_imagenet.pt` and `beitv2_base_patch16_224_pt1k.pth` for Data2Vec2Image and BEiTv2, respectively.

Then run the following to start training:

```shell
python src/train.py --config-path configs/training --config-name s_smke.yaml base_dir=<path_to_store_all_data>
```

The trained model will be stored under `<path_to_store_all_data>/models/S-SMKE/model-{step}-{loss}-train.ckpt`.

| Model | ImageNet-1K | COCO R@1 | Flickr30K R@1 | #params | weights |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [S-SMKE](src/models/S_SMKE.py) | 37.1 | 44.6 | 61.8 | 117M | - |

### ImageNet-1K Classification

For finetuning S-SMKE on ImageNet-1K, run:

```shell
python src/train.py \
    --config-path configs/fine_tuning \
    --config-name imagenet.yaml base_dir=<path_to_store_all_data> \
    model.model_path=<path_to_store_all_data>/models/S-SMKE/model-{step}-{loss}-train.ckpt \
    model.model_name=S-SMKE \
    model.linear_classifier=False
```

For linear evaluation, set `model.linear_classifier=True`.

| Model | Resolution | finetune acc@1 | lin eval acc@1 | #params | weights |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [ImageClassification](src/models/image_classification.py) | 224x224 | 75.5 | 65.0 | 43.9M | - |

### GLUE

To finetune S-SMKE on **all** GLUE tasks, navigate to [`src/sweep_glue.sh`](src/sweep_glue.sh), and set the
variables `model` and `model_name` to:

```shell
#!/bin/bash

model="<path_to_store_all_data>/models/S-SMKE/model-{step}-{loss}-train.ckpt"
model_name="S-SMKE"

...
```

Then run:

```shell
cd src
python sweep_glue.sh
```

| Model | Task |Metric | Score | #params | weights |
|:----------------------------------------|:----------:|:-----:|:-------:|:-------:|-------------------|
| [TextClassification](src/models/text_classification.py) | MNLI | Accuracy | 73.88 | 67M | - |
| [TextClassification](src/models/text_classification.py) | QNLI | Accuracy | 78.71 | 67M | - |
| [TextClassification](src/models/text_classification.py) | RTE | Accuracy | 51.6 | 67M | - |
| [TextClassification](src/models/text_classification.py) | MRPC | F1 | 79.9 | 67M | - |
| [TextClassification](src/models/text_classification.py) | QQP | F1 | 81.2 | 67M | - |
| [TextClassification](src/models/text_classification.py) | STS-B | Spearman | 57.5 | 67M | - |
| [TextClassification](src/models/text_classification.py) | CoLA | MCC | **14.2** | 67M | - |
| [TextClassification](src/models/text_classification.py) | SST | Accuracy | 83.5 | 67M | - |
| [TextClassification](src/models/text_classification.py) | WNLI | Accuracy | 45.0 | 67M | - |

### Retrieval

The finetune S-SMKE on Flickr30K and COCO image-text retrieval, navigate to [`src/sweep_retrieval.sh`](src/sweep_retrieval.sh), and set the
variables `model` and `model_name` to:

```shell
#!/bin/bash

model="<path_to_store_all_data>/models/S-SMKE/model-{step}-{loss}-train.ckpt"
model_name="S-SMKE"

...
```

Then run:

```shell
cd src
python sweep_retrieval.sh
```

To generate the image-text retrieval scores published in research papers like BEiT-3 run the following (COCO):

```shell
python src/run_image_text_retrieval.py \
    --config-path configs \
    --config-name coco_flickr_retrieval.yaml \
    model_path=<path_to_store_all_data>/models/retrieval_finetune_coco/model-{step}-{loss}-val.ckpt \
    model_name=S-SMKE
```

And for Flickr30K replace `model_path` with `model_path=<path_to_store_all_data>/models/retrieval_finetune_flickr30k/model-{step}-{loss}-val.ckpt`.

#### Flickr30K
| Model | Resolution | IR@1 | TR@1 | #params | weights |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [RetrievalFinetune](src/models/retrieval_finetune.py) | 224x224 | 64.6 | 82.0 | 117M | - |

#### COCO
| Model | Resolution | IR@1 | TR@1 | #params | weights |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [RetrievalFinetune](src/models/retrieval_finetune.py) | 224x224 | 39.8 | 56.2 | 117M | - |

## Limitations

- Not applicable to visual reasoning, visual question answering, image captioning
- Low performance on unimodal downstream tasks, especially CoLA (GLUE)