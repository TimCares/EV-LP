# Efficient Vision-Language Pretraining (EV-LP)
This repository is the official Pytorch implementation of
"Leveraging Pretrained Unimodal Models for Efficient Vision-Language Pretraining".
See [here](docs/thesis.pdf) for details.

## Models
**Note:** Due to the amount of downstream tasks and running costs we do not host pretrained weights yet.
### Pretrained
| Model | ImageNet-1K | COCO R@1 | Flickr30K R@1 | #params | weights |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [S-SMKE](src/models/S_SMKE.py) | 37.1 | 44.6 | 61.8 | 117M | - |

### Image Classification
| Model | Resolution | finetune acc@1 | lin eval acc@1 | #params | weights |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [ImageClassification](src/models/image_classification.py) | 224x224 | 75.5 | 65.0 | 43.9M | - |

### GLUE
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

### Flickr30K Retrieval
| Model | Resolution | IR@1 | TR@1 | #params | weights |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [RetrievalFinetune](src/models/retrieval_finetune.py) | 224x224 | 64.6 | 82.0 | 117M | - |

### COCO Retrieval
| Model | Resolution | IR@1 | TR@1 | #params | weights |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [RetrievalFinetune](src/models/retrieval_finetune.py) | 224x224 | 39.8 | 56.2 | 117M | - |

## Limitations

- Not applicable to visual reasoning, visual question answering, image captioning
- Low performance on unimodal downstream tasks, especially CoLA (GLUE)