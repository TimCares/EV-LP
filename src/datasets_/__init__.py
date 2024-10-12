from .base_datasets import (
    BaseDataset,
    ImageDataset,
    BaseImageText,
)

from .unimodal_datasets import (
    ImageNetDataset,
    CIFARDataset,
    UNIMODAL_DATASET_REGISTRY,
)

from .multimodal_datasets import (
    COCOCaptions, 
    Flickr30Dataset,
    ConceptualCaptions,
    MULTIMODAL_DATASET_REGISTRY,
)

from .glue import (
    CoLA,
    SST,
    QNLI,
    RTE,
    MRPC,
    QQP,
    STSB,
    MNLI,
    GLUE_DATASET_REGISTRY,
)

from .dummy import DummyDataset, DUMMY_DATASET_REGISTRY

from .masked_lm import MaskedLMDataset

DATASET_REGISTRY = UNIMODAL_DATASET_REGISTRY | MULTIMODAL_DATASET_REGISTRY  | GLUE_DATASET_REGISTRY | \
    DUMMY_DATASET_REGISTRY | {'masked_lm': MaskedLMDataset}
