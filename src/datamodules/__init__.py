from .unimodal_datamodules import (
    BaseDataModule,
    MaskedLMDataModule,
    CIFARDataModule,
    ImageNetDataModule,
    UNIMODAL_DATAMODULE_REGISTRY
)

from .multimodal_datamodules import (
    COCOCaptionsDataModule,
    ConceptualCaptionsDataModule,
    Flickr30DataModule,
    MULTIMODAL_DATAMODULE_REGISTRY
)

from .dummy import (
    DummyDataModule,
    DUMMY_DATAMODULE_REGISTRY
)

from .glue import (
    GLUE_DATAMODULE_REGISTRY
)

from .multi_data_module import MultiDataModule

DATAMODULE_REGISTRY = UNIMODAL_DATAMODULE_REGISTRY | MULTIMODAL_DATAMODULE_REGISTRY \
    | DUMMY_DATAMODULE_REGISTRY | GLUE_DATAMODULE_REGISTRY # combine them