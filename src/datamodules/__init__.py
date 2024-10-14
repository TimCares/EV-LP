from .unimodal_datamodules import (
    BaseDataModule,
    MaskedLMDataModule,
    CIFARDataModule,
    ImageNetDataModule,
)

from .multimodal_datamodules import (
    COCOCaptionsDataModule,
    ConceptualCaptionsDataModule,
    Flickr30KDataModule,
)

from .dummy import DummyDataModule
from .glue import GLUEDataModule

from .multi_data_module import MultiDataModule
