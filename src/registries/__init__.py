from .registry import (
    MODEL_REGISTRY,
    MODEL_CONFIG_REGISTRY,
    DATASET_REGISTRY,
    DATAMODULE_REGISTRY,
    register_model,
    register_model_config,
    register_dataset,
    register_datamodule,
)

from models import * # to register models if not imported through entrypoint or scripts used by the entrypoint
from datasets import * # see above
from datamodules import * # see above