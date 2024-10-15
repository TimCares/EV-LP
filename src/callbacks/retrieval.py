"""
This module contains a callback that performs image-text retrieval on the validation set of a datamodule.
"""
import logging
from pytorch_lightning import Callback, LightningDataModule, Trainer, LightningModule
from typing import *
import torch
from pytorch_lightning.utilities import rank_zero_only
from run_image_text_retrieval import zero_shot_retrieval

logger = logging.getLogger(__name__)

class RetrievalCallback(Callback):
    def __init__(self, datamodule:LightningDataModule, name:str):
        """Callback to perform image-text retrieval on the validation set of the datamodule.

        Args:
            datamodule (LightningDataModule): The datamodule to perform retrieval on.
            name (str): The name under which to log the average retrieval score.
        """        
        super().__init__()
        self.datamodule = datamodule
        self.name = name

    @torch.no_grad()
    @rank_zero_only
    def on_validation_start(self, trainer:Trainer, pl_module:LightningModule, **kwargs) -> None:
        """Pytorch Lightning hook that runs before the validation loop starts. Used to run image-text retrieval
        during (before) the validation loop.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The model/module to be evaluated.
        """       
        result = zero_shot_retrieval(pl_module.module.model, self.datamodule.val_dataloader(), pl_module.device)

        pl_module.log(
            f"val/{self.name}",
            result['average_score'],
            rank_zero_only=True,
            logger=True,
            on_epoch=True,
        )
