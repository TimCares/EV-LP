"""
This module provides utility callbacks for PyTorch Lightning.
"""
import logging
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import *
from time import time
import os

logger = logging.getLogger(__name__)

class WallClockCallback(Callback):
    def __init__(self):
        """
        This callback measures the wall clock time of the GPU during training, and logs the total GPU wall clock time
        at the end of training. It also logs the average GPU wall clock time per batch.
        """        
        self.gpu_wall_clock_time = 0.0
        self.start_time = None
        self.n_batches = 0

    def on_train_batch_start(self, trainer:Trainer, pl_module:LightningModule, batch:Any, batch_idx:int) -> None:
        self.start_time = time()

    def on_train_batch_end(self, trainer:Trainer, pl_module:LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        elapsed_time = time() - self.start_time
        self.gpu_wall_clock_time += elapsed_time
        self.n_batches += 1

    def on_train_end(self, trainer:Trainer, pl_module:LightningModule) -> None:
        logger.info(f"GPU Wall Clock Time: {self.gpu_wall_clock_time/60:.2f} minutes")
        logger.info(f"GPU Wall Clock Time: {self.gpu_wall_clock_time/3600:.2f} hours")
        trainer.logger.experiment.log({"gpu_wall_clock_time": self.gpu_wall_clock_time})

        logger.info(f"Average GPU Wall Clock Time per batch: {self.gpu_wall_clock_time/self.n_batches:.2f} seconds")
        trainer.logger.experiment.log({"gpu_wall_clock_time_per_batch": self.gpu_wall_clock_time/self.n_batches})

    @property
    def state_key(self) -> str:
        return f"WallClock[gpu_wall_clock_time={self.gpu_wall_clock_time}, n_batches={self.n_batches}]"

    def load_state_dict(self, state_dict):
        self.gpu_wall_clock_time = state_dict["gpu_wall_clock_time"]
        self.n_batches = state_dict["n_batches"]

    def state_dict(self):
        sd = {
            "gpu_wall_clock_time": self.gpu_wall_clock_time,
            "n_batches": self.n_batches
        }
        return sd
    

class GracefulStoppingCallback(Callback):
    def __init__(self, ckpt_path:os.PathLike):
        """This callback saves the model checkpoint when a SIGTERM signal is received. This is useful for
        gracefully stopping when using spot instances on cloud providers.

        Args:
            ckpt_path (os.PathLike): The path where the checkpoint should be saved.
        """        
        self.ckpt_path = ckpt_path

    def on_train_batch_start(self, trainer:Trainer, pl_module:LightningModule, batch:Any, batch_idx:int) -> None:
        if trainer.received_sigterm:
            logger.info("Received SIGTERM. Gracefully stopping and saving checkpoint...")
            trainer.save_checkpoint(filepath=self.ckpt_path)
            trainer.should_stop = True
            logger.info("Checkpoint saved.")
