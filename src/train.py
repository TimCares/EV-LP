import hydra
from omegaconf import OmegaConf, open_dict, DictConfig
import os
import torch
from typing import List
import logging
from pytorch_lightning import seed_everything, Trainer, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
import sys
sys.path.append("beit2")
from models import MODEL_REGISTRY
from datamodules import DATAMODULE_REGISTRY, MultiDataModule
from callbacks import (
    ImageNetZeroShotCallback,
    WallClockCallback,
)

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=os.path.join("..", "configs"))
def main(cfg: DictConfig) -> None:
    """
    Central entry point for training.
    """

    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)

    # model setup
    if 'cfg' in MODEL_REGISTRY[cfg.model_name].keys():
        cfg_cls = MODEL_REGISTRY[cfg.model_name]['cfg']
        cfg.model = OmegaConf.merge(cfg_cls(), cfg)
    module_cls = MODEL_REGISTRY[cfg.model_name]['module']

    module = module_cls(cfg)

    # callbacks setup
    callbacks = [
        ModelSummary(),
        LearningRateMonitor(logging_interval="step"),
        WallClockCallback(),
    ]

    if "imagenet_zero_shot_callback" in cfg and cfg.imagenet_zero_shot_callback:
        callbacks.append(ImageNetZeroShotCallback(
            data_path=cfg.data_path,
        ))
    
    if 'checkpoint' in cfg:
        common_checkpoint_args = OmegaConf.to_container(cfg.checkpoint.common, resolve=True)
        for ckpt in cfg.checkpoint.checkpoints:
            args = OmegaConf.to_container(ckpt, resolve=True) | common_checkpoint_args
            callbacks.append(ModelCheckpoint(**args))

    # strategy setup
    torch.set_float32_matmul_precision("high") # or: "highest"
    trainer_args = OmegaConf.to_container(cfg.lightning_trainer, resolve=True)
    if 'strategy' not in trainer_args:
        if 'deepspeed' in trainer_args:
            from pytorch_lightning.strategies import DeepSpeedStrategy
            trainer_args['strategy'] = DeepSpeedStrategy(**trainer_args.pop('deepspeed'))
        elif 'ddp' in trainer_args:
            from pytorch_lightning.strategies import DDPStrategy
            trainer_args['strategy'] = DDPStrategy(**trainer_args.pop('ddp'))
        else:
            trainer_args['strategy'] = 'auto'

    # datamodule setup
    dataloader_args = cfg.data.dataloader
    common_args = cfg.data.common

    datamodules:List[LightningDataModule] = []
    for datamodule_key in cfg.data.datamodules.keys():
        dataset_args = cfg.data.datamodules[datamodule_key]
        with open_dict(dataset_args):
            dataset_args.update(dataloader_args)
            dataset_args.update(common_args)
        dm_ = DATAMODULE_REGISTRY[datamodule_key](**dataset_args)
        datamodules.append(dm_)
        logger.info(f"Train datamodule {datamodule_key}: {dataset_args}")

    if len(datamodules) == 1:
        datamodule = datamodules[0]
    else:
        datamodule = MultiDataModule(datamodules=datamodules, **dataloader_args)

    logger.info("Setting up datamodules:")
    datamodule.prepare_data()
    datamodule.setup("fit")
    logger.info("Datamodule setup complete.")

    # trainer setup & training
    trainer = Trainer(
        **trainer_args,
        enable_checkpointing=ckpt in cfg.checkpoint.checkpoints,
        callbacks=callbacks,
    )

    trainer.fit(module,
                datamodule=datamodule)

if __name__ == "__main__":
    main()