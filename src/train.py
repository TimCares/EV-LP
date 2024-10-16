"""
This script is the central entry point for training.
"""
import hydra
from omegaconf import OmegaConf, open_dict, DictConfig
import os
import torch
from typing import List
import logging
from pytorch_lightning import seed_everything, Trainer, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
from registries import MODEL_REGISTRY, MODEL_CONFIG_REGISTRY, DATAMODULE_REGISTRY, DATASET_REGISTRY
from datamodules import MultiDataModule
from datasets_ import TextMixin, ImageMixin
from datasets_ import data_utils
from callbacks import (
    ImageNetZeroShotCallback,
    WallClockCallback,
)
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=os.path.join("..", "configs"))
def main(cfg: DictConfig) -> None:
    """
    Central entry point for training.
    """

    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)

    ### model setup
    if cfg.model_name in MODEL_CONFIG_REGISTRY.keys(): # some models don't have a config
        cfg_cls = MODEL_CONFIG_REGISTRY[cfg.model_name]
        # override config with CLI and .yaml config args
        cfg.model = OmegaConf.merge(cfg_cls(), cfg.model)
    # get the model and pass the config
    module = MODEL_REGISTRY[cfg.model_name](cfg)

    ### callbacks setup
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

    ### strategy setup
    torch.set_float32_matmul_precision("high") # or: "highest"
    trainer_args = OmegaConf.to_container(cfg.lightning_trainer, resolve=True)
    if 'strategy' not in trainer_args:
        if 'deepspeed' in trainer_args:
            assert torch.cuda.is_available(), "No CUDA device available."
            from pytorch_lightning.strategies import DeepSpeedStrategy
            trainer_args['strategy'] = DeepSpeedStrategy(**trainer_args.pop('deepspeed'))
        elif 'ddp' in trainer_args:
            assert torch.cuda.is_available(), "No CUDA device available."
            from pytorch_lightning.strategies import DDPStrategy
            trainer_args['strategy'] = DDPStrategy(**trainer_args.pop('ddp'))
        else:
            trainer_args['strategy'] = 'auto'

    ### datamodule setup
    dataloader_args = cfg.data.dataloader
    common_args = cfg.data.common if 'common' in cfg.data else {}

    # we only use one in this work
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    datamodules:List[LightningDataModule] = []
    for datamodule_key in cfg.data.datamodules.keys():
        dataset_args = cfg.data.datamodules[datamodule_key]
        with open_dict(dataset_args):
            dataset_args.update(dataloader_args)
            dataset_args.update(common_args)
            # dataset and its datamodule are registered in DATASET_REGISTRY and DATAMODULE_REGISTRY
            # under the same key -> check if the dataset is a TextMixin dataset and pass the tokenizer
            if issubclass(DATASET_REGISTRY[datamodule_key], TextMixin):
                dataset_args['tokenizer'] = tokenizer
            # check if the dataset is an ImageMixin dataset -> datamodule potentially needs transforms
            if issubclass(DATASET_REGISTRY[datamodule_key], ImageMixin):
                if 'train_transforms' in dataset_args:
                    train_transforms = data_utils.create_transforms_from_config(dataset_args.pop('train_transforms'), train=True)
                    dataset_args['train_transforms'] = train_transforms
                if 'eval_transforms' in dataset_args:
                    eval_transforms = data_utils.create_transforms_from_config(dataset_args.pop('eval_transforms'), train=False)
                    dataset_args['eval_transforms'] = eval_transforms
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

    ### trainer setup & training
    trainer = Trainer(
        **trainer_args,
        enable_checkpointing=ckpt in cfg.checkpoint.checkpoints,
        callbacks=callbacks,
    )

    trainer.fit(module,
                datamodule=datamodule)

if __name__ == "__main__":
    main()
