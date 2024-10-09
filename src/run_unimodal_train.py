import sys
sys.path.append("beit2")
import hydra
from omegaconf import OmegaConf, DictConfig
import os
import logging
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from models import MODEL_REGISTRY
from datamodules import DATAMODULE_REGISTRY
from callbacks import WallClockCallback, RetrievalCallback
from fairseq.dataclass.utils import merge_with_parent

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=os.path.join("..", "configs"))
def main(cfg: DictConfig) -> None:
    
    if 'cfg' in MODEL_REGISTRY[cfg.model_name].keys():
        cfg_cls = MODEL_REGISTRY[cfg.model_name]['cfg']
        cfg.model = merge_with_parent(dc=cfg_cls(), cfg=cfg.model, remove_missing=False)
    module_cls = MODEL_REGISTRY[cfg.model_name]['module']
    
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')

    module = module_cls(cfg)

    OmegaConf.resolve(cfg=cfg) # resolving done in-place

    callbacks = [
        ModelSummary(),
        LearningRateMonitor(logging_interval="step"),
        WallClockCallback(),
    ]

    if 'checkpoint' in cfg:
        common_checkpoint_args = OmegaConf.to_container(cfg.checkpoint.common, resolve=True)
        for ckpt in cfg.checkpoint.checkpoints:
            args = OmegaConf.to_container(ckpt, resolve=True) | common_checkpoint_args
            callbacks.append(ModelCheckpoint(**args))

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

    datamodule_args = OmegaConf.to_container(cfg.data, resolve=True)
    name = datamodule_args.pop('_name')
    datamodule = DATAMODULE_REGISTRY[name](**datamodule_args)
    logger.info(f"Datamodule {name}: {datamodule_args}")

    callbacks.append(RetrievalCallback(datamodule=datamodule, name=name))

    logger.info("Setting up datamodules:")
    datamodule.prepare_data()
    datamodule.setup("fit")
    logger.info("Datamodule setup complete.")

    trainer = Trainer(
        **trainer_args,
        enable_checkpointing=True,
        callbacks=callbacks,
    )

    if 'load_checkpoint' in cfg and cfg.load_checkpoint is not None:
        logger.info(f'Resuming from checkpoint: {cfg.load_checkpoint}')
        ckpt_path = os.path.join(cfg.model_path, cfg.load_checkpoint)
    else:
        ckpt_path = None

    trainer.fit(module,
                datamodule=datamodule,
                ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
