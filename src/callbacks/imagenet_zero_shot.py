"""
This module provides a callback for CLIP zero-shot image classification on the ImageNet-1k validation set.
"""
import logging
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_only
import sys
import os
sys.path.append('..')
from registries import DATAMODULE_REGISTRY
from data.imagenet_zeroshot_data import (
    imagenet_classnames,
    openai_imagenet_template,
)
from rich.progress import track
from utils import pad_text_sequence # src/utils.py
from transformers import BertTokenizer
from datasets_ import data_utils
from datamodules import ImageNetDataModule
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def _zero_shot_classifier(pl_module:LightningModule, device:str='cuda') -> torch.Tensor:
    """Generates the ImageNet-1K class prototypes, which can be viewed as the zero-shot classifier.
    See https://arxiv.org/pdf/2103.00020 for more.

    Args:
        pl_module (LightningModule): The model/module to be evaluated.
        device (str, optional): On which device to run the evaluation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The class prototypes, i.e. the zero-shot classifier. Shape: (emb_dim, num_classes).
            Each imagenet class is represented by a single prototype embedding.
    """    
    MAX_SEQ_LEN = 22 # max sequence length for OpenAI's templates with BERT tokenizer
    tokenizer:BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    zeroshot_weights = []
    for classname in track(imagenet_classnames, description="Building classifier"):
        # "texts" is a list of 80 tokenized sentences that "describe" the class, e.g. "a picture of a dog" for the class "dog"
        texts = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(template(classname)))
                 for template in openai_imagenet_template]
        padding_masks = []
        for i in range(len(texts)):
            language_tokens, padding_mask = pad_text_sequence(tokens=texts[i], max_seq_len=MAX_SEQ_LEN,
                                                              pad_idx=tokenizer.pad_token_id, bos_idx=tokenizer.cls_token_id,
                                                              eos_idx=tokenizer.sep_token_id,)
            
            texts[i] = language_tokens
            padding_masks.append(padding_mask)

        texts = torch.tensor(texts, dtype=torch.long)
        padding_masks = torch.tensor(padding_masks, dtype=torch.long)
        assert texts.size(1) == MAX_SEQ_LEN
        assert padding_masks.size(1) == MAX_SEQ_LEN

        texts = texts.to(device)
        padding_masks = padding_masks.to(device)
        # all 80 descriptions are encoded, averaged, and normed to get the prototype embedding for the current class
        class_embeddings = pl_module.model.encode_text(text=texts, padding_mask=padding_masks)['x'] # (80, emb_dim)
        class_embedding = class_embeddings.mean(dim=0) # (emb_dim,)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def _n_correct(output:torch.Tensor, target:torch.Tensor, topk:Tuple[int]=(1,)) -> List[float]:
    """Computes the top-k accuracy for the given output and target tensors.

    Args:
        output (torch.Tensor): The predicted logits, shape (batch_size, num_classes).
        target (torch.Tensor): The target labels, shape (batch_size,).
        topk (Tuple[int], optional): The top-k accuracies to compute. Defaults to (1,) for just top-1 accuracy.

    Returns:
        List[float]: The top-k accuracies, in the order specified by the topk argument.
    """    
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        correct[:k].reshape(-1).float().sum().cpu().item()
        for k in topk
    ]


@rank_zero_only
def run_imagenet_zero_shot(
    pl_module:LightningModule,
    dataloader:DataLoader,
    device:str='cuda',
) -> Dict[str, float]:
    """This function actually performs the zero-shot classification.

    Args:
        trainer (Trainer): The trainer object.
        pl_module (LightningModule): The model/module to be evaluated.
        device (str, optional): On which device to run the evaluation. Defaults to 'cuda'.

    Returns:
        Dict[str, float]: A dictionary containing the top-1 accuracy (key "imagenet--clip--zeroshot-val-top1")
            and the top-5 accuracy (key "imagenet--clip--zeroshot-val-top5").
    """    
    classifier = _zero_shot_classifier(pl_module, device)
    top1, top5, n = 0.0, 0.0, 0.0
    for sample in track(dataloader, description=f"ImageNet-1k CLIP zero-shot validation"):
        images = sample["image"]
        target = sample["target"]
        images = images.to(device)
        target = target.to(device)
        if pl_module.dtype == torch.float16: # when using deep speed
            images = images.half()
        image_features = pl_module.model.encode_image(image=images)['x'] # should normalize the features, shape (batch_size, emb_dim)
        logits = 100.0 * image_features @ classifier # (batch_size, num_classes)

        # measure accuracy
        acc1, acc5 = _n_correct(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    results = {}
    results[f"imagenet--clip--zeroshot-val-top1"] = top1
    results[f"imagenet--clip--zeroshot-val-top5"] = top5
    return results

class ImageNetZeroShotCallback(Callback):
    def __init__(
            self,
            data_path:os.PathLike,):
        """Callback for CLIP zero-shot image classification on the ImageNet-1k validation set.

        Args:
            data_path (os.PathLike): The path where the data is stored.
        """        
        super().__init__()
        eval_transform = data_utils.get_transform_pretraining(train=False, size=224)
        imagenet_args = {
            'data_path': data_path,
            'eval_transforms': eval_transform,
            'batch_size':256,
            'num_workers':5,
            'shuffle':False,
            'drop_last':False,
        }
        self.datamodule:ImageNetDataModule = DATAMODULE_REGISTRY['ImageNet'](**imagenet_args)

    def validate(self, trainer:Trainer, pl_module:LightningModule) -> None:
        """Run the zero-shot validation on the ImageNet-1k validation set and log the results.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The model/module to be evaluated.
        """        
        metrics = run_imagenet_zero_shot(
            pl_module=pl_module,
            dataloader=self.datamodule.val_dataloader(),
            device=pl_module.device,
        )

        for metric_key in metrics:
            pl_module.log(
                f"val/{metric_key}",
                metrics[metric_key],
                logger=True,
                rank_zero_only=True,
                on_epoch=True,
            )

    @torch.no_grad()
    @rank_zero_only
    def on_validation_start(self, trainer:Trainer, pl_module:LightningModule, **kwargs) -> None:
        """Pytorch Lightning hook that runs before the validation loop starts. Used to run the zero-shot validation
        during (before) the validation loop.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The model/module to be evaluated.
        """        
        self.datamodule.prepare_data()
        self.datamodule.setup(stage='fit')
        
        self.validate(trainer, pl_module)

        # destroy the dataloader to free up memory, as it is only required for validation
        self.datamodule.teardown(stage='fit')
