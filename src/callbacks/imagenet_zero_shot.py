import logging
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
import sys
import os
sys.path.append('..')
from datamodules import DATAMODULE_REGISTRY
from data.imagenet_zeroshot_data import (
    imagenet_classnames,
    openai_imagenet_template,
)
from rich.progress import track
from utils import pad_text_sequence # src/utils.py
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

def _zero_shot_classifier(pl_module, device):
    MAX_SEQ_LEN = 22 # max sequence length for OpenAI's templates with BERT tokenizer
    tokenizer:BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    zeroshot_weights = []
    for classname in track(imagenet_classnames, description="Building classifier"):
        texts = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(template(classname)))
                 for template in openai_imagenet_template]
        padding_masks = []
        for i in range(len(texts)):
            language_tokens, padding_mask = pad_text_sequence(tokens=texts[i], num_max_bpe_tokens=MAX_SEQ_LEN,
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
        # check for fp16 not needed here -> texts is long tensor and will be converted by embedding table to correct dtype
        class_embeddings = pl_module.model.encode_text(text=texts, padding_mask=padding_masks)['x']
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def _n_correct(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        correct[:k].reshape(-1).float().sum().cpu().item()
        for k in topk
    ]


@rank_zero_only
def run_imagenet_zero_shot(pl_module,
                             dataloader:DataLoader,
                             device,):
    classifier = _zero_shot_classifier(pl_module, device)
    top1, top5, n = 0.0, 0.0, 0.0
    for sample in track(dataloader, description=f"ImageNet-1k CLIP zero-shot validation"):
        images = sample["image"]
        target = sample["target"]
        images = images.to(device)
        target = target.to(device)
        if pl_module.dtype == torch.float16: # when using deep speed
            images = images.half()
        image_features = pl_module.model.encode_image(image=images)['x']
        logits = 100.0 * image_features @ classifier

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
            data_path: os.PathLike,):
        super().__init__()
        imagenet_args = {
            'data_path': data_path,
            'pretraining': True,
            'batch_size':256,
            'num_workers':5,
            'shuffle':False,
            'drop_last':False,
        }
        self.datamodule = DATAMODULE_REGISTRY['imagenet'](**imagenet_args)

    def validate(self, trainer, pl_module) -> None:
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
    def on_validation_start(self, trainer, pl_module, **kwargs) -> None:

        for name_key in self.datamodules.keys(): # setup datamodules
            self.datamodules[name_key].prepare_data()
            self.datamodules[name_key].setup(stage='fit')
            self.datamodules[name_key].setup(stage='test')
        
        self.validate(trainer, pl_module)

        for name_key in self.datamodules.keys(): # cleanup datamodules
            self.datamodules[name_key].teardown(stage='fit')
            self.datamodules[name_key].teardown(stage='test')
