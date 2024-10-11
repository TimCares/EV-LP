from omegaconf import OmegaConf
import os
import torch
import logging
from functools import partial
from collections import namedtuple
import logging
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from collections import OrderedDict
from typing import List, Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
from data2vec_fairseq.models.data2vec2 import Data2VecMultiModel
from data2vec_fairseq.models.data2vec2 import Data2VecMultiConfig
from data2vec_fairseq.data.modality import Modality
from fairseq.data import Dictionary
from fairseq.dataclass.utils import merge_with_parent
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling

logger = logging.getLogger(__name__)

def load_model(pretrained_model_cfg:DictConfig,
               model_state_dict:OrderedDict[str, torch.Tensor]) -> Data2VecMultiModel:
    
    pretrained_model_cfg = merge_with_parent(Data2VecMultiConfig(), pretrained_model_cfg, remove_missing=True)

    logger.info(f"Modality used: {pretrained_model_cfg.supported_modality}")
    if pretrained_model_cfg.supported_modality.name.lower() == 'text':
        Task = namedtuple('Task', 'source_dictionary')

        dictionary = Dictionary.load(os.path.join('..', 'data', "dict.txt"))
        dictionary.add_symbol("<mask>")
        dummy_task = Task(source_dictionary=dictionary)
    else:
        dummy_task = False

    model = Data2VecMultiModel.build_model(pretrained_model_cfg, task=dummy_task)

    result = model.load_state_dict(model_state_dict)
    logger.info(f'Loaded state dict, result: {result}')
    return model


def load_pretrained_d2v_model(state_dict_path:str, keep_decoder:bool=False, remove_dropout:bool=False) -> Data2VecMultiModel:
    model_meta_data = torch.load(state_dict_path)
    pretrained_model_cfg = OmegaConf.create(model_meta_data['cfg']['model'])
    if remove_dropout:
        for k in pretrained_model_cfg.keys():
            if 'drop' in k:
                pretrained_model_cfg[k] = 0.0
    model = load_model(pretrained_model_cfg=pretrained_model_cfg, model_state_dict=model_meta_data['model'])

    # removes decoder, and all encoders with modality != supported modality
    model.remove_pretraining_modules(modality=pretrained_model_cfg.supported_modality, keep_decoder=keep_decoder)

    return model


def pad_text_sequence(tokens:List[int],
                      num_max_bpe_tokens:int,
                      pad_idx:int,
                      bos_idx:int,
                      eos_idx:int) -> Tuple[List[int], List[int]]:
    
    if len(tokens) > num_max_bpe_tokens - 2:
        tokens = tokens[:num_max_bpe_tokens - 2]
    tokens = ([bos_idx] if tokens[0]!=bos_idx else []) + tokens + ([eos_idx] if tokens[-1]!=eos_idx else [])
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (num_max_bpe_tokens - num_tokens)
    language_tokens = tokens + [pad_idx] * (num_max_bpe_tokens - num_tokens)

    return language_tokens, padding_mask


def prepare_output(out:List[torch.Tensor], modality:Optional[Modality]=None, norm:bool=True) -> torch.Tensor:
    if norm:
        out = [
            F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2)
            for tl in out  # BTC -> BCT -> BTC
        ]

    y = out[0].float()
    for tl in out[1:]:
        y.add_(tl.float())
    y = y.div_(len(out))

    if modality is not None and modality == Modality.IMAGE:
        y = F.layer_norm(y, y.shape[-1:])
    return y

def freeze_module(module:nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

def unfreeze_module(module:nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True
    module.train()

def load_beit2_teacher(sd_path:str, **kwargs) -> VisionTransformerForMaskedImageModeling:
    sd = torch.load(sd_path)['model']
    for key in list(sd.keys()):
        if "cls_pt_layers" in key:
            del sd[key]

    beit2 = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    result = beit2.load_state_dict(sd)
    logger.info(f"Loaded BEiT2 teacher state dict with result: {result}")
    del beit2.lm_head
    return beit2
