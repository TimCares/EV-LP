from omegaconf import OmegaConf
import torch
import logging
from functools import partial
import logging
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from collections import OrderedDict
from typing import List, Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
from data2vec_fairseq.models.data2vec2 import Data2VecMultiModel
from data2vec_fairseq.models.data2vec2 import Data2VecMultiConfig
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from enum import Enum, auto

logger = logging.getLogger(__name__)

class Modality(Enum):
    """
    Enum class to set the modalities of data.
    """    
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()
    VL = auto() # Vision-Language
    VA = auto() # Vision-Audio
    LA = auto() # Language-Audio
    DUMMY = auto()

def load_model(pretrained_model_cfg:DictConfig,
               model_state_dict:OrderedDict[str, torch.Tensor]) -> Data2VecMultiModel:
    """Loads a pretrained Data2Vec2 model.

    Args:
        pretrained_model_cfg (DictConfig): OmegaConf config of the model, is merged with the dataclass for the Data2Vec2 model.
        model_state_dict (OrderedDict[str, torch.Tensor]): Pytorch state dict of the (pretrained) model.

    Returns:
        Data2VecMultiModel: The initialized Data2Vec2 model.
    """    
    
    pretrained_model_cfg = OmegaConf.merge(Data2VecMultiConfig(), pretrained_model_cfg)

    model = Data2VecMultiModel.build_model(pretrained_model_cfg)

    result = model.load_state_dict(model_state_dict)
    logger.info(f'Loaded Data2Vec2 state dict, result: {result}')
    return model


def load_pretrained_d2v_model(state_dict_path:str, keep_decoder:bool=False, remove_dropout:bool=False) -> Data2VecMultiModel:
    """Loads a pretrained Data2Vec2 model from a state dict.

    Args:
        state_dict_path (str): Path to the state dict.
        keep_decoder (bool, optional): Whether to keep the decoder of the Data2Vec2 model. Defaults to False, meaning the decoder is removed.
            Should be set to False if the model is used for inference only.
        remove_dropout (bool, optional): If the dropout layers of the Data2Vec2 model should be set to p=0.0. Defaults to False.

    Returns:
        Data2VecMultiModel: The pretrained Data2Vec2 model.
    """    
    model_meta_data = torch.load(state_dict_path)
    pretrained_model_cfg = OmegaConf.create(model_meta_data['cfg']['model'])
    if remove_dropout:
        for k in pretrained_model_cfg.keys():
            if 'drop' in k:
                pretrained_model_cfg[k] = 0.0 # set dropout to 0.0
    model = load_model(pretrained_model_cfg=pretrained_model_cfg, model_state_dict=model_meta_data['model'])

    # removes decoder, and all encoders with modality != supported modality
    model.remove_pretraining_modules(modality=pretrained_model_cfg.supported_modality, keep_decoder=keep_decoder)

    return model


def pad_text_sequence(tokens:List[int],
                      num_max_bpe_tokens:int,
                      pad_idx:int,
                      bos_idx:int,
                      eos_idx:int) -> Tuple[List[int], List[int]]:
    """Pads a text sequence with padding tokens, and adds the beginning and end of sequence tokens if not already present.

    Args:
        tokens (List[int]): The tokens of the text sequence, each token is an integer and one element in the list.
        num_max_bpe_tokens (int): The length to which the sequence should be padded.
        pad_idx (int): The padding token index.
        bos_idx (int): The beginning of sequence token index.
        eos_idx (int): The end of sequence token index.

    Returns:
        Tuple[List[int], List[int]]: A tuple of the padded text sequence (tuple[0]) and the padding mask (tuple[1]).
    """    
    if len(tokens) > num_max_bpe_tokens - 2:
        tokens = tokens[:num_max_bpe_tokens - 2]
    tokens = ([bos_idx] if tokens[0]!=bos_idx else []) + tokens + ([eos_idx] if tokens[-1]!=eos_idx else [])
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (num_max_bpe_tokens - num_tokens)
    language_tokens = tokens + [pad_idx] * (num_max_bpe_tokens - num_tokens)

    return language_tokens, padding_mask


def prepare_output(out:List[torch.Tensor], modality:Optional[Modality]=None, norm:bool=True) -> torch.Tensor:
    """Prepares the layer activations of a model for the Data2Vec loss.

    Args:
        out (List[torch.Tensor]): The layer activations of the model. Each element in the list is a tensor of shape (B, T, C),
            and represents the activations of one layer.
        modality (Optional[Modality], optional): The modality of the model that generated the activations. 
            If it is Modality.IMAGE and not None, then layer norm will be applied after averaging along the layer dimension.
            Defaults to None.
        norm (bool, optional): Whether to apply instance norm along the time dimension. Instance norm is applied
            on each layer seperately. Defaults to True.

    Returns:
        torch.Tensor: The average activations of the layers for each time step. Output shape is (B, T, C) and can be used for
            the mse loss.
    """    
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
    """Puts a module in evaluation mode and freezes the parameters.

    Args:
        module (nn.Module): The module to freeze.
    """    
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

def unfreeze_module(module:nn.Module) -> None:
    """Puts a module in training mode and unfreezes the parameters.

    Args:
        module (nn.Module): The module to unfreeze.
    """    
    for param in module.parameters():
        param.requires_grad = True
    module.train()

def load_beit2_teacher(sd_path:str, **kwargs) -> VisionTransformerForMaskedImageModeling:
    """Loads a pretrained BEiT2 model from a state dict.

    Args:
        sd_path (str): The path to the state dict.

    Returns:
        VisionTransformerForMaskedImageModeling: The loaded BEiT2 model.
    """    
    sd = torch.load(sd_path)['model']
    for key in list(sd.keys()):
        if "cls_pt_layers" in key: # remove the patch aggregation layers, only needed for pretraining
            del sd[key]

    beit2 = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    result = beit2.load_state_dict(sd)
    logger.info(f"Loaded BEiT2 teacher state dict with result: {result}")
    del beit2.lm_head
    return beit2

# from BEiT-3 repo: https://github.com/microsoft/unilm/blob/master/beit3/modeling_utils.py
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def init_weights(m:nn.Module) -> None:
    """Initializes the weights of a model in place.

    Args:
        m (nn.Module): The model to initialize.
    """    
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
