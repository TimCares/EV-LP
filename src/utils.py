import torch
import logging
from typing import List, Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
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

def pad_text_sequence(tokens:List[int],
                      max_seq_len:int,
                      pad_token_id:int,
                      cls_token_id:int,
                      sep_token_id:int) -> Tuple[List[int], List[int]]:
    """Pads a text sequence with padding tokens, and adds the beginning and end of sequence tokens if not already present.

    Args:
        tokens (List[int]): The tokens of the text sequence, each token is an integer and one element in the list.
        max_seq_len (int): The length to which the sequence should be padded.
        pad_token_id (int): The padding token index.
        cls_token_id (int): The beginning of sequence token index.
        sep_token_id (int): The end of sequence token index.

    Returns:
        Tuple[List[int], List[int]]: A tuple of the padded text sequence (tuple[0]) and the padding mask (tuple[1]).
    """    
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[:max_seq_len - 2]
    tokens = ([cls_token_id] if tokens[0]!=cls_token_id else []) + tokens + ([sep_token_id] if tokens[-1]!=sep_token_id else [])
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_seq_len - num_tokens)
    language_tokens = tokens + [pad_token_id] * (max_seq_len - num_tokens)

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
