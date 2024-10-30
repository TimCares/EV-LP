import torch
from typing import Dict
from modules.layers import LayerNormLastBlock
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
import torch.nn as nn
import os
import logging

logger = logging.getLogger(__name__)

def preprocess_data2vec2_image_state_dict(state_dict:Dict[str, torch.Tensor], n_layers:int=None) -> Dict[str, torch.Tensor]:
    """Adjusts the keys of the state_dict of the Data2Vec2Image model to match the keys of the timm VisionTransformer model.

    Args:
        state_dict (Dict[str, torch.Tensor]): The state_dict with the pretrained weights of the Data2Vec2Image model.
        n_layers (int, optional): The number of layers to reuse. If all layers (12) should be used, set to None. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: The adjusted state_dict, which can be applied to the timm VisionTransformer model.
    """    
    new_sd = {}
    for key in list(state_dict.keys()):
        if 'decoder' in key or '_ema' in key:
            continue
        if 'blocks' in key:
            block_idx = int(key.split('.')[1])
            if n_layers is not None and block_idx > n_layers-1:
                continue
        new_key = key
        new_key = new_key.replace('modality_encoders.IMAGE.', '')
        new_key = new_key.replace('extra_tokens', 'cls_token')
        new_key = new_key.replace('local_encoder', 'patch_embed')
        new_key = new_key.replace('fixed_positional_encoder.positions', 'pos_embed')
        new_key = new_key.replace('context_encoder.norm', 'norm_pre')

        new_sd[new_key] = state_dict[key]
    return new_sd

def get_data2vec_image_model(state_dict_path:os.PathLike, n_layers:int=12) -> VisionTransformer:
    """Creates a timm VisionTransformer model with the weights of the Data2Vec2Image model.

    Args:
        state_dict_path (os.PathLike): Path to the state dict of the Data2Vec2Image model.
        n_layers (int, optional): The number of layers to reuse. Has a maximum of 12 layers. Defaults to 12.

    Returns:
        VisionTransformer: The timm VisionTransformer model with the weights of the Data2Vec2Image model.
    """
    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        global_pool='',
        embed_dim=768,
        depth=n_layers,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_norm=False,
        init_values=None,
        class_token=True,
        no_embed_class=True, # in Data2Vec2Image the class token is added after the positional encoding
        pre_norm=True,
        fc_norm=False,
        block_fn=LayerNormLastBlock,
    )
    vit.norm = nn.Identity() # Data2Vec2Image does not have a norm layer after the last block

    # workaround, as timm ViT does NOT use a bias in the patch embedding IF embeddings are normed before being passed
    # to the blocks
    # however, in Data2Vec2Image the patch embedding has a bias AND the embeddings are normed before being passed to the blocks
    vit.patch_embed = PatchEmbed(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        bias=True, # this is the important part
        dynamic_img_pad=False
    )

    state_dict = torch.load(state_dict_path)['model']
    state_dict = preprocess_data2vec2_image_state_dict(state_dict, n_layers=n_layers)
    result = vit.load_state_dict(state_dict)
    logger.info(f"Loaded Data2VecImage state dict with result: {result}")
    return vit
