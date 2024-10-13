"""
This module contains implementations of the Cross-Modal Late Interaction (CMLI) method for image-text retrieval
from the paper FILIP: https://arxiv.org/abs/2111.07783
Code for cmli logits heavily borrowed from x-clip -> https://github.com/lucidrains/x-clip
(https://github.com/lucidrains/x-clip/blob/main/x_clip/x_clip.py)
"""
import torch
import einops
import opt_einsum
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def max_neg_value(dtype:torch.dtype) -> torch.Tensor:
    """Returns the maximum negative value for a given dtype.

    Args:
        dtype (torch.dtype): The pytorch dtype.

    Returns:
        torch.Tensor: The maximum negative value for the given dtype.
    """    
    return -torch.finfo(dtype).max

def masked_mean(t:torch.Tensor, mask:torch.Tensor, dim:int=1, eps:int=1e-6) -> torch.Tensor:
    """Computes the masked mean of a tensor along a given dimension.
    Indices where the mask is True (1) are ignored in the mean computation.

    Args:
        t (torch.Tensor): The tensor to compute the mean of.
        mask (torch.Tensor): The mask tensor, where 1 indicates that the value should be masked/ignored.
        dim (int, optional): The dimension along which to compute the mean. Defaults to 1.
        eps (int, optional): A small value to avoide divion by 0. Defaults to 1e-6.

    Returns:
        torch.Tensor: The masked mean of the tensor along the given dimension.
    """    
    t = t.masked_fill(mask.bool(), 0.) # prepare sum, masked values are set to 0
    numer = t.sum(dim = dim)
    denom = (~mask.bool()).sum(dim = dim).clamp(min = eps) # get the number of non-masked values, clamp to avoid division by 0
    return numer / denom

def reduce_token_similarity(
        sim:torch.Tensor,
        padding_mask:torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Calculates the image-to-text and text-to-image similarity scores by reducing the similarity tensor
    along the corresponding dimensions.

    Args:
        sim (torch.Tensor): The similarity tensor, of shape (batch_size, batch_size, num_tokens-1, num_patches).
            The CLS token is removed from the input features, so -1 in the num_tokens dimension.
        padding_mask (torch.Tensor): The text padding mask, of shape (batch_size, num_tokens-1).

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the image-to-text (key "i2t") and text-to-image (key "t2i") similarity scores.
    """    
    # find the most similar image patch for each token
    text_to_image = einops.reduce(sim, '... t i -> ... t', 'max')
    text_to_image_mask = einops.rearrange(padding_mask, 'b t -> b 1 t')
    # compute the mean similarity score for each each text to each image, by averaging the similarity scores of all non-padding tokens
    # to their most similar image patch for each text-image pair
    text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1) # (batch_size, batch_size)

    image_to_text_mask = einops.rearrange(padding_mask, 'b t -> b 1 t 1')
    masked_sim = sim.masked_fill(image_to_text_mask.bool(), max_neg_value(sim.dtype))
    # find the most similar token for each image patch and
    # compute the mean similarity score for each image to each text, by averaging the similarity scores of all image patches
    # to their most similar token
    image_to_text = einops.reduce(einops.reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')

    return {"i2t": image_to_text, "t2i": text_to_image}

def to_half(text_features:torch.Tensor, image_features:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast image and text features to half precision. This is used to reduce memory usage.

    Args:
        text_features (torch.Tensor): The text features.
        image_features (torch.Tensor): The image features.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of the text (tuple[0]) and image (tuple[1]) features in half precision.
    """    
    text_features = text_features.half()
    image_features = image_features.half()
    return text_features, image_features

def remove_cls(
    text_features:torch.Tensor, 
    image_features:torch.Tensor, 
    padding_mask:torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Removes the CLS token, i.e. the first token, from the input features and padding mask.

    Args:
        text_features (torch.Tensor): The text features.
        image_features (torch.Tensor): The image features.
        padding_mask (torch.Tensor): The padding mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of the text (tuple[0])
            and image (tuple[1]) features, and the padding mask (tuple[2]) with the CLS token removed.
    """    
    text_features = text_features[:, 1:]
    image_features = image_features[:, 1:]
    padding_mask = padding_mask[:, 1:]
    return text_features, image_features, padding_mask

def infer_cmli_logits(
    text_features:torch.Tensor,
    image_features:torch.Tensor,
    padding_mask:torch.Tensor,
    logit_scale:float|torch.Tensor=1.0,
) -> Dict[str, torch.Tensor]:
    """Computes Cross-Modal Late Interaction (CMLI) between text and image features.
    Features are first cast to half precision and the CLS token is removed.
    Features should already be normalized.

    Args:
        text_features (torch.Tensor): The text features, of shape (batch_size, num_tokens, feature_dim).
        image_features (torch.Tensor): The image features, of shape (batch_size, num_patches+1, feature_dim).
            The first element in the second dimension is the CLS token, so +1.
        padding_mask (torch.Tensor): The text padding mask, of shape (batch_size, num_tokens).
        logit_scale (float | torch.Tensor, optional): By which factor to scale the similarity scores. Only useful
            during training. Defaults to 1.0, i.e. no scaling.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the image-to-text (key "i2t") and text-to-image (key "t2i") similarity scores.
    """    
    
    text_features, image_features = to_half(text_features, image_features)
    text_features, image_features, padding_mask = remove_cls(text_features, image_features, padding_mask)
    
    # compute the similarity scores between all text tokens and image patches for each possible text-image pair
    # -> (batch_size, batch_size, num_tokens, num_patches)
    sim = logit_scale * opt_einsum.contract('x t d, y i d -> x y t i', text_features, image_features)

    return reduce_token_similarity(sim, padding_mask) # generate the image-to-text and text-to-image similarity scores
