"""
This module contains the implementation of various loss functions used in the project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import gather_features, GatherLayer
from .cmli import infer_cmli_logits
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def mask_eos(padding_masks:torch.Tensor) -> torch.Tensor:
    """Mask the last non-padding token in the input tensor, i.e. the end of sequence token.

    Args:
        padding_masks (torch.Tensor): The padding masks, of shape (batch_size, num_tokens).
            Non-masked time steps are indicated by 0, masked time steps by 1.

    Returns:
        torch.Tensor: The padding masks with the last non-padding token masked.
    """
    # (padding_masks == 0).cumsum(dim=1) -> yields a tensor where the first occurence of 0 is 1, the second 2, etc.
    # argmax(dim=1) -> yields the index of the last occurence of 0, which is the last non-padding token (the end of sequence token)
    last_zero_indices = (padding_masks == 0).cumsum(dim=1).argmax(dim=1)
    padding_masks[torch.arange(padding_masks.size(0)), last_zero_indices] = 1
    return padding_masks

class CachedLabelContrastiveLoss(nn.Module):
    def __init__(
            self,
            cache_labels:bool=False,
            rank:int=0,
            world_size:int=1,
    ):
        """Supertype for any contrastive-based loss that (potentially) works in a distributed setting
        and allows for caching of labels.

        Args:
            cache_labels (bool, optional): Whether to cache the labels for the loss of the subtype. Defaults to False.
            rank (int, optional): The rank of the current worker/device. Defaults to 0.
            world_size (int, optional): How many devices/workers there are. Defaults to 1, i.e. no distributed setting
                and only one device.
        """        
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_labels(self, num_logits:int, device:str) -> torch.Tensor:
        """Gets the labels for the loss calculation. If caching is enabled, the labels are cached for the given device.

        Args:
            num_logits (int): How many classes there are. In the case of contrastive losses, this is the number of
                negative samples plus the one positive sample. If e.g. batch size is 256 with 2 devices, then num_logits
                would be 512.
            device (str): On which device the labels should be generated.

        Returns:
            torch.Tensor: The labels for the loss calculation.
        """
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels


# The implementation code is modified from open_clip (https://github.com/mlfoundations/open_clip.git)
class ClipLoss(CachedLabelContrastiveLoss):
    """
    Implements the image-text contrastive loss from CLIP. Supports distributed training.
    """    
    def forward(
        self,
        image_features:torch.Tensor,
        text_features:torch.Tensor,
        logit_scale:torch.Tensor,
        gather:bool=True
    ) -> Dict[str, torch.Tensor]:
        """Computes the image-text contrastive loss.
        Features are expected to be normalized.

        Args:
            image_features (torch.Tensor): The image features, of shape (batch_size, feature_dim).
            text_features (torch.Tensor): The text features, of shape (batch_size, feature_dim).
            logit_scale (torch.Tensor): By which factor to scale the similarity scores. Only useful during training.
                If no scaling is desired, set to 1.
            gather (bool, optional): Whether to gather the features if in a distributed setting. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the total loss (key "loss"), the image loss (key "image_loss"),
                the text loss (key "text_loss"), the logits per image (key "logits_per_image"),
                the logits per text (key "logits_per_text"), and the targets/labels (key "targets").
        """        
        device = image_features.device
        if self.world_size > 1 and gather:
            all_image_features, all_text_features = gather_features(
                image_features, text_features
            )

            logits_per_image = logit_scale * image_features @ all_text_features.T
            logits_per_text = logit_scale * text_features @ all_image_features.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        labels = self.get_labels(logits_per_image.shape[0], device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        total_loss = (image_loss + text_loss) / 2
        
        out_dict = {
            'loss': total_loss,
            'image_loss': image_loss,
            'text_loss': text_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict
    

class KDClipLoss(CachedLabelContrastiveLoss):
    """
    Computes the Contrastive Distillation loss between image features from the student and teacher (target) model,
    and text features from the student and image features (target) from the teacher model. Supports distributed training.
    """    
    def forward(
        self,
        input_image:torch.Tensor,
        input_text:torch.Tensor,
        target:torch.Tensor,
        logit_scale:torch.Tensor,
        gather:bool=True
    ) -> Dict[str, torch.Tensor]:
        """Computes the loss.
        Features are expected to be normalized.

        Args:
            input_image (torch.Tensor): The image features of the student, of shape (batch_size, feature_dim).
            input_text (torch.Tensor): The text features of the student, of shape (batch_size, feature_dim).
            target (torch.Tensor): The image features of the teacher, of shape (batch_size, feature_dim).
            logit_scale (torch.Tensor): By which factor to scale the similarity scores. Only useful during training.
                If no scaling is desired, set to 1.
            gather (bool, optional): Whether to gather the features if in a distributed setting. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the total loss (key "loss"), the image loss (key "image_loss"),
                the text loss (key "text_loss"), the logits per image (key "logits_per_image"),
                the logits per text (key "logits_per_text"), and the targets/labels (key "targets").
        """    
        device = input_image.device
        if self.world_size > 1 and gather:
            all_target = GatherLayer.apply(target)
            all_target = torch.cat(all_target)

            logits_per_image = logit_scale * input_image @ all_target.T
            logits_per_text = logit_scale * input_text @ all_target.T
        else:
            logits_per_image = logit_scale * input_image @ target.T
            logits_per_text = logit_scale * input_text @ target.T

        labels = self.get_labels(logits_per_image.shape[0], device)

        image_loss = F.cross_entropy(logits_per_image, labels) # image-to-image loss (i2i)
        text_loss = F.cross_entropy(logits_per_text, labels) # text-to-image loss (t2i)

        total_loss = (image_loss + text_loss) / 2
        
        out_dict = {
            'loss': total_loss,
            'image_loss': image_loss,
            'text_loss': text_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict


class ClipMomentumMemoryBankLoss(nn.Module):
    def __init__(
        self,
        embed_size:int=768,
        size:int=16384, # 2^14
        half_precision:bool=False,
        device:str='cuda',
        world_size:int=1
    ):
        """Implements the image-text contrastive loss with memory bank, maintained by a momentum encoder.

        Args:
            embed_size (int, optional): Dimension of the embeddings. Defaults to 768.
            size (int, optional): Size of the memory bank. Defaults to 16384.
            device (str, optional): Device on which the memory bank should be stored. Defaults to 'cuda'.
            world_size (int, optional): How many devices there are. Defaults to 1.
        """        
        super().__init__()
        self.world_size = world_size
        self.size = size
        assert self.size > 0, "Size of memory bank must be larger than batch size"
        
        if half_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # init random memory bank for image representations
        imb_tmp = torch.rand((self.size, embed_size), dtype=dtype, device=device, requires_grad=False)
        imb_tmp = imb_tmp / imb_tmp.norm(dim=-1, keepdim=True)
        self.register_buffer('image_memory_bank', imb_tmp) # not a parameter, so register as buffer

        # init random memory bank for text representations
        tmb_tmp = torch.rand((self.size, embed_size), dtype=dtype, device=device, requires_grad=False)
        tmb_tmp = tmb_tmp / tmb_tmp.norm(dim=-1, keepdim=True)
        self.register_buffer('text_memory_bank', tmb_tmp) # not a parameter, so register as buffer
        self.index_pointer = 0 # we fill from the beginning

    def _update(self, img_emb:torch.Tensor, text_emb:torch.Tensor) -> None:
        """Updates the memory bank with new image and text embeddings. They should be normalized
        and come from a momentum encoder.

        Args:
            img_emb (torch.Tensor): The image embeddings.
            text_emb (torch.Tensor): The text embeddings.
        """        
        if self.world_size > 1:
            img_emb, text_emb = gather_features(
                img_emb,
                text_emb,
            )

        bsz = img_emb.shape[0]
        assert bsz == text_emb.shape[0] # ensure image and text have the same batch size
        assert self.size % bsz == 0 # ensure they fit

        end_idx = self.index_pointer + bsz # find the end index by advancing the pointer by the batch size
        # detach to avoid backpropagation on older embeddings
        self.image_memory_bank[self.index_pointer:end_idx] = img_emb.detach()
        self.text_memory_bank[self.index_pointer:end_idx] = text_emb.detach()

        self.index_pointer = end_idx % self.size # advance the pointer and wrap around if necessary
    
    def forward(
            self,
            image_features:torch.Tensor,
            text_features:torch.Tensor,
            image_features_m:torch.Tensor,
            text_features_m:torch.Tensor,
            logit_scale:torch.Tensor,) -> Dict[str, torch.Tensor]:
        return self.compute_loss(
            logit_scale=logit_scale, 
            image_features=image_features,
            text_features=text_features,
            image_features_m=image_features_m,
            text_features_m=text_features_m,
        )

    def compute_loss(
            self,
            image_features:torch.Tensor,
            text_features:torch.Tensor,
            image_features_m:torch.Tensor,
            text_features_m:torch.Tensor,
            logit_scale:torch.Tensor,) -> Dict[str, torch.Tensor]:
        """Computes the image-text contrastive loss with memory bank.

        Args:
            image_features (torch.Tensor): The image features of the current batch, of shape (batch_size, feature_dim).
            text_features (torch.Tensor): The text features of the current batch, of shape (batch_size, feature_dim).
            image_features_m (torch.Tensor): The image features of the current batch, produced by the momentum encoder,
                of shape (batch_size, feature_dim).
            text_features_m (torch.Tensor): The text features of the current batch, produced by the momentum encoder,
                of shape (batch_size, feature_dim).
            logit_scale (torch.Tensor): The scaling factor for the logits. Only useful during training. If no scaling is desired,
                set to 1.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the total loss (key "loss"),
                the logits per image (key "logits_per_image"), the logits per text (key "logits_per_text"),
                and the targets/labels (key "targets").
        """        
        device = image_features.device
        
        # compute the contrastive loss between the current image embeddings with the current text embeddings of the momentum encoder
        # AND the complete text memory bank
        logits_per_image = logit_scale * image_features @ torch.cat([text_features_m, self.text_memory_bank], dim=0).t()
        # compute the contrastive loss between the current text embeddings with the current image embeddings of the momentum encoder
        # AND the complete image memory bank
        logits_per_text = logit_scale * text_features @ torch.cat([image_features_m, self.image_memory_bank], dim=0).t()

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        itc_loss = (
            F.cross_entropy(logits_per_image.float(), labels) # image-to-text loss
            + F.cross_entropy(logits_per_text.float(), labels) # text-to-image loss
        ) / 2

        out_dict = {
            'loss': itc_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict
    

class KDClipMomentumMemoryBankLoss(nn.Module):
    def __init__(
        self,
        embed_size:int=768,
        size:int=16384, # 2^14
        device:str='cuda',
        world_size:int=1,
    ):
        """Implements the contrastive distillation loss with memory bank. Representations passed to the
        "_update" amd "forward" methods should be normalized and come from a frozen teacher model.

        Args:
            embed_size (int, optional): Dimension of the embeddings. Defaults to 768.
            size (int, optional): Size of the memory bank. Defaults to 16384.
            device (str, optional): Device on which the memory bank should be stored. Defaults to 'cuda'.
            world_size (int, optional): How many devices there are. Defaults to 1.
        """
        super().__init__()
        self.world_size = world_size
        self.size = size
        
        # init random memory bank, stores image embeddings of the teacher
        target_mb_ = torch.rand((self.size, embed_size), device=device, requires_grad=False)
        target_mb_ = target_mb_ / target_mb_.norm(dim=-1, keepdim=True)
        self.register_buffer('target_memory_bank', target_mb_) # not a parameter, so register as buffer
        self.index_pointer = 0

    def _update(self, target:torch.Tensor) -> None:
        """Updates the memory bank with new (image) target embeddings.
        They should be normalized and come from a frozen teacher model.

        Args:
            target (torch.Tensor): The target embeddings.
        """        
        if self.world_size > 1:
            all_target = GatherLayer.apply(target)
            all_target = torch.cat(all_target)

        bsz = all_target.shape[0]
        assert self.size % bsz == 0 # ensure they fit

        end_idx = self.index_pointer + bsz # find the end index by advancing the pointer by the batch size
        # detach to avoid backpropagation on older embeddings
        self.target_memory_bank[self.index_pointer:end_idx] = all_target.detach()

        self.index_pointer = end_idx % self.size # advance the pointer and wrap around if necessary
    
    def forward(
            self,
            input_image:torch.Tensor,
            input_text:torch.Tensor,
            target:torch.Tensor,
            logit_scale:torch.Tensor,) -> Dict[str, torch.Tensor]:
        return self.compute_loss(
            logit_scale=logit_scale, 
            input_image=input_image,
            input_text=input_text,
            target=target,
        )

    def compute_loss(
            self,
            input_image:torch.Tensor,
            input_text:torch.Tensor,
            target:torch.Tensor,
            logit_scale:torch.Tensor,) -> Dict[str, torch.Tensor]:
        """Computes the contrastive distillation loss with memory bank.

        Args:
            input_image (torch.Tensor): The image features of the current batch, produced by the student model,
                of shape (batch_size, feature_dim).
            input_text (torch.Tensor): The text features of the current batch, produced by the student model,
                of shape (batch_size, feature_dim).
            target (torch.Tensor): The image features of the current batch, produced by the teacher model,
                of shape (batch_size, feature_dim).
            logit_scale (torch.Tensor): The scaling factor for the logits. Only useful during training. If no scaling is desired,
                set to 1.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the total loss (key "loss"), the image loss (key "image_loss"),
                the text loss (key "text_loss"), the logits per image (key "logits_per_image"),
                the logits per text (key "logits_per_text"), and the targets/labels (key "targets").
        """        
        device = input_image.device
        
        all_target = torch.cat([target, self.target_memory_bank], dim=0).t()
        logits_per_image = logit_scale * input_image @ all_target
        logits_per_text = logit_scale * input_text @ all_target

        labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels) # image-to-image loss (i2i)
        text_loss = F.cross_entropy(logits_per_text, labels) # text-to-image loss (t2i)

        total_loss = (image_loss + text_loss) / 2

        out_dict = {
            'loss': total_loss,
            'image_loss': image_loss,
            'text_loss': text_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict


class CMLILoss(CachedLabelContrastiveLoss):
    """
    Implements the Cross-Modal Late Interaction (CMLI) loss from FILIP: https://arxiv.org/pdf/2111.07783.
    Supports distributed training.
    """
    def _gather(
        self,
        image_features:torch.Tensor,
        text_features:torch.Tensor,
        padding_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Garthers the features if in a distributed setting. Otherwise, returns the input features.

        Args:
            image_features (torch.Tensor): The image features to gather.
            text_features (torch.Tensor): The text features to gather.
            padding_mask (torch.Tensor): The padding mask to gather.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of the gathered image features (tuple[0]),
                text features (tuple[1]), and padding mask (tuple[2]).
        """        
        if self.world_size > 1:
            all_image_features, all_text_features, all_padding_mask = gather_features(
                image_features, text_features, padding_mask
            )
        else:
            all_image_features = image_features
            all_text_features = text_features
            all_padding_mask = padding_mask
        return all_image_features, all_text_features, all_padding_mask
    
    def forward(
        self,
        image_features:torch.Tensor,
        text_features:torch.Tensor,
        padding_mask:torch.Tensor,
        logit_scale:torch.Tensor=torch.tensor(1.0),
    ) -> Dict[str, torch.Tensor]:
        """Computes the Cross-Modal Late Interaction (CMLI) loss.

        Args:
            image_features (torch.Tensor): The image features, of shape (batch_size, num_patches+1, feature_dim).
            text_features (torch.Tensor): The text features, of shape (batch_size, num_tokens, feature_dim).
            padding_mask (torch.Tensor): The padding mask for the text features, of shape (batch_size, num_tokens).
            logit_scale (torch.Tensor, optional): The scaling factor for the logits. Only useful during training.
                If no scaling is desired, set to "torch.tensor(1.0)"/"1". Defaults to torch.tensor(1.0).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the total loss (key "loss"), the logits per image (key "logits_per_image"),
                the logits per text (key "logits_per_text"), and the targets/labels (key "targets").
        """        

        # eos token is does not represent an actual token that is part of a sentence.
        # we only want the similarity between actual (sub-)words and image patches
        padding_mask = mask_eos(padding_mask)
        image_features, text_features, padding_mask = self._gather(
            image_features, text_features, padding_mask
        )

        # actually compute the similarity scores between all possible text-image pairs in the batch
        cmli_logits = infer_cmli_logits(
            text_features=text_features,
            image_features=image_features,
            padding_mask=padding_mask,
            logit_scale=logit_scale
        )

        logits_per_image = cmli_logits['i2t']
        
        logits_per_text = cmli_logits['t2i']

        labels = self.get_labels(logits_per_image.shape[0], logits_per_image.device)

        cmli_loss = (
            F.cross_entropy(logits_per_image, labels) + # image-to-text loss (i2t)
            F.cross_entropy(logits_per_text, labels) # text-to-image loss (t2i)
            ) / 2
        
        out_dict = {
            'loss': cmli_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict
