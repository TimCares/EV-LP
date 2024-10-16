"""
Various layers used in (multimodal) Transformer models.
Build using timm, VLMo and BEiT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import LayerScale
from timm.models.vision_transformer import Block as TimmBlock
from timm.layers import Mlp, DropPath
from typing import Optional
import torch.distributed as dist
from typing import Tuple, Union

class LayerNormLastBlock(TimmBlock):
    """
    Used for Data2Vec2Image model, where LayerNorm is applied after the attention block.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_x = self.attn(x)
        x = x + self.drop_path1(att_x)
        r = x = self.norm1(x)
        x = self.mlp(x)
        x = self.norm2(r + self.drop_path2(x))
        return x

class Mlp_(Mlp):
    """
    Transformer MLP block that also returns the raw output of the first linear layer.
    """    
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the MLP block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the raw output of
                the first linear layer (tuple[0] -> shape (B, T, D_ff)) and the output of the second linear
                layer (tuple[1] -> shape (B, T, D)).
        """        
        x = self.fc1(x)
        x_interm = x
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x_interm, x


class Attention(nn.Module):
    def __init__(
        self,
        dim:int,
        num_heads:int=12,
        qkv_bias:bool=False,
        qk_scale:float=None,
        attn_drop:float=0.0,
        proj_drop:float=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))
        
        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp_,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask)))
        x_res = x
        x_interm, x = self.mlp(self.norm2(x))
        x = x_res + self.drop_path2(self.ls2(x))
        return x_interm, x


# GatherLayer and  gather_features copied from BEiT-3 -> https://github.com/microsoft/unilm/blob/master/beit3/utils.py
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all devices with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """
    @staticmethod
    def forward(ctx, x:torch.Tensor) -> Tuple[torch.Tensor]:
        """Gathers all tensors from all devices.

        Args:
            x (torch.Tensor): The tensor to gather from all devices.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the gathered tensors. One element for each device.
        """        
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)
    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        """Computes the gradients for the gather operation.

        Args:
            *grads (Any): The gradients to reduce.

        Returns:
            torch.Tensor: The gradients for the gather operation on the current device.
        """        
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_features(
        image_features:torch.Tensor,
        text_features:torch.Tensor,
        padding_mask:torch.Tensor=None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Gathers the features from all devices. Useful for e.g. contrastive learning in distributed settings.

    Args:
        image_features (torch.Tensor): The image features to gather.
        text_features (torch.Tensor): The text features to gather.
        padding_mask (torch.Tensor, optional): The padding mask to gather. Defaults to None, meaning no padding mask is gathered.

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: A tuple containing the gathered
            image features (tuple[0]) and text features (tuple[1]). If a padding mask is provided,
            the tuple will also contain the gathered padding mask (tuple[2]).
            Results will be of shape (B*P, T, D) for image and text features, and (B*P, T) for the padding mask, where B is the batch size
            and P is the number of devices.
    """    
    gathered_image_features = GatherLayer.apply(image_features)
    gathered_text_features = GatherLayer.apply(text_features)
    all_image_features = torch.cat(gathered_image_features)
    all_text_features = torch.cat(gathered_text_features)
    
    if padding_mask is not None:
        gathered_padding_mask = GatherLayer.apply(padding_mask)
        all_padding_mask = torch.cat(gathered_padding_mask)
        return all_image_features, all_text_features, all_padding_mask

    return all_image_features, all_text_features


class Pooler(nn.Module):
    def __init__(self, hidden_size:int) -> None:
        """A simple pooling layer that pools the first token of the input sequence.
        Consists of a linear layer and a tanh activation function. Useful as an intermediate for e.g. image-text matching,
        or for classification tasks (e.g. in BERT).

        Args:
            hidden_size (int): Dimension of the input features.
        """        
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0] # hidden_states has shape (batch_size, seq_len, hidden_size)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output # shape (batch_size, hidden_size)
    

# copied from VLMo repo -> https://github.com/microsoft/unilm/blob/master/vlmo/vlmo/modules/multiway_transformer.py
class MoMEBlock(nn.Module):
    def __init__(
        self,
        dim:int,
        num_heads:int,
        mlp_ratio:float=4.0,
        qkv_bias:bool=False,
        qk_scale:bool=None,
        drop:float=0.0,
        attn_drop:float=0.0,
        drop_path:float=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_vlffn:bool=False,
        layer_scale_init_values:float=0.1,
        max_text_len:int=40,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_text = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_imag = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_vl = None
        if with_vlffn:
            self.mlp_vl = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vl = norm_layer(dim)
        
        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        self.max_text_len = max_text_len

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None, modality_type:str=None) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask))

        if modality_type == "image":
            x = x + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x)))
        elif modality_type == "text":
            x = x + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x)))
        else:
            if self.mlp_vl is None:
                x_text = x[:, : self.max_text_len]
                x_imag = x[:, self.max_text_len :]
                x_text = x_text + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x_text)))
                x_imag = x_imag + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
                x = torch.cat([x_text, x_imag], dim=1)
            else:
                x = x + self.drop_path(self.gamma_2 * self.mlp_vl(self.norm2_vl(x)))

        return x
