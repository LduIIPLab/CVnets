
import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from ..layers import (
    get_normalization_layer,
    ConvLayer,
    Dropout,
    LinearSelfAttention,
)

from ..modules import BaseModule
from ..misc.profiler import module_profile


class LocationPreservingVit(BaseModule):
    """
    This class defines the location preserving vision transformer with linear self-attention in `EHFR_Net paper <>`_
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        ffn_latent_dim: int,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.1,
        ffn_dropout: Optional[float] = 0.0,
        norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        attn_unit1 = LinearSelfAttention(
            opts, embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn1 = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim
            ),
            attn_unit1,
            Dropout(p=dropout),
        )

        attn_unit2 = LinearSelfAttention(
            opts, embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn2 = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim
            ),
            attn_unit2,
            Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim
            ),
            ConvLayer(
                opts=opts,
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=True,
            ),
            Dropout(p=ffn_dropout),
            ConvLayer(
                opts=opts,
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=False,
            ),
            Dropout(p=dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name1 = attn_unit1.__repr__()
        self.attn_fn_name2 = attn_unit1.__repr__()
        self.norm_name = norm_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn1={}, attn_fn1={}, norm_layer={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.attn_fn_name1,
            self.attn_fn_name1,
            self.norm_name,
        )

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        res = x
        x_patch = torch.transpose(x, 2, 3)
        x_patch = self.pre_norm_attn1(x_patch)
        x_patch = torch.transpose(x_patch, 2, 3)
        x = self.pre_norm_attn2(x)
        x = x + x_patch
        x = x + res

        x = x + self.pre_norm_ffn(x)

        return x

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        input = torch.transpose(input, 2, 3)
        out1, p_mha1, m_mha1 = module_profile(module=self.pre_norm_attn1, x=input)
        input = torch.transpose(input, 2, 3)
        out2, p_mha2, m_mha2 = module_profile(module=self.pre_norm_attn2, x=input)
        out, p_ffn, m_ffn = module_profile(module=self.pre_norm_ffn, x=out2)

        macs = m_mha1 + m_mha2 + m_ffn
        params = p_mha1 + p_mha2 + p_ffn

        return input, params, macs


