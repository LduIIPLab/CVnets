import numpy as np
from torch import nn, Tensor
import math
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, Union, Sequence

from .transformer import LocationPreservingVit
from .mobilenetv2 import InvertedResidual
from .base_module import BaseModule
from ..misc.profiler import module_profile
from ..layers import get_normalization_layer


class HBlock(BaseModule):
    """
    This class defines the `HBlock block`

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        out_channels (int): :math:`C_{out}` from an expected input of size :math:`(N, C_{out}, H, W)`
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

        .. note::
            If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_local_blocks: int = 1,
        n_attn_blocks: Optional[int] = 2,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        attn_dropout: Optional[float] = 0.0,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
        dilation: Optional[int] = 1,
        expand_ratio: Optional[Union[int, float, tuple, list]] = 2,
        *args,
        **kwargs
    ) -> None:
        attn_unit_dim = out_channels

        super(HBlock, self).__init__()
        self.local_acq, out_channels = self._build_local_layer(
            opts=opts,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            n_layers=n_local_blocks,
            dilation=dilation,
        )

        self.global_acq, attn_unit_dim = self._build_attn_layer(
            opts=opts,
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_attn_blocks = n_attn_blocks
        self.n_local_blocks = n_local_blocks

    def _build_local_layer(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Optional[Union[int, float, tuple, list]],
        n_layers: int,
        dilation: int = 1,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(expand_ratio, (int, float)):
            expand_ratio = [expand_ratio] * n_layers
        elif isinstance(expand_ratio, (list, tuple)):
            pass
        else:
            raise NotImplementedError

        local_acq = []
        if stride == 2 and n_layers != 0:
            local_acq.append(
                InvertedResidual(
                    opts=opts,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expand_ratio=expand_ratio[0],
                    dilation=dilation,
                )
            )

            for i in range(1, n_layers):
                local_acq.append(
                    InvertedResidual(
                        opts=opts,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=1,
                        expand_ratio=expand_ratio[i],
                        dilation=dilation,
                    )
                )

        else:
            for i in range(n_layers):
                local_acq.append(
                    InvertedResidual(
                        opts=opts,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=1,
                        expand_ratio=expand_ratio[i],
                        dilation=dilation,
                    )
                )

        return nn.Sequential(*local_acq), out_channels

    def _build_attn_layer(
        self,
        opts,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        attn_norm_layer: str,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_acq = [
            LocationPreservingVit(
                opts=opts,
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            )
            for block_idx in range(n_layers)
        ]
        global_acq.append(
            get_normalization_layer(
                opts=opts, norm_type=attn_norm_layer, num_features=d_model
            )
        )

        return nn.Sequential(*global_acq), d_model

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Local information acquisition:"

        if isinstance(self.local_acq, nn.Sequential):
            for m in self.local_acq:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.local_acq)

        repr_str += "\n\t Global information acquisition with patch size of {}x{}".format(
            self.patch_h,
            self.patch_w,
        )
        if isinstance(self.global_acq, nn.Sequential):
            for m in self.global_acq:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_acq)

        repr_str += "\n)"
        return repr_str

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)

        fm = self.local_acq(x)
        fm_local = fm

        # convert feature map to patches
        patches, output_size = self.unfolding_pytorch(fm)

        # learn global representations on all patches
        patches = self.global_acq(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = fm + fm_local

        return fm

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        params = macs = 0.0
        input = self.resize_input_if_needed(input)

        fm, p, m = module_profile(module=self.local_acq, x=input)
        params += p
        macs += m

        patches, output_size = self.unfolding_pytorch(feature_map=fm)

        patches, p, m = module_profile(module=self.global_acq, x=patches)
        params += p
        macs += m

        fm = self.folding_pytorch(patches=patches, output_size=output_size)

        return fm, params, macs
