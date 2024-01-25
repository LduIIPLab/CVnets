
from torch import nn
import argparse
from typing import Dict, Tuple

from . import register_cls_models
from .base_cls import BaseEncoder
from .config.ehfr_net import get_configuration
from ...layers import ConvLayer, LinearLayer, GlobalPool, Identity
from ...modules import HBlock as Block


@register_cls_models("ehfr_net")
class EHFR_Net(BaseEncoder):
    """
    This class defines the EHFR_Net architecture
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 101)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        ehfr_net_config = get_configuration(opts=opts)
        image_channels = ehfr_net_config["layer0"]["img_channels"]
        out_channels = ehfr_net_config["layer0"]["out_channels"]

        super().__init__(*args, **kwargs)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer4"],
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer5"],
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }

        self.classifier = nn.Sequential(
            GlobalPool(pool_type=pool_type, keep_dim=False),
            LinearLayer(in_features=out_channels, out_features=num_classes, bias=True),
        )

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.classification.ehfr_net.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.ehfr_net.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between FFN ehfr_net. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.ehfr_net.dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.ehfr_net.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier. Defaults to 1.0",
        )
        group.add_argument(
            "--model.classification.ehfr_net.attn-norm-layer",
            type=str,
            default="layer_norm_2d",
            help="Norm layer in attention block. Defaults to LayerNorm",
        )
        return parser

    def _make_hblock(
        self, opts, input_channel, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []

        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = getattr(opts, "model.classification.ehfr_net.dropout", 0.0)

        block.append(
            Block(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=cfg.get("stride", 1),
                ffn_multiplier=ffn_multiplier,
                n_local_blocks=cfg.get("n_local_blocks", 1),
                n_attn_blocks=cfg.get("n_attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=getattr(
                    opts, "model.classification.ehfr_net.ffn_dropout", 0.0
                ),
                attn_dropout=getattr(
                    opts, "model.classification.ehfr_net.attn_dropout", 0.0
                ),
                attn_norm_layer=getattr(
                    opts, "model.classification.ehfr_net.attn_norm_layer", "layer_norm_2d"
                ),
                expand_ratio=cfg.get("expand_ratio", 4),
                dilation=prev_dilation,
            )
        )

        input_channel = cfg.get("out_channels")

        return nn.Sequential(*block), input_channel
