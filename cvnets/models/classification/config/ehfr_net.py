
from typing import Dict

from utils.math_utils import make_divisible, bound_fn


def get_configuration(opts) -> Dict:

    width_multiplier = getattr(opts, "model.classification.ehfr_net.width_multiplier", 1.0)

    ffn_multiplier = (
        2
    )

    layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=16))
    config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": layer_0_dim,
        },
        "layer1": {
            "out_channels": int(make_divisible(32 * width_multiplier, divisor=16)),  # 128 * 128
            "expand_ratio": 1,
            "stride": 1,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 2,
            "n_attn_blocks": 1,
            "patch_h": 2,
            "patch_w": 2,
        },
        "layer2": {
            "out_channels": int(make_divisible(48 * width_multiplier, divisor=16)),  # 64 * 64
            "expand_ratio": [1, 1, 3],
            "stride": 2,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 3,
            "n_attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
        },
        "layer3": {  # 28x28
            "out_channels": int(make_divisible(80 * width_multiplier, divisor=16)),  # 32 * 32
            "expand_ratio": 3,
            "stride": 2,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 3,
            "n_attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
        },
        "layer4": {  # 14x14
            "out_channels": int(make_divisible(160 * width_multiplier, divisor=16)),  # 16 * 16
            "expand_ratio": [6, 2.5, 2.3],
            "stride": 2,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 3,
            "n_attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
        },
        "layer5": {  # 7x7
            "out_channels": int(make_divisible(320 * width_multiplier, divisor=16)),  # 8 * 8
            "expand_ratio": 6,
            "stride": 2,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 1,
            "n_attn_blocks": 1,
            "patch_h": 2,
            "patch_w": 2,
        },
        "last_layer_exp_factor": 4,
    }

    return config
