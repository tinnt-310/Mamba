"""Serializable architecture configuration for the 2D RGB-D pipeline."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math


DEFAULT_CONFIG = {
    "input_size": (180, 320),
    "heatmap_size": (90, 160),
    "out_indices": (1, 2, 3, 4),
    "geometry_channels": (32, 48, 96, 128, 192),
    "fusion_channels": (32, 32, 32, 32),
    "decoder_width": 32,
    "num_keypoints": 9,
    "pretrained": False,
    "pretrained_path": None,
    "modality": "rgbd",
    "fusion_mode": "legacy",
    "skip_fusion_scales": (),
    "detail_decoder": False,
    "global_ssm": {"state_dim": 4, "expand": 1, "directions": 1, "backend": "torch"},
    "local": {
        "enabled": True, "level": .30, "margin_ratio": .25,
        "layout": "five", "overlap_ratio": .50, "state_dim": 16,
        "directions": 4, "backend": "torch", "embedding": True,
        "microbatch_size": 8,
        "mode": "legacy", "feature_source": "f1", "roi_size": 13,
        "conditioning": False,
    },
}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _merge(base, overrides, path="config"):
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{path} must be a mapping")
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"Unknown {path} key: {key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{path}.{key}")
        else:
            base[key] = deepcopy(value)


def make_config(overrides=None):
    """Deep merge defaults without mutating callers, then validate dimensions.

    Lists from JSON are canonicalized to tuples so checkpoint architecture
    comparisons have one representation.
    """
    config = deepcopy(DEFAULT_CONFIG)
    if overrides is not None:
        _merge(config, overrides)
    for key in ("input_size", "heatmap_size", "out_indices",
                "geometry_channels", "fusion_channels"):
        try:
            config[key] = tuple(config[key])
        except TypeError as exc:
            raise ValueError(f"{key} must be a sequence") from exc
    for key in ("input_size", "heatmap_size"):
        if len(config[key]) != 2:
            raise ValueError(f"{key} must have height and width")
        for value in config[key]:
            _positive_int(value, key)
    indices = config["out_indices"]
    if (not indices or any(isinstance(i, bool) or not isinstance(i, int) or
                           i not in range(5) for i in indices) or
            indices != tuple(sorted(set(indices)))):
        raise ValueError("out_indices must be unique increasing integers in [0,4]")
    if len(config["geometry_channels"]) != 5:
        raise ValueError("geometry_channels must specify all five encoder stages")
    for value in config["geometry_channels"]:
        _positive_int(value, "geometry_channels")
        if value < 2:
            raise ValueError("geometry_channels must be at least 2")
    if len(config["fusion_channels"]) != len(indices):
        raise ValueError("fusion_channels must match the selected scale count")
    for value in config["fusion_channels"]:
        _positive_int(value, "fusion_channels")
    for key in ("decoder_width", "num_keypoints"):
        _positive_int(config[key], key)
    if not isinstance(config["pretrained"], bool):
        raise ValueError("pretrained must be boolean")
    if config['pretrained_path'] is not None and not isinstance(config['pretrained_path'], str):
        raise ValueError('pretrained_path must be a local checkpoint path or None')
    if config['modality'] not in ('rgbd', 'rgb') or config['fusion_mode'] not in ('legacy','quality'):
        raise ValueError('Unsupported modality/fusion mode')
    config['skip_fusion_scales'] = tuple(config['skip_fusion_scales'])
    if any(isinstance(i,bool) or not isinstance(i,int) or not 0 <= i < len(indices) for i in config['skip_fusion_scales']):
        raise ValueError('skip_fusion_scales indexes selected fusion scales')
    if not isinstance(config['detail_decoder'],bool):
        raise ValueError('detail_decoder must be boolean')
    global_ssm = config["global_ssm"]
    for key in ("state_dim", "expand"):
        _positive_int(global_ssm[key], f"global_ssm.{key}")
    if global_ssm["directions"] not in (1, 4):
        raise ValueError("global_ssm.directions must be 1 or 4")
    if global_ssm["backend"] not in ("torch", "mamba"):
        raise ValueError("global_ssm.backend must be torch or mamba")
    local = config["local"]
    if local['mode'] not in ('legacy','residual') or local['feature_source'] not in ('f1','detail'):
        raise ValueError('Unsupported local mode/feature source')
    _positive_int(local['roi_size'],'local.roi_size')
    if local['roi_size']%2 != 1 or local['roi_size'] > 65:
        raise ValueError('local.roi_size must be odd and <=65')
    if not isinstance(local['conditioning'],bool):
        raise ValueError('local.conditioning must be boolean')
    if local['mode']=='residual' and (not local['embedding'] or local['layout'] != 'one'):
        raise ValueError('Residual local requires semantic ID and one ROI')
    for key in ("enabled", "embedding"):
        if not isinstance(local[key], bool):
            raise ValueError(f"local.{key} must be boolean")
    _positive_int(local["state_dim"], "local.state_dim")
    _positive_int(local["microbatch_size"], "local.microbatch_size")
    if local["microbatch_size"] > 8:
        raise ValueError("local.microbatch_size must be at most 8")
    if local["directions"] != 4:
        raise ValueError("local.directions must be 4")
    if local["backend"] not in ("torch", "mamba"):
        raise ValueError("local.backend must be torch or mamba")
    if local["layout"] not in ("one", "five"):
        raise ValueError("local.layout must be one or five")
    for key in ("level", "margin_ratio", "overlap_ratio"):
        value = local[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"local.{key} must be finite")
    if not 0 < local["level"] <= 1:
        raise ValueError("local.level must be in (0,1]")
    if local["margin_ratio"] < 0:
        raise ValueError("local.margin_ratio must be nonnegative")
    if not 0 <= local["overlap_ratio"] <= 1:
        raise ValueError("local.overlap_ratio must be in [0,1]")
    return config
