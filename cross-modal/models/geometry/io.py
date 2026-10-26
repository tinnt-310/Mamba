"""Load metric sensor depth and its own camera intrinsics without resampling."""
import json
from pathlib import Path

import numpy as np
import torch


def load_depth_frame(root: str | Path, frame_id: str):
    """Return depth [1,H,W] and K [3,3]; stack frames along batch dimension.

    Reads only depth_sensor/<id>.npy and poses/<id>.json, with top-level 'K'.
    No fixed calibration, visualization depth, filling, or unit conversion.
    """
    frame_id = str(frame_id)
    if not frame_id or frame_id in (".", "..") or any(c in frame_id for c in '/\\:'):
        raise ValueError("frame_id must be a single filename stem")
    root = Path(root)
    depth = np.load(root / 'depth_sensor' / f'{frame_id}.npy', allow_pickle=False)
    if depth.ndim != 2 or depth.dtype != np.float32 or min(depth.shape) <= 0:
        raise ValueError("sensor depth must be a nonempty float32 [H,W] array in meters")
    with (root / 'poses' / f'{frame_id}.json').open(encoding='utf-8') as file:
        K = np.asarray(json.load(file)['K'], dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("each pose JSON must contain a finite 3x3 K")
    return torch.from_numpy(depth.copy()).unsqueeze(0), torch.from_numpy(K.copy())
