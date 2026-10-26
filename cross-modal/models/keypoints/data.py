"""Paired RGB/metric-depth keypoints and a persisted, source-preserving subset.

Dataset construction and reads belong inside Slurm jobs on this cluster.
RGB uses ImageNet channel normalization. Sensor depth stays in meters and is
resized by nearest sampling; point centers and intrinsics use the same affine.
Visibility 1 and 2 are labeled points; visibility 0, missing/nonfinite labels
and labels outside the original pixel footprint receive zero training weight.
"""
from __future__ import annotations

import json
import hashlib
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from .heatmaps import gaussian_heatmaps, transform_points


MAX_SAMPLES = 5000
SPLITS = ("train", "val", "test")


def _frame_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("frame IDs must be safe filename stems without path traversal")
    return value


def _ids(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("frame IDs must be a list")
    result = [_frame_id(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError("frame IDs must be unique")
    return result


def _names(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("keypoint_names must be a nonempty sequence")
    if any(not isinstance(name, str) or not name.strip() for name in values) or len(set(values)) != len(values):
        raise ValueError("keypoint names must be unique nonempty strings")
    return list(values)


def _source_splits(root):
    result = {}
    seen = set()
    for split in SPLITS:
        with (root / f"{split}.txt").open(encoding="utf-8") as file:
            values = _ids([line.strip() for line in file if line.strip()])
        if seen.intersection(values):
            raise ValueError("source train/val/test splits must be disjoint")
        seen.update(values)
        result[split] = values
    return result


def _read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _source_fingerprints(source):
    return {split: hashlib.sha256(("\n".join(source[split]) + "\n").encode("utf-8")).hexdigest() for split in SPLITS}


def _point_names(root, frame_id):
    annotation = _read_json(root / "annotations" / f"{_frame_id(frame_id)}.json")
    if annotation.get("frame_id") != frame_id:
        raise ValueError("annotation frame_id does not match requested frame")
    return _names([point["name"] for point in annotation["keypoints_2d"]])


def create_subset_manifest(root, path, limit=5000, seed=20260922):
    """Sample each source split independently and persist exact selected IDs.

    No images are copied or decoded. The default subset contains exactly
    4000 train, 500 validation and 500 test frames. Smaller limits are useful
    for engineering smoke jobs; limits above the user's 5000-frame budget
    are rejected, including when the source dataset contains more frames.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SAMPLES:
        raise ValueError("subset limit must be an integer between 1 and 5000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    root, path = Path(root).resolve(), Path(path)
    source = _source_splits(root)
    counts = (limit * 8 // 10, limit // 10, limit - limit * 8 // 10 - limit // 10)
    rng = random.Random(seed)
    selected = {}
    for split, count in zip(SPLITS, counts):
        if count > len(source[split]):
            raise ValueError(f"source {split} split has too few samples for the requested subset")
        # Sort after random selection to make data iteration stable and readable.
        selected[split] = sorted(rng.sample(source[split], count))
    first = next(frame_id for split in SPLITS for frame_id in selected[split])
    manifest = {"manifest_version": 1, "dataset_root": str(root), "limit": limit,
                "seed": seed, "source_split_sha256": _source_fingerprints(source),
                "keypoint_names": _point_names(root, first), **selected}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path, *, root=None):
    """Validate the persisted cap, disjoint IDs, and original split membership."""
    manifest = _read_json(Path(path))
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != 1:
        raise ValueError("unsupported subset manifest")
    limit = manifest.get("limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SAMPLES:
        raise ValueError("subset manifest limit must be between 1 and 5000")
    root_value = manifest.get("dataset_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("manifest must identify its dataset_root")
    dataset_root = Path(root_value).resolve()
    if root is not None and Path(root).resolve() != dataset_root:
        raise ValueError("requested dataset root does not match the persisted subset provenance")
    source = _source_splits(dataset_root)
    if manifest.get("source_split_sha256") != _source_fingerprints(source):
        raise ValueError("source split provenance differs from the persisted subset")
    _names(manifest.get("keypoint_names"))
    seen = set()
    counts = (limit * 8 // 10, limit // 10, limit - limit * 8 // 10 - limit // 10)
    for split, count in zip(SPLITS, counts):
        values = _ids(manifest.get(split))
        if len(values) != count:
            raise ValueError("subset split counts must preserve the declared 80/10/10 allocation")
        if seen.intersection(values):
            raise ValueError("subset train/val/test splits must be disjoint")
        if not set(values).issubset(source[split]):
            raise ValueError(f"subset {split} IDs do not belong to the original source split")
        seen.update(values)
    if len(seen) != limit or len(seen) > MAX_SAMPLES:
        raise ValueError("manifest sample count does not match its declared limit")
    return manifest


def _size(value, name):
    value = tuple(value)
    if len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value):
        raise ValueError(f"{name} must contain two positive integers")
    return value


class RGBDKeypointDataset(Dataset):
    def __init__(self, root, ids, keypoint_names=None, input_size=(180, 320), heatmap_size=(90, 160), sigma=2):
        self.root = Path(root).resolve()
        self.ids = _ids(ids)
        if len(self.ids) > MAX_SAMPLES:
            raise ValueError("dataset cannot index more than the authorized 5000 samples")
        if keypoint_names is None:
            if not self.ids:
                raise ValueError("empty datasets require explicit keypoint_names")
            keypoint_names = _point_names(self.root, self.ids[0])
        self.keypoint_names = _names(keypoint_names)
        self.input_size = _size(input_size, "input_size")
        self.heatmap_size = _size(heatmap_size, "heatmap_size")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        self.sigma = float(sigma)

    def __len__(self):
        return len(self.ids)

    def _path(self, relative):
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("sample paths must stay inside dataset_root") from exc
        return path

    def __getitem__(self, index):
        frame_id = self.ids[index]
        annotation = _read_json(self._path(f"annotations/{frame_id}.json"))
        if annotation.get("frame_id") != frame_id:
            raise ValueError("annotation frame_id does not match requested frame")
        expected = {"rgb": f"rgb/{frame_id}.png", "depth_sensor": f"depth_sensor/{frame_id}.npy", "poses": f"poses/{frame_id}.json"}
        if not isinstance(annotation.get("paths"), dict) or any(annotation["paths"].get(name) != relative for name, relative in expected.items()):
            raise ValueError("annotation RGB/depth/pose paths must match their frame ID")
        rgb_path = self._path(expected["rgb"])
        with Image.open(rgb_path) as image:
            image = image.convert("RGB")
            original_hw = (image.height, image.width)
            image = image.resize((self.input_size[1], self.input_size[0]), Image.Resampling.BILINEAR)
            rgb_array = np.asarray(image, dtype=np.float32).copy()
        rgb = torch.from_numpy(rgb_array).permute(2, 0, 1) / 255
        mean, std = rgb.new_tensor((.485, .456, .406))[:, None, None], rgb.new_tensor((.229, .224, .225))[:, None, None]
        rgb = (rgb - mean) / std
        depth_array = np.load(self._path(expected["depth_sensor"]), allow_pickle=False)
        if depth_array.ndim != 2 or depth_array.dtype != np.float32 or tuple(depth_array.shape) != original_hw:
            raise ValueError("paired sensor depth must be float32 [H,W] with the original RGB shape")
        depth = torch.from_numpy(depth_array.copy())[None, None]
        depth = F.interpolate(depth, size=self.input_size, mode="nearest")[0]
        K = torch.as_tensor(_read_json(self._path(expected["poses"]))["K"], dtype=torch.float32)
        if K.shape != (3, 3) or not torch.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not torch.allclose(K[2], K.new_tensor((0, 0, 1))):
            raise ValueError("each pose must provide finite pinhole K with positive focal lengths")
        sy, sx = self.input_size[0] / original_hw[0], self.input_size[1] / original_hw[1]
        resize = K.new_tensor(((sx, 0, (sx - 1) / 2), (0, sy, (sy - 1) / 2), (0, 0, 1)))
        K = resize @ K
        labels = annotation.get("keypoints_2d")
        if not isinstance(labels, list):
            raise ValueError("annotation must provide keypoints_2d records")
        by_name = {}
        for point in labels:
            name = point.get("name")
            if not isinstance(name, str) or not name or name in by_name:
                raise ValueError("annotation keypoint names must be unique")
            by_name[name] = point
        original_points = torch.zeros(len(self.keypoint_names), 2, dtype=torch.float32)
        weights = torch.zeros(len(self.keypoint_names), dtype=torch.float32)
        visibility = torch.zeros(len(self.keypoint_names), dtype=torch.int64)
        for keypoint, name in enumerate(self.keypoint_names):
            point = by_name.get(name)
            if point is None:
                continue
            visible = point.get("visibility", 0)
            if isinstance(visible, bool) or not isinstance(visible, int) or visible not in (0, 1, 2):
                raise ValueError("keypoint visibility must be 0, 1 or 2")
            visibility[keypoint] = visible
            uv = np.asarray(point.get("uv", (np.nan, np.nan)), dtype=np.float64)
            if uv.shape != (2,):
                raise ValueError("keypoint uv must contain two coordinates")
            if np.isfinite(uv).all():
                original_points[keypoint] = torch.as_tensor(uv, dtype=torch.float32)
                inside = -.5 <= uv[0] < original_hw[1] - .5 and -.5 <= uv[1] < original_hw[0] - .5
                weights[keypoint] = float(visible > 0 and inside)
        points = transform_points(original_points, original_hw, self.input_size)
        target_points = transform_points(original_points, original_hw, self.heatmap_size)
        target = gaussian_heatmaps(target_points, self.heatmap_size, sigma=self.sigma, weights=weights)
        return {"frame_id": frame_id, "rgb": rgb, "depth": depth, "K": K,
                "points": points, "weights": weights, "visibility": visibility,
                "original_points": original_points, "original_size": torch.tensor(original_hw, dtype=torch.int64),
                "target": target, "rgb_path": str(rgb_path)}
