"""Native-size adaptive patches with one shared four-direction selective SSM."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from models.fusion.ssm import SimpleSelectiveSSM, scan_image, merge_image
from .config import make_config
from .regions import Box, RegionRecord, extract_regions


@dataclass(frozen=True)
class PatchPrediction:
    batch_index: int
    keypoint_index: int
    box: Box
    prediction: torch.Tensor


@dataclass(frozen=True)
class LocalRefinementOutput:
    heatmaps: torch.Tensor
    support: torch.Tensor
    coverage: torch.Tensor
    regions: list[RegionRecord]
    patches: list[PatchPrediction]


class FourDirectionLocalMamba(nn.Module):
    """Share one SSM across raster, column and reversed sequences."""
    def __init__(self, channels=32, state_dim=16, backend="torch"):
        super().__init__()
        self.channels = channels
        self.ssm = SimpleSelectiveSSM(channels, state_dim=state_dim, backend=backend)
        self.norm = nn.LayerNorm(channels)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        if x.ndim != 4 or min(x.shape) <= 0 or x.shape[1] != self.channels:
            raise ValueError(f"Expected nonempty [N,{self.channels},h,w] local patches")
        if not x.is_floating_point():
            raise ValueError("local patches must be floating point")
        n, c, h, w = x.shape
        sequences = scan_image(x, directions=4)
        scanned = self.ssm(sequences.reshape(n * 4, c, h * w))
        aligned = merge_image(scanned.reshape(n, 4, c, h * w), h, w)
        normalized = self.norm(aligned.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.project(normalized)


def merge_patch_predictions(patches, *, batch_size, num_keypoints, feature_hw):
    """Scatter predictions to F1, averaging overlaps with a separate bool mask.

    The sum uses differentiable index_add; integer coverage never participates
    in autograd. At least one prediction is needed to determine dtype/device.
    """
    if not patches:
        raise ValueError("patch predictions must not be empty")
    h, w = feature_hw
    if min(batch_size, num_keypoints, h, w) <= 0:
        raise ValueError("merge dimensions must be positive")
    reference = patches[0].prediction
    total = batch_size * num_keypoints * h * w
    coverage = torch.zeros(total, dtype=torch.int64, device=reference.device)
    values, destinations = [], []
    for patch in patches:
        b, k, box, prediction = (patch.batch_index, patch.keypoint_index,
                                  patch.box, patch.prediction)
        if not 0 <= b < batch_size or not 0 <= k < num_keypoints:
            raise ValueError("patch batch or keypoint index is outside the merge grid")
        if not (0 <= box.x1 < box.x2 <= w and 0 <= box.y1 < box.y2 <= h):
            raise ValueError("patch box is outside the feature grid")
        if prediction.shape != (1, 1, box.height, box.width):
            raise ValueError("patch prediction shape differs from its native box")
        if prediction.device != reference.device or prediction.dtype != reference.dtype:
            raise ValueError("patch predictions must share dtype and device")
        rows = torch.arange(box.y1, box.y2, device=reference.device)[:, None]
        columns = torch.arange(box.x1, box.x2, device=reference.device)[None, :]
        indices = ((b * num_keypoints + k) * h * w + rows * w + columns).flatten()
        destinations.append(indices)
        values.append(prediction.flatten())
    indices = torch.cat(destinations)
    coverage.index_add_(0, indices, torch.ones_like(indices))
    # Normalize each contribution first: sum(P)/coverage can overflow even
    # when every P and the resulting mean are finite in the source dtype.
    contributions = torch.cat(values) / coverage[indices].to(reference.dtype)
    means = reference.new_zeros(total).index_add(0, indices, contributions)
    shape = (batch_size, num_keypoints, h, w)
    coverage = coverage.reshape(shape)
    return means.reshape(shape), coverage > 0, coverage


class AdaptiveLocalRefinement(nn.Module):
    """Slice F1 without resizing, process equal shapes, then merge predictions.

    ``enabled`` is accepted as part of the shared serialized configuration;
    the containing model decides whether this module is instantiated.
    """
    def __init__(self, channels, num_keypoints, **local_config):
        super().__init__()
        if (isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0 or
                isinstance(num_keypoints, bool) or not isinstance(num_keypoints, int) or num_keypoints <= 0):
            raise ValueError("channels and num_keypoints must be positive integers")
        self.config = make_config({"local": local_config})["local"]
        self.channels, self.num_keypoints = channels, num_keypoints
        self.microbatch_size = self.config["microbatch_size"]
        self.profile = False
        self.last_timings = {}
        self.local_mamba = FourDirectionLocalMamba(channels, self.config["state_dim"],
                                                  self.config["backend"])
        self.keypoint_embedding = (nn.Embedding(num_keypoints, channels)
                                   if self.config["embedding"] else None)
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, global_heatmaps, f1, *, return_patches=False):
        if (f1.ndim != 4 or min(f1.shape) <= 0 or f1.shape[1] != self.channels or
                not f1.is_floating_point() or not torch.isfinite(f1).all()):
            raise ValueError(f"f1 must be finite nonempty [B,{self.channels},Hf,Wf]")
        if (global_heatmaps.ndim != 4 or global_heatmaps.shape[:2] !=
                (f1.shape[0], self.num_keypoints)):
            raise ValueError("global heatmaps must match F1 batch and configured keypoints")
        if self.profile:
            if f1.is_cuda:
                torch.cuda.synchronize(f1.device)
            before_regions = perf_counter()
        regions = extract_regions(
            global_heatmaps.detach(), tuple(f1.shape[-2:]),
            level=self.config["level"], margin_ratio=self.config["margin_ratio"],
            layout=self.config["layout"], overlap_ratio=self.config["overlap_ratio"])
        if self.profile:
            if f1.is_cuda:
                torch.cuda.synchronize(f1.device)
            before_scan = perf_counter()
        groups = defaultdict(list)
        for region in regions:
            for box in region.patch_boxes:
                groups[(box.height, box.width)].append((region.batch_index,
                                                         region.keypoint_index, box))
        predictions = []
        for records in groups.values():
            for start in range(0, len(records), self.microbatch_size):
                microbatch = records[start:start + self.microbatch_size]
                crops = torch.cat([f1[b:b+1, :, box.y1:box.y2, box.x1:box.x2]
                                   for b, k, box in microbatch], dim=0)
                if self.keypoint_embedding is not None:
                    ids = torch.tensor([k for b, k, box in microbatch],
                                       dtype=torch.int64, device=f1.device)
                    crops = crops + self.keypoint_embedding(ids)[:, :, None, None]
                heatmaps = self.head(self.local_mamba(crops))
                predictions.extend(PatchPrediction(b, k, box, heatmaps[i:i+1])
                                   for i, (b, k, box) in enumerate(microbatch))
        if self.profile:
            if f1.is_cuda:
                torch.cuda.synchronize(f1.device)
            before_merge = perf_counter()
        heatmaps, support, coverage = merge_patch_predictions(
            predictions, batch_size=f1.shape[0], num_keypoints=self.num_keypoints,
            feature_hw=tuple(f1.shape[-2:]))
        if self.profile:
            if f1.is_cuda:
                torch.cuda.synchronize(f1.device)
            after_merge = perf_counter()
            self.last_timings = {
                "regions_seconds": before_scan - before_regions,
                "scan_seconds": before_merge - before_scan,
                "merge_seconds": after_merge - before_merge,
            }
        return LocalRefinementOutput(heatmaps, support, coverage, regions,
                                      predictions if return_patches else [])
