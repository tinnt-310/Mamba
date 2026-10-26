"""Connected adaptive regions and native-size half-open patch boxes."""

from collections import deque
from dataclasses import dataclass
import math
import operator

import numpy as np
import torch


@dataclass(frozen=True)
class Box:
    """Integer pixel-edge bounds: [x1, x2) by [y1, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        for name in ("x1", "y1", "x2", "y2"):
            try:
                value = operator.index(getattr(self, name))
            except TypeError as exc:
                raise ValueError("Box coordinates must be integers") from exc
            object.__setattr__(self, name, value)
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box must have positive width and height")

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


@dataclass(frozen=True)
class RegionRecord:
    batch_index: int
    keypoint_index: int
    global_box: Box
    feature_box: Box
    patch_boxes: list[Box]


def _grid_hw(hw):
    if len(hw) != 2:
        raise ValueError("Grid size must be (height, width)")
    try:
        height, width = (operator.index(size) for size in hw)
    except TypeError as exc:
        raise ValueError("Grid dimensions must be integers") from exc
    if height <= 0 or width <= 0:
        raise ValueError("Grid dimensions must be positive")
    return height, width


def _region_parameters(level, margin_ratio):
    if not math.isfinite(level) or not 0 < level <= 1:
        raise ValueError("Region level must be finite and in (0, 1]")
    if not math.isfinite(margin_ratio) or margin_ratio < 0:
        raise ValueError("Region margin_ratio must be finite and nonnegative")


def _peak_component(array, level, margin_ratio):
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("Heatmap must be a nonempty two-dimensional grid")
    if not np.isfinite(array).all():
        raise ValueError("Heatmap responses must be finite (no NaN or Inf)")
    height, width = array.shape
    lo, hi = float(array.min()), float(array.max())
    if hi - lo <= 1e-6:
        return Box(0, 0, width, height)
    # The input is detached float32 as specified; higher precision arithmetic
    # prevents finite opposite-sign float32 extrema overflowing the difference.
    normalized = (array.astype(np.float64, copy=False) - lo) / (hi - lo)
    mask = normalized >= level
    peak_y, peak_x = divmod(int(array.argmax()), width)
    visited = np.zeros(array.shape, dtype=bool)
    visited[peak_y, peak_x] = True
    queue = deque([(peak_y, peak_x)])
    min_x = max_x = peak_x
    min_y = max_y = peak_y
    while queue:
        y, x = queue.popleft()
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    margin_x = math.ceil((max_x - min_x + 1) * margin_ratio)
    margin_y = math.ceil((max_y - min_y + 1) * margin_ratio)
    return Box(max(0, min_x - margin_x), max(0, min_y - margin_y),
               min(width, max_x + 1 + margin_x), min(height, max_y + 1 + margin_y))


def peak_component_region(heatmap, level=0.30, margin_ratio=0.25):
    """Return the expanded 4-connected threshold component of the first peak."""
    _region_parameters(level, margin_ratio)
    if isinstance(heatmap, torch.Tensor):
        array = heatmap.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        array = np.asarray(heatmap, dtype=np.float32)
    return _peak_component(array, level, margin_ratio)


def scale_box(box, src_hw, dst_hw):
    """Map edges outward using floor/ceil, independently of point centers."""
    src_h, src_w = _grid_hw(src_hw)
    dst_h, dst_w = _grid_hw(dst_hw)
    # Integer division implements exact floor/ceil without floating edge drift.
    x1 = min(dst_w - 1, max(0, box.x1 * dst_w // src_w))
    y1 = min(dst_h - 1, max(0, box.y1 * dst_h // src_h))
    x2 = min(dst_w, max(x1 + 1, (box.x2 * dst_w + src_w - 1) // src_w))
    y2 = min(dst_h, max(y1 + 1, (box.y2 * dst_h + src_h - 1) // src_h))
    return Box(x1, y1, x2, y2)


def make_patch_boxes(region, layout="five", overlap_ratio=0.50):
    """Produce corners then center, stably deduplicated, without resizing."""
    if layout not in ("one", "five"):
        raise ValueError("Patch layout must be 'one' or 'five'")
    if not math.isfinite(overlap_ratio) or not 0 <= overlap_ratio <= 1:
        raise ValueError("Patch overlap_ratio must be finite and in [0, 1]")
    if layout == "one":
        return [region]
    width = min(region.width, math.ceil(region.width / (2 - overlap_ratio)))
    height = min(region.height, math.ceil(region.height / (2 - overlap_ratio)))
    left, right = region.x1, region.x2 - width
    top, bottom = region.y1, region.y2 - height
    center_x = left + round((region.width - width) / 2)
    center_y = top + round((region.height - height) / 2)
    origins = ((left, top), (right, top), (left, bottom), (right, bottom), (center_x, center_y))
    patches, seen = [], set()
    for x, y in origins:
        box = Box(x, y, x + width, y + height)
        if box not in seen:
            patches.append(box)
            seen.add(box)
    return patches


def extract_regions(global_heatmaps, feature_hw, level=0.30, margin_ratio=0.25,
                    layout="five", overlap_ratio=0.50):
    """Extract all records after a single detached float32 CPU transfer."""
    _region_parameters(level, margin_ratio)
    _grid_hw(feature_hw)
    if isinstance(global_heatmaps, torch.Tensor):
        arrays = global_heatmaps.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        arrays = np.asarray(global_heatmaps, dtype=np.float32)
    if arrays.ndim != 4 or not all(arrays.shape):
        raise ValueError("Global heatmaps must have nonempty shape [B, K, H, W]")
    if not np.isfinite(arrays).all():
        raise ValueError("Global heatmaps must contain finite responses")
    records = []
    for batch_index in range(arrays.shape[0]):
        for keypoint_index in range(arrays.shape[1]):
            global_box = _peak_component(arrays[batch_index, keypoint_index], level, margin_ratio)
            feature_box = scale_box(global_box, arrays.shape[-2:], feature_hw)
            patches = make_patch_boxes(feature_box, layout, overlap_ratio)
            records.append(RegionRecord(batch_index, keypoint_index, global_box, feature_box, patches))
    return records
