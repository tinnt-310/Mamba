"""Reconstructable prediction artifacts, holdout metrics and visualization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .heatmaps import decode_heatmaps, transform_points


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return _numpy(value).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _box(box):
    if isinstance(box, (list, tuple)):
        return [int(value) for value in box]
    return [int(getattr(box, key)) for key in ("x1", "y1", "x2", "y2")]


def _artifact_paths(path):
    path = Path(path)
    if path.suffix in (".npz", ".json"):
        path = path.with_suffix("")
    return path.with_suffix(".npz"), path.with_suffix(".json")


def _coordinate_artifacts(maps, support, prefix, input_hw, original_hw, arrays):
    coordinates = {}
    grid_hw = tuple(maps.shape[-2:])
    for name, subpixel in (("argmax", False), ("subpixel", True)):
        decoded = decode_heatmaps(maps, support=support, subpixel=subpixel)
        coords = _numpy(decoded.coords)[0]
        input_coords = transform_points(coords, grid_hw, input_hw)
        original_coords = transform_points(coords, grid_hw, original_hw)
        used = _numpy(decoded.subpixel_used)[0]
        peak = _numpy(decoded.peak_response)[0]
        for space, values in (("grid", coords), ("input", input_coords), ("original", original_coords)):
            arrays[f"{prefix}_{name}_{space}"] = values
        coordinates[name] = {"grid": coords, "input": input_coords, "original": original_coords,
                             "peak_response": peak, "subpixel_used": used}
        if subpixel:
            arrays[f"{prefix}_peak_response"] = peak
            arrays[f"{prefix}_subpixel_used"] = used
    return coordinates


def save_prediction_artifact(path, sample, stage, config, keypoint_names):
    """Save one unbatched sample and one batch-one prediction as NPZ plus JSON.

    NPZ contains only numeric arrays and can be opened with allow_pickle=False.
    JSON retains source RGB, full config, label names and all region/patch boxes.
    Both decoders are saved in heatmap, input-image and original-image spaces.
    """
    maps = stage.global_heatmaps
    if maps.ndim != 4 or maps.shape[0] != 1:
        raise ValueError("prediction artifacts require a batch-one global heatmap")
    if len(keypoint_names) != maps.shape[1] or len(set(keypoint_names)) != len(keypoint_names):
        raise ValueError("keypoint names must uniquely match the predicted heatmaps")
    rgb = sample["rgb"]
    if rgb.ndim != 3:
        raise ValueError("prediction artifacts require an unbatched sample")
    input_hw = tuple(int(value) for value in rgb.shape[-2:])
    original_hw = tuple(int(value) for value in _numpy(sample["original_size"]).reshape(-1))
    if len(original_hw) != 2 or min(original_hw) <= 0:
        raise ValueError("original_size must contain positive height and width")
    arrays = {"global_heatmaps": _numpy(maps)[0], "weights": _numpy(sample["weights"]),
              "original_points": _numpy(sample["original_points"]), "points": _numpy(sample["points"])}
    coordinates = {"global": _coordinate_artifacts(maps, None, "global", input_hw, original_hw, arrays)}
    regions, patches = [], []
    feature_hw = None
    local = stage.local
    if local is not None:
        if local.heatmaps.shape[:2] != maps.shape[:2] or local.support.shape != local.heatmaps.shape or local.coverage.shape != local.heatmaps.shape:
            raise ValueError("local heatmap/support/coverage shapes must match batch and keypoints")
        feature_hw = tuple(int(value) for value in local.heatmaps.shape[-2:])
        arrays.update(local_heatmaps=_numpy(local.heatmaps)[0], support=_numpy(local.support)[0], coverage=_numpy(local.coverage)[0])
        coordinates["local"] = _coordinate_artifacts(local.heatmaps, local.support, "local", input_hw, original_hw, arrays)
        for region in local.regions:
            if region.batch_index != 0:
                raise ValueError("artifact regions must belong to batch index zero")
            regions.append({"batch_index": 0, "keypoint_index": int(region.keypoint_index),
                            "global_box": _box(region.global_box), "feature_box": _box(region.feature_box),
                            "patch_boxes": [_box(box) for box in region.patch_boxes]})
        for patch_index, patch in enumerate(local.patches or []):
            if patch.batch_index != 0:
                raise ValueError("artifact patches must belong to batch index zero")
            key = f"patch_{patch_index:04d}_heatmap"
            arrays[key] = _numpy(patch.prediction)
            patches.append({"batch_index": 0, "keypoint_index": int(patch.keypoint_index), "box": _box(patch.box), "array_key": key})
    metadata = {"artifact_version": 1, "frame_id": sample["frame_id"], "keypoint_names": list(keypoint_names),
                "config": config, "input_size": input_hw, "original_size": original_hw,
                "global_size": tuple(maps.shape[-2:]), "feature_size": feature_hw,
                "rgb_path": str(sample.get("rgb_path", "")), "regions": regions, "patches": patches,
                "coordinates": coordinates, "points": arrays["points"], "original_points": arrays["original_points"],
                "weights": arrays["weights"]}
    for timing_key in ("runtime_seconds", "peak_vram_bytes"):
        if timing_key in sample:
            metadata[timing_key] = sample[timing_key]
    npz_path, json_path = _artifact_paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(_jsonable(metadata), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"npz": str(npz_path), "json": str(json_path)}


def _error_summary(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {"count": int(errors.size), "mean": float(errors.mean()) if errors.size else None,
            "median": float(np.median(errors)) if errors.size else None,
            "p95": float(np.percentile(errors, 95)) if errors.size else None}


def compute_metrics(predictions, targets, weights, keypoint_names=None):
    """Euclidean pixel errors on labeled points, without hiding invalid outputs."""
    predictions, targets, weights = (_numpy(value) for value in (predictions, targets, weights))
    if predictions.shape != targets.shape or predictions.ndim < 2 or predictions.shape[-1] != 2 or weights.shape != predictions.shape[:-1]:
        raise ValueError("predictions/targets must share [...,K,2] and weights [...,K]")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("metric weights must be finite and nonnegative")
    valid = weights > 0
    if not np.isfinite(predictions[valid]).all() or not np.isfinite(targets[valid]).all():
        raise ValueError("valid metric points must have finite predictions and targets")
    count = predictions.shape[-2]
    names = list(keypoint_names) if keypoint_names is not None else [str(index) for index in range(count)]
    if len(names) != count or len(set(names)) != count:
        raise ValueError("keypoint names must uniquely match the metric keypoint count")
    errors = np.linalg.norm(predictions - targets, axis=-1)
    result = _error_summary(errors[valid])
    result["per_keypoint"] = {name: _error_summary(errors[..., keypoint][valid[..., keypoint]]) for keypoint, name in enumerate(names)}
    return result


def _metadata(value):
    if isinstance(value, dict):
        return value
    _, path = _artifact_paths(value)
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _correlation(x, y):
    if len(x) < 2:
        return None
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def aggregate_region_statistics(artifacts):
    """Summarize saved artifacts without recomputing model predictions.

    Accept JSON/NPZ paths or loaded metadata dictionaries. Recall checks the
    ground-truth center against the union of native patch pixel footprints.
    Report subpixel use per axis to retain independent-axis fallback behavior.
    """
    if isinstance(artifacts, (str, Path, dict)):
        artifacts = [artifacts]
    records = [_metadata(value) for value in artifacts]
    areas, widths, heights, patch_areas, patch_counts = [], [], [], [], []
    roi_hits = roi_total = 0
    subpixel = {"global": [], "local": []}
    area_pairs, response_pairs, error_pairs = [], [], []
    runtimes, vram = [], []
    for metadata in records:
        if "runtime_seconds" in metadata:
            runtimes.append(float(metadata["runtime_seconds"]))
        if "peak_vram_bytes" in metadata:
            vram.append(float(metadata["peak_vram_bytes"]))
        for stage in subpixel:
            if stage in metadata["coordinates"]:
                subpixel[stage].extend(np.asarray(metadata["coordinates"][stage]["subpixel"]["subpixel_used"], dtype=bool).reshape(-1).tolist())
        feature_hw = metadata.get("feature_size")
        if feature_hw is None:
            continue
        weights = np.asarray(metadata["weights"], dtype=float)
        gt = np.asarray(metadata["original_points"], dtype=float)
        feature_points = transform_points(gt, metadata["original_size"], feature_hw)
        by_keypoint = {int(region["keypoint_index"]): region for region in metadata["regions"]}
        for keypoint in np.flatnonzero(weights > 0):
            roi_total += 1
            region = by_keypoint.get(int(keypoint))
            if region is not None:
                x, y = feature_points[keypoint]
                roi_hits += int(any(box[0] - .5 <= x < box[2] - .5 and box[1] - .5 <= y < box[3] - .5 for box in region["patch_boxes"]))
        global_coordinates = metadata["coordinates"]["global"]["argmax"]
        global_points = np.asarray(global_coordinates["original"], dtype=float)
        global_responses = np.asarray(global_coordinates["peak_response"], dtype=float)
        for region in metadata["regions"]:
            x1, y1, x2, y2 = region["feature_box"]
            width, height = x2 - x1, y2 - y1
            area = width * height
            widths.append(width)
            heights.append(height)
            areas.append(area)
            patch_counts.append(len(region["patch_boxes"]))
            patch_areas.extend((box[2] - box[0]) * (box[3] - box[1]) for box in region["patch_boxes"])
            keypoint = int(region["keypoint_index"])
            if weights[keypoint] > 0:
                area_pairs.append(area)
                response_pairs.append(global_responses[keypoint])
                error_pairs.append(float(np.linalg.norm(global_points[keypoint] - gt[keypoint])))
    def mean(values):
        return float(np.mean(values)) if values else None
    result = {"image_count": len(records), "region_count": len(areas), "patch_count": len(patch_areas),
              "roi_recall": roi_hits / roi_total if roi_total else None, "roi_hits": roi_hits, "roi_valid_keypoints": roi_total,
              "mean_region_width": mean(widths), "mean_region_height": mean(heights), "mean_region_area": mean(areas),
              "max_region_area": max(areas) if areas else None, "mean_patch_area": mean(patch_areas),
              "mean_patches_per_region": mean(patch_counts), "max_patches_per_region": max(patch_counts) if patch_counts else None,
              "mean_sequence_length": mean(patch_areas), "max_sequence_length": max(patch_areas) if patch_areas else None,
              "region_area_global_error_correlation": _correlation(area_pairs, error_pairs),
              "peak_response_global_error_correlation": _correlation(response_pairs, error_pairs),
              "region_area_peak_response_correlation": _correlation(area_pairs, response_pairs),
              "mean_runtime_seconds": mean(runtimes), "peak_vram_bytes": max(vram) if vram else None}
    for stage, used in subpixel.items():
        rate = mean(used)
        result[f"{stage}_subpixel_axis_rate"] = rate
        result[f"{stage}_fallback_axis_rate"] = 1 - rate if rate is not None else None
    return result


def visualize_artifact(path, output_path):
    """Render source RGB, saved boxes and predictions; never adjust with GT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    npz_path, json_path = _artifact_paths(path)
    metadata = _metadata(json_path)
    if not metadata.get("rgb_path"):
        raise ValueError("artifact does not identify the original source RGB image")
    with Image.open(metadata["rgb_path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with np.load(npz_path, allow_pickle=False) as arrays:
        panels = 3 if "local_heatmaps" in arrays else 2
        figure, axes = plt.subplots(1, panels, figsize=(6 * panels, 5))
        axes[0].imshow(rgb)
        for stage, color, marker in (("global", "cyan", "x"), ("local", "lime", "+")):
            if stage in metadata["coordinates"]:
                coords = np.asarray(metadata["coordinates"][stage]["subpixel"]["original"])
                axes[0].scatter(coords[:, 0], coords[:, 1], c=color, marker=marker, label=stage)
                for name, (x, y) in zip(metadata["keypoint_names"], coords):
                    axes[0].annotate(name, (x, y), color=color, fontsize=6)
        original_h, original_w = metadata["original_size"]
        for region in metadata["regions"]:
            gh, gw = metadata["global_size"]
            x1, y1, x2, y2 = region["global_box"]
            axes[0].add_patch(Rectangle((x1 * original_w / gw - .5, y1 * original_h / gh - .5), (x2 - x1) * original_w / gw, (y2 - y1) * original_h / gh, fill=False, edgecolor="cyan", linewidth=.7))
            fh, fw = metadata["feature_size"]
            for x1, y1, x2, y2 in region["patch_boxes"]:
                axes[0].add_patch(Rectangle((x1 * original_w / fw - .5, y1 * original_h / fh - .5), (x2 - x1) * original_w / fw, (y2 - y1) * original_h / fh, fill=False, edgecolor="yellow", linewidth=.5))
        axes[0].set_title(f"{metadata['frame_id']}: saved subpixel predictions")
        axes[0].legend(loc="best", fontsize=8)
        axes[1].imshow(arrays["global_heatmaps"].max(axis=0), cmap="magma")
        axes[1].set_title("Global heatmaps (maximum across keypoints)")
        if panels == 3:
            local = np.where(arrays["support"], arrays["local_heatmaps"], -np.inf).max(axis=0)
            axes[2].imshow(np.ma.masked_invalid(local), cmap="magma")
            axes[2].set_title("Local heatmaps within saved support")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=140)
        plt.close(figure)
    return str(output_path)
