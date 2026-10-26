# Dense depth / geometry branch

## Inspected RGB and fusion contract

`models/geometry/se_resnet.py` wraps timm `seresnet50` by default, with
`seresnet101` and `seresnet152` also supported. It returns a **tuple** of NCHW
features, selected with `out_indices`. Default indices are `(0,1,2,3,4)`.
There is no assembled training model or dataset in this repository. The README
example consumes all five stages through `MultiScaleRGBDCompensation`.

| Index | RGB stage | RGB channels | Reduction | Native 720x1280 H,W | Geometry stage/channels |
|---|---|---:|---:|---|---|
| 0 | act1 | 64 | 2 | 360,640 | stem / 32 |
| 1 | layer1 | 256 | 4 | 180,320 | stages.0 / 48 |
| 2 | layer2 | 512 | 8 | 90,160 | stages.1 / 96 |
| 3 | layer3 | 1024 | 16 | 45,80 | stages.2 / 128 |
| 4 | layer4 | 2048 | 32 | 23,40 | stages.3 / 192 |

Each row maps directly to the corresponding geometry output. The geometry
7x7/2/padding-3 stem, 3x3/2/padding-1 pool and residual stride schedule
1,2,2,2 match the RGB grid origin and reductions, including odd dimensions.
No feature resizing hides mismatches. `from_rgb_backbone` reads the selected
indices and checks reductions. Select indices in increasing order.

Geometry widths are configurable; the RGB channels are not copied into a
second SE-ResNet. GroupNorm and SiLU follow geometry convolutions. GroupNorm
normalizes learned activations; input XYZ retains its absolute metric scale.

`MS2FusionBranches` takes matching NCHW tensors and internally flattens both
to B,C,L in the same raster order. `CrossParametricSSM` also supports four
directions through the same `scan_image` function for both inputs. No new scan
or token permutation is introduced. `GeometryProjector` supplies paired 1x1
convolutions for direct CP/SP integration; `MultiScaleRGBDCompensation` already
contains its own projections.

## Geometry contract

- Depth is optical-axis Z in meters, shaped B,1,H,W. K is required per frame,
  B,3,3, on the same device. No dataset-wide K is stored.
- X=(u-cx)Z/fx, Y=(v-cy)Z/fy, integer pixel centers; X right, Y down, Z forward.
- Finite positive depth defines the boolean validity mask. Invalid inputs
  become zero before back-projection; invalid XYZ/normals/mask channels are zero.
- Default input is exactly XYZ + normal + depth-valid mask, seven channels.
  A valid center with unusable neighbors retains XYZ/mask and has zero normal.
  `xyz_to_normals` separately exposes the stricter normal-valid mask.
- Central differences use four axial neighbors at radius 1 by default. All
  neighbors must be valid and within 0.02 meters of the center. Borders,
  degenerate cross products and nonfinite normals are invalid. `radius`,
  `depth_threshold`, `eps` (cross-product magnitude in square meters), and
  `point_into_surface` are configurable.
- Default normals face the camera: a front plane has normal (0,0,-1).
  `point_into_surface=True` gives (0,0,+1). This is a central-difference
  approximation inspired by normalSpeed, not a reproduction of its estimator.
- Geometry arithmetic promotes half/bfloat16 to float32; float64 is retained.
  Use autocast around the branch for mixed-precision CNN execution.
- Invalid depth cannot inject NaN/Inf into the CNN. Malformed K and numerical
  overflow in back-projection raise errors. Arbitrarily corrupted model weights
  are not sanitized. Entirely missing depth can still produce learned feature
  activations through affine normalization; feature values are not validity masks.
- No RGB/ImageNet normalization, per-image min-max, hole filling, extra priors,
  point sampling, or hidden depth-unit conversion is applied.

RGB and depth must be registered to the same pixel grid. Equal shapes alone
cannot establish physical registration. If the host pipeline crops/resizes,
apply the same transform to both inputs and update K. For a crop subtract its
left/top from cx/cy. For half-pixel resize use cx'=(cx+0.5)*sx-0.5,
cy'=(cy+0.5)*sy-0.5 and fx'=fx*sx, fy'=fy*sy. Use the convention of the actual
resampler. The provided loader performs no spatial transforms.

## Runnable integration

Install the existing RGB dependency and geometry I/O dependencies:

```powershell
python -m pip install -e ".[geometry,test]"
python -m unittest discover -s tests -v
```

```python
import torch
from models.geometry.se_resnet import SEResNetBackbone
from models.geometry import DenseGeometryEncoder, DepthGeometryBranch, GeometryProjector
from models.geometry.io import load_depth_frame
from models.fusion.ms2fusion import MS2FusionBranches

rgb_encoder = SEResNetBackbone(pretrained=False).eval()
depth_encoder = DepthGeometryBranch(
    DenseGeometryEncoder.from_rgb_backbone(rgb_encoder)
).eval()
widths = (8, 8, 8, 8, 8)  # small CPU demonstration; choose training widths explicitly
project = GeometryProjector(rgb_encoder.stage_channels, depth_encoder.stage_channels, widths)
blocks = torch.nn.ModuleList(MS2FusionBranches(c, state_dim=2) for c in widths)

rgb = torch.rand(1, 3, 17, 25)
depth = torch.ones(1, 1, 17, 25)  # meters
K = torch.tensor([[[500., 0., 12.], [0., 500., 8.], [0., 0., 1.]]])
with torch.no_grad():
    rgb_features = rgb_encoder(rgb)
    geo_features = depth_encoder(depth, K)
    r, g = project(rgb_features, geo_features)
    outputs = tuple(block(a, b) for block, a, b in zip(blocks, r, g))

# Real frame loading, using your own root and ids:
# samples = [load_depth_frame(root, frame_id) for frame_id in frame_ids]
# depth, K = (torch.stack(items) for items in zip(*samples))
# rgb must contain the corresponding registered RGB frames.
```

The loader reads `depth_sensor/<id>.npy` (float32 H,W meters) and top-level
`K` from `poses/<id>.json`. It never reads `depth_vis`. Move models, RGB,
depth and K to the same device. Geometry-only tensor modules require PyTorch;
file loading additionally requires NumPy, and the existing RGB wrapper needs timm.

For existing compensation, pass the unprojected `rgb_features, geo_features`
directly to `MultiScaleRGBDCompensation(rgb_encoder.stage_channels,
depth_encoder.stage_channels, channels=widths)`. Do not apply projections twice.

Native-resolution geometry is supported; the pure Python selective-scan backend
is intended for small correctness tests. Use the existing CUDA backend for
large Mamba feature maps. Tests cover real RGB forwards on small even/odd grids,
native-resolution operator shapes with meta tensors, forward/backward through
CP/SP, and four-direction compensation. No training quality or sensor-noise
benchmark is claimed; normal thresholds should be validated on your sensor.

## Official sources inspected

All implementation here is newly written dense PyTorch code. The following
concepts informed it; no upstream point encoder or task head is copied.

- [FFB6D YCB preprocessing](https://github.com/ethnhe/FFB6D/blob/master/ffb6d/datasets/ycb/ycb_dataset.py):
  `dpt_2_pcld` uses K and depth scale; valid flattened indices select matching
  XYZ/RGB/normals. It constructs image/point neighbor indices across scales.
  [RandLANet](https://github.com/ethnhe/FFB6D/blob/master/ffb6d/models/RandLA/RandLANet.py)
  encodes distance, relative XYZ, center XYZ and neighbor XYZ through learned
  layers and attentive pooling. [FFB6D model](https://github.com/ethnhe/FFB6D/blob/master/ffb6d/models/ffb6d.py)
  transfers features using indexed interpolation/pooling. Adopted: metric local
  geometry and spatial correspondence. Replaced point processing with a dense CNN.
- [PVN3D utilities](https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/utils/basic_utils.py)
  back-project selected pixels and estimate PCL normals.
  [Dataset](https://github.com/ethnhe/PVN3D/blob/master/pvn3d/datasets/ycb/ycb_dataset.py)
  associates normals and RGB through the same indices and builds metric
  keypoint/center offsets. [Model](https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/pvn3d.py)
  learns local PointNet++ geometry and predicts offsets. Adopted XYZ/normals as
  learned geometric inputs; no voting or keypoint heads are added.
- [DenseFusion dataset](https://github.com/j96w/DenseFusion/blob/master/datasets/ycb/dataset.py)
  back-projects depth using pixel coordinates and scale.
  [Network](https://github.com/j96w/DenseFusion/blob/master/lib/network.py)
  gathers image embeddings by `choose` before combining them with learned point
  features. Adopted correspondence discipline, implemented as a retained grid.
- [normalSpeed](https://github.com/hfutcgncas/normalSpeed): `depth_normal` expects
  millimeters and fx/fy; its example uses kernel 5, range 2000 mm, difference
  threshold 20 mm and outward normals. Its [C++ implementation](https://github.com/hfutcgncas/normalSpeed/blob/master/normalSpeed/normal.cpp)
  uses `kernel_size` as an offset radius for eight surrounding samples, solves
  a local depth-slope system, scales slopes by fx/fy, and median-filters normal
  components. Adopted discontinuity rejection and
  explicit orientation. Our estimator remains in meters, uses four axial
  samples, and imposes no default 2-meter range cutoff.
- [DFormerv2 GeoPriorGen](https://github.com/VCIP-RGBD/DFormer/blob/main/models/encoders/DFormerv2.py):
  derives dense attention decay from depth differences and image positions.
  Adopted only the dense representation principle, without its attention prior.
