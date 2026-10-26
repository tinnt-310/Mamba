import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.geometry.se_resnet import SEResNetBackbone
from models.geometry import (depth_to_xyz, build_valid_mask, build_geometry_tensor,
                             xyz_to_normals, DenseGeometryEncoder, DepthGeometryBranch,
                             GeometryProjector)
from models.geometry.io import load_depth_frame
from models.fusion import MultiScaleRGBDCompensation
from models.fusion.ms2fusion import MS2FusionBranches


def intrinsics(batch=2):
    k = torch.tensor([[500., 0., 5.], [0., 450., 4.], [0., 0., 1.]]).repeat(batch, 1, 1)
    if batch > 1:
        k[1, 0, 0], k[1, 1, 2] = 700., 3.
    return k


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_backprojection_reprojection_and_per_frame_K(self):
        depth = torch.rand(2, 1, 9, 11) + 0.5
        k = intrinsics()
        xyz = depth_to_xyz(depth, k)
        torch.testing.assert_close(xyz[:, 2:3], depth)
        u = k[:, 0, 0, None, None] * xyz[:, 0] / xyz[:, 2] + k[:, 0, 2, None, None]
        v = k[:, 1, 1, None, None] * xyz[:, 1] / xyz[:, 2] + k[:, 1, 2, None, None]
        torch.testing.assert_close(u, torch.arange(11.).expand(2, 9, 11), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v, torch.arange(9.)[None, :, None].expand(2, 9, 11), atol=1e-6, rtol=1e-6)
        same_depth = torch.ones_like(depth)
        points = depth_to_xyz(same_depth, k)
        self.assertFalse(torch.equal(points[0], points[1]))

    def test_shapes_invalid_depth_and_finite_backward(self):
        depth = torch.ones(2, 1, 9, 11)
        depth[0, 0, 3, 2:7] = torch.tensor([0., -1., float('nan'), float('inf'), -float('inf')])
        depth.requires_grad_()
        mask = build_valid_mask(depth)
        xyz = depth_to_xyz(depth, intrinsics())
        normal, nv = xyz_to_normals(xyz, mask)
        geometry = build_geometry_tensor(depth, intrinsics())
        self.assertEqual(xyz.shape, (2, 3, 9, 11))
        self.assertEqual(normal.shape, xyz.shape)
        self.assertEqual(mask.shape, (2, 1, 9, 11))
        self.assertEqual(nv.shape, mask.shape)
        self.assertEqual(geometry.shape, (2, 7, 9, 11))
        self.assertTrue(torch.isfinite(geometry).all())
        self.assertTrue((geometry[:, :6].masked_select(~mask.expand(-1, 6, -1, -1)) == 0).all())
        self.assertFalse(nv[0, 0, 2, 4])
        geometry.square().mean().backward()
        self.assertTrue(torch.isfinite(depth.grad).all())

    def test_plane_orientation_edges_holes_and_degenerate(self):
        depth = torch.ones(1, 1, 9, 11)
        xyz = depth_to_xyz(depth, intrinsics(1))
        normal, valid = xyz_to_normals(xyz, build_valid_mask(depth))
        expected = torch.tensor([0., 0., -1.])[None, :, None, None].expand(1, 3, 7, 9)
        torch.testing.assert_close(normal[:, :, 1:-1, 1:-1], expected)
        self.assertFalse(valid[:, :, 0].any())
        opposite, _ = xyz_to_normals(xyz, build_valid_mask(depth), point_into_surface=True)
        torch.testing.assert_close(opposite, -normal)
        depth[:, :, :, 6:] = 2
        _, valid = xyz_to_normals(depth_to_xyz(depth, intrinsics(1)), build_valid_mask(depth))
        self.assertFalse(valid[:, :, :, 5:7].any())
        depth.zero_()
        g = build_geometry_tensor(depth, intrinsics(1))
        self.assertEqual(g.count_nonzero().item(), 0)
        degenerate = torch.ones(1, 3, 5, 5)
        n, v = xyz_to_normals(degenerate, torch.ones(1, 1, 5, 5, dtype=torch.bool))
        self.assertFalse(v.any())
        self.assertEqual(n.count_nonzero().item(), 0)
        for shape in ((1, 1), (2, 5)):
            g = build_geometry_tensor(torch.ones(1, 1, *shape), intrinsics(1))
            self.assertTrue(torch.isfinite(g).all())
            self.assertEqual(g[:, 3:6].count_nonzero().item(), 0)

    def test_tilted_plane_and_radius(self):
        k = intrinsics(1)
        u = torch.arange(11.)[None, None, None, :]
        # Plane Z = 1 + 0.2 X has camera-facing normal (0.2, 0, -1).
        depth = (1 / (1 - 0.2 * (u - 5) / 500)).expand(1, 1, 9, 11)
        normal, valid = xyz_to_normals(depth_to_xyz(depth, k), build_valid_mask(depth), radius=2)
        expected = torch.tensor([0.2, 0., -1.])
        expected = expected / expected.norm()
        torch.testing.assert_close(normal[0, :, 4, 5], expected, atol=1e-5, rtol=1e-5)
        self.assertTrue(valid[0, 0, 4, 5])

    def test_validation(self):
        depth = torch.ones(1, 1, 5, 5)
        for k in (intrinsics(2), torch.zeros(1, 3, 3), torch.full((1, 3, 3), float('nan'))):
            with self.assertRaises(ValueError):
                depth_to_xyz(depth, k)
        with self.assertRaises(ValueError):
            DenseGeometryEncoder(out_indices=(4, 1))
        with self.assertRaises(ValueError):
            GeometryProjector((4,), (4, 8), (4,))

    def test_real_rgb_alignment_and_mamba_gradients(self):
        rgb_encoder = SEResNetBackbone(pretrained=False).eval()
        geo_encoder = DenseGeometryEncoder.from_rgb_backbone(rgb_encoder)
        branch = DepthGeometryBranch(geo_encoder)
        self.assertEqual(rgb_encoder.stage_channels, (64, 256, 512, 1024, 2048))
        self.assertEqual(branch.stage_reductions, rgb_encoder.stage_reductions)
        for h, w in ((32, 48), (33, 49), (1, 1)):
            with torch.no_grad():
                rgb = rgb_encoder(torch.rand(1, 3, h, w))
                geo = branch(torch.ones(1, 1, h, w), intrinsics(1))
            self.assertIsInstance(geo, tuple)
            self.assertEqual([x.shape[2:] for x in rgb], [x.shape[2:] for x in geo])
        # Real forward at native sensor size; meta tensors avoid large RGB allocations.
        rgb_meta = SEResNetBackbone(pretrained=False).eval().to('meta')
        geo_meta = DenseGeometryEncoder.from_rgb_backbone(rgb_meta).to('meta')
        # Encoder validates actual geometry values, so execute its layers on meta.
        r = rgb_meta(torch.empty(1, 3, 720, 1280, device='meta'))
        x = geo_meta.stem(torch.empty(1, 7, 720, 1280, device='meta'))
        shapes = [x.shape[2:]]
        x = geo_meta.pool(x)
        for stage in geo_meta.stages:
            x = stage(x)
            shapes.append(x.shape[2:])
        self.assertEqual([t.shape[2:] for t in r], shapes)
        self.assertEqual(tuple(shapes[-1]), (23, 40))
        with torch.no_grad():
            rgb = rgb_encoder(torch.rand(1, 3, 9, 13))
        geo = branch(torch.ones(1, 1, 9, 13), intrinsics(1))
        projector = GeometryProjector(rgb_encoder.stage_channels, branch.stage_channels, (4,) * 5)
        rp, gp = projector(rgb, geo)
        blocks = torch.nn.ModuleList(MS2FusionBranches(4, state_dim=2, expand=1) for _ in rp)
        loss = 0
        for r, g, block in zip(rp, gp, blocks):
            self.assertEqual(r.shape, g.shape)
            out = block(r, g)
            self.assertTrue(all(torch.isfinite(t).all() for t in out.values()))
            loss = loss + sum(t.square().mean() for t in out.values())
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()))
        self.assertGreater(branch.encoder.stem[0].weight.grad.abs().sum().item(), 0)
        fusion = MultiScaleRGBDCompensation(rgb_encoder.stage_channels, branch.stage_channels, (4,) * 5,
                                            state_dim=2, expand=1, directions=4)
        fused, _ = fusion(rgb, tuple(g.detach() for g in geo))
        self.assertTrue(all(torch.isfinite(t).all() for t in fused))

    def test_selected_stages_and_all_invalid_autocast(self):
        rgb = SEResNetBackbone(pretrained=False, out_indices=(1, 3, 4)).eval()
        branch = DepthGeometryBranch(DenseGeometryEncoder.from_rgb_backbone(rgb))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            features = branch(torch.zeros(1, 1, 17, 23), intrinsics(1))
            loss = sum(f.float().square().mean() for f in features)
        loss.backward()
        self.assertEqual(len(features), 3)
        self.assertTrue(all(torch.isfinite(f).all() for f in features))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()))

    def test_sensor_loader_uses_frame_intrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'depth_sensor').mkdir()
            (root / 'poses').mkdir()
            for i, k in enumerate(intrinsics()):
                np.save(root / 'depth_sensor' / f'{i}.npy', np.ones((9, 11), np.float32))
                (root / 'poses' / f'{i}.json').write_text(json.dumps({'K': k.tolist()}))
            frames = [load_depth_frame(root, str(i)) for i in range(2)]
            depth, k = (torch.stack(items) for items in zip(*frames))
            self.assertEqual(depth.shape, (2, 1, 9, 11))
            torch.testing.assert_close(k, intrinsics())


if __name__ == '__main__':
    unittest.main()
