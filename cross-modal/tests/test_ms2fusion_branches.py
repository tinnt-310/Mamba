"""Numerical parity against actual pinned DBISSF_SS/DBISSF_Attention source."""
import ast
import math
from pathlib import Path
import unittest
import torch
from torch import nn
from einops import rearrange, repeat
from models.fusion.ms2fusion import MS2FusionBranches, SPSSM


def upstream_classes():
    root = Path(__file__).resolve().parents[1] / 'references'
    scope = dict(torch=torch, nn=nn, math=math, rearrange=rearrange, repeat=repeat,
                 F=torch.nn.functional)
    scan_path = root / 'mamba/mamba_ssm/ops/selective_scan_interface.py'
    tree = ast.parse(scan_path.read_text(encoding='utf-8'))
    scan = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'selective_scan_ref')
    exec(compile(ast.Module(body=[scan], type_ignores=[]), str(scan_path), 'exec'), scope)
    scope['selective_scan_fn_v1'] = scope['selective_scan_ref']
    path = root / 'MS2Fusion/RGB-T OD/models/common.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
             and n.name in ('DBISSF_Attention', 'DBISSF_SS')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['DBISSF_SS']


class MS2FusionParityTests(unittest.TestCase):
    def test_cp_sp_image_outputs_and_all_gradients_match_source(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)
        upstream = upstream_classes()(d_model=17, d_state=3, ssm_ratio=2, scan=False, dropout=0)
        local = MS2FusionBranches(17, state_dim=3)
        pairs = []

        def bind(dst, src):
            for (dn, d), (sn, s) in zip(dst.named_parameters(), src.named_parameters()):
                self.assertEqual(d.numel(), s.numel(), (dn, sn))
                with torch.no_grad():
                    d.copy_(s.reshape_as(d))
                pairs.append((d, s))

        for dn, sn in [('rgb_in', 'in_proj'), ('depth_in', 'in_proj_modalx'),
                       ('shared_in', 'in_proj_share'), ('rgb_out', 'out_proj_rgb'),
                       ('depth_out', 'out_proj_e'), ('shared_out', 'out_proj_share')]:
            bind(getattr(local, dn), getattr(upstream, sn))
        bind(local.local[0], upstream.conv2d)
        core = upstream.CMA_ssm
        for ssm, suffix in [(local.cp_ssm.rgb_ssm, '1'), (local.cp_ssm.depth_ssm, '2'),
                            (local.sp_ssm.shared_ssm, 'share')]:
            bind(ssm.x_proj, getattr(core, 'x_proj_' + suffix))
            bind(ssm.dt_proj, getattr(core, 'dt_proj_' + suffix))
            for name in ('A_log', 'D'):
                d, s = getattr(ssm, name), getattr(core, name + '_' + suffix)
                with torch.no_grad():
                    d.copy_(s)
                pairs.append((d, s))
        for dn, sn in [('rgb_norm', 'out_norm_1'), ('depth_norm', 'out_norm_2'),
                       ('shared_norm', 'out_norm_share')]:
            bind(getattr(local, dn), getattr(core, sn))
        self.assertEqual(len(pairs), len(list(local.parameters())))
        self.assertEqual(len(pairs), len(list(upstream.parameters())))
        r, d = (torch.randn(2, 17, 2, 3, requires_grad=True) for _ in range(2))
        expected = upstream(r, d)
        features = local(r, d)
        actual = [features[k] for k in ('rgb_cross', 'rgb_shared', 'depth_cross', 'depth_shared')]
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)
        ga = torch.autograd.grad(sum(x.square().sum() for x in actual), (r, d, *(x[0] for x in pairs)))
        ge = torch.autograd.grad(sum(x.square().sum() for x in expected), (r, d, *(x[1] for x in pairs)))
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e.reshape_as(a), atol=3e-5, rtol=3e-4)

    def test_shared_parameters_joint_embedding_and_separate_inputs(self):
        sp = SPSSM(4, state_dim=3)
        r, d, sr, sd = (torch.randn(1, 4, 5, requires_grad=True) for _ in range(4))
        seen = []
        hook = sp.shared_ssm.x_proj.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
        try:
            yr, yd = sp(r, d, sr, sd)
        finally:
            hook.remove()
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], (sr + sd).transpose(1, 2))
        self.assertFalse(torch.allclose(yr, yd))
        grads = torch.autograd.grad(yr.square().sum() + yd.square().sum(), (r, d, sr, sd))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
        torch.testing.assert_close(grads[2], grads[3])


if __name__ == '__main__':
    unittest.main()
