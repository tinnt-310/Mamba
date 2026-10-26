import ast
from pathlib import Path
import unittest

import torch
from torch import nn

from models.fusion import RGBDCompensation, MultiScaleRGBDCompensation
from models.fusion.ssm import SimpleSelectiveSSM, selective_scan, scan_image, merge_image
from models.fusion.ms2fusion.cross_parametric import exchange_readouts
from models.fusion.ms2fusion import CPSSM
from models.fusion.reliability import compensate


class CompensationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        torch.set_num_threads(1)

    def test_standalone_cp_block(self):
        cp = CPSSM(4, state_dim=3)
        r = torch.randn(2, 4, 7, requires_grad=True)
        d = torch.randn(2, 4, 7, requires_grad=True)
        rc, dc = cp(r, d)
        rp, dp = cp.rgb_ssm.parameterize(r), cp.depth_ssm.parameterize(d)
        # Explicit upstream C-only call contract, independent of exchange helper.
        expected_r = selective_scan(r, rp.delta, -cp.rgb_ssm.A_log.exp(),
                                    rp.B, dp.C, cp.rgb_ssm.D)
        expected_d = selective_scan(d, dp.delta, -cp.depth_ssm.A_log.exp(),
                                    dp.B, rp.C, cp.depth_ssm.D)
        torch.testing.assert_close(rc, expected_r)
        torch.testing.assert_close(dc, expected_d)
        rc.square().mean().backward(retain_graph=True)
        self.assertGreater(d.grad.abs().sum().item(), 0)
        (rc.square().mean() + dc.square().mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in cp.parameters()))

    def test_main_block_calls_cp_forward(self):
        model = RGBDCompensation(3, 1, 4, state_dim=3)
        calls = []
        cp = model.cross_parametric.cp_ssm
        handle = cp.register_forward_hook(lambda module, args, output: calls.append(output))
        try:
            fused, aux = model(torch.randn(1, 3, 2, 3), torch.randn(1, 1, 2, 3))
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(fused.shape, (1, 4, 2, 3))
        self.assertEqual(len(aux), 8)
        # State is registered once, and can be restored without missing keys.
        restored = RGBDCompensation(3, 1, 4, state_dim=3)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(any('cross_parametric.cp_ssm.rgb_ssm.A_log' == k
                            for k in model.state_dict()))

    def test_scan_order_and_rectangular_roundtrip(self):
        x = torch.arange(6.).reshape(1, 1, 2, 3)
        scans = scan_image(x, 4)
        self.assertEqual(scans[0, 1, 0].tolist(), [0, 3, 1, 4, 2, 5])
        for shape in [(2, 3, 3, 5), (1, 2, 1, 1), (1, 2, 5, 1)]:
            x = torch.randn(shape).transpose(2, 3).requires_grad_()
            for k in (1, 4):
                y = merge_image(scan_image(x, k), *x.shape[2:])
                torch.testing.assert_close(y, x)
                torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0], torch.ones_like(x))

    def test_recurrence_against_official_reference_and_gradients(self):
        # Load only the actual pure reference function via AST: do not import or
        # execute the upstream module's CUDA/Triton imports.
        path = Path(__file__).resolve().parents[1] / 'references/mamba/mamba_ssm/ops/selective_scan_interface.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'selective_scan_ref')
        scope = {'torch': torch, 'F': torch.nn.functional, 'rearrange': lambda x, pattern: x[:, None]}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        ref = scope['selective_scan_ref']
        shapes = [(2, 3, 7), (2, 3, 7), (3, 2), (2, 2, 7), (2, 2, 7), (3,)]
        args = [torch.randn(s).requires_grad_() for s in shapes]
        with torch.no_grad():
            args[1].copy_(args[1].sigmoid())
            args[2].copy_(-args[2].exp())
        expected = ref(*args)
        actual = selective_scan(*args)
        torch.testing.assert_close(actual, expected)
        grad_actual = torch.autograd.grad(actual.square().sum(), args)
        grad_expected = torch.autograd.grad(expected.square().sum(), args)
        for a, e in zip(grad_actual, grad_expected):
            torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-5)

    def test_double_precision_gradcheck(self):
        args = [torch.randn(s, dtype=torch.double, requires_grad=True)
                for s in [(1, 2, 3), (1, 2, 3), (2, 2), (1, 2, 3), (1, 2, 3), (2,)]]
        self.assertTrue(torch.autograd.gradcheck(selective_scan, tuple(args)))

    def test_c_only_exchange_and_dependency(self):
        r, d = SimpleSelectiveSSM(4, 3), SimpleSelectiveSSM(4, 3)
        x, z = torch.randn(2, 4, 6), torch.randn(2, 4, 6, requires_grad=True)
        rp, dp = r.parameterize(x), d.parameterize(z)
        rc, dc = exchange_readouts(rp, dp)
        self.assertIs(rc.delta, rp.delta)
        self.assertIs(rc.B, rp.B)
        self.assertIs(rc.C, dp.C)
        self.assertIs(dc.C, rp.C)
        cross = r.run(x, rc)
        donor_grad = torch.autograd.grad(cross.square().sum(), z)[0]
        self.assertGreater(donor_grad.abs().sum().item(), 0)
        tied_rc, _ = exchange_readouts(rp, rp)
        torch.testing.assert_close(r.run(x, tied_rc), r(x))

    def test_gate_limits_and_full_block_gradients(self):
        m = RGBDCompensation(3, 1, 4, 6, state_dim=3)
        r = torch.randn(2, 3, 3, 5, requires_grad=True)
        d = torch.randn(2, 1, 3, 5, requires_grad=True)
        fused, aux = m(r, d)
        self.assertEqual(fused.shape, (2, 6, 3, 5))
        for name in ('rgb', 'depth'):
            s, c = aux[name + '_self'], aux[name + '_cross']
            torch.testing.assert_close(compensate(s, c, torch.zeros_like(s)), s)
            torch.testing.assert_close(compensate(s, c, torch.ones_like(s)), c)
        for name in ('gate_d2r', 'gate_r2d'):
            self.assertTrue(((aux[name] >= 0) & (aux[name] <= 1)).all())
        self.assertFalse(torch.equal(aux['gate_d2r'], aux['gate_r2d']))
        self.assertTrue(set(map(id, m.compensation.d2r.parameters())).isdisjoint(map(id, m.compensation.r2d.parameters())))
        fused.square().mean().backward()
        for tensor in (r, d):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_identical_streams_equal_self_and_cross(self):
        m = RGBDCompensation(4, 4, 4, state_dim=3, directions=4)
        m.cp.depth_in.load_state_dict(m.cp.rgb_in.state_dict())
        m.cp.depth_ssm.load_state_dict(m.cp.rgb_ssm.state_dict())
        x = torch.randn(1, 4, 3, 5)
        _, aux = m(x, x)
        for name in ('rgb', 'depth'):
            torch.testing.assert_close(aux[name + '_self'], aux[name + '_cross'])

    def test_multiscale_and_validation(self):
        m = MultiScaleRGBDCompensation([4, 8], [2, 4], [4, 6], state_dim=2, directions=4)
        r = [torch.randn(1, 4, 3, 5), torch.randn(1, 8, 1, 1)]
        d = [torch.randn(1, 2, 3, 5), torch.randn(1, 4, 1, 1)]
        y, aux = m(r, d)
        self.assertEqual([t.shape for t in y], [torch.Size([1, 4, 3, 5]), torch.Size([1, 6, 1, 1])])
        self.assertIsNot(m.blocks[0].cp.rgb_ssm, m.blocks[1].cp.rgb_ssm)
        with self.assertRaises(ValueError):
            m(r[:1], d)
        with self.assertRaisesRegex(ValueError, 'spatial'):
            m.blocks[0](r[0], torch.randn(1, 2, 5, 3))
        with self.assertRaisesRegex(ValueError, 'channels'):
            m.blocks[0](r[0], torch.randn(1, 3, 3, 5))

    def test_cpu_autocast(self):
        m = RGBDCompensation(3, 1, 4, state_dim=2)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y, _ = m(torch.randn(1, 3, 2, 3), torch.randn(1, 1, 2, 3))
            loss = y.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_optional_cuda_backend(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except ImportError:
            self.skipTest('mamba_ssm CUDA extension unavailable')
        m = SimpleSelectiveSSM(4, 3).cuda()
        x = torch.randn(2, 4, 7, device='cuda', requires_grad=True)
        expected = m(x)
        m.backend = 'mamba'
        actual = m(x)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        params = (x, *m.parameters())
        ge = torch.autograd.grad(expected.square().sum(), params)
        ga = torch.autograd.grad(actual.square().sum(), params)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, rtol=2e-3, atol=2e-3)


if __name__ == '__main__':
    unittest.main()
