"""Isolated FP32 correctness/validation/benchmark worker; run by run_audit.py."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST'):
    raise SystemExit('Numerical work requires a Slurm allocation')
if os.environ['SLURM_JOB_ID'] == '2082' or platform.node().split('.')[0] == 'hpc01':
    raise SystemExit('Audit must not execute in job 2082 or on hpc01')

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from models.fusion.ssm.simple_ssm import SimpleSelectiveSSM
from models.keypoints.ablation import initialize_local
from models.keypoints.checkpoints import load_checkpoint
from models.keypoints.config import make_config
from models.keypoints.data import load_manifest, RGBDKeypointDataset
from models.keypoints.experiment import seed_everything, make_optimizer, evaluate_model, validate_checkpoint_manifest
from models.keypoints.heatmaps import decode_heatmaps, transform_points
from models.keypoints.local_losses import local_patch_loss, weighted_heatmap_loss
from models.keypoints.local_refinement import FourDirectionLocalMamba
from models.keypoints.model import RGBDKeypointModel, StageOutput

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True, type=Path)
parser.add_argument('--action', required=True)
parser.add_argument('--backend', choices=('torch', 'mamba'), default='torch')
parser.add_argument('--model', choices=('A1', 'A4'), default='A1')
parser.add_argument('--round', type=int, default=0)
args = parser.parse_args()
ROOT = args.root.resolve()
SEED = 20260922
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')
seed_everything(SEED)


def save_json(name, value):
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_backend(model, backend):
    modules = [m for m in model.modules() if isinstance(m, SimpleSelectiveSSM)]
    if not modules:
        raise AssertionError('No selective scan modules found')
    for module in modules:
        module.backend = backend
    if hasattr(model, 'config') and 'global_ssm' in model.config:
        model.config['global_ssm']['backend'] = backend
        model.config['local']['backend'] = backend
    return len(modules)


def make_model(kind, backend):
    seed_everything(SEED)
    model, payload = load_checkpoint(ROOT / 'checkpoint.pt')
    manifest = load_manifest(ROOT / 'manifest.json')
    validate_checkpoint_manifest(payload, manifest)
    if model.local_enabled or model.config['out_indices'] != (1, 2, 3, 4):
        raise AssertionError('Snapshot must be the trained four-scale global A1 checkpoint')
    if kind == 'A4':
        config = copy.deepcopy(model.config)
        config['local'].update(enabled=True, layout='five', embedding=True)
        model = RGBDKeypointModel(make_config(config))
        initialize_local(model, SEED)
        load_checkpoint(ROOT / 'checkpoint.pt', model=model, allow_local_missing=True)
    select_backend(model, backend)
    model.set_phase('joint' if kind == 'A4' else 'global')
    return model.float().cuda(), payload, manifest


def dataset(manifest, model, split, count=None):
    if split not in ('train', 'val'):
        raise AssertionError('Audit may not read the test dataset')
    ids = manifest[split] if count is None else manifest[split][:count]
    return RGBDKeypointDataset(manifest['dataset_root'], ids, manifest['keypoint_names'],
                              input_size=model.config['input_size'], heatmap_size=model.config['heatmap_size'])


def inference_sample(root, frame_id, input_hw):
    """Production-equivalent RGB/depth/K preprocessing without labels or targets."""
    root = Path(root)
    with Image.open(root / 'rgb' / f'{frame_id}.png') as source:
        source = source.convert('RGB')
        original_hw = (source.height, source.width)
        pixels = np.asarray(source.resize((input_hw[1], input_hw[0]), Image.Resampling.BILINEAR),
                            dtype=np.float32).copy()
    rgb = torch.from_numpy(pixels).permute(2, 0, 1) / 255
    rgb = (rgb - rgb.new_tensor((.485, .456, .406))[:, None, None]) / rgb.new_tensor((.229, .224, .225))[:, None, None]
    depth_array = np.load(root / 'depth_sensor' / f'{frame_id}.npy', allow_pickle=False)
    if depth_array.dtype != np.float32 or tuple(depth_array.shape) != original_hw:
        raise ValueError('Invalid RGB/depth pair')
    depth = F.interpolate(torch.from_numpy(depth_array.copy())[None, None], size=input_hw, mode='nearest')[0]
    K = torch.tensor(json.loads((root / 'poses' / f'{frame_id}.json').read_text())['K'], dtype=torch.float32)
    sy, sx = input_hw[0]/original_hw[0], input_hw[1]/original_hw[1]
    K = K.new_tensor(((sx, 0, (sx-1)/2), (0, sy, (sy-1)/2), (0, 0, 1))) @ K
    return {'rgb': rgb, 'depth': depth, 'K': K, 'original_size': original_hw, 'frame_id': frame_id}


def gpu_inputs(sample):
    return [sample[key].unsqueeze(0).cuda() for key in ('rgb', 'depth', 'K')]


def forward(model, inputs, patches=False):
    return (model.forward_stages(*inputs, return_patches=patches) if model.local_enabled
            else StageOutput(model(*inputs), None))


def loss_for(model, stage, sample):
    weights = sample['weights'][None].cuda()
    loss = weighted_heatmap_loss(stage.global_heatmaps, sample['target'][None].cuda(), weights, positive_weight=100.)
    if stage.local is not None:
        points = transform_points(sample['points'][None].cuda(), model.config['input_size'], stage.local.heatmaps.shape[-2:])
        loss = loss + local_patch_loss(stage.local.patches, points, weights, stage.local.heatmaps.shape[-2:], positive_weight=100.)
    return loss


def decode(stage, sample):
    maps = stage.local.heatmaps if stage.local is not None else stage.global_heatmaps
    support = stage.local.support if stage.local is not None else None
    coords = decode_heatmaps(maps, support=support, subpixel=True).coords
    return transform_points(coords, maps.shape[-2:], tuple(sample['original_size'])).cpu()


def grads(model, extra=None):
    result = {name: None if param.grad is None else param.grad.detach().cpu().clone()
              for name, param in model.named_parameters()}
    if extra is not None:
        result['input'] = extra.grad.detach().cpu().clone()
    return result


def assert_outputs(actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError('Nonfinite output')
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    return float((actual-expected).abs().max())


def compare_grads(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError('Gradient parameter names differ')
    worst, finite_count = 0., 0
    for name, reference in expected.items():
        candidate = actual[name]
        if (reference is None) != (candidate is None):
            raise AssertionError(f'Gradient participation differs: {name}')
        if reference is None:
            continue
        if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
            raise AssertionError(f'Nonfinite gradient: {name}')
        norm = float(torch.linalg.vector_norm(reference.double()))
        error = float(torch.linalg.vector_norm((candidate-reference).double()))
        if norm < 1e-8:
            if error > 1e-6:
                raise AssertionError(f'Near-zero gradient mismatch {name}: {error}')
        else:
            relative = error/norm
            worst = max(worst, relative)
            if relative > 2e-3:
                raise AssertionError(f'Gradient relative L2 mismatch {name}: {relative}')
        finite_count += 1
    if finite_count == 0:
        raise AssertionError('No active gradients compared')
    return {'worst_relative_l2': worst, 'finite_gradient_tensors': finite_count}


def environment():
    import selective_scan_cuda as extension
    from mamba_ssm.ops import selective_scan_interface as interface
    path = Path(extension.__file__).resolve()
    if path.suffix != '.so' or not path.is_relative_to(ROOT / 'environment'):
        raise AssertionError(f'Expected native extension in private environment: {path}')
    if interface.selective_scan_cuda is not extension:
        raise AssertionError('Mamba interface is not using the loaded native extension')
    if 'RTX A5000' not in torch.cuda.get_device_name(0):
        raise AssertionError('Wrong GPU model for the planned benchmark')
    if torch.__version__.split('+')[0] != '2.11.0' or torch.version.cuda != '13.0' or sys.version_info[:2] != (3, 13):
        raise AssertionError('Python/PyTorch/CUDA versions changed')
    if not Path(torch.__file__).resolve().is_relative_to(ROOT / 'environment'):
        raise AssertionError('PyTorch was not imported from the private environment')
    counts = {'fwd': 0, 'bwd': 0}
    originals = {name: getattr(extension, name) for name in counts}
    def counted(name):
        def call(*values, **kwargs):
            counts[name] += 1
            if values[0].device.type != 'cuda' or values[0].dtype != torch.float32:
                raise AssertionError('Native selective-scan did not receive CUDA FP32 input')
            return originals[name](*values, **kwargs)
        return call
    for name in counts:
        setattr(extension, name, counted(name))
    module = SimpleSelectiveSSM(32, 4, backend='mamba').cuda()
    x = torch.randn(1, 32, 127, device='cuda', requires_grad=True)
    module(x).square().sum().backward()
    module.zero_grad(set_to_none=True); x.grad = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        module(x).square().sum().backward()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(ROOT / 'selective-scan-trace.json'))
    kernels = sorted({event.name for event in prof.events()
                      if str(event.device_type).endswith('CUDA') and 'selective_scan' in event.name})
    if not counts['fwd'] or not counts['bwd'] or not kernels:
        raise AssertionError(f'Native execution not proven: counters={counts}, kernels={kernels}')
    info = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'torch_file': torch.__file__,
            'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'compute_capability': torch.cuda.get_device_capability(0),
            'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI, 'cuda_home': os.environ['CUDA_HOME'],
            'extension': str(path), 'extension_sha256': sha(path), 'native_calls': counts,
            'cuda_kernels': kernels, 'dtype': 'float32', 'autocast': False, 'tf32': False,
            'packages': {name: importlib.metadata.version(name) for name in ('torch', 'timm', 'mamba-ssm', 'triton')}}
    save_json('environment.json', info)


def scan():
    results = []
    for index, (batch, length, state_dim) in enumerate(((1, 1, 4), (1, 7, 16), (1, 3600, 4), (32, 3600, 16))):
        seed_everything(SEED + index)
        module = SimpleSelectiveSSM(32, state_dim, backend=args.backend).cuda()
        x = torch.randn(batch, 32, length, device='cuda', requires_grad=True)
        y = module(x)
        loss = y.square().sum()/batch
        loss.backward()
        results.append({'output': y.detach().cpu(), 'loss': loss.detach().cpu(), 'grads': grads(module, x),
                        'state_sha256': state_sha(module), 'shape': list(x.shape), 'state_dim': state_dim})
        del module, x, y, loss
        gc.collect(); torch.cuda.empty_cache()
    seed_everything(SEED)
    module = FourDirectionLocalMamba(32, 16, backend=args.backend).cuda()
    x = torch.randn(8, 32, 15, 23, device='cuda', requires_grad=True)
    y = module(x); loss = y.square().sum()/8; loss.backward()
    results.append({'output': y.detach().cpu(), 'loss': loss.detach().cpu(), 'grads': grads(module, x),
                    'state_sha256': state_sha(module), 'shape': list(x.shape), 'directions': 4})
    torch.save(results, ROOT / f'scan-{args.backend}.pt')


def correctness():
    import models.keypoints.local_refinement as local_module
    result = []
    reference = (torch.load(ROOT / f'correctness-{args.model}-torch.pt', weights_only=False)
                 if args.backend == 'mamba' else None)
    for index in range(3):
        model, payload, manifest = make_model(args.model, args.backend)
        model.train()
        data = dataset(manifest, model, 'train', 3)
        sample = data[index]
        inference = inference_sample(manifest['dataset_root'], sample['frame_id'], model.config['input_size'])
        for key in ('rgb', 'depth', 'K'):
            torch.testing.assert_close(inference[key], sample[key], rtol=0, atol=0, equal_nan=True)
        digest = state_sha(model)
        original_regions = local_module.extract_regions
        if args.model == 'A4' and reference is not None:
            fixed_maps = reference[index]['global'].cuda()
            def fixed_regions(maps, hw, **kwargs):
                return original_regions(fixed_maps, hw, **kwargs)
            local_module.extract_regions = fixed_regions
        try:
            stage = forward(model, gpu_inputs(sample), patches=True)
            loss = loss_for(model, stage, sample)
            loss.backward()
            row = {'global': stage.global_heatmaps.detach().cpu(), 'loss': loss.detach().cpu(),
                   'grads': grads(model), 'state_sha256': digest, 'frame_id': sample['frame_id'],
                   'checkpoint_epoch': payload['epoch']}
            if stage.local is not None:
                row.update(local=stage.local.heatmaps.detach().cpu(), support=stage.local.support.cpu(),
                           coverage=stage.local.coverage.cpu(),
                           patches=[(p.batch_index, p.keypoint_index, p.box.x1, p.box.y1, p.box.x2, p.box.y2)
                                    for p in stage.local.patches])
            result.append(row)
        finally:
            local_module.extract_regions = original_regions
        del model, stage, loss
        gc.collect(); torch.cuda.empty_cache()
    torch.save(result, ROOT / f'correctness-{args.model}-{args.backend}.pt')


def compare():
    report = {}
    for label in ('scan', 'correctness-A1', 'correctness-A4'):
        reference = torch.load(ROOT / f'{label}-torch.pt', map_location='cpu', weights_only=False)
        candidate = torch.load(ROOT / f'{label}-mamba.pt', map_location='cpu', weights_only=False)
        if len(reference) != len(candidate):
            raise AssertionError('Correctness sample counts differ')
        rows = []
        for actual, expected in zip(candidate, reference):
            if actual['state_sha256'] != expected['state_sha256']:
                raise AssertionError('Models did not start with identical weights')
            row = {'gradient': compare_grads(actual['grads'], expected['grads'])}
            for name in ('output', 'global', 'local', 'loss'):
                if name in expected:
                    row[name + '_max_abs_error'] = assert_outputs(actual[name], expected[name])
            for name in ('support', 'coverage'):
                if name in expected and not torch.equal(actual[name], expected[name]):
                    raise AssertionError(f'Local {name} differs')
            if actual.get('patches') != expected.get('patches'):
                raise AssertionError('Local patches differ despite fixed ROI input')
            rows.append(row)
        report[label] = rows
    report['pass'] = True
    save_json('correctness.json', report)


def validation():
    model, payload, manifest = make_model('A1', args.backend)
    data = dataset(manifest, model, 'val')
    if len(data) != 500:
        raise AssertionError('Validation must use all 500 authorized frames')
    result = evaluate_model(model, data, device='cuda', decoder='subpixel')
    result.update(checkpoint_epoch=payload['epoch'], checkpoint_sha256=sha(ROOT / 'checkpoint.pt'))
    save_json(f'validation-{args.backend}.json', result)


def compare_validation():
    reference, candidate = [json.loads((ROOT / f'validation-{b}.json').read_text()) for b in ('torch', 'mamba')]
    if reference['checkpoint_sha256'] != candidate['checkpoint_sha256']:
        raise AssertionError('Validation checkpoints differ')
    result = {name + '_absolute_delta': abs(candidate['selected'][name] - reference['selected'][name])
              for name in ('mean', 'p95')}
    result.update(torch=reference['selected'], mamba=candidate['selected'])
    result['pass'] = result['mean_absolute_delta'] <= .01 and result['p95_absolute_delta'] <= .05
    save_json('validation-comparison.json', result)
    if not result['pass']:
        raise AssertionError(f'Validation metric gate failed: {result}')


def sync():
    torch.cuda.synchronize()


def summarize(rows, memory):
    values = np.asarray([row['seconds'] for row in rows], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or not (values > 0).all():
        raise AssertionError('Invalid timing samples')
    return {'mean_seconds': float(values.mean()), 'p50_seconds': float(np.percentile(values, 50)),
            'p95_seconds': float(np.percentile(values, 95)), 'images_per_second': float(1/values.mean()),
            'fps': float(1/values.mean()), 'count': len(rows), **memory, 'samples': rows}


def memory():
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def bench_inference(model, manifest, full):
    ids = manifest['val'][:100]
    def read(frame_id):
        return inference_sample(manifest['dataset_root'], frame_id, model.config['input_size'])
    cached = None if full else [read(frame_id) for frame_id in ids]
    model.eval()
    with torch.inference_mode():
        for index in range(10):
            sample = read(ids[index]) if full else cached[index]
            stage = forward(model, gpu_inputs(sample))
            if full:
                decode(stage, sample)
        del stage
        sync(); torch.cuda.reset_peak_memory_stats()
        rows = []
        for index, frame_id in enumerate(ids):
            if full:
                sync(); start = time.perf_counter()
                sample = read(frame_id)
                inputs = gpu_inputs(sample)
            else:
                sample = cached[index]
                inputs = gpu_inputs(sample)
                sync(); start = time.perf_counter()
            stage = forward(model, inputs)
            coordinates = decode(stage, sample) if full else None
            sync(); elapsed = time.perf_counter() - start
            if not torch.isfinite(stage.global_heatmaps).all() or (coordinates is not None and not torch.isfinite(coordinates).all()):
                raise AssertionError('Nonfinite inference result')
            rows.append({'frame_id': frame_id, 'seconds': elapsed})
            del stage, inputs
    return summarize(rows, memory())


def bench_training(model, manifest, full):
    model.train()
    data = dataset(manifest, model, 'train', 32)
    cached = None if full else [data[index] for index in range(32)]
    optimizer = make_optimizer(model, lr=3e-4 if args.model == 'A1' else 1e-4)
    initial_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    def one(index, measured=False):
        if full:
            sync(); total_start = time.perf_counter()
            sample = data[index]
            inputs = gpu_inputs(sample)
        else:
            sample = cached[index]
            inputs = gpu_inputs(sample)
            # Transfer supervision outside the forward/backward interval as well.
            sample = dict(sample)
            for key in ('target', 'weights', 'points'):
                sample[key] = sample[key].cuda()
            sync(); total_start = time.perf_counter()
        if full:
            stage = forward(model, inputs, patches=True)
            loss = loss_for(model, stage, sample)
            (loss/8).backward()
            component = {}
        else:
            sync(); forward_start = time.perf_counter()
            stage = forward(model, inputs, patches=True)
            sync(); forward_seconds = time.perf_counter() - forward_start
            loss_start = time.perf_counter()
            loss = loss_for(model, stage, sample)
            sync(); loss_seconds = time.perf_counter() - loss_start
            backward_start = time.perf_counter()
            (loss/8).backward()
            sync(); backward_seconds = time.perf_counter() - backward_start
            component = {'forward_seconds': forward_seconds, 'loss_seconds': loss_seconds,
                         'backward_seconds': backward_seconds}
        if not full:
            sync(); measured_seconds = time.perf_counter() - total_start
        step_seconds = None
        if (index+1) % 8 == 0:
            sync(); step_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            sync(); step_seconds = time.perf_counter() - step_start
        if full:
            sync(); measured_seconds = time.perf_counter() - total_start
        if not torch.isfinite(loss):
            raise AssertionError('Nonfinite training loss')
        return {'frame_id': sample['frame_id'], 'seconds': measured_seconds,
                'optimizer_step_seconds': step_seconds, 'loss': float(loss.detach()), **component}
    optimizer.zero_grad(set_to_none=True)
    for index in range(16):
        one(index)
    model.load_state_dict(initial_model, strict=True)
    optimizer.load_state_dict(initial_optimizer)
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(rng_cpu); torch.cuda.set_rng_state_all(rng_cuda)
    del initial_model, initial_optimizer
    sync(); torch.cuda.reset_peak_memory_stats()
    rows = [one(index, True) for index in range(32)]
    result = summarize(rows, memory())
    steps = [{'seconds': row['optimizer_step_seconds']} for row in rows if row['optimizer_step_seconds'] is not None]
    result['optimizer_steps'] = summarize(steps, {})
    result['accumulation'] = 8
    return result


def benchmark():
    result = {'model': args.model, 'backend': args.backend, 'round': args.round, 'dtype': 'float32',
              'tf32': False, 'autocast': False, 'batch_size': 1, 'checkpoint_sha256': sha(ROOT / 'checkpoint.pt'),
              'A4_local_trained': False, 'scope_note': 'training_compute excludes optimizer and data transfer; training_full includes them'}
    digests = []
    for scope, function, full in (('inference_forward', bench_inference, False), ('inference_full', bench_inference, True),
                                  ('training_compute', bench_training, False), ('training_full', bench_training, True)):
        model, payload, manifest = make_model(args.model, args.backend)
        digest = state_sha(model); digests.append(digest)
        print(f'{args.model} {args.backend} round {args.round}: {scope}', flush=True)
        result[scope] = function(model, manifest, full)
        del model
        gc.collect(); torch.cuda.empty_cache()
        save_json(f'bench-progress-{args.model}-{args.backend}-{args.round}.json', result)
    if len(set(digests)) != 1:
        raise AssertionError('Benchmark scopes did not start with identical model weights')
    result['initial_state_sha256'] = digests[0]
    save_json(f'bench-{args.model}-{args.backend}-{args.round}.json', result)


def selftest():
    """Reject silent false passes before spending time on the full audit."""
    def must_reject(function):
        try:
            function()
        except AssertionError:
            return
        raise AssertionError('Audit check silently accepted invalid evidence')
    must_reject(lambda: assert_outputs(torch.tensor([float('nan')]), torch.ones(1)))
    must_reject(lambda: assert_outputs(torch.zeros(2), torch.ones(2)))
    must_reject(lambda: compare_grads({'p': torch.zeros(2)}, {'p': torch.ones(2)}))
    must_reject(lambda: compare_grads({'p': None}, {'p': torch.ones(2)}))
    compare_grads({'p': torch.ones(2)}, {'p': torch.ones(2)})
    must_reject(lambda: summarize([{'seconds': 0.}], {}))
    manifest = load_manifest(ROOT / 'manifest.json')
    if [len(manifest[s]) for s in ('train', 'val', 'test')] != [4000, 500, 500]:
        raise AssertionError('Unexpected authorized manifest sizes')
    save_json('harness-selftest.json', {'pass': True, 'rejection_cases': 5,
                                       'manifest_counts': [4000, 500, 500]})


actions = {'environment': environment, 'scan': scan, 'correctness': correctness, 'compare': compare,
           'validation': validation, 'compare-validation': compare_validation, 'benchmark': benchmark,
           'selftest': selftest}
try:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in audit allocation')
    actions[args.action]()
    print(f'PASS {args.action} {args.model} {args.backend} {args.round}', flush=True)
except BaseException as error:
    save_json(f'failure-{args.action}-{args.model}-{args.backend}-{args.round}.json',
              {'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
    raise
