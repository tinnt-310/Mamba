"""Slurm-only train/evaluate/infer CLI for 2D RGB-D keypoints."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .runtime import ComputeBudgetExpired, check_deadline

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/work/tinnt/BlenderProc/output/keypoints_bbox_50000_job_1634')
DEFAULT_MANIFEST = PROJECT / 'runs' / 'subset5000.json'
SEED = 20260922
PHASE_DEFAULTS = {'global': (200, 3e-4, 20), 'local': (50, 3e-4, 10), 'joint': (50, 1e-4, 10)}


def require_compute_node():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST'):
        raise RuntimeError('Numerical work must run on a Slurm compute node; submit scripts/slurm/*.slurm per /home/tinnt/slurm.md')


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '1'))))


def make_optimizer(model, *, lr, weight_decay=1e-4):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (no_decay if getattr(parameter, '_no_weight_decay', False) else decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError('phase has no trainable parameters')
    return torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                             {'params': no_decay, 'weight_decay': 0.}], lr=lr)


def metric_improved(candidate, best):
    if candidate.get('mean') is None or not math.isfinite(candidate['mean']):
        return False
    return best is None or (candidate['mean'], candidate['p95']) < (best['mean'], best['p95'])


def accumulation_window(index, count, accumulation):
    start = (index // accumulation) * accumulation
    return min(accumulation, count - start)


def validate_checkpoint_manifest(payload, manifest):
    if payload['split_ids'] != {s: manifest[s] for s in ('train','val','test')}:
        raise ValueError('checkpoint split IDs differ from the persisted manifest')
    if payload['keypoint_names'] != manifest['keypoint_names']:
        raise ValueError('checkpoint ordered keypoint names differ from the manifest')


class FeatureCache:
    """Bounded CPU LRU, used only while global parameters are frozen."""
    def __init__(self, max_bytes=8 * 1024**3):
        self.max_bytes, self.bytes = max_bytes, 0
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def get_or_compute(self, model, batch, device):
        if model.phase != 'local':
            raise ValueError('Feature cache requires frozen global/local phase')
        if getattr(model,'residual_local',None) is not None or model.config.get('detail_decoder'):
            raise ValueError('Legacy feature cache cannot store detail/residual features')
        # Inputs plus model identity/parameter versions invalidate stale entries
        # after source, preprocessing, augmentation or checkpoint changes.
        provenance = tuple(batch.get('cache_key',batch.get('rgb_path',batch['frame_id'])))
        versions = tuple((id(p),p._version) for module in model._global_modules() for p in module.parameters())
        key = (provenance,tuple(batch['frame_id']),versions)
        if key in self.entries:
            self.hits += 1
            item = self.entries.pop(key); self.entries[key] = item
        else:
            self.misses += 1
            with torch.no_grad():
                fused = model.encode_fused(batch['rgb'].to(device), batch['depth'].to(device), batch['K'].to(device))
                maps = model.global_from_fused(fused)
            item = (fused[0].detach().cpu(), maps.detach().cpu())
            size = sum(t.numel() * t.element_size() for t in item)
            if size <= self.max_bytes:
                while self.entries and self.bytes + size > self.max_bytes:
                    old = self.entries.popitem(last=False)[1]
                    self.bytes -= sum(t.numel() * t.element_size() for t in old)
                self.entries[key] = item; self.bytes += size
        return tuple(t.to(device) for t in item)


def _json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _dataset(root, manifest, split, config, max_samples=None):
    from .data import RGBDKeypointDataset
    if config['num_keypoints'] != len(manifest['keypoint_names']):
        raise ValueError('model keypoint count differs from manifest semantic names')
    ids = manifest[split]
    if max_samples is not None:
        ids = ids[:max_samples]
    if not ids:
        raise ValueError(f'empty {split} split')
    return RGBDKeypointDataset(root, ids, manifest['keypoint_names'],
                              input_size=config['input_size'], heatmap_size=config['heatmap_size'])


def _sample(batch, index):
    return {key: value[index] for key, value in batch.items()}


def _region_summary(stage, batch, input_hw):
    from .heatmaps import transform_points
    if stage.local is None:
        return None
    local = stage.local
    points = transform_points(batch['points'].to(local.heatmaps.device), input_hw, local.heatmaps.shape[-2:])
    valid = batch['weights'].to(points.device) > 0
    hits = valid.new_zeros(valid.shape)
    areas, patch_areas, counts = [], [], []
    for region in local.regions:
        bi, ki = region.batch_index, region.keypoint_index
        x, y = points[bi, ki].tolist()
        box = region.feature_box
        areas.append(box.width * box.height)
        counts.append(len(region.patch_boxes))
        for patch in region.patch_boxes:
            patch_areas.append(patch.width * patch.height)
            if patch.x1-.5 <= x < patch.x2-.5 and patch.y1-.5 <= y < patch.y2-.5:
                hits[bi, ki] = True
    return {'valid': int(valid.sum()), 'hits': int((hits & valid).sum()),
            'areas': areas, 'patch_areas': patch_areas, 'counts': counts}


@torch.no_grad()
def evaluate_model(model, dataset, *, device='cpu', decoder='subpixel', artifact_dir=None):
    from .heatmaps import decode_heatmaps, transform_points
    from .outputs import compute_metrics, save_prediction_artifact
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    predictions = {'global_argmax': [], 'global_subpixel': []}
    evaluate_local = model.local_enabled and model.phase != 'global'
    if evaluate_local:
        predictions.update(local_argmax=[], local_subpixel=[])
    targets, weights = [], []
    region_rows, subpixel_counts = [], {}
    artifact_paths = []
    durations = []
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        check_deadline()
        rgb, depth, K = (batch[key].to(device) for key in ('rgb', 'depth', 'K'))
        if rgb.is_cuda: torch.cuda.synchronize()
        started = time.perf_counter()
        if evaluate_local:
            stage = model.forward_stages(rgb, depth, K, return_patches=artifact_dir is not None)
        else:
            from .model import StageOutput
            stage = StageOutput(model(rgb, depth, K), None)
        if rgb.is_cuda: torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        for prefix, maps, support in [('global', stage.global_heatmaps, None),
                                      ('local', stage.local.heatmaps if stage.local else None,
                                       stage.local.support if stage.local else None)]:
            if maps is None: continue
            decoded = decode_heatmaps(maps, support=support, subpixel=True)
            subpixel_counts.setdefault(prefix, [0, 0])
            subpixel_counts[prefix][0] += int(decoded.subpixel_used.sum())
            subpixel_counts[prefix][1] += decoded.subpixel_used.numel()
            for method, coords in [('argmax', decoded.argmax), ('subpixel', decoded.coords)]:
                original_hw = tuple(int(v) for v in batch['original_size'][0])
                original = transform_points(coords[0], maps.shape[-2:], original_hw)
                predictions[f'{prefix}_{method}'].append(original.cpu())
        targets.append(batch['original_points'][0]); weights.append(batch['weights'][0])
        row = _region_summary(stage, batch, rgb.shape[-2:])
        if row: region_rows.append(row)
        if artifact_dir is not None:
            sample = _sample(batch, 0)
            sample['runtime_seconds'] = durations[-1]
            sample['peak_vram_bytes'] = int(torch.cuda.max_memory_allocated(device)) if rgb.is_cuda else 0
            saved = save_prediction_artifact(Path(artifact_dir) / batch['frame_id'][0], sample,
                                             stage, model.config, dataset.keypoint_names)
            artifact_paths.append(saved['json'])
    gt, validity = torch.stack(targets), torch.stack(weights)
    metrics = {key: compute_metrics(torch.stack(value), gt, validity, dataset.keypoint_names)
               for key, value in predictions.items()}
    if artifact_dir is not None:
        from .outputs import aggregate_region_statistics
        metrics['artifact_statistics'] = aggregate_region_statistics(artifact_paths)
    metrics['selected'] = metrics[('local' if evaluate_local else 'global') + '_' + decoder]
    if region_rows:
        valid = sum(row['valid'] for row in region_rows)
        areas = [v for row in region_rows for v in row['areas']]
        patch_areas = [v for row in region_rows for v in row['patch_areas']]
        counts = [v for row in region_rows for v in row['counts']]
        metrics['regions'] = {'roi_recall': sum(row['hits'] for row in region_rows) / valid if valid else None,
                              'mean_area': float(np.mean(areas)), 'mean_patch_area': float(np.mean(patch_areas)),
                              'mean_patch_count': float(np.mean(counts)),
                              'sequence_length_mean': float(np.mean(patch_areas)),
                              'sequence_length_max': int(max(patch_areas))}
    metrics['subpixel'] = {key: {'used_axes': values[0], 'total_axes': values[1],
                               'used_fraction': values[0]/values[1], 'fallback_fraction': 1-values[0]/values[1]}
                           for key, values in subpixel_counts.items()}
    metrics['runtime_seconds_per_image'] = float(np.mean(durations))
    metrics['peak_vram_bytes'] = int(torch.cuda.max_memory_allocated(device)) if str(device).startswith('cuda') else 0
    metrics['images'] = len(dataset)
    return metrics


def train_epoch(model, dataset, optimizer, *, phase, device, accumulation=8,
                positive_weight=100., lambda_local=1., cache=None, shuffle=True,
                progress_path=None, epoch=None):
    from .heatmaps import transform_points
    from .local_losses import local_patch_loss, weighted_heatmap_loss
    from .model import StageOutput
    model.train(); model.set_phase(phase)
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)
    optimizer.zero_grad(set_to_none=True)
    total, global_total, local_total = 0., 0., 0.
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        check_deadline()
        if phase == 'local':
            f1, maps = cache.get_or_compute(model, batch, device)
            stage = model.refine_cached(f1, maps, return_patches=True)
        elif phase == 'global':
            stage = StageOutput(model(*(batch[key].to(device) for key in ('rgb', 'depth', 'K'))), None)
        else:
            stage = model.forward_stages(*(batch[key].to(device) for key in ('rgb', 'depth', 'K')), return_patches=True)
        valid = batch['weights'].to(device)
        global_loss = stage.global_heatmaps.sum() * 0.
        local_loss = global_loss
        if phase != 'local':
            global_loss = weighted_heatmap_loss(stage.global_heatmaps, batch['target'].to(device), valid, positive_weight=positive_weight)
        if phase != 'global':
            if stage.local is None:
                raise ValueError('local/joint phase requires an enabled local module')
            points = transform_points(batch['points'].to(device), model.config['input_size'], stage.local.heatmaps.shape[-2:])
            local_loss = local_patch_loss(stage.local.patches, points, valid, stage.local.heatmaps.shape[-2:], positive_weight=positive_weight)
        loss = global_loss + lambda_local * local_loss
        if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss for {batch["frame_id"]}')
        (loss / accumulation_window(index, len(loader), accumulation)).backward()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach()); global_total += float(global_loss.detach()); local_total += float(local_loss.detach())
        if (index+1) % 100 == 0 or index+1 == len(loader):
            progress = {'phase': phase, 'epoch': epoch, 'samples_done': index+1,
                        'samples_total': len(loader), 'mean_loss': total/(index+1),
                        'elapsed_seconds': time.perf_counter()-started,
                        'job_id': os.environ.get('SLURM_JOB_ID')}
            if progress_path is not None: _json_write(progress_path, progress)
            print(json.dumps({'training_progress': progress}), flush=True)
    return {'loss': total/len(loader), 'global_loss': global_total/len(loader), 'local_loss': local_total/len(loader)}


def train_run(model, root, manifest, output, *, phase='global', epochs=None, lr=None,
              accumulation=8, positive_weight=100., lambda_local=1., max_samples=None,
              decoder='subpixel', resume_payload=None, seed=SEED, cache_gib=8., evaluate_every=5):
    from .checkpoints import save_checkpoint
    epochs_default, lr_default, patience = PHASE_DEFAULTS[phase]
    epochs = epochs_default if epochs is None else epochs
    lr = lr_default if lr is None else lr
    if epochs < 1 or accumulation < 1 or evaluate_every < 1:
        raise ValueError('epochs, accumulation and evaluation interval must be positive')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.set_phase(phase)
    optimizer = make_optimizer(model, lr=lr)
    start_epoch = 1
    best, best_epoch = None, 0
    if resume_payload is not None:
        if resume_payload['phase'] != phase: raise ValueError('resume phase mismatch; use --warm-start for phase transitions')
        validate_checkpoint_manifest(resume_payload, manifest)
        if resume_payload['optimizer'] is None:
            raise ValueError('resume requires a training checkpoint with optimizer state')
        optimizer.load_state_dict(resume_payload['optimizer'])
        start_epoch = resume_payload['epoch'] + 1
        best = resume_payload.get('extra', {}).get('best_metric')
        best_epoch = resume_payload.get('extra', {}).get('best_epoch', 0)
        if best is not None:
            source_best = Path(resume_payload['_checkpoint_path']).parent/'best.pt'
            target_best = output/'best.pt'
            if not source_best.exists():
                raise ValueError('resume requires companion best.pt to preserve checkpoint selection')
            from .checkpoints import load_checkpoint
            # A strict independent reload validates every tensor without changing
            # the resumed model parameters or consuming its RNG stream.
            with torch.random.fork_rng(devices=[]):
                best_model, best_payload = load_checkpoint(source_best)
            validate_checkpoint_manifest(best_payload, manifest)
            if best_model.config != model.config:
                raise ValueError('resume companion best.pt config differs from resumed model')
            if best_payload['phase'] != phase:
                raise ValueError('resume companion best.pt phase differs from resumed phase')
            if best_payload['epoch'] != best_epoch:
                raise ValueError('resume companion best.pt epoch differs from recorded best epoch')
            if best_payload['metric'] != best:
                raise ValueError('resume companion best.pt metric differs from recorded best metric')
            if best_payload.get('training', {}) != resume_payload.get('training', {}):
                raise ValueError('resume companion best.pt training settings differ from resumed checkpoint')
            if source_best.resolve() != target_best.resolve():
                shutil.copy2(source_best, target_best)
            del best_model, best_payload
    train_data = _dataset(root, manifest, 'train', model.config, max_samples)
    val_data = _dataset(root, manifest, 'val', model.config, max_samples)
    cache = FeatureCache(int(cache_gib*1024**3)) if phase == 'local' else None
    training = {'seed': seed, 'epochs': epochs, 'lr': lr, 'accumulation': accumulation,
                'weight_decay': 1e-4, 'gradient_clip': 5., 'positive_weight': positive_weight,
                'lambda_local': lambda_local, 'decoder': decoder, 'evaluate_every': evaluate_every,
                'early_stopping_epochs': patience, 'max_samples': max_samples, 'cache_gib': cache_gib}
    _json_write(output/'config.json', {'model': model.config, 'training': training,
                                     'manifest': str(DEFAULT_MANIFEST), 'dataset_root': str(root)})
    history_path = output/'history.jsonl'
    for epoch in range(start_epoch, epochs+1):
        check_deadline()
        began = time.perf_counter()
        losses = train_epoch(model, train_data, optimizer, phase=phase, device=device,
                             accumulation=accumulation, positive_weight=positive_weight,
                             lambda_local=lambda_local, cache=cache,
                             progress_path=output/'active_epoch.json', epoch=epoch)
        row = {'epoch': epoch, 'phase': phase, **losses, 'seconds': time.perf_counter()-began}
        if (epoch == start_epoch or epoch % evaluate_every == 0 or epoch == epochs
                or (best is not None and epoch-best_epoch >= patience)):
            metrics = evaluate_model(model, val_data, device=device, decoder=decoder)
            row['validation'] = metrics
            chosen = metrics['selected']
            if metric_improved(chosen, best):
                best, best_epoch = chosen, epoch
                save_checkpoint(output/'best.pt', model, optimizer, epoch=epoch, phase=phase,
                                keypoint_names=manifest['keypoint_names'],
                                split_ids={s:manifest[s] for s in ('train','val','test')},
                                metric=chosen, training=training,
                                extra={'best_metric': best, 'best_epoch': best_epoch})
        else:
            chosen = best
        save_checkpoint(output/'last.pt', model, optimizer, epoch=epoch, phase=phase,
                        keypoint_names=manifest['keypoint_names'],
                        split_ids={s:manifest[s] for s in ('train','val','test')},
                        metric=chosen, training=training, extra={'best_metric': best, 'best_epoch': best_epoch})
        if cache: row['cache'] = {'bytes': cache.bytes, 'hits': cache.hits, 'misses': cache.misses}
        with history_path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        if best is not None and epoch-best_epoch >= patience: break
    if best is None: raise RuntimeError('No valid validation metric; no best checkpoint can be selected')
    _json_write(output/'completed.json', {'best_checkpoint': str(output/'best.pt'),
                                        'best_epoch': best_epoch, 'best_metric': best,
                                        'phase': phase, 'epochs_requested': epochs})
    return output/'best.pt'


def _config_from_args(args):
    from .config import make_config
    overrides = json.loads(Path(args.config).read_text()) if getattr(args, 'config', None) else {}
    if getattr(args, 'scales', None):
        count = args.scales
        overrides['out_indices'] = [3,4] if count == 2 else [1,2,3,4]
        overrides['fusion_channels'] = [32]*count
    local = overrides.setdefault('local', {})
    for flag, key in [('local','enabled'), ('embedding','embedding')]:
        value = getattr(args, flag, None)
        if value is not None: local[key] = value == 'on'
    if getattr(args,'patches',None): local['layout'] = args.patches
    if getattr(args,'backend',None):
        overrides.setdefault('global_ssm', {})['backend'] = args.backend
        local['backend'] = args.backend
    return make_config(overrides)


def build_parser():
    parser = argparse.ArgumentParser(description='RGB-D keypoints, 5000-only experiment; numerical commands require Slurm')
    sub = parser.add_subparsers(dest='command', required=True)
    def common(command):
        command.add_argument('--data-root', type=Path, default=DATA_ROOT)
        command.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
        command.add_argument('--seed', type=int, default=SEED)
        command.add_argument('--device', choices=['cpu','cuda'], default='cuda')
        command.add_argument('--max-samples', type=int, help='smoke subset per split, drawn only from persisted 5000')
        command.add_argument('--decoder', choices=['argmax','subpixel'], default='subpixel')
    def architecture(command):
        command.add_argument('--config', type=Path)
        command.add_argument('--scales', type=int, choices=[2,4])
        command.add_argument('--local', choices=['on','off'])
        command.add_argument('--patches', choices=['one','five'])
        command.add_argument('--embedding', choices=['on','off'])
        command.add_argument('--backend', choices=['torch','mamba'])
    prepare = sub.add_parser('prepare'); common(prepare)
    prepare.add_argument('--limit', type=int, default=5000)
    validate = sub.add_parser('validate'); common(validate)
    validate.add_argument('--output', type=Path, default=PROJECT/'runs'/'dataset_validation.json')
    train = sub.add_parser('train'); common(train); architecture(train)
    train.add_argument('--phase', choices=list(PHASE_DEFAULTS), default='global')
    train.add_argument('--epochs', type=int); train.add_argument('--lr', type=float)
    train.add_argument('--accumulation', type=int, default=8)
    train.add_argument('--positive-weight', type=float, default=100.)
    train.add_argument('--lambda-local', type=float, default=1.)
    train.add_argument('--cache-gib', type=float, default=8.)
    train.add_argument('--evaluate-every', type=int, default=5)
    train.add_argument('--output', type=Path, default=PROJECT/'runs'/'A4')
    initialize = train.add_mutually_exclusive_group()
    initialize.add_argument('--resume', type=Path); initialize.add_argument('--warm-start', type=Path)
    for name in ('evaluate','infer'):
        command = sub.add_parser(name); common(command)
        command.add_argument('--checkpoint', required=True, type=Path)
        command.add_argument('--split', choices=['train','val','test'], default='val')
        command.add_argument('--output', type=Path, default=PROJECT/'runs'/name)
        if name == 'infer': command.add_argument('--frame-id')
    visualize = sub.add_parser('visualize')
    visualize.add_argument('--artifact', required=True, type=Path)
    visualize.add_argument('--output', required=True, type=Path)
    benchmark = sub.add_parser('benchmark'); common(benchmark); architecture(benchmark)
    benchmark.add_argument('--images', type=int, default=1)
    benchmark.add_argument('--output', type=Path, default=PROJECT/'runs'/'benchmark.json')
    ablate = sub.add_parser('ablate'); common(ablate); architecture(ablate)
    ablate.add_argument('--output', type=Path, default=PROJECT/'runs'/'ablation5000')
    ablate.add_argument('--global-epochs', type=int, default=200)
    ablate.add_argument('--local-epochs', type=int, default=50)
    ablate.add_argument('--joint-epochs', type=int, default=50)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    require_compute_node()
    if getattr(args,'max_samples',None) is not None and args.max_samples < 1:
        raise ValueError('--max-samples must be positive')
    if args.command == 'visualize':
        from .outputs import visualize_artifact
        visualize_artifact(args.artifact, args.output); return
    seed_everything(args.seed)
    from .data import create_subset_manifest, load_manifest
    if args.command == 'prepare':
        manifest = create_subset_manifest(args.data_root, args.manifest, limit=args.limit, seed=args.seed)
        print(json.dumps({'manifest':str(args.manifest), 'counts':{s:len(manifest[s]) for s in ('train','val','test')}})); return
    manifest = load_manifest(args.manifest, root=args.data_root)
    if args.command == 'validate':
        from .validation import validate_subset
        report = validate_subset(args.data_root, manifest)
        _json_write(args.output, report); print(json.dumps(report)); return
    if args.command in ('train','benchmark','ablate'):
        from .model import RGBDKeypointModel
        config = _config_from_args(args)
    if args.command == 'train':
        from .checkpoints import load_checkpoint
        resume = None
        if args.resume:
            model, resume = load_checkpoint(args.resume, restore_rng=True)
        else:
            model = RGBDKeypointModel(config)
            if args.warm_start:
                source_model, source = load_checkpoint(args.warm_start)
                validate_checkpoint_manifest(source, manifest)
                allow_local = not source_model.local_enabled and model.local_enabled
                load_checkpoint(args.warm_start, model=model, allow_local_missing=allow_local)
                del source_model
        model.to(args.device)
        train_run(model, args.data_root, manifest, args.output, phase=args.phase,
                  epochs=args.epochs, lr=args.lr, accumulation=args.accumulation,
                  positive_weight=args.positive_weight, lambda_local=args.lambda_local,
                  max_samples=args.max_samples, decoder=args.decoder, resume_payload=resume,
                  seed=args.seed, cache_gib=args.cache_gib, evaluate_every=args.evaluate_every)
    elif args.command in ('evaluate','infer'):
        from .checkpoints import load_checkpoint
        model, payload = load_checkpoint(args.checkpoint)
        validate_checkpoint_manifest(payload, manifest)
        model.to(args.device)
        data = _dataset(args.data_root, manifest, args.split, model.config, args.max_samples)
        if getattr(args, 'frame_id', None):
            if args.frame_id not in manifest[args.split]: raise ValueError('frame ID is outside selected split')
            from .data import RGBDKeypointDataset
            data = RGBDKeypointDataset(args.data_root, [args.frame_id], manifest['keypoint_names'],
                                      input_size=model.config['input_size'], heatmap_size=model.config['heatmap_size'])
        metrics = evaluate_model(model, data, device=args.device, decoder=args.decoder, artifact_dir=args.output/'artifacts')
        _json_write(args.output/'metrics.json', metrics); print(json.dumps(metrics, allow_nan=False))
    elif args.command == 'benchmark':
        from .benchmark import benchmark_model
        model = RGBDKeypointModel(config).to(args.device)
        data = _dataset(args.data_root, manifest, 'val', config, args.images)
        report = benchmark_model(model, data, device=args.device)
        _json_write(args.output, report); print(json.dumps(report, allow_nan=False))
    elif args.command == 'ablate':
        from .ablation import run_ablation
        run_ablation(args, manifest, config)


if __name__ == '__main__':
    try:
        sys.exit(main() or 0)
    except ComputeBudgetExpired as exc:
        print(json.dumps({'status': 'allocation_budget_reached', 'resume_required': True,
                          'job_id': os.environ.get('SLURM_JOB_ID'), 'reason': str(exc)}), flush=True)
        sys.exit(75)
