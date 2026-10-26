"""Controlled A0–A4 experiment and validation-only adaptive region selection."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import torch


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read_lock(path):
    try:
        with path.open(encoding='utf-8') as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError('provenance JSON must contain an object')
        return value
    except (OSError, ValueError) as exc:
        raise ValueError(f'campaign provenance lock is invalid: {path}; use a new output directory') from exc


def _immutable_json(path, value):
    value = _canonical(value)
    if path.exists():
        if _read_lock(path) != value:
            raise ValueError(f'campaign provenance differs from {path}; use a new output directory')
        return value
    try:
        with path.open('x', encoding='utf-8') as destination:
            destination.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    except FileExistsError:
        if _read_lock(path) != value:
            raise ValueError(f'campaign provenance differs from {path}; use a new output directory')
    return value


def _local_artifacts(output):
    return any((output / run / phase).exists()
               for run in ('A2', 'A3', 'A4') for phase in ('local', 'joint'))


def lock_campaign_protocol(args, manifest, base_config):
    """Bind an output to one exact protocol before model construction/training.

    Locks are canonical JSON so list/tuple configs compare equally, while IDs
    and semantic keypoint order remain exact. Legacy artifacts cannot supply
    sufficient provenance and must be rerun in a new output directory.
    """
    from .config import make_config
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = str(Path(args.data_root).resolve())
    if root != str(Path(manifest['dataset_root']).resolve()):
        raise ValueError('campaign dataset root differs from manifest; use a new output directory')
    persisted_manifest = deepcopy(manifest)
    persisted_manifest['dataset_root'] = root
    protocol = _canonical({
        'protocol_version': 1, 'base_config': make_config(base_config),
        'seed': args.seed, 'decoder': args.decoder,
        'epochs': {'global': args.global_epochs, 'local': args.local_epochs, 'joint': args.joint_epochs},
        'max_samples': args.max_samples, 'data_root': root, 'manifest': persisted_manifest,
    })
    path = output / 'campaign_protocol.json'
    if not path.exists() and any(output.iterdir()):
        raise ValueError('existing campaign output has no immutable protocol; use a new output directory')
    return _immutable_json(path, protocol)


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
    except OSError as exc:
        raise ValueError('campaign A1 parent checkpoint is missing or unreadable; use a new output directory') from exc
    return digest.hexdigest()


def lock_local_provenance(output, a1_checkpoint, seed, *, region_selection=None):
    """Lock and verify A1 bytes, common local seed, and validation ROI record."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'local_provenance.json'
    fingerprint = _checkpoint_sha256(a1_checkpoint)
    if path.exists():
        existing = _read_lock(path)
        if (existing.get('provenance_version') != 1 or
                existing.get('a1_checkpoint_sha256') != fingerprint or
                existing.get('local_init_seed') != seed):
            raise ValueError('campaign local parent fingerprint or seed changed; use a new output directory')
        if region_selection is not None and existing.get('region_selection') != _canonical(region_selection):
            raise ValueError('campaign locked region configuration changed; use a new output directory')
        result = existing
    else:
        if _local_artifacts(output):
            raise ValueError('existing local campaign artifacts have no parent provenance; use a new output directory')
        if region_selection is None:
            raise ValueError('validation region selection is required to establish local provenance')
        result = {'provenance_version': 1, 'a1_checkpoint_sha256': fingerprint,
                  'local_init_seed': seed, 'region_selection': _canonical(region_selection)}
    region = result.get('region_selection')
    if (not isinstance(region, dict) or region.get('split') != 'val' or
            not isinstance(region.get('selected'), dict) or
            not {'level', 'margin_ratio'}.issubset(region['selected'])):
        raise ValueError('campaign local provenance has no valid locked region selection; use a new output directory')
    from .config import make_config
    try:
        make_config({'local': {key: region['selected'][key] for key in ('level', 'margin_ratio')}})
    except ValueError as exc:
        raise ValueError('campaign locked region settings are invalid; use a new output directory') from exc
    roi_path = output / 'region_selection.json'
    if roi_path.exists() and _read_lock(roi_path) != result['region_selection']:
        raise ValueError('campaign region artifact differs from its locked selection; use a new output directory')
    return _immutable_json(path, result)


def ablation_configs(base, *, level=.30, margin=.25):
    from .config import make_config
    base = make_config(base)
    result = {}
    for run in ('A0','A1','A2','A3','A4'):
        config = deepcopy(base)
        config['local'].update(enabled=run in ('A2','A3','A4'), layout='one' if run=='A2' else 'five',
                               embedding=run=='A4', level=level, margin_ratio=margin)
        if run=='A0':
            config['out_indices'] = tuple(base['out_indices'][-2:])
            config['fusion_channels'] = tuple(base['fusion_channels'][-2:])
        result[run] = make_config(config)
    return result


def initialize_local(model, seed):
    """Construct common local weights independently of the optional embedding."""
    options = {k:v for k,v in model.config['local'].items() if k!='enabled'}
    options['embedding'] = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        common = type(model.local_refinement)(channels=model.config['fusion_channels'][0],
                                             num_keypoints=model.config['num_keypoints'], **options)
        common_state = common.state_dict()
        current = model.local_refinement.state_dict()
        for key, value in current.items():
            if key in common_state:
                current[key] = common_state[key].to(value.device)
            else:
                # A separate deterministic seed prevents embedding from shifting SSM init.
                digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
                generator = torch.Generator(device=value.device).manual_seed(seed+digest)
                current[key] = torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype) * .02
        model.local_refinement.load_state_dict(current, strict=True)


@torch.no_grad()
def select_region_config(model, dataset, *, device, recall_target=.95):
    from torch.utils.data import DataLoader
    from .heatmaps import transform_points
    from .regions import extract_regions
    model.eval()
    cached = []
    for batch in DataLoader(dataset,batch_size=1,shuffle=False):
        fused = model.encode_fused(*(batch[key].to(device) for key in ('rgb','depth','K')))
        maps = model.global_from_fused(fused).detach().cpu()
        feature_hw = fused[0].shape[-2:]
        points = transform_points(batch['points'], model.config['input_size'], feature_hw)
        cached.append((maps, feature_hw, points, batch['weights']>0))
    def measure(level, margin):
        hits = valid = 0; area_sum = count = 0
        for maps, hw, points, validity in cached:
            records = extract_regions(maps, hw, level=level, margin_ratio=margin, layout='five', overlap_ratio=.5)
            for record in records:
                area_sum += record.feature_box.width*record.feature_box.height; count += 1
                bi,ki = record.batch_index,record.keypoint_index
                if not validity[bi,ki]: continue
                valid += 1
                x,y = points[bi,ki].tolist()
                hits += any(box.x1-.5<=x<box.x2-.5 and box.y1-.5<=y<box.y2-.5 for box in record.patch_boxes)
        return {'level':level,'margin_ratio':margin,'roi_recall':hits/valid if valid else 0.,
                'valid_keypoints':valid,'mean_area':area_sum/count if count else 0.}
    default = measure(.30,.25)
    candidates = [default]
    if default['roi_recall'] < recall_target:
        candidates.extend(measure(level,margin) for level in (.15,.30,.50) for margin in (.25,.50,1.)
                          if (level,margin)!=(.30,.25))
    passing = [row for row in candidates if row['roi_recall']>=recall_target]
    selected = min(passing,key=lambda row:(row['mean_area'],-row['roi_recall'])) if passing else min(candidates,key=lambda row:(-row['roi_recall'],row['mean_area']))
    return {'selected':selected,'candidates':candidates,'target_recall':recall_target,
            'recall_target_met':bool(passing),'split':'val'}


def run_ablation(args, manifest, base_config):
    from .experiment import train_run,evaluate_model,_dataset,_json_write,seed_everything
    from .model import RGBDKeypointModel
    from .checkpoints import load_checkpoint
    from .benchmark import benchmark_model
    output = Path(args.output)
    protocol = lock_campaign_protocol(args, manifest, base_config)
    lineage = None
    if (output / 'local_provenance.json').exists():
        # Check even completed local campaigns before any global phase can run.
        lineage = lock_local_provenance(output, output / 'A1' / 'global' / 'best.pt', args.seed)
    elif _local_artifacts(output):
        raise ValueError('existing local campaign artifacts have no parent provenance; use a new output directory')
    def phase_run(model, phase_output, phase, epochs):
        phase_output = Path(phase_output)
        resumed = None
        if (phase_output/'last.pt').exists():
            model, resumed = load_checkpoint(phase_output/'last.pt', model=model, restore_rng=True)
            if resumed['split_ids'] != {s:manifest[s] for s in ('train','val','test')}:
                raise ValueError('campaign resume split mismatch')
            if resumed['keypoint_names'] != manifest['keypoint_names']:
                raise ValueError('campaign resume semantic keypoint names mismatch')
            if resumed.get('training',{}).get('max_samples') != args.max_samples:
                raise ValueError('campaign resume smoke/full sample scope mismatch')
            training = resumed.get('training', {})
            if (resumed.get('phase') != phase or training.get('seed') != args.seed or
                    training.get('decoder') != args.decoder or training.get('epochs') != epochs):
                raise ValueError('campaign phase checkpoint differs from locked training protocol; use a new output directory')
            if (phase_output/'completed.json').exists():
                best = phase_output / 'best.pt'
                if not best.exists():
                    raise ValueError('completed campaign phase has no best checkpoint; use a new output directory')
                _, best_payload = load_checkpoint(best, model=model)
                if (best_payload.get('phase') != phase or best_payload.get('training') != training or
                        best_payload.get('split_ids') != resumed['split_ids'] or
                        best_payload.get('keypoint_names') != resumed['keypoint_names']):
                    raise ValueError('campaign best checkpoint differs from completed phase provenance; use a new output directory')
                return best
        return train_run(model,args.data_root,manifest,phase_output,phase=phase,epochs=epochs,
                         max_samples=args.max_samples,decoder=args.decoder,seed=args.seed,resume_payload=resumed)
    configs = ablation_configs(base_config)
    report = {'sample_counts':{s:len(manifest[s]) for s in ('train','val','test')},
              'seed':args.seed,'max_samples':args.max_samples,
              'campaign_protocol_sha256':hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
              'protocol':'A0/A1 from scratch; A2-A4 shared A1, shared local init, locked validation region; test after lock',
              'runs':{}}
    for run in ('A0','A1'):
        seed_everything(args.seed)
        model = RGBDKeypointModel(configs[run]).to(args.device)
        best = phase_run(model,output/run/'global','global',args.global_epochs)
        report['runs'][run]={'checkpoint':str(best)}
        _json_write(output/'progress.json',report)
        del model
        if args.device=='cuda': torch.cuda.empty_cache()
    parent = report['runs']['A1']['checkpoint']
    if lineage is None:
        a1,_ = load_checkpoint(parent); a1.to(args.device)
        roi = select_region_config(a1,_dataset(args.data_root,manifest,'val',a1.config,args.max_samples),device=args.device)
        del a1
        lineage = lock_local_provenance(output, parent, args.seed, region_selection=roi)
    else:
        lineage = lock_local_provenance(output, parent, args.seed)
        roi = lineage['region_selection']
    _json_write(output/'region_selection.json',roi)
    selected = roi['selected']
    configs = ablation_configs(base_config,level=selected['level'],margin=selected['margin_ratio'])
    report['region_selection']=roi
    report['a1_checkpoint_sha256'] = lineage['a1_checkpoint_sha256']
    report['local_init_seed'] = lineage['local_init_seed']
    # Validation inference artifacts can be produced now; test remains untouched.
    for run in ('A2','A3','A4'):
        seed_everything(args.seed)
        model = RGBDKeypointModel(configs[run])
        initialize_local(model,args.seed)
        # Recheck the parent immediately before each local stage as well.
        lock_local_provenance(output, parent, args.seed)
        load_checkpoint(parent,model=model,allow_local_missing=True)
        model.to(args.device)
        local_best = phase_run(model,output/run/'local','local',args.local_epochs)
        model,_ = load_checkpoint(local_best); model.to(args.device)
        joint_best = phase_run(model,output/run/'joint','joint',args.joint_epochs)
        report['runs'][run]={'checkpoint':str(joint_best),'local_checkpoint':str(local_best),
                            'a1_checkpoint_sha256':lineage['a1_checkpoint_sha256'],
                            'local_init_seed':lineage['local_init_seed'],
                            'region':{'level':selected['level'], 'margin_ratio':selected['margin_ratio'],
                                      'overlap_ratio':configs[run]['local']['overlap_ratio']}}
        _json_write(output/'progress.json',report)
        del model
        if args.device=='cuda': torch.cuda.empty_cache()
    report['configuration_locked']=True
    _json_write(output/'progress.json',report)
    for run in ('A0','A1','A2','A3','A4'):
        model,payload = load_checkpoint(report['runs'][run]['checkpoint']); model.to(args.device)
        test_data = _dataset(args.data_root,manifest,'test',model.config,args.max_samples)
        metrics = evaluate_model(model,test_data,device=args.device,decoder=args.decoder,
                                 artifact_dir=output/run/'test_artifacts')
        benchmark = benchmark_model(model,_dataset(args.data_root,manifest,'val',model.config,1),device=args.device)
        report['runs'][run].update(metrics=metrics,benchmark=benchmark)
        _json_write(output/run/'test_metrics.json',metrics)
        _json_write(output/'report.json',report)
        del model
        if args.device=='cuda': torch.cuda.empty_cache()
    for method in ('argmax','subpixel'):
        baseline = report['runs']['A1']['metrics']['global_'+method]['mean']
        for run in ('A2','A3','A4'):
            refined = report['runs'][run]['metrics']['local_'+method]['mean']
            report['runs'][run].setdefault('improvement_vs_A1',{})[method] = baseline-refined
    report['research_conclusion'] = 'Holdout improvement is assessed separately for each decoder; smoke results do not establish generalization.'
    _json_write(output/'report.json',report)
    print(str(output/'report.json'),flush=True)
