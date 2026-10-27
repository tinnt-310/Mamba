"""Versioned checkpoints with strict reload and restricted local warm-start."""
from __future__ import annotations
import json
from pathlib import Path
import random
import numpy as np
import torch


def _canonical(value):
    return json.loads(json.dumps(value))


def _global_config(config):
    return {k: v for k, v in _canonical(config).items() if k != 'local'}


def save_checkpoint(path, model, optimizer, *, epoch, phase, keypoint_names,
                    split_ids, metric, training=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'config_version': (4 if (model.config.get('global_ssm',{}).get('topology','legacy')!='legacy' or
             model.config.get('decoder',{}).get('kind','legacy')!='legacy' or
             model.config.get('local',{}).get('core')=='hybrid') else (3 if (model.config.get('modality','rgbd') != 'rgbd' or
             model.config.get('fusion_mode','legacy') != 'legacy' or
             model.config.get('detail_decoder',False) or model.config.get('skip_fusion_scales') or
             model.config.get('decoder',{}).get('merge','mean') != 'mean' or
             model.config.get('decoder',{}).get('activation','gelu') != 'gelu' or
             model.config.get('decoder',{}).get('depth',2) != 2 or
             model.config.get('decoder',{}).get('residual',False) or
             model.config.get('local',{}).get('mode','legacy') != 'legacy') else 2)), 'config': model.config,
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'epoch': int(epoch), 'phase': phase,
        'keypoint_names': list(keypoint_names), 'split_ids': split_ids,
        'metric': metric, 'training': training or {}, 'extra': extra or {},
        'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
                'python': random.getstate(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, model=None, allow_local_missing=False,
                    map_location='cpu', restore_rng=False):
    # Checkpoints are trusted local experiment artifacts; metadata includes RNG.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or 'config' not in payload:
        raise ValueError('checkpoint must contain a model config')
    if payload.get('config_version', 1) not in (1, 2, 3, 4):
        raise ValueError('unsupported checkpoint config_version')
    payload['_checkpoint_path'] = str(Path(path).resolve())
    from .config import make_config
    from .model import RGBDKeypointModel
    raw_config = payload['config']
    config = make_config(raw_config)
    if 'local' not in raw_config:
        config['local']['enabled'] = False
    if model is None:
        from copy import deepcopy
        construction = deepcopy(config)
        construction['pretrained'] = False
        construction['pretrained_path'] = None
        model = RGBDKeypointModel(construction)
        model.config = config
    elif allow_local_missing:
        if _global_config(model.config) != _global_config(config):
            raise ValueError('global config mismatch during local warm-start')
    elif _canonical(model.config) != _canonical(config):
        raise ValueError('checkpoint config differs from target model config')
    state = payload.get('model_state', payload.get('model_state_dict', payload.get('state_dict')))
    if not isinstance(state, dict):
        raise ValueError('checkpoint missing model_state')
    if allow_local_missing:
        if any(key.startswith('local_refinement.') for key in state):
            raise ValueError('local warm-start requires a global-only checkpoint')
        expected = model.state_dict()
        missing = set(expected) - set(state)
        unexpected = set(state) - set(expected)
        illegal = {key for key in missing if not key.startswith('local_refinement.')}
        if illegal or unexpected:
            raise RuntimeError(f'global checkpoint mismatch: missing={sorted(illegal)}, unexpected={sorted(unexpected)}')
        # Fill only explicitly permitted local keys and then load strictly.
        complete = {**state, **{key: expected[key] for key in missing}}
        model.load_state_dict(complete, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    if not allow_local_missing:
        model.set_phase(payload.get('phase', 'global'))
    if restore_rng and 'rng' in payload:
        rng = payload['rng']
        torch.set_rng_state(rng['torch'].cpu())
        np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])
        if rng.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
    return model, payload
