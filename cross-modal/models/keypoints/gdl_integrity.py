"""CPU-compute integrity seals for completed trials and resumable allocations."""
from __future__ import annotations
import json
import math
from pathlib import Path
from .campaign import read,dump,sha

_SHAPES={}


def expected_shapes(config):
    from copy import deepcopy
    from .model import RGBDKeypointModel
    key=json.dumps(canonical(config),sort_keys=True)
    if key not in _SHAPES:
        construction=deepcopy(config);construction['pretrained']=False;construction['pretrained_path']=None
        model=RGBDKeypointModel(construction)
        _SHAPES[key]={name:tuple(value.shape) for name,value in model.state_dict().items()}
    return _SHAPES[key]


def canonical(value):
    return json.loads(json.dumps(value))


def verify_task_source(task):
    source=Path(task['source_directory']);manifest=source.parent/'source_manifest.json'
    if sha(manifest)!=task['source_manifest_sha256']:raise ValueError('Task source manifest changed')
    for name,digest in read(manifest)['files'].items():
        if sha(source/name)!=digest:raise ValueError('Task source content changed: '+name)


def audit_checkpoint(root,task,path,*,horizon=False,expected_metric=None):
    import torch
    from .config import make_config
    torch.set_num_threads(1)
    value=torch.load(path,map_location='cpu',weights_only=False)
    training=task['training'];extra=value.get('extra',{})
    origin=None
    if canonical(value.get('training'))!=canonical(training):
        # Successive training intentionally copies immutable best checkpoints.
        # Accept only an exact ancestor file, never a merely similar recipe.
        ancestor=training.get('resume_trial')
        visited=set()
        while ancestor and ancestor not in visited and Path(path).name!='last.pt':
            visited.add(ancestor)
            ancestor_task=read(root/'architecture_gdl/tasks'/ancestor/'task.json')['training']
            if canonical(value.get('training'))==canonical(ancestor_task):
                original=root/'trials'/ancestor/Path(path).name
                if sha(original)!=sha(path):raise ValueError('Inherited checkpoint differs from immutable ancestor')
                origin=ancestor;break
            ancestor=ancestor_task.get('resume_trial')
        if origin is None:raise ValueError('Checkpoint training recipe mismatch')
    if canonical(make_config(value['config']))!=canonical(make_config(training['config'])):
        raise ValueError('Checkpoint architecture mismatch')
    if extra.get('manifest_sha256')!=sha(root/'manifest50k.json'):raise ValueError('Checkpoint manifest mismatch')
    if not value.get('model_state') or not value.get('optimizer') or not value.get('rng'):
        raise ValueError('Incomplete model/optimizer/RNG checkpoint')
    shapes=expected_shapes(make_config(training['config']))
    if set(value['model_state'])!=set(shapes) or any(tuple(t.shape)!=shapes[k] for k,t in value['model_state'].items()):
        raise ValueError('Checkpoint tensor keys/shapes differ from declared architecture')
    if not all(key in extra for key in ('updates','epoch','batch_cursor')):raise ValueError('Sampler/progress missing')
    updates=extra['updates']
    if not isinstance(updates,int) or updates<0 or updates>training['max_updates']:raise ValueError('Invalid progress')
    if horizon and updates!=training['max_updates']:raise ValueError('Checkpoint has not reached matched horizon')
    if extra['epoch']<0 or extra['batch_cursor']<0:raise ValueError('Invalid sampler cursor')
    rng=value['rng']
    if not all(name in rng and rng[name] is not None for name in ('torch','numpy','python','cuda')):
        raise ValueError('Full RNG state missing')
    def finite(item):
        if isinstance(item,torch.Tensor) and item.is_floating_point() and not torch.isfinite(item).all():
            raise ValueError('Nonfinite model/optimizer tensor')
        if isinstance(item,dict):
            for child in item.values():finite(child)
        elif isinstance(item,(list,tuple)):
            for child in item:finite(child)
    finite(value['model_state']);finite(value['optimizer'])
    if expected_metric is not None and canonical(value['metric'])!=canonical(expected_metric):
        raise ValueError('Checkpoint metric does not match selected validation artifact')
    return dict(sha256=sha(path),updates=updates,epoch=extra['epoch'],batch_cursor=extra['batch_cursor'],origin_trial=origin,
        scheduler=dict(kind='cosine_from_update' if training.get('cosine_learning_rate') else 'fixed',
                       horizon=training['max_updates'],learning_rate=training.get('learning_rate',3e-4)),
        model_optimizer_rng_sampler_finite=True)


def audit_training(root,task,*,resume=False):
    verify_task_source(task)
    trial=root/'trials'/task['id'];folder=root/'architecture_gdl/tasks'/task['id']
    specification=folder/'task.json'
    persisted={k:v for k,v in task.items() if k!='resume_verified_updates'}
    if read(specification)!=persisted:raise ValueError('Immutable task differs from registry')
    files=[trial/'last.pt']
    complete=trial/'completed.json'
    if not resume:
        value=read(complete)
        if value.get('status')!='complete' or value.get('updates')!=task['training']['max_updates']:
            raise ValueError('Incomplete matched training')
        files.extend([complete,trial/'history.jsonl',trial/'best.pt'])
        if value.get('best_admissible') is not None:files.append(trial/'best_admissible.pt')
    fingerprints={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files}
    seal_path=folder/('resume_integrity.json' if resume else 'artifact_integrity.json')
    if seal_path.exists():
        cached=read(seal_path)
        if cached.get('task_sha256')==sha(specification) and cached.get('fingerprints')==fingerprints:
            return cached
    # CPU controller executes this only after Slurm reports the worker terminal.
    last=audit_checkpoint(root,task,trial/'last.pt',horizon=not resume)
    checkpoints={'last':last}
    if not resume:
        checkpoints['best']=audit_checkpoint(root,task,trial/'best.pt',expected_metric=value['best'])
        if value.get('best_admissible') is not None:
            checkpoints['best_admissible']=audit_checkpoint(root,task,trial/'best_admissible.pt',expected_metric=value['best_admissible'])
        history=[json.loads(line) for line in (trial/'history.jsonl').read_text().splitlines() if line.strip()]
        if not history or history[-1]['updates']!=task['training']['max_updates']:
            raise ValueError('Final full validation has not been recorded')
        for name in ('OLD','NEW'):
            metrics=history[-1]['validation']['final'][name]
            if not all(k in metrics for k in ('mean','median','p95','PCK05')):raise ValueError('Validation source metrics missing')
            if not all(math.isfinite(metrics[k]) and metrics[k]>=0 for k in ('mean','median','p95','PCK05')) or metrics['PCK05']>1:
                raise ValueError('Invalid validation metrics')
        if task['training'].get('reference_metrics') and value.get('best_admissible') is not None:
            from .acceptance import acceptable
            if not acceptable(value['best_admissible'],read(task['training']['reference_metrics']),
                              local=task['training'].get('selection_local',False)):
                raise ValueError('Selected checkpoint does not pass locked gates')
    result=dict(passed=True,task_sha256=sha(specification),source_manifest_sha256=task['source_manifest_sha256'],
        manifest_sha256=sha(root/'manifest50k.json'),checkpoints=checkpoints,fingerprints=fingerprints,
        file_hashes={str(p):sha(p) for p in files},resume=resume)
    dump(seal_path,result);return result
