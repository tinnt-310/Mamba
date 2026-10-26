"""Single-GPU, resumable trials over the locked synthetic manifest."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint, save_checkpoint
from .config import make_config
from .heatmaps import decode_heatmaps
from .local_losses import weighted_heatmap_loss
from .model import RGBDKeypointModel
from .multisource import MultiSourceDataset, digest, dump, load_manifest, read
from .acceptance import acceptable


def loader(dataset,batch=8,workers=4,shuffle=False,seed=20261025):
    options = dict(batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,
                   generator=torch.Generator().manual_seed(seed))
    if workers:
        options.update(prefetch_factor=2,persistent_workers=False)
    return DataLoader(dataset,**options)


def transfer(batch):
    return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def final_maps(stage):
    return stage.final_heatmaps if stage.final_heatmaps is not None else stage.global_heatmaps


def coordinate_loss(maps,points,input_hw,weights,radius=4):
    """Differentiable local expectation; detached window avoids distant peaks."""
    b,k,h,w = maps.shape
    peaks = maps.detach().flatten(-2).argmax(-1)
    yy,xx = torch.meshgrid(torch.arange(h,device=maps.device),torch.arange(w,device=maps.device),indexing='ij')
    mask = ((xx-peaks.remainder(w)[...,None,None]).abs()<=radius)&(
        (yy-peaks.div(w,rounding_mode='floor')[...,None,None]).abs()<=radius)
    scores = maps.float().clamp_min(1e-6).log().masked_fill(~mask,-torch.inf)
    probability = scores.flatten(-2).softmax(-1).reshape_as(scores)
    xy = torch.stack(((probability*xx).sum((-2,-1)),(probability*yy).sum((-2,-1))),-1)
    target = (points+.5)*points.new_tensor([w/input_hw[1],h/input_hw[0]])-.5
    error = ((xy-target)/xy.new_tensor([w,h])).square().sum(-1)
    return (error*weights).sum()/weights.sum().clamp_min(1)


def metrics(errors,normalized):
    e,n = np.asarray(errors),np.asarray(normalized)
    if not e.size:
        raise ValueError('No valid labels in metric')
    return dict(mean=float(e.mean()),median=float(np.median(e)),p95=float(np.percentile(e,95)),
                PCK05=float((n<=.05).mean()),within_1px=float((e<1).mean()),
                within_2px=float((e<2).mean()),within_5px=float((e<5).mean()),points=len(e))


@torch.no_grad()
def evaluate(model,dataset,batch_size=8,workers=4,records_path=None):
    model.eval()
    buckets = {stage:{source:[[],[]] for source in ('OLD','NEW')} for stage in ('global','final')}
    roi_hits,roi_count = 0,0
    rows = []
    for batch in loader(dataset,batch_size,workers):
        batch = transfer(batch)
        stage = model.forward_stages(*(batch[k] for k in ('rgb','depth','K')))
        for name,maps in (('global',stage.global_heatmaps),('final',final_maps(stage))):
            support = stage.local.support if name=='final' and stage.local is not None else None
            coords = decode_heatmaps(maps.float(),support=support).coords
            size = batch['original_size']
            scale = torch.stack((size[:,1]/maps.shape[-1],size[:,0]/maps.shape[-2]),-1)
            original = (coords+.5)*scale[:,None]-.5
            errors = torch.linalg.vector_norm(original-batch['original_points'],dim=-1).cpu().numpy()
            normalized = errors/batch['bbox_diagonal'].cpu().numpy()[:,None]
            valid = batch['weights'].cpu().numpy()>0
            if records_path is not None and name=='final':
                predictions = original.cpu().numpy()
                for i,uid in enumerate(batch['frame_id']):
                    row = dict(uid=uid,source=batch['source'][i],coords=predictions[i].tolist(),
                               errors=errors[i].tolist(),normalized=normalized[i].tolist(),valid=valid[i].tolist())
                    if 'diagnostics' in batch:
                        row.update(json.loads(batch['diagnostics'][i]))
                    rows.append(row)
            for i,source in enumerate(batch['source']):
                buckets[name][source][0].extend(errors[i][valid[i]].tolist())
                buckets[name][source][1].extend(normalized[i][valid[i]].tolist())
        if stage.residual is not None:
            maps = stage.global_heatmaps
            target = ((batch['points']+.5)*batch['points'].new_tensor(
                [maps.shape[-1]/dataset.input_size[1],maps.shape[-2]/dataset.input_size[0]])-.5).round().long()
            inside = (target[...,0]>=0)&(target[...,0]<maps.shape[-1])&(target[...,1]>=0)&(target[...,1]<maps.shape[-2])
            linear = target[...,1].clamp(0,maps.shape[-2]-1)*maps.shape[-1]+target[...,0].clamp(0,maps.shape[-1]-1)
            hit = stage.residual.support.flatten(-2).gather(-1,linear[...,None]).squeeze(-1)&inside
            valid = batch['weights']>0
            roi_hits += int((hit&valid).sum()); roi_count += int(valid.sum())
    result = {}
    for stage,by_source in buckets.items():
        values = {s:metrics(*v) for s,v in by_source.items() if v[0]}
        values['balanced_mean'] = float(np.mean([m['mean'] for m in values.values()]))
        values['balanced_p95'] = float(np.mean([metrics(*v)['p95'] for v in by_source.values() if v[0]]))
        result[stage] = values
    result['roi_recall'] = roi_hits/roi_count if roi_count else None
    if records_path is not None:
        dump(records_path,dict(metrics=result,records=rows))
    return result


def optimizer(model,phase):
    groups = []
    global_ids = {id(p) for m in model._global_modules() for p in m.parameters()}
    for backbone in (False,True):
        for no_decay in (False,True):
            params = [p for p in model.parameters() if p.requires_grad and (id(p) in global_ids)==backbone
                      and bool(getattr(p,'_no_weight_decay',False))==no_decay]
            if params:
                groups.append(dict(params=params,lr=3e-5 if phase=='joint' and backbone else 3e-4,
                                   weight_decay=0. if no_decay else 1e-4))
    return torch.optim.AdamW(groups)


def checkpoint(path,model,opt,manifest,task,epoch,extra):
    save_checkpoint(path,model,opt,epoch=epoch,phase=extra.get('active_phase',task.get('phase','global')),keypoint_names=manifest['keypoint_names'],
                    split_ids={s:[r['uid'] for r in manifest[s]] for s in ('train','val','test')},
                    metric=extra.get('best'),training=task,
                    extra={**extra,'manifest_sha256':extra['manifest_sha256']})


def train(task,campaign):
    seed = task.get('seed',20261025)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    output = campaign/'trials'/task['id']; output.mkdir(parents=True,exist_ok=True)
    manifest = load_manifest(campaign/'manifest50k.json')
    manifest_hash = digest(campaign/'manifest50k.json')
    config = make_config(task['config'])
    phase = task.get('phase','global')
    last = output/'last.pt'
    state = dict(epoch=0,batch_cursor=0,updates=0,best=None,best_epoch=0,
                 manifest_sha256=manifest_hash,gradient_diagnostics=[])
    if last.exists():
        model,payload = load_checkpoint(last,restore_rng=True)
        if payload['training'] != task or payload['extra']['manifest_sha256'] != manifest_hash:
            raise ValueError('Trial/manifest changed during resume')
        state.update(payload['extra'])
        phase=state.get('active_phase',phase)
    else:
        model = RGBDKeypointModel(config)
        parent = task.get('parent')
        if parent:
            _,payload = load_checkpoint(campaign/'trials'/parent/'best.pt')
            if payload['extra']['manifest_sha256'] != manifest_hash:
                raise ValueError('Parent manifest mismatch')
            incoming = payload['model_state']; expected = model.state_dict()
            missing = set(expected)-set(incoming)
            unexpected = set(incoming)-set(expected)
            if unexpected or any(not k.startswith('residual_local.') for k in missing):
                raise ValueError(f'Unsafe global warm-start: missing={missing}, unexpected={unexpected}')
            model.load_state_dict({**expected,**incoming},strict=True)
    model.cuda(); model.set_phase(phase)
    state['active_phase']=phase
    opt = optimizer(model,phase)
    if 'learning_rate' in task:
        for group in opt.param_groups:
            group['lr'] = task['learning_rate']
    if last.exists():
        opt.load_state_dict(payload['optimizer'])
    records = manifest['train']
    if task.get('train_limit'):
        rng = random.Random(seed)
        records = sum((rng.sample([r for r in records if r['source']==s],task['train_limit']//2)
                       for s in ('OLD','NEW')),[])
    train_data = MultiSourceDataset(manifest,'train',config,augment=task.get('augment',False) and phase!='local',records=records)
    val_data = MultiSourceDataset(manifest,'val',config)
    batch_size,workers = task.get('batch',8),task.get('workers',4)
    effective = task.get('effective_batch',8)
    if effective%batch_size:
        raise ValueError('Effective batch must be divisible by microbatch')
    accumulation = effective//batch_size
    amp = task.get('precision','fp32')=='bf16'
    limit = float(os.environ.get('MAMBA_JOB_SECONDS','3600'))
    started = time.monotonic(); stop_at = started+max(1,limit-180)
    import psutil
    cpu_process=psutil.Process()
    cpu_process.cpu_percent()
    history = output/'history.jsonl'
    reference = read(campaign/'trials'/task['selection_reference']/'completed.json')['best'] if task.get('selection_reference') else None
    if reference is not None and state['best'] is None:
        initial = evaluate(model,val_data,batch_size,workers)
        state['best'],state['best_epoch'] = initial,0
        checkpoint(output/'best.pt',model,opt,manifest,task,0,state)
        if acceptable(initial,reference):
            state['best_admissible'] = initial
            checkpoint(output/'best_admissible.pt',model,opt,manifest,task,0,state)
    telemetry = subprocess.Popen(['nvidia-smi',
                                 '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw',
                                 '--format=csv','-l','5'],stdout=(output/f'gpu-{os.environ["SLURM_JOB_ID"]}.csv').open('w'))
    # Freeze BN only for local; use deterministic epoch sampling for resumability.
    try:
        while state['updates'] < task.get('max_updates',100000):
            epoch = state['epoch']
            train_data.epoch = epoch
            model.train(); model.set_phase(phase)
            batches = loader(train_data,batch_size,workers,True,seed+epoch)
            opt.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            totals,steps,data_wait = 0.,0,0.
            for index,batch in enumerate(batches):
                if index < state['batch_cursor']:
                    continue
                data_wait += time.perf_counter()-step_start
                if task.get('cosine_learning_rate'):
                    fraction = state['updates']/task['max_updates']
                    rate = task['min_learning_rate']+(task['learning_rate']-task['min_learning_rate'])*(.5+.5*np.cos(np.pi*fraction))
                    for group in opt.param_groups: group['lr'] = rate
                batch = transfer(batch)
                context = torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext()
                with context:
                    stage = model.forward_stages(*(batch[k] for k in ('rgb','depth','K')))
                    output_maps = final_maps(stage)
                    heat_loss = weighted_heatmap_loss(output_maps.float(),batch['target'],batch['weights'])
                    weight = task.get('coordinate_weight',0.)
                    loss = heat_loss
                    if weight:
                        loss = loss+weight*coordinate_loss(output_maps,batch['points'],config['input_size'],batch['weights'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                if phase=='joint' and len(state['gradient_diagnostics'])<8:
                    shared = [p for m in model._global_modules() for p in m.parameters() if p.requires_grad]
                    global_loss = weighted_heatmap_loss(stage.global_heatmaps.float(),batch['target'],batch['weights'])
                    left = torch.autograd.grad(global_loss,shared,retain_graph=True,allow_unused=True)
                    right = torch.autograd.grad(loss,shared,retain_graph=True,allow_unused=True)
                    a = torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p,g in zip(shared,left)])
                    b = torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p,g in zip(shared,right)])
                    state['gradient_diagnostics'].append(dict(global_norm=float(a.norm()),final_norm=float(b.norm()),
                        cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0))))
                # A final partial accumulation window gets its actual denominator.
                window = min(accumulation,len(batches)-(index//accumulation)*accumulation)
                (loss/window).backward()
                totals += float(loss.detach()); steps += 1
                if (index+1)%accumulation and index+1<len(batches):
                    step_start = time.perf_counter(); continue
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
                opt.step(); opt.zero_grad(set_to_none=True)
                state['updates'] += 1; state['batch_cursor'] = index+1
                scheduled=task.get('local_after_updates')
                if scheduled and phase=='global' and state['updates']>=scheduled:
                    parent_metrics=evaluate(model,val_data,batch_size,workers)
                    state['local_parent_metrics']=parent_metrics
                    phase='local'; state['active_phase']=phase
                    model.set_phase(phase); opt=optimizer(model,phase)
                    train_data.augment=False
                    # A fresh loader is required after changing augmentation/phase.
                    checkpoint(last,model,opt,manifest,task,epoch,state)
                    break
                if state['updates']%200 == 0:
                    checkpoint(last,model,opt,manifest,task,epoch,state)
                    children=cpu_process.children(recursive=True)
                    dump(output/'progress.json',dict(epoch=epoch,updates=state['updates'],loss=totals/max(steps,1),
                         data_wait_seconds=data_wait,elapsed_seconds=time.monotonic()-started,
                         peak_vram_bytes=torch.cuda.max_memory_allocated(),cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         cpu_percent_parent=cpu_process.cpu_percent(),
                         cpu_seconds_live_workers=sum(sum(p.cpu_times()[:2]) for p in children if p.is_running()),
                         rss_live_workers=sum(p.memory_info().rss for p in children if p.is_running())))
                if time.monotonic()>=stop_at or state['updates']>=task.get('max_updates',100000):
                    break
                step_start = time.perf_counter()
            end_epoch = state['batch_cursor'] >= len(batches)
            should_validate = end_epoch or time.monotonic()>=stop_at or state['updates']>=task.get('max_updates',100000)
            if should_validate:
                result = evaluate(model,val_data,batch_size,workers)
                candidate = result['final']
                score = (candidate['balanced_mean'],candidate['balanced_p95'])
                previous = (state['best']['final']['balanced_mean'],state['best']['final']['balanced_p95']) if state['best'] else (float('inf'),float('inf'))
                if score < previous:
                    state['best'],state['best_epoch'] = result,epoch
                    checkpoint(output/'best.pt',model,opt,manifest,task,epoch,state)
                if reference is not None and acceptable(result,reference):
                    admitted = state.get('best_admissible')
                    if admitted is None or score < (admitted['final']['balanced_mean'],admitted['final']['balanced_p95']):
                        state['best_admissible'] = result
                        checkpoint(output/'best_admissible.pt',model,opt,manifest,task,epoch,state)
                row = dict(epoch=epoch,updates=state['updates'],loss=totals/max(steps,1),validation=result,
                           elapsed_seconds=time.monotonic()-started,data_wait_seconds=data_wait)
                with history.open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps(row,allow_nan=False),flush=True)
            if end_epoch:
                state['epoch'] += 1; state['batch_cursor'] = 0
            checkpoint(last,model,opt,manifest,task,state['epoch'],state)
            if state['updates']>=task.get('max_updates',100000):
                dump(output/'completed.json',dict(status='complete',updates=state['updates'],best=state['best']))
                return
            if state['best'] and state['epoch']-state['best_epoch']>=20:
                dump(output/'completed.json',dict(status='early_stopped',updates=state['updates'],best=state['best']))
                return
            if time.monotonic()>=stop_at:
                dump(output/'slice_complete.json',dict(status='resume_required',updates=state['updates'],best=state['best']))
                return
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=10)


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST'):
        raise RuntimeError('Slurm compute required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=Path,required=True)
    args = parser.parse_args()
    task = read(args.task)
    if task.get('kind','train')=='train':
        train(task,args.campaign)
    else:
        from .trial_validation import run
        run(task,args.campaign)


if __name__=='__main__':
    main()
