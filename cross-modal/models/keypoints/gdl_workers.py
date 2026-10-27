"""Compute-node workers: native correctness, wall-time calibration and locked evaluation."""
from __future__ import annotations
from copy import deepcopy
import gc
import json
from pathlib import Path
import time
import torch
from .campaign import read,dump,sha
from .config import make_config
from .model import RGBDKeypointModel
from .multisource import MultiSourceDataset,load_manifest
from .trial_runner import loader,transfer,optimizer,final_maps,evaluate,checkpoint
from .local_losses import weighted_heatmap_loss
from .checkpoints import load_checkpoint,save_checkpoint
from .gdl_search import base_config


def native_scan_parity():
    from models.fusion.ms2fusion.cross_parametric import CrossParametricSSM
    cpu=CrossParametricSSM(4,2,1,4,'torch')
    gpu=CrossParametricSSM(4,2,1,4,'mamba').cuda();gpu.load_state_dict(cpu.state_dict())
    r=torch.randn(2,4,5,7,requires_grad=True);d=torch.randn_like(r,requires_grad=True)
    rg=r.detach().cuda().requires_grad_(True);dg=d.detach().cuda().requires_grad_(True)
    left=cpu(r,d);right=gpu(rg,dg)
    for key in left:torch.testing.assert_close(left[key],right[key].cpu(),atol=2e-5,rtol=2e-4)
    sum(v.square().mean() for v in left.values()).backward()
    sum(v.square().mean() for v in right.values()).backward()
    torch.testing.assert_close(r.grad,rg.grad.cpu(),atol=2e-5,rtol=2e-3)
    torch.testing.assert_close(d.grad,dg.grad.cpu(),atol=2e-5,rtol=2e-3)
    return dict(passed=True,shape=[2,4,5,7],four_directions=True,forward_atol=2e-5,backward_atol=2e-5)


def gradient_norm(module):
    values=[p.grad.detach().float() for p in module.parameters() if p.grad is not None]
    if any(not torch.isfinite(v).all() for v in values):raise ValueError('Nonfinite branch gradients')
    return float(sum(v.square().sum() for v in values).sqrt()) if values else 0.


def profile(root,task,out):
    torch.set_num_threads(1)
    torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025)
    parity=native_scan_parity()
    manifest=load_manifest(root/'manifest50k.json')
    specs=task.get('configs') or [base_config(root,g) for g in ('g0','g1','g2')]
    if not task.get('configs'):
        specs += [base_config(root,'g0',d) for d in ('d1','d2')]
        specs += [base_config(root,'g0','d0',core) for core in ('conv','ssm','hybrid')]
    results={}
    for config in specs:
        g=config['global_ssm']['topology'];d=config['decoder']['kind']
        local=config['local']['core'] if config['local']['enabled'] else None
        key=f'{g}_{d}_{local or "none"}'
        cfg=make_config(config);model=RGBDKeypointModel(cfg).cuda()
        phase='local' if local else 'global';model.set_phase(phase).train();opt=optimizer(model,phase)
        train_data=MultiSourceDataset(manifest,'train',cfg,augment=not local)
        batches=iter(loader(train_data,8,4,True))
        times=[];branch_norms={};torch.cuda.reset_peak_memory_stats()
        for step in range(30):
            torch.cuda.synchronize();start=time.perf_counter();batch=transfer(next(batches))
            opt.zero_grad(set_to_none=True)
            stage=model.forward_stages(*(batch[k] for k in ('rgb','depth','K')))
            loss=weighted_heatmap_loss(final_maps(stage),batch['target'],batch['weights'])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            if step in (0,1,2,9,29):
                norms={name:gradient_norm(module) for name,module in model.named_modules()
                       if name.endswith(('cnn','ssm','rgb_detail','depth_detail','cnn_output'))}
                branch_norms[str(step)]=norms
            opt.step();torch.cuda.synchronize()
            if step>=5:times.append(time.perf_counter()-start)
        peak=torch.cuda.max_memory_allocated()
        if local=='hybrid':
            if not all(any(v>0 for n,v in branch_norms['29'].items() if n.endswith(name)) for name in ('.cnn','.ssm')):
                raise ValueError('Hybrid Local branch gradient missing after optimizer steps')
        if d=='d2' and not local:
            if not all(any(v>0 for n,v in branch_norms['29'].items() if n.endswith(name)) for name in ('rgb_detail','depth_detail')):
                raise ValueError('D2 shallow modality gradient missing')
        model.eval()
        with torch.no_grad():
            args=tuple(batch[k][:1] for k in ('rgb','depth','K'))
            invalid=[final_maps(model.forward_stages(args[0],torch.full_like(args[1],v),args[2]))
                     for v in (0.,float('nan'),float('inf'))]
            if not all(torch.isfinite(v).all() for v in invalid):raise ValueError('Invalid depth prediction')
            if not all(torch.equal(invalid[0],v) for v in invalid[1:]):raise ValueError('Invalid encoding parity')
        torch.cuda.synchronize();start=time.perf_counter()
        metrics=evaluate(model,MultiSourceDataset(manifest,'val',cfg),8,4)
        torch.cuda.synchronize();validation_seconds=time.perf_counter()-start
        start=time.perf_counter()
        save_checkpoint(out/(key+'.pt'),model,opt,epoch=0,phase=phase,keypoint_names=manifest['keypoint_names'],
            split_ids={s:[r['uid'] for r in manifest[s]] for s in ('train','val','test')},metric=metrics,
            extra=dict(profile=True,updates=30,manifest_sha256=sha(root/'manifest50k.json')))
        checkpoint_seconds=time.perf_counter()-start
        results[key]=dict(wall_seconds_per_update=sum(times)/len(times),full_validation_seconds=validation_seconds,
            checkpoint_seconds=checkpoint_seconds,peak_vram_bytes=peak,gradient_steps=branch_norms,
            invalid_encoding_parity=True,correctness=True,temporary_steps=30,
            interpretation='Wall time includes real train input IO; full 5K validation and checkpoint measured separately')
        dump(out/'profile.json',dict(variants=results,native_scan_parity=parity,gpu=torch.cuda.get_device_name()))
        del model,opt,batches,batch,stage,loss,invalid;gc.collect();torch.cuda.empty_cache()


def subset(root,config):
    import hashlib
    manifest=load_manifest(root/'manifest50k.json');records=[]
    for source in ('OLD','NEW'):
        pool=[r for r in manifest['val'] if r['source']==source]
        records+=sorted(pool,key=lambda r:hashlib.sha256(('modality-v1:'+r['uid']).encode()).hexdigest())[:256]
    return MultiSourceDataset(manifest,'val',config,records=records,diagnostics=True)


def evidence(root,task,out):
    from .modality_diagnostics import run
    parent=task['parent'];checkpoint_path=Path(parent['checkpoint'])
    run(root,checkpoint_path,out/'modality.json')
    model,_=load_checkpoint(checkpoint_path);model.cuda().eval();data=subset(root,model.config)
    clean=evaluate(model,data,8,4,records_path=out/'clean_predictions.json')
    gate_summary=None
    if model.config['local']['core']=='hybrid':
        gate=model.residual_local.mamba.last_gate
        gate_summary=dict(mean=float(gate.mean()),fraction_below_005=float((gate<.05).float().mean()),
                          fraction_above_095=float((gate>.95).float().mean()))
    rows={};checks=[]
    # No optimizer changes to the trained checkpoint: interventions on inference only.
    if model.config['decoder']['kind']=='d2':
        for name in ('rgb','depth'):
            setattr(model.decoder,'mask_'+name,True)
            rows['detail_'+name]=evaluate(model,data,8,4,records_path=out/('detail_'+name+'_predictions.json'))
            setattr(model.decoder,'mask_'+name,False)
    if model.config['local']['core']=='hybrid':
        for name in ('cnn','ssm'):
            model.residual_local.mamba.mask_branch=name
            rows['local_'+name]=evaluate(model,data,8,4,records_path=out/('local_'+name+'_predictions.json'))
        model.residual_local.mamba.mask_branch=None
    calibration=load_manifest(root/'manifest50k.json')
    batch=transfer(next(iter(loader(MultiSourceDataset(calibration,'train',model.config),8,0))))
    model.requires_grad_(True);model.zero_grad(set_to_none=True)
    loss=weighted_heatmap_loss(final_maps(model.forward_stages(*(batch[k] for k in ('rgb','depth','K')))),batch['target'],batch['weights'])
    loss.backward()
    norms={name:gradient_norm(module) for name,module in model.named_modules()
           if name.endswith(('rgb_detail','depth_detail','mamba.cnn','mamba.ssm'))}
    if model.config['decoder']['kind']=='d2':
        checks += [norms.get('decoder.'+name+'_detail',0)>0 for name in ('rgb','depth')]
    if model.config['local']['core']=='hybrid':
        checks += [norms.get('residual_local.mamba.'+name,0)>0 for name in ('cnn','ssm')]
    from .regression import paired_rows,summarize
    clean_rows=read(out/'clean_predictions.json')['records'];comparisons={}
    for condition in rows:
        changed=read(out/(condition+'_predictions.json'))['records'];pairs=paired_rows(clean_rows,changed)
        comparisons[condition]={s:summarize([p for p in pairs if p[0]['source']==s]) for s in ('OLD','NEW')}
        # Branch utility is accuracy evidence; numerical sensitivity alone is insufficient.
        checks += [comparisons[condition][s]['delta']['mean']>0 for s in ('OLD','NEW')]
    dump(out/'branches.json',dict(passed=all(checks),gradient_norms=norms,gate_summary_last_clean_batch=gate_summary,clean=clean,
        interventions=rows,paired_comparisons=comparisons,checkpoint_sha256=sha(checkpoint_path)))
    # Full validation paired predictions preserve keypoint and ROI evidence, not just the subset.
    model.zero_grad(set_to_none=True)
    evaluate(model,MultiSourceDataset(calibration,'val',model.config,diagnostics=True),8,4,
             records_path=out/'full_validation_predictions.json')
    gd=read(root/'architecture_gdl/state.json')['gd_parents'][str(parent['seed'])]
    del model;gc.collect();torch.cuda.empty_cache()
    global_model,_=load_checkpoint(gd['checkpoint']);global_model.cuda().eval()
    evaluate(global_model,MultiSourceDataset(calibration,'val',global_model.config,diagnostics=True),8,4,
             records_path=out/'parent_validation_predictions.json')


def intervals(root,task,out):
    from .modality_diagnostics import add_intervals
    from .regression import paired_rows,summarize,bootstrap,geometry_metrics
    folder=root/'architecture_gdl/tasks'/task['evidence_task'];add_intervals(folder/'modality.json')
    branches=read(folder/'branches.json')
    clean=read(folder/'clean_predictions.json')['records']
    for name in branches['paired_comparisons']:
        changed=read(folder/(name+'_predictions.json'))['records'];pairs=paired_rows(clean,changed)
        for source in ('OLD','NEW'):
            branches['paired_comparisons'][name][source].update(bootstrap([p for p in pairs if p[0]['source']==source]))
    dump(folder/'branches.json',branches)
    parent=read(folder/'parent_validation_predictions.json')['records']
    final=read(folder/'full_validation_predictions.json')['records'];pairs=paired_rows(parent,final)
    comparisons={s:{**summarize([p for p in pairs if p[0]['source']==s]),**bootstrap([p for p in pairs if p[0]['source']==s])} for s in ('OLD','NEW')}
    mm={s:geometry_metrics([r for r in final if r['source']==s]) for s in ('OLD','NEW')}
    per_keypoint={s:[dict(keypoint=i,mean=sum(r['errors'][i] for r in final if r['source']==s and r['valid'][i])/
        max(1,sum(1 for r in final if r['source']==s and r['valid'][i]))) for i in range(9)] for s in ('OLD','NEW')}
    dump(out/'paired_summary.json',dict(comparisons=comparisons,synthetic_lateral_mm=mm,keypoints=per_keypoint,
        limitation='Validation selected checkpoints; paired render groups do not prove physical scene independence'))


def clean_benchmark(root,checkpoint_path,out,label):
    from .trial_validation import benchmark as measure
    lock=read(root/'architecture_gdl/legacy_reference_lock.json')
    key=str(checkpoint_path.resolve())
    if label!='candidate' and (key not in lock['checkpoints'] or sha(checkpoint_path)!=lock['checkpoints'][key]):
        raise ValueError('Locked legacy checkpoint hash mismatch')
    model,_=load_checkpoint(checkpoint_path);model.cuda().eval()
    manifest=load_manifest(root/'manifest50k.json')
    data=MultiSourceDataset(manifest,'val',model.config,inference_only=True)
    timing=measure(model,data,1,2)
    batch=transfer(next(iter(loader(data,1,0))))
    event_times=[];local_times=[]
    with torch.no_grad():
        args=tuple(batch[k] for k in ('rgb','depth','K'))
        for i in range(550):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();stage=model.forward_stages(*args);b.record();b.synchronize()
            if i>=50:event_times.append(a.elapsed_time(b))
        if model.local_enabled:
            fused,detail=model._encode(*args);maps=model.global_from_fused(fused,detail)
            for i in range(550):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();model.refine_cached(fused[0],maps,detail=detail);b.record();b.synchronize()
                if i>=50:local_times.append(a.elapsed_time(b))
    import numpy as np
    timing.update(model_cuda_p50_ms=float(np.median(event_times)),
        local_incremental_p50_ms=float(np.median(local_times)) if local_times else 0.,
        checkpoint_sha256=sha(checkpoint_path),gpu=torch.cuda.get_device_name())
    dump(out/(label+'.json'),timing)


def benchmark(root,task,out):
    """Each model uses a fresh subprocess; baseline and candidate buffers never coexist."""
    import os,subprocess,sys
    seed=task['parent']['seed']
    paths=dict(candidate=task['parent']['checkpoint'],
        baseline=str(root/'trials'/f'F_baseline_{seed}'/'best.pt'),
        legacy_candidate=str(root/'trials'/f'F_candidate_{seed}'/'best.pt'))
    for label,path in paths.items():
        command=[sys.executable,'-m','models.keypoints.gdl_workers','benchmark-one',str(root),path,str(out),label]
        subprocess.run(command,check=True,env=os.environ.copy())
    import statistics
    values={name:read(out/(name+'.json')) for name in paths}
    c,b=values['candidate'],values['baseline']
    latency=lambda v:statistics.median(r['batch_p50_ms'] for r in v['gpu_ready'])
    memory=lambda v:max(r['allocated_bytes'] for r in v['gpu_ready'])
    dump(out/'benchmark.json',dict(passed=latency(c)<=latency(b) and memory(c)<=memory(b),models=values,
        latency_delta_ms=latency(c)-latency(b),allocated_delta_bytes=memory(c)-memory(b),
        rule='Unchanged strict legacy baseline release envelope; batch-one FP32 same reader; GPU-ready includes decode/copy'))


def verify(root,task,out):
    freeze_path=root/'architecture_gdl/freeze.json';freeze=read(freeze_path)
    if sha(root/'manifest50k.json')!=freeze['manifest_sha256']:raise ValueError('Frozen manifest mismatch')
    torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025)
    model=RGBDKeypointModel(make_config(freeze['config'])).cuda().set_phase('joint').train()
    manifest=load_manifest(root/'manifest50k.json')
    batch=transfer(next(iter(loader(MultiSourceDataset(manifest,'train',model.config),2,0))))
    opt=optimizer(model,'joint')
    def step(net,optim):
        optim.zero_grad(set_to_none=True)
        maps=final_maps(net.forward_stages(*(batch[k] for k in ('rgb','depth','K'))))
        loss=weighted_heatmap_loss(maps,batch['target'],batch['weights']);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);optim.step()
        return float(loss.detach())
    for _ in range(3):step(model,opt)
    path=out/'resume.pt'
    metadata=dict(epoch=0,phase='joint',keypoint_names=manifest['keypoint_names'],
        split_ids={s:[r['uid'] for r in manifest[s]] for s in ('train','val','test')},metric={},
        extra=dict(updates=3,batch_cursor=3,manifest_sha256=freeze['manifest_sha256']))
    save_checkpoint(path,model,opt,**metadata)
    expected_loss=step(model,opt);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    restored,payload=load_checkpoint(path,restore_rng=True);restored.cuda().train().set_phase('joint')
    restored_opt=optimizer(restored,'joint');restored_opt.load_state_dict(payload['optimizer'])
    actual_loss=step(restored,restored_opt);maximum=0.
    for name,value in restored.state_dict().items():
        reference=expected[name].to(value.device)
        if value.is_floating_point():
            maximum=max(maximum,float((value-reference).abs().max()))
            torch.testing.assert_close(value,reference,atol=1e-5,rtol=1e-4)
        elif not torch.equal(value,reference):raise ValueError('Resume buffer mismatch')
    model.eval();restored.eval()
    with torch.no_grad():
        args=tuple(batch[k] for k in ('rgb','depth','K'))
        torch.testing.assert_close(final_maps(model.forward_stages(*args)),final_maps(restored.forward_stages(*args)),atol=1e-5,rtol=1e-4)
    dump(out/'verification.json',dict(passed=True,freeze_sha256=sha(freeze_path),resume_max_difference=maximum,
        uninterrupted_loss=expected_loss,resumed_loss=actual_loss,native_scan_parity=native_scan_parity()))


def final(root,task,out):
    from .trial_validation import real_evaluation
    from .regression import geometry_metrics
    lock=read(root/'architecture_gdl/final_lock.json');entry=lock['checkpoints'][str(task['seed'])]
    if sha(entry['checkpoint'])!=entry['checkpoint_sha256'] or sha(root/'manifest50k.json')!=lock['manifest_sha256']:
        raise ValueError('Final checkpoint or manifest lock mismatch')
    model,_=load_checkpoint(entry['checkpoint']);model.cuda().eval()
    manifest=load_manifest(root/'manifest50k.json')
    test=evaluate(model,MultiSourceDataset(manifest,'test',model.config,diagnostics=True),8,4,records_path=out/'synthetic_predictions.json')
    records=read(out/'synthetic_predictions.json')['records']
    mm={s:geometry_metrics([r for r in records if r['source']==s]) for s in ('OLD','NEW')}
    real=real_evaluation(model,root)
    del model;gc.collect();torch.cuda.empty_cache()
    benchmark(root,task,out);timing=read(out/'benchmark.json')
    baseline=read(root/'trials'/f'E_F_baseline_{task["seed"]}'/'metrics.json')
    legacy=read(root/'trials'/f'E_F_candidate_{task["seed"]}'/'metrics.json')
    reasons=[]
    for source in ('OLD','NEW'):
        c,b=test['final'][source],baseline['synthetic_test']['final'][source]
        if c['p95']>b['p95'] or c['PCK05']<b['PCK05']:reasons.append('synthetic '+source+' P95/PCK')
    c,b=real['pixel'],baseline['D455']['pixel']
    if not(c['mean']<b['mean'] and c['p95']<=b['p95'] and c['PCK05']>=b['PCK05']):reasons.append('D455 mean/P95/PCK')
    if not timing['passed']:reasons.append('legacy baseline runtime/VRAM')
    dump(out/'metrics.json',dict(accepted=not reasons,reasons=reasons,synthetic_test=test,synthetic_lateral_mm=mm,
        D455=real,benchmark=timing,legacy_baseline_metrics=baseline,legacy_candidate_metrics=legacy,
        checkpoint_sha256=entry['checkpoint_sha256'],seed=task['seed']))


if __name__=='__main__':
    import sys,os
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm compute required')
    if sys.argv[1]!='benchmark-one':raise ValueError('Unknown subprocess role')
    clean_benchmark(Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]),sys.argv[5])
