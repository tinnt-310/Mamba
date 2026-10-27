"""Slurm CPU controller for bounded, independent Mamba RGB-D experiments."""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .campaign_budget import Budget,Reservation,TERMINAL
from .acceptance import acceptable

PROJECT = Path(__file__).resolve().parents[2]
CAPS = dict(profile=12,pilot=72,confirm=144,runtime=36,final=24)


def read(path):
    return json.loads(Path(path).read_text())


def dump(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def base_config():
    return dict(global_ssm=dict(backend='mamba'),local=dict(enabled=False,backend='mamba'))


def task(identifier,config,*,hours=6,phase='pilot',branch=None,**kwargs):
    return dict(id=identifier,config=config,hours=hours,budget_phase=phase,branch=branch,
                max_updates=30000,train_limit=8000,seed=20261025,batch=8,workers=4,
                effective_batch=8,**kwargs)


def pilot_tasks(root,profile):
    base=base_config(); augmented=deepcopy(base)
    pre=deepcopy(base); pre['pretrained_path']=str(root/'pretrained/seresnet50_a1_0-ffa00869.pth')
    rgb=deepcopy(base); rgb['modality']='rgb'
    quality=deepcopy(base); quality['fusion_mode']='quality'
    high=deepcopy(base); high.update(input_size=[360,640],heatmap_size=[180,320])
    detail=deepcopy(base); detail['detail_decoder']=True
    skipped=deepcopy(base)
    components=profile['components_mean_ms']
    index=max(range(4),key=lambda i:components[f'fusion_{i}'])
    skipped['skip_fusion_scales']=[index]
    tasks=[task('P_base',base,branch='A'),task('P_aug',augmented,branch='A',augment=True),
           task('P_pretrained',pre,branch='A',augment=True),
           task('P_rgb',rgb,hours=9,branch='C'),task('P_quality',quality,hours=9,branch='C'),
           task('P_highres',high,branch='D'),task('P_detail_decoder',detail,branch='D'),
           task('P_skip_fusion',skipped,branch='D')]
    for name,source,conditioning,weight in [('f1','f1',False,0),('detail','detail',False,0),
            ('condition','detail',True,0),('coordinate01','detail',True,.1),('coordinate1','detail',True,1.)]:
        config=deepcopy(base)
        config['local'].update(enabled=True,mode='residual',layout='one',embedding=True,
                               feature_source=source,conditioning=conditioning)
        t=task('P_local_'+name,config,hours=3,branch='B',parent='P_base',
               coordinate_weight=weight,requires=['P_base'])
        t.update(phase='local',max_updates=10000)
        tasks.append(t)
    return tasks


def update_accounting(budget):
    for record in budget.records:
        if record.terminal or not record.job_id:
            continue
        text=subprocess.check_output(['sacct','-j',record.job_id,'-n','-P',
             '--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode'],text=True)
        for line in text.splitlines():
            fields=line.split('|')
            if fields[0]==record.job_id:
                state=fields[1].split()[0].split('+')[0]
                # Accounting may lag; retain full reservation until a terminal row exists.
                record.state=state
                record.elapsed=max(record.elapsed,int(fields[2] or 0))
                break


def validation(root,t):
    path=root/'trials'/t['id']/'last.pt'
    # Lightweight controller reads worker-emitted JSON, never tensors.
    lines=root/'trials'/t['id']/'history.jsonl'
    if not lines.exists(): return None
    values=[json.loads(line) for line in lines.read_text().splitlines() if line.strip()]
    if not values: return None
    return min((r['validation'] for r in values),key=lambda m:(m['final']['balanced_mean'],m['final']['balanced_p95']))


def finished(root,t,budget):
    if any(r.trial==t['id'] and not r.terminal for r in budget.records): return False
    if (root/'trials'/t['id']/'completed.json').exists(): return True
    records=[r for r in budget.records if r.trial==t['id']]
    if not records: return False
    if any(r.state=='FAILED' for r in records):
        approved=set(read(root/'approved_retry_jobs.json')) if (root/'approved_retry_jobs.json').exists() else set()
        if any(r.state=='FAILED' and r.job_id not in approved for r in records):
            return True  # Fail closed except explicitly audited infrastructure fixes.
    return budget.remaining_trial(t['id'],int(t['hours']*3600))<300 or len(records)>=8


def succeeded(root,t,budget):
    """Terminal scheduling status is distinct from a successful trial artifact."""
    if not finished(root,t,budget): return False
    records=[r for r in budget.records if r.trial==t['id']]
    if records and records[-1].state!='COMPLETED': return False
    path=root/'trials'/t['id']/'completed.json'
    return path.exists() and read(path).get('status') in ('complete','early_stopped')


def submit(root,t,budget,state):
    if any(r.trial==t['id'] and not r.terminal for r in budget.records): return False
    seconds=budget.remaining_trial(t['id'],int(t['hours']*3600))
    seconds=min(12*3600,(seconds//60)*60)
    seconds=min(seconds,((budget.total_seconds-sum(r.charge for r in budget.records))//60)*60)
    if t['budget_phase'] in budget.phase_seconds:
        seconds=min(seconds,((budget.phase_seconds[t['budget_phase']]-sum(r.charge for r in budget.records
                     if r.phase==t['budget_phase']))//60)*60)
    # Branch caps include reservations of every trial in that branch.
    branch=t.get('branch')
    if t['budget_phase']=='pilot' and branch:
        ids={x['id'] for x in state['tasks'] if x.get('branch')==branch and x['budget_phase']=='pilot'}
        remaining=18*3600-sum(r.charge for r in budget.records if r.trial in ids)
        seconds=min(seconds,(remaining//60)*60)
    if seconds<300:
        # Active reservations may release unused time; don't end a trial until
        # those reservations have actually been reconciled by Slurm accounting.
        if not any(not r.terminal for r in budget.records):
            dump(root/'trials'/t['id']/'completed.json',dict(status='budget_exhausted'))
        return False
    reservation=budget.reserve(t['id'],t['budget_phase'],seconds,int(t['hours']*3600))
    if reservation is None: return False
    path=root/'tasks'/f'{t["id"]}.json'
    if path.exists() and read(path)!=t: raise ValueError('Immutable task changed')
    dump(path,t)
    # Persist reservation before contacting Slurm; a crash cannot double-spend it.
    state['reservations']=budget.serialize(); dump(root/'state.json',state)
    command=['sbatch','--parsable','--job-name=mb-'+t['id'][:11],
             '--export=NONE',
             '--time='+str(seconds//60),
             '--output='+str(root/'logs'/f'{t["id"]}-%j.out'),
             '--error='+str(root/'logs'/f'{t["id"]}-%j.err'),
             str(root/'source/scripts/slurm/improve_worker.slurm'),str(root),str(path),str(seconds)]
    try:
        reservation.job_id=subprocess.check_output(command,text=True).strip().split(';')[0]
        reservation.state='PENDING'
    except subprocess.CalledProcessError:
        reservation.state='FAILED'; reservation.elapsed=0
        raise
    state['reservations']=budget.serialize(); dump(root/'state.json',state)
    print(json.dumps(dict(submitted=reservation.job_id,trial=t['id'],reserved_seconds=seconds)),flush=True)
    return True


def controller(root,baseline_job):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Controller requires CPU compute allocation')
    root=root.resolve(); (root/'logs').mkdir(exist_ok=True)
    lock=(root/'controller.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(root/'state.json') if (root/'state.json').exists() else dict(
        phase='preflight',tasks=[],reservations=[dict(trial='historical_profile',phase='profile',
                  seconds=1800,job_id=baseline_job,state='PENDING',elapsed=0)],
        total_gpu_hours=288,max_concurrency=4)
    if (root/'manual_gpu_jobs.json').exists():
        known={r['job_id'] for r in state['reservations']}
        state['reservations'].extend(r for r in read(root/'manual_gpu_jobs.json') if r['job_id'] not in known)
    budget=Budget(state['reservations'],phase_hours=CAPS)
    if any(not r.job_id and not r.terminal for r in budget.records):
        raise RuntimeError('Unreconciled submission reservation; inspect Slurm before resuming controller')
    while True:
        update_accounting(budget)
        if state['phase']=='preflight':
            if (root/'manifest50k.json').exists() and (root/'preflight_passed.json').exists() and (root/'gpu_preflight_passed.json').exists() and (root/'baseline/A4_NEW/profile.json').exists() and (root/'baseline/training_profile.json').exists():
                if not (root/'source').exists(): raise RuntimeError('Immutable source snapshot missing')
                if not (root/'pretrained/provenance.json').exists(): raise RuntimeError('Pretrained hash/provenance not verified')
                state['tasks']=pilot_tasks(root,read(root/'baseline/A1_NEW/profile.json'))
                configurations=read(root/'baseline/training_profile.json')['configurations']
                selected=max(configurations,key=lambda r:r['images_per_second'])
                for t in state['tasks']:
                    t.update(batch=selected['batch'],workers=selected['workers'])
                state['training_execution']=selected
                state['phase']='pilot'
        tasks=state['tasks']
        completed={t['id'] for t in tasks if finished(root,t,budget)}
        successful={t['id'] for t in tasks if succeeded(root,t,budget)}
        state['unsuccessful_trials']=sorted(set(state.get('unsuccessful_trials',[]))|(completed-successful))
        # Submit ready trials; all phases share this same four-slot ledger.
        for t in tasks:
            if t['id'] in completed or not set(t.get('requires',[]))<=completed: continue
            if not set(t.get('requires',[]))<=successful:
                dump(root/'trials'/t['id']/'completed.json',dict(status='skipped',reason='Required parent did not succeed'))
                continue
            if any(not (root/'trials'/parent/'best.pt').exists() for parent in t.get('requires',[])):
                dump(root/'trials'/t['id']/'completed.json',dict(status='skipped',reason='Required parent checkpoint missing'))
                continue
            submit(root,t,budget,state)
        if tasks and all(t['id'] in completed for t in tasks):
            if state['phase']=='pilot':
                base=next(t for t in tasks if t['id']=='P_base')
                bm=validation(root,base) if base['id'] in successful else None
                if bm is None: raise RuntimeError('Baseline pilot failed; no candidate can be selected')
                local_values=[(t,validation(root,t)) for t in tasks if t.get('phase')=='local'
                              and t['id'] in successful and acceptable(validation(root,t),bm,True)]
                if local_values and not state.get('joint_attempted'):
                    local_best=min(local_values,key=lambda p:p[1]['final']['balanced_mean'])[0]
                    joint=deepcopy(local_best)
                    joint.update(id='P_local_joint',phase='joint',parent=local_best['id'],
                                 requires=[local_best['id']],hours=3)
                    state['tasks'].append(joint); state['joint_attempted']=True
                    state['reservations']=budget.serialize(); dump(root/'state.json',state)
                    continue
                scores=[(t,validation(root,t) if t['id'] in successful else None) for t in tasks]
                accepted=[(t,m) for t,m in scores if t['id']!='P_base' and acceptable(m,bm,t.get('phase')=='local')]
                winner=min(accepted,key=lambda pair:(pair[1]['final']['balanced_mean'],pair[1]['final']['balanced_p95']))[0] if accepted else base
                dump(root/'pilot_selection.json',dict(baseline=base['id'],winner=winner['id'],
                     results={t['id']:m for t,m in scores},selection='Synthetic validation only; per-source PCK/P95 guarded'))
                # Composition is rebuilt from scratch and checked against its parent recipe.
                recipe=[(t,m) for t,m in accepted if t.get('branch')=='A']
                arch=[(t,m) for t,m in accepted if t.get('branch')=='D']
                fusion=[(t,m) for t,m in accepted if t.get('branch')=='C']
                selected=[]
                for label,values in [('recipe',recipe),('architecture',arch),('fusion',fusion)]:
                    if not values: continue
                    choice=min(values,key=lambda p:p[1]['final']['balanced_mean'])[0]
                    selected.append(dict(label=label,choice=choice))
                state['pilot_tasks']=deepcopy(tasks)
                state['pilot_winner']=winner
                state['composition_queue']=selected
                state['composition_parent']=base
                state['composition_tasks']=[]
                state['tasks']=[]
                state['phase']='next_composition'
            elif state['phase']=='composition':
                t=tasks[0]
                if t['id'] in successful and acceptable(validation(root,t),validation(root,state['composition_parent']),t.get('phase')=='local'):
                    state['composition_parent']=t
                state['composition_tasks'].append(t)
                state['phase']='next_composition'
            elif state['phase']=='confirm':
                dump(root/'final_lock.json',dict(tasks=tasks,selection=state['selection'],
                     manifest_sha256=sha(root/'manifest50k.json'),source_sha256=read(root/'source_manifest.json')['sha256'],
                     policy='Frozen before synthetic test/D455; no real-driven retuning'))
                benchmark=[]
                for t in tasks:
                    if t['id'] in successful and (root/'trials'/t['id']/'best.pt').exists():
                        benchmark.append(dict(id='E_'+t['id'],kind='evaluate_final',checkpoint_trial=t['id'],
                              hours=4,budget_phase='final',config=t['config']))
                if not benchmark:
                    raise RuntimeError('No final checkpoint; cannot evaluate or report an achieved improvement')
                if 'F_candidate_20261025' in successful:
                    benchmark.append(dict(id='R_sweep',kind='runtime',hours=18,budget_phase='runtime',
                                          checkpoint_trial='F_candidate_20261025',config=state['selection']['candidate']['config']))
                state['tasks']=benchmark; state['phase']='evaluation'
            elif state['phase']=='evaluation':
                from .campaign_report import report
                errors=state['unsuccessful_trials']
                state['failed_tasks']=errors
                report(root,budget.serialize(),task_errors=errors)
                state['phase']='completed_with_errors' if errors else 'complete'
        if state['phase']=='next_composition':
            queue=state['composition_queue']
            if queue:
                item=queue.pop(0); chosen=item['choice']; label=item['label']
                parent=state['composition_parent']; config=deepcopy(parent['config'])
                augment=parent.get('augment',False)
                keys={'recipe':['pretrained_path'],'architecture':['input_size','heatmap_size','detail_decoder','skip_fusion_scales'],
                      'fusion':['modality','fusion_mode']}[label]
                for key in keys:
                    if key in chosen['config']: config[key]=chosen['config'][key]
                if label=='recipe': augment=chosen.get('augment',False)
                t=task('M_'+label,config,hours=3,phase='runtime',augment=augment)
                state['tasks']=[t]; state['phase']='composition'
            else:
                locals_=[t for t in state['pilot_tasks'] if t.get('phase')=='local' and
                         acceptable(validation(root,t),validation(root,next(x for x in state['pilot_tasks'] if x['id']=='P_base')),True)]
                if locals_ and not state.get('composed_local_attempted'):
                    chosen=min(locals_,key=lambda t:validation(root,t)['final']['balanced_mean'])
                    parent=state['composition_parent']; config=deepcopy(parent['config'])
                    config['local']=deepcopy(chosen['config']['local'])
                    t=task('M_local',config,hours=3,phase='runtime',parent=parent['id'],
                           augment=False,coordinate_weight=chosen.get('coordinate_weight',0.))
                    t.update(phase='local',max_updates=10000)
                    state['tasks']=[t]; state['phase']='composition'; state['composed_local_attempted']=True
                else:
                    state['phase']='select_final'
        if state['phase']=='select_final':
            base=next(t for t in state['pilot_tasks'] if t['id']=='P_base')
            winner=state['pilot_winner']; bm=validation(root,base)
            candidates=[t for t in state.get('composition_tasks',[]) if succeeded(root,t,budget) and acceptable(validation(root,t),bm)]
            if candidates:
                all_candidates=candidates+[winner]
                winner=min(all_candidates,key=lambda t:validation(root,t)['final']['balanced_mean'])
            # Local is trained only against a matched global parent, including full runs.
            state['selection']=dict(baseline=base,candidate=winner)
            full=[]
            for label,chosen in [('baseline',base),('candidate',winner)]:
                for seed in (20261025,20261026,20261027):
                    t=deepcopy(chosen)
                    t.update(id=f'F_{label}_{seed}',budget_phase='confirm',hours=24,train_limit=None,
                             max_updates=100000,seed=seed)
                    t.pop('branch',None); t.pop('requires',None)
                    if t.get('phase') in ('local','joint'):
                        # Full candidate learns its own global recipe first, then
                        # freezes that exact parent for residual training, within
                        # the same 24-hour / 100k-update trial allocation.
                        t.update(phase='global',local_after_updates=75000)
                        t.pop('parent',None)
                    full.append(t)
            state['selection']['final_tasks']=deepcopy(full)
            state['tasks']=full; state['phase']='confirm'
        state['reservations']=budget.serialize(); state['budget']=budget.summary()
        dump(root/'state.json',state)
        dump(root/'status.json',dict(phase=state['phase'],budget=budget.summary(),
             trials=[dict(id=t['id'],finished=finished(root,t,budget)) for t in state['tasks']]))
        if state['phase'] in ('complete','completed_with_errors'): return
        # Detect a stalled campaign instead of silently spinning after exhausted admission.
        if not any(not r.terminal for r in budget.records) and tasks and not all(t['id'] in completed for t in tasks):
            if all(budget.remaining_trial(t['id'],int(t['hours']*3600))<300 for t in tasks if t['id'] not in completed):
                raise RuntimeError('No remaining trial budget; inspect unresolved trial results')
        time.sleep(30)


def snapshot(root):
    if (root/'source').exists(): raise ValueError('Snapshot already exists; refusing overwrite')
    source=root/'source'; source.mkdir()
    for folder in ('models','scripts','tests','references'):
        shutil.copytree(PROJECT/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml',source/'pyproject.toml')
    files={str(path.relative_to(source)):sha(path) for path in sorted(source.rglob('*')) if path.is_file()}
    combined=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    dump(root/'source_manifest.json',dict(sha256=combined,files=files))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['snapshot','controller'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline-job',default='5705')
    args=parser.parse_args()
    if args.command=='snapshot': snapshot(args.root)
    else: controller(args.root,args.baseline_job)
