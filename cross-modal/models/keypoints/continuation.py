"""Resume the existing campaign with validation-only diagnostics and a bounded ledger."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .campaign import read,dump,sha,snapshot,update_accounting
from .campaign_budget import Budget

SEEDS=(20261025,20261026,20261027)
PHASE_HOURS={'next_diagnostic':12,'next_train':48,'next_confirm':72,'next_runtime':18,'next_final':12}


def initialize(root):
    out=root/'continuation'; out.mkdir(exist_ok=True)
    if (out/'state.json').exists(): raise ValueError('Continuation already initialized; resume controller')
    old=read(root/'state.json')
    if any(r['state'] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED') for r in old['reservations']):
        raise ValueError('Reconcile historical active reservations first')
    snapshot(out)
    dump(out/'policy.json',dict(max_gpu_hours=162,total_gpu_hours=288,concurrency=4,
        phase_hours=PHASE_HOURS,bootstrap_repeats=2000,bootstrap_seed=20261026,
        priority='CI excludes zero and max(0.05px,5%) mean/P95 regression or 0.1pp PCK regression',
        selection='Synthetic validation only; never read test or real labels before final lock'))
    tasks=[]
    for seed in SEEDS:
        for label,which in [('baseline','best'),('candidate','best'),('candidate','last')]:
            tasks.append(dict(id=f'{label}_{seed}_{which}',kind='export',seed=seed,label=label,which=which,
                              phase='next_diagnostic',seconds=1200))
    dump(out/'state.json',dict(phase='diagnostic',tasks=tasks,reservations=old['reservations'],
                              historical_jobs=[r['job_id'] for r in old['reservations']]))


def outcome(root,t,budget):
    records=[r for r in budget.records if r.trial=='next_'+t['id']]
    if any(not r.terminal for r in records): return 'running'
    folder=root/'continuation'/'tasks'/t['id']
    if (folder/'completed.json').exists(): return read(folder/'completed.json')['status']
    if any(r.state not in ('COMPLETED',) for r in records): return 'failed'
    if records: return 'incomplete'
    return 'pending'


def submit(root,t,budget,state):
    out=root/'continuation'
    spent=sum(r.charge for r in budget.records if r.phase.startswith('next_'))
    if spent+t['seconds']>162*3600: return False
    reservation=budget.reserve('next_'+t['id'],t['phase'],t['seconds'])
    if reservation is None: return False
    path=out/'tasks'/t['id']/'task.json'
    if path.exists() and read(path)!=t: raise ValueError('Task mutation forbidden')
    dump(path,t); state['reservations']=budget.serialize(); dump(out/'state.json',state)
    command=['sbatch','--parsable','--export=NONE','--partition=gpu','--gres=gpu:nvidia_rtx_a5000:1',
             '--cpus-per-task=8','--mem=64G','--time='+str(t['seconds']//60),
             '--job-name=mb-next-'+t['id'][:6],
             '--output='+str(out/'tasks'/t['id']/'%j.out'),
             '--error='+str(out/'tasks'/t['id']/'%j.err'),
             str(out/'source/scripts/slurm/improve_continue.slurm'),str(root),'worker',str(path),str(t['seconds'])]
    try:
        reservation.job_id=subprocess.check_output(command,text=True).strip().split(';')[0]
        reservation.state='PENDING'
    except subprocess.CalledProcessError:
        reservation.state='FAILED'; raise
    finally:
        state['reservations']=budget.serialize(); dump(out/'state.json',state)
    print(json.dumps(dict(submitted=reservation.job_id,task=t['id'])),flush=True)
    return True


def worker(root,path):
    import torch
    from .checkpoints import load_checkpoint
    from .multisource import MultiSourceDataset,load_manifest
    from .trial_runner import evaluate
    t=read(path); out=root/'continuation'; folder=path.parent
    torch.set_num_threads(1)
    if t['kind']=='export':
        checkpoint=root/'trials'/f'F_{t["label"]}_{t["seed"]}'/(t['which']+'.pt')
        manifest=load_manifest(root/'manifest50k.json')
        model,payload=load_checkpoint(checkpoint); model.cuda().eval()
        data=MultiSourceDataset(manifest,'val',model.config,diagnostics=True)
        target=out/'diagnostics'/(t['id']+'.json')
        if not target.exists():
            pending=target.with_suffix('.pending.json')
            metrics=evaluate(model,data,records_path=pending)
            if t['which']=='best':
                old=read(root/'trials'/f'F_{t["label"]}_{t["seed"]}'/'completed.json')['best']
                for source in ('OLD','NEW'):
                    for key in ('mean','p95','PCK05'):
                        if abs(metrics['final'][source][key]-old['final'][source][key])>1e-5:
                            raise ValueError(f'Historical FP32 parity failed: {source} {key}')
            value=read(pending); value.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                manifest_sha256=sha(root/'manifest50k.json'),phase=payload['phase'],updates=payload['extra'].get('updates'))
            dump(target,value); pending.unlink()
        else:
            value=read(target)
            if value.get('checkpoint_sha256')!=sha(checkpoint) or value.get('manifest_sha256')!=sha(root/'manifest50k.json'):
                raise ValueError('Prediction provenance mismatch')
    elif t['kind']=='runtime':
        from .trial_validation import run
        run(dict(id='N_runtime',kind='runtime',checkpoint_trial=f'F_candidate_{t["seed"]}'),root)
        result=root/'trials/N_runtime/completed.json'
        if not result.exists(): return
        dump(folder/'completed.json',read(result)); return
    elif t['kind']=='train':
        from .trial_runner import train
        train(t['training'],root)
        result=root/'trials'/t['training']['id']/'completed.json'
        if not result.exists(): return
        dump(folder/'completed.json',read(result)); return
    else: raise ValueError('Unknown continuation task')
    dump(folder/'completed.json',dict(status='complete'))


def controller(root):
    out=root/'continuation'; lock=(out/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(out/'state.json'); budget=Budget(state['reservations'],phase_hours=PHASE_HOURS)
    if any(not r.job_id and not r.terminal for r in budget.records):
        raise RuntimeError('Unreconciled reservation: inspect Slurm before resume')
    while True:
        update_accounting(budget)
        statuses={t['id']:outcome(root,t,budget) for t in state['tasks']}
        for t in state['tasks']:
            if statuses[t['id']]=='pending': submit(root,t,budget,state)
        statuses={t['id']:outcome(root,t,budget) for t in state['tasks']}
        failed=[k for k,v in statuses.items() if v in ('failed','incomplete','completed_with_errors')]
        done=all(v not in ('pending','running') for v in statuses.values())
        if done and state['phase']=='diagnostic' and not failed:
            from .regression import analyze
            diagnostic=analyze(root)
            state['diagnostic']=dict(training_priority=diagnostic['training_priority'])
            if diagnostic['training_priority']:
                state['phase']='training'; state['tasks']=[]
                for seed in SEEDS:
                    original=read(root/'tasks'/f'F_candidate_{seed}.json')
                    for label,rate in [('control',3e-4),('low_lr',3e-5)]:
                        training={**original,'id':f'N_{label}_{seed}','parent':f'F_candidate_{seed}',
                            'selection_reference':f'F_baseline_{seed}','learning_rate':rate,
                            'max_updates':10000,'cosine_learning_rate':label=='low_lr','min_learning_rate':3e-6}
                        state['tasks'].append(dict(id=training['id'],kind='train',training=training,
                                                 phase='next_train',seconds=7200))
            else:
                state['phase']='runtime'
                state['tasks']=[dict(id='runtime',kind='runtime',seed=20261025,phase='next_runtime',seconds=3600)]
        elif done and state['phase']=='training' and not failed:
            # Admission uses synthetic validation only. Preserve every trial checkpoint.
            from .acceptance import acceptable
            selection={}
            for seed in SEEDS:
                reference=read(root/'trials'/f'F_baseline_{seed}'/'completed.json')['best']
                possible=[]
                for label in ('control','low_lr'):
                    trial=f'N_{label}_{seed}'; record=read(root/'trials'/trial/'completed.json')
                    # Read worker-emitted metadata, never tensor weights on controller.
                    history=root/'trials'/trial/'history.jsonl'
                    admitted=[json.loads(line)['validation'] for line in history.read_text().splitlines()
                              if acceptable(json.loads(line)['validation'],reference)]
                    if admitted:
                        score=min(admitted,key=lambda x:x['final']['balanced_mean'])
                        possible.append((score['final']['balanced_mean'],trial))
                selection[str(seed)]=min(possible)[1] if possible else None
            dump(out/'validation_selection.json',selection)
            state['phase']='runtime'
            state['tasks']=[dict(id='runtime',kind='runtime',seed=20261025,phase='next_runtime',seconds=3600)]
        elif done:
            state['phase']='completed_with_errors' if failed else 'complete'
            state['failed_tasks']=failed
        state['reservations']=budget.serialize(); state['budget']=budget.summary()
        state['continuation_budget']=dict(allocated_gpu_hours=sum(r.elapsed for r in budget.records if r.phase.startswith('next_'))/3600,
            charged_or_reserved_gpu_hours=sum(r.charge for r in budget.records if r.phase.startswith('next_'))/3600,limit_gpu_hours=162)
        dump(out/'state.json',state)
        dump(root/'status.json',dict(phase=state['phase'],budget=state['budget'],continuation_budget=state['continuation_budget'],
            tasks=statuses,historical_status='Training/final evaluation completed; historical runtime job 5934 failed',
            details='continuation/state.json'))
        if state['phase'] in ('complete','completed_with_errors'):
            from .regression import write_report
            write_report(root,state)
            return
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['initialize','controller','worker'])
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--task',type=Path)
    args=parser.parse_args()
    if args.command=='initialize': initialize(args.root)
    else:
        if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm allocation required')
        if args.command=='controller': controller(args.root)
        else: worker(args.root,args.task)
