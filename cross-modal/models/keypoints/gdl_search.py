"""Slurm-only staged G–D–L search with immutable tasks and validation-only selection."""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import time
import traceback
from .campaign import read,dump,sha,snapshot
from .acceptance import acceptable
from .gdl_recovery import output as slurm_output,SlurmUnavailable,recover_submissions,token,heartbeat

SEEDS=(20261025,20261026)
DEFAULT_MAX_CAMPAIGN_GPUS=4
TERMINAL={'COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL',
          'DEADLINE','REVOKED','SUBMIT_FAILED'}


def campaign_gpu_limit(state):
    value=state.get('max_campaign_gpus',DEFAULT_MAX_CAMPAIGN_GPUS)
    if type(value) is not int or not 1<=value<=6:raise ValueError('Invalid authorized campaign GPU limit')
    return value


def pilot_seeds(state):
    values=state.get('pilot_seeds',[SEEDS[0]])
    if not values or len(set(values))!=len(values) or any(type(v) is not int or v not in SEEDS for v in values):
        raise ValueError('Pilot seeds must be distinct confirmed Global seeds')
    return tuple(sorted(values))


def requested_gpus(queue):
    # Count devices, including multi-GPU jobs outside this campaign and pending allocations.
    return sum(int(match.group(1)) for line in queue.splitlines() for gres in line.split('|')[1:2]
               for match in re.finditer(r'(?:^|,)\s*(?:gres/)?gpu:(?:[^,:()]+:)?(\d+)(?:\([^)]*\))?(?=,|$)',gres))


def base_config(root,g='g0',d='d0',local=None):
    config=dict(global_ssm=dict(backend='mamba',directions=4,topology=g,window_size=8),
        decoder=dict(kind=d),local=dict(enabled=local is not None,backend='mamba',mode='residual',
        layout='one',embedding=True,feature_source='f1',roi_size=13,core=local or 'conv'),
        pretrained_path=str(root/'pretrained/seresnet50_a1_0-ffa00869.pth'))
    if d=='d2': config.update(heatmap_size=[180,320],target_sigma=4.)
    return config


def train_task(root,identifier,config,seed=20261025,updates=15000,limit=8000,phase='global',**extra):
    if extra.get('parent_checkpoint') and Path(extra['parent_checkpoint']).is_file():
        extra['parent_checkpoint_sha256']=sha(extra['parent_checkpoint'])
    return dict(id=identifier,kind='train',gpu=True,training=dict(id=identifier,config=config,
        seed=seed,subset_seed=20261025,batch=8,effective_batch=8,workers=4,precision='fp32',
        max_updates=updates,train_limit=limit,phase=phase,augment=phase!='local',early_stopping=False,
        validation_interval=1000 if limit else 5000,shared_gdl_initialization=True,stage='exploratory',**extra))


def result(root,task):
    path=root/'trials'/task['id']/'completed.json'
    if not path.exists(): raise ValueError('Missing completed training '+task['id'])
    value=read(path)
    if value['status']!='complete' or value['updates']!=task['training']['max_updates']:
        raise ValueError('Training did not complete matched horizon '+task['id'])
    return value


def selected(root,task):
    value=result(root,task)
    admissible=value.get('best_admissible')
    if task['training'].get('reference_metrics'):
        if not admissible: return None
        metric,name=admissible,'best_admissible'
    else:metric,name=value['best'],'best'
    checkpoint=root/'trials'/task['id']/(name+'.pt')
    proof=read(root/'architecture_gdl/tasks'/task['id']/'artifact_integrity.json')
    digest=sha(checkpoint)
    if digest!=proof['checkpoints'][name]['sha256']:raise ValueError('Selected checkpoint hash changed')
    return dict(trial=task['id'],metrics=metric,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        config=task['training']['config'],seed=task['training']['seed'])


def write_reference(root,parent):
    path=root/'architecture_gdl/references'/(parent['trial']+'.json')
    if path.exists():
        if read(path)!=parent['metrics']: raise ValueError('Immutable parent metrics changed')
    else: dump(path,parent['metrics'])
    return str(path)


def score(parent):
    values=parent['metrics']['final']
    return (values['balanced_mean'],values['balanced_p95'],parent['trial'])


def history_rows(path):
    text=Path(path).read_text();lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def calibration_seconds(root,state,task):
    if task['kind']!='train': return 3600
    training=task['training']; phase=training['phase']
    g=training['config']['global_ssm']['topology']; d=training['config']['decoder']['kind']
    local=training['config']['local']['core'] if phase=='local' else None
    key=f'{g}_{d}_{local or "none"}'
    profiles=state.get('calibration',{})
    profile=profiles.get(key,profiles.get(f'g0_d0_{local or "none"}'))
    if not profile: raise ValueError('Missing measured wall-time calibration '+key)
    validations=math.ceil(training['max_updates']/training['validation_interval'])
    estimate=training['max_updates']*profile['wall_seconds_per_update']+validations*profile['full_validation_seconds']
    estimate+=(math.ceil(training['max_updates']/200)+validations)*profile['checkpoint_seconds']
    # Refine with complete pilot allocations, including startup/IO/checkpoint/validation.
    for old in state.get('all_tasks',[]):
        tr=old.get('training',{})
        if tr.get('phase')==phase and tr.get('config')==training['config']:
            records=[r for r in state['ledger'] if r['task']==old['id'] and r.get('state')=='COMPLETED']
            if records and tr.get('max_updates'):
                estimate=max(estimate,sum(r['elapsed_seconds'] for r in records)*training['max_updates']/tr['max_updates'])
    return min(43200,max(900,int(math.ceil((estimate*1.35+180)/60))*60))


def reconcile(state):
    for row in state['ledger']:
        if row.get('state') in TERMINAL or not row.get('job_id'): continue
        output=slurm_output(['sacct','-X','-j',row['job_id'],'-n','-P',
            '--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode'])
        for line in output.splitlines():
            fields=line.split('|')
            if fields[0]!=row['job_id']: continue
            row['state']=fields[1].split()[0].split('+')[0]
            row['elapsed_seconds']=int(fields[2] or 0)
            if len(fields)>4:row['exit_code']=fields[4]
            if row['gpu'] and fields[3] and 'gres/gpu=1' not in fields[3]:
                raise ValueError('GPU allocation does not match single-GPU ledger')
    state['allocated_gpu_hours']=sum(r.get('elapsed_seconds',0) for r in state['ledger'] if r['gpu'])/3600
    state['reserved_or_charged_gpu_hours']=sum((r.get('elapsed_seconds',0) if r.get('state') in TERMINAL
        else r['seconds']) for r in state['ledger'] if r['gpu'])/3600


def task_status(root,state,task):
    records=[r for r in state['ledger'] if r['task']==task['id']]
    if any(r.get('state') not in TERMINAL for r in records): return 'RUNNING'
    out=root/'architecture_gdl/tasks'/task['id']
    if (out/'completed.json').exists():
        proof=read(out/'completed.json')
        if not proof.get('passed') or proof.get('task_sha256')!=sha(out/'task.json'):return 'FAILED'
        if task['kind']=='train':
            from .gdl_integrity import audit_training
            try:audit_training(root,task)
            except Exception as error:
                state.setdefault('artifact_errors',{})[task['id']]=str(error);return 'FAILED'
        return 'VERIFIED'
    if not records: return 'READY'
    last=records[-1]
    if task['kind']=='train' and last['state'] in ('COMPLETED','TIMEOUT','NODE_FAIL','PREEMPTED','BOOT_FAIL'):
        checkpoint=root/'trials'/task['id']/'last.pt'
        if checkpoint.exists():
            from .gdl_integrity import audit_training
            try:proof=audit_training(root,task,resume=True)
            except Exception as error:
                state.setdefault('artifact_errors',{})[task['id']]=str(error);return 'FAILED'
            updates=proof['checkpoints']['last']['updates']
            progress=updates>last.get('start_updates',0)
            if progress or len(records)<3:
                task['resume_verified_updates']=updates
                return 'READY'
    if last['state'] in ('NODE_FAIL','PREEMPTED','BOOT_FAIL','SUBMIT_FAILED') and len(records)<3:return 'READY'
    return 'FAILED'


def assert_source(root,state):
    manifest=read(Path(state['source']).parent/'source_manifest.json')
    for name,digest in manifest['files'].items():
        if sha(Path(state['source'])/name)!=digest: raise ValueError('Execution snapshot changed: '+name)
    if sha(root/'manifest50k.json')!=state['manifest_sha256']: raise ValueError('Dataset manifest changed')


def submit(root,state,task):
    # External requested devices count too; never exceed the authorized runtime limit.
    queue=slurm_output(['squeue','--me','--noheader','--format=%i|%b|%T'])
    active_gpu=requested_gpus(queue)
    own_gpu=sum(1 for row in state['ledger'] if row['gpu'] and row.get('state') not in TERMINAL)
    if task.get('gpu') and max(active_gpu,own_gpu)>=campaign_gpu_limit(state): return
    if any(not row.get('job_id') and row.get('state') not in TERMINAL for row in state['ledger']):
        return
    if task.get('official'):
        freeze=root/'architecture_gdl/freeze.json'; proof=root/'architecture_gdl/tasks/verify_frozen/verification.json'
        if not freeze.exists() or not proof.exists() or not read(proof).get('passed') or read(proof)['freeze_sha256']!=sha(freeze):
            raise ValueError('Official training requires current frozen GPU verification')
        frozen=read(freeze);training=task['training'];local=training['phase']=='local'
        if (training['config']!=frozen['config' if local else 'global_config'] or
            training['max_updates']!=frozen['local_updates' if local else 'global_updates'] or
            training['seed'] not in frozen['seeds'] or training['train_limit'] is not None):
            raise ValueError('Official task differs from frozen architecture/recipe')
        if sha(Path(state['source']).parent/'source_manifest.json')!=frozen['source_manifest_sha256']:
            raise ValueError('Official source differs from verified freeze')
    if task['kind']=='final':
        lock_path=root/'architecture_gdl/final_lock.json'
        if not lock_path.exists():raise ValueError('Final evaluation requires final checkpoint lock')
        locked=read(lock_path);entry=locked['checkpoints'][str(task['seed'])]
        if (entry['checkpoint']!=task['parent']['checkpoint'] or sha(entry['checkpoint'])!=entry['checkpoint_sha256'] or
            sha(root/'manifest50k.json')!=locked['manifest_sha256']):
            raise ValueError('Final evaluation task differs from checkpoint/manifest lock')
    folder=root/'architecture_gdl/tasks'/task['id'];folder.mkdir(parents=True,exist_ok=True)
    task.setdefault('source_directory',state['source'])
    task.setdefault('source_manifest_sha256',sha(Path(task['source_directory']).parent/'source_manifest.json'))
    path=folder/'task.json'
    persisted={k:v for k,v in task.items() if k!='resume_verified_updates'}
    if path.exists() and read(path)!=persisted: raise ValueError('Immutable task changed')
    dump(path,persisted)
    seconds=calibration_seconds(root,state,task)
    attempt=1+sum(r['task']==task['id'] for r in state['ledger'])
    record=dict(task=task['id'],gpu=bool(task.get('gpu')),seconds=seconds,state='RESERVED',elapsed_seconds=0,
        created_epoch=time.time(),submission_token=token(root,task['id'],attempt),attempt=attempt,
        start_updates=task.get('resume_verified_updates',0),source_directory=task['source_directory'])
    state['ledger'].append(record);dump(root/'architecture_gdl/state.json',state)
    command=['sbatch','--parsable','--export=NONE','--partition='+('gpu' if task.get('gpu') else 'compute'),
        '--cpus-per-task='+('8' if task.get('gpu') else '4'),'--mem='+('64G' if task.get('gpu') else '16G'),
        '--time='+str(seconds//60),'--job-name=gdl-'+hashlib.sha256(task['id'].encode()).hexdigest()[:10],
        '--comment='+record['submission_token'],'--signal=USR1@180',
        '--output='+str(folder/'%j.out'),'--error='+str(folder/'%j.err')]
    if task.get('gpu'): command.append('--gres=gpu:nvidia_rtx_a5000:1')
    command.extend([str(Path(task['source_directory'])/'scripts/slurm/gdl.slurm'),str(root),'worker',str(path),str(seconds),task['source_directory'],
                    'gpu' if task.get('gpu') else 'cpu'])
    try:
        record['job_id']=slurm_output(command).strip().split(';')[0];record['state']='PENDING'
    except SlurmUnavailable:
        record['state']='SUBMISSION_UNKNOWN'
        dump(root/'architecture_gdl/state.json',state);raise
    dump(root/'architecture_gdl/state.json',state)


def queue(state,phase,tasks):
    state['stage_history'].append(dict(phase=state['phase'],tasks=state['tasks'],
        decisions={k:state.get(k) for k in ('global_parents','gd_parents','local_choices','finalist_index')}))
    state['phase']=phase;state['tasks']=tasks;state['all_tasks'].extend(tasks)


def confirmation(root,config,label,parents=None,updates=100000):
    tasks=[]
    for seed in SEEDS:
        extra={}
        if parents:
            extra['reference_metrics']=write_reference(root,parents[str(seed)])
        tasks.append(train_task(root,f'GDL_{label}_{seed}',deepcopy(config),seed,updates,None,**extra))
    return tasks


def parents_by_seed(root,tasks):
    values={str(t['training']['seed']):selected(root,t) for t in tasks}
    return values if set(values)==set(map(str,SEEDS)) and all(values.values()) else None


def converged(root,tasks):
    for task in tasks:
        rows=history_rows(root/'trials'/task['id']/'history.jsonl')
        recent=rows[-3:]
        if len(recent)<3:return False
        means=[r['validation']['final']['balanced_mean'] for r in recent]
        if means[-1]==min(means) and (means[0]-means[-1])/max(means[0],1e-12)>=.01:return False
    return True


def extend_confirmation(root,state):
    if converged(root,state['tasks']): return False
    if state.get('extensions',{}).get(state['phase']):
        state.update(phase='INSUFFICIENT_CONVERGENCE_EVIDENCE',reason='Matched extended confirmation is still improving; no architecture freeze')
        return True
    state.setdefault('extensions',{})[state['phase']]=True
    tasks=[]
    for old in state['tasks']:
        new=deepcopy(old);tr=new['training'];tr['resume_trial']=old['id']
        tr['max_updates']=int(tr['max_updates']*1.5);new['id']=tr['id']=old['id']+'_extended'
        tasks.append(new)
    state['stage_history'].append(dict(phase=state['phase'],tasks=state['tasks']))
    state['tasks']=tasks;state['all_tasks'].extend(tasks)
    return True


def advance(root,state):
    phase=state['phase'];tasks=[t for t in state['tasks'] if state.get('task_status',{}).get(t['id'],'VERIFIED')=='VERIFIED']
    out=root/'architecture_gdl'
    if phase=='profile':
        state['calibration']=read(out/'tasks/profile/profile.json')['variants']
        queue(state,'global_pilot',[train_task(root,f'GDL_A_{g}_20261025',base_config(root,g)) for g in ('g0','g1','g2')])
    elif phase=='global_pilot':
        review_global_pilots(root,state,tasks)
        queue(state,'global_control_confirmation',confirmation(root,base_config(root),'Acontrol'))
    elif phase=='global_control_confirmation':
        state['global_control']=parents_by_seed(root,tasks)
        state['global_control_tasks']=tasks
        if not state.get('pilot_gate_reviewed'):
            pilot=next(s['tasks'] for s in state['stage_history'] if s['phase']=='global_pilot')
            review_global_pilots(root,state,pilot)
        if not state.get('global_candidates'):
            if extend_confirmation(root,state):return
            state['global_parents']=state['global_control'];queue_decoders(root,state);return
        state['global_challenger']=state['global_candidates'][state.get('global_candidate_index',0)]
        new=confirmation(root,state['global_challenger'],'Achallenger',state['global_control'],
                         updates=state.get('global_horizon',100000))
        if state.get('global_candidate_index',0):
            for t in new:t['id']=t['training']['id']=t['id']+'_alternate'
        if state.get('global_horizon'):
            for t in new:
                old=next(o for o in state['global_challenger_tasks'] if o['training']['seed']==t['training']['seed'])
                t['training']['resume_trial']=old['id'];t['id']=t['training']['id']=old['id']+'_extended'
        queue(state,'global_challenger_confirmation',new)
    elif phase=='global_challenger_confirmation':
        if not converged(root,tasks+state['global_control_tasks']):
            if state.get('global_horizon'):
                state.update(phase='INSUFFICIENT_CONVERGENCE_EVIDENCE',reason='Global matched 150K confirmation still improving');return
            state['global_horizon']=150000;state['global_challenger_tasks']=tasks
            extended=[]
            for old in state['global_control_tasks']:
                new=deepcopy(old);new['training']['resume_trial']=old['id'];new['training']['max_updates']=150000
                new['id']=new['training']['id']=old['id']+'_extended';extended.append(new)
            queue(state,'global_control_confirmation',extended);return
        challenger=parents_by_seed(root,tasks)
        if not challenger and state.get('global_candidate_index',0)+1<len(state.get('global_candidates',[])):
            state['global_candidate_index']=state.get('global_candidate_index',0)+1
            state['global_challenger']=state['global_candidates'][state['global_candidate_index']]
            alternate=confirmation(root,state['global_challenger'],'Achallenger_alt',state['global_control'],
                state.get('global_horizon',100000))
            queue(state,'global_challenger_confirmation',alternate);return
        state['global_parents']=challenger or state['global_control']
        state['global_runnerup']=state['global_control'] if challenger else None
        queue_decoders(root,state)
    elif phase=='decoder_profile':
        state['calibration'].update(read(out/'tasks/profile_decoder/profile.json')['variants'])
        queue(state,'decoder_pilot',stage_pilot_tasks(root,state,'decoder_pilot'))
    elif phase=='decoder_pilot':
        valid=review_stage_pilots(root,state,tasks,'decoder_pilot')
        if valid:
            winner=valid[0];state['decoder_candidates']=[p['config'] for p in valid]
            state['decoder_challenger']=winner['config'];state['decoder_candidate_index']=0
            horizon=next(t['training']['max_updates'] for t in state['global_control_tasks'])
            queue(state,'decoder_confirmation',confirmation(root,winner['config'],'Bconfirm',state['global_parents'],horizon))
        else:
            state['gd_parents']=state['global_parents'];queue_local(root,state)
    elif phase=='decoder_confirmation':
        if not converged(root,tasks):
            state.update(phase='INSUFFICIENT_CONVERGENCE_EVIDENCE',reason='Decoder confirmation still improving at matched Global horizon; no asymmetric extension');return
        parents=parents_by_seed(root,tasks)
        if not parents and state.get('decoder_candidate_index',0)+1<len(state.get('decoder_candidates',[])):
            index=state['decoder_candidate_index']=state.get('decoder_candidate_index',0)+1
            horizon=next(t['training']['max_updates'] for t in state['global_control_tasks'])
            queue(state,'decoder_confirmation',confirmation(root,state['decoder_candidates'][index],
                'Bconfirm_alternate',state['global_parents'],horizon));return
        state['gd_parents']=parents or state['global_parents'];queue_local(root,state)
    elif phase=='local_profile':
        state['calibration'].update(read(out/'tasks/profile_local/profile.json')['variants'])
        queue(state,'local_pilot',stage_pilot_tasks(root,state,'local_pilot'))
    elif phase=='local_pilot':
        valid=review_stage_pilots(root,state,tasks,'local_pilot')
        if not valid:
            state.update(phase='NO_VALID_LOCAL',reason='No Local met strict 5%/per-source P95/PCK gate; no official GDL');return
        chosen=valid[:2];state['local_choices']=[p['config']['local']['core'] for p in chosen]
        confirm=[]
        for core in state['local_choices']:
            confirm.extend(local_confirmation(root,state,core))
        queue(state,'local_confirmation',confirm)
    elif phase=='local_confirmation':
        # All competing heads get the same extension when any seed has not stabilized.
        if extend_confirmation(root,state):return
        winners=[]
        for core in state['local_choices']:
            group=[t for t in tasks if t['training']['config']['local']['core']==core]
            parents=parents_by_seed(root,group)
            if parents:winners.append(parents)
        if not winners:
            state.update(phase='NO_VALID_LOCAL',reason='No Local passed both confirmation seeds; no fallback official');return
        winners.sort(key=lambda p:sum(score(v)[0] for v in p.values())/2)
        state['finalists']=winners;state['finalist_index']=0;queue_evidence(root,state)
    elif phase=='evidence':
        index=state['finalist_index']
        queue(state,'evidence_cpu',[dict(id=f'intervals_{index}_{seed}',kind='intervals',gpu=False,
            evidence_task=f'evidence_{index}_{seed}') for seed in state['final_parents']])
    elif phase=='evidence_cpu':
        failures=[]
        for seed in state['final_parents']:
            folder=out/'tasks'/f'evidence_{state["finalist_index"]}_{seed}'
            value=read(folder/'modality.json')
            for condition in ('depth_invalid','depth_features_zero','rgb_features_zero'):
                for source in ('OLD','NEW'):
                    if value['paired_comparisons'][condition][source]['ci95']['mean'][0]<=0:
                        failures.append(f'{seed}/{condition}/{source}: modality benefit not established')
            branch=read(folder/'branches.json')
            if not branch['passed']:failures.append(seed+': dead/ineffective branch diagnostic')
        if failures:
            reject_finalist(root,state,'MODALITY_OR_BRANCH_GATE_FAILED',failures);return
        queue(state,'benchmark',[dict(id=f'benchmark_{state["finalist_index"]}_{seed}',kind='benchmark',gpu=True,parent=parent)
                               for seed,parent in state['final_parents'].items()])
    elif phase=='benchmark':
        values=[read(out/'tasks'/t['id']/'benchmark.json') for t in tasks]
        if not all(v['passed'] for v in values):
            reject_finalist(root,state,'LATENCY_VRAM_GATE_FAILED','Final GDL exceeds locked legacy baseline runtime envelope');return
        config=state['final_parents']['20261025']['config']
        global_tasks=[t for t in state['all_tasks'] if t['id']==state['gd_parents']['20261025']['trial']]
        local_tasks=[t for t in state['all_tasks'] if t['id']==state['final_parents']['20261025']['trial']]
        freeze=dict(config=config,global_config=state['gd_parents']['20261025']['config'],
            global_updates=global_tasks[0]['training']['max_updates'],local_updates=local_tasks[0]['training']['max_updates'],
            source_manifest_sha256=sha(Path(state['source']).parent/'source_manifest.json'),
            manifest_sha256=state['manifest_sha256'],seeds=[20261025,20261026,20261027],
            legacy_reference_lock_sha256=sha(out/'legacy_reference_lock.json'),
            local_required=True,global_only_fallback=False,official_initialization='new matched seed; RGB pretrained only',
            selection='best admissible validation checkpoint; strict legacy gates',gpu_hour_limit=None)
        dump(out/'freeze.json',freeze)
        queue(state,'verification',[dict(id='verify_frozen',kind='verify',gpu=True)])
    elif phase=='verification':
        proof=read(out/'tasks/verify_frozen/verification.json')
        if not proof.get('passed') or proof['freeze_sha256']!=sha(out/'freeze.json'):raise ValueError('Invalid frozen proof')
        freeze=read(out/'freeze.json')
        official=[train_task(root,f'GDL_OFF_GD_{seed}',freeze['global_config'],seed,freeze['global_updates'],None) for seed in freeze['seeds']]
        for t in official:t.update(official=True);t['training']['stage']='official'
        queue(state,'official_global',official)
    elif phase=='official_global':
        state['official_gd']={str(t['training']['seed']):selected(root,t) for t in tasks}
        freeze=read(out/'freeze.json');official=[]
        for seed,parent in state['official_gd'].items():
            task=train_task(root,f'GDL_OFF_L_{seed}',freeze['config'],int(seed),freeze['local_updates'],None,'local',
                parent=parent['trial'],parent_checkpoint=parent['checkpoint'],reference_metrics=write_reference(root,parent),selection_local=True)
            task.update(official=True);task['training']['stage']='official';official.append(task)
        queue(state,'official_local',official)
    elif phase=='official_local':
        finals={str(t['training']['seed']):selected(root,t) for t in tasks}
        if not all(finals.values()):
            state.update(phase='OFFICIAL_LOCAL_GATE_FAILED',reason='At least one official Local seed failed; no Global-only fallback');return
        locks={seed:{**p,'checkpoint_sha256':sha(p['checkpoint'])} for seed,p in finals.items()}
        dump(out/'final_lock.json',dict(checkpoints=locks,freeze_sha256=sha(out/'freeze.json'),manifest_sha256=state['manifest_sha256']))
        queue(state,'final',[dict(id='final_'+seed,kind='final',gpu=True,parent=p,seed=int(seed)) for seed,p in locks.items()])
    elif phase=='final':
        metrics=[read(out/'tasks'/t['id']/'metrics.json') for t in tasks]
        accepted=all(m['accepted'] for m in metrics)
        dump(out/'release.json',dict(accepted=accepted,default_model_changed=False,rows=metrics,
            limitation='Observed D455 session and render-grouped synthetic split; no independent unseen-domain claim'))
        state.update(phase='COMPLETED',release_accepted=accepted)


def admissible_pilot(root,task,control):
    parent=selected(root,task)
    if acceptable(parent['metrics'],control['metrics']):return parent
    # Both best and final checkpoints exist; final may preserve tails better than best-mean.
    history=root/'trials'/task['id']/'history.jsonl'
    rows=history_rows(history)
    if rows and rows[-1]['updates']==task['training']['max_updates'] and acceptable(rows[-1]['validation'],control['metrics']):
        checkpoint=root/'trials'/task['id']/'last.pt'
        return dict(parent,metrics=rows[-1]['validation'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
    return None


def review_global_pilots(root,state,tasks):
    control=selected(root,next(t for t in tasks if t['training']['config']['global_ssm']['topology']=='g0'))
    candidates=[];decisions=[]
    for task in tasks:
        if task['training']['config']['global_ssm']['topology']=='g0':continue
        if state.get('task_status',{}).get(task['id'])=='FAILED' or task['id'] in state.get('failed_tasks',{}):continue
        parent=admissible_pilot(root,task,control)
        decisions.append(dict(trial=task['id'],gate_passed=parent is not None))
        if parent:candidates.append(parent)
    state['global_candidates']=[p['config'] for p in sorted(candidates,key=score)]
    state['pilot_gate_reviewed']=True;state['global_pilot_decisions']=decisions


def stage_pilot_tasks(root,state,phase):
    if phase not in ('decoder_pilot','local_pilot'):raise ValueError('Unsupported parallel pilot phase')
    tasks=[]
    for seed in pilot_seeds(state):
        parent=state['global_parents' if phase=='decoder_pilot' else 'gd_parents'][str(seed)]
        variants=('d0','d1','d2') if phase=='decoder_pilot' else ('conv','ssm','hybrid')
        for variant in variants:
            if phase=='decoder_pilot':
                config=base_config(root,parent['config']['global_ssm']['topology'],variant)
                task=train_task(root,f'GDL_B_{variant}_{seed}',config,seed,phase='decoder',
                    parent=parent['trial'],parent_checkpoint=parent['checkpoint'])
            else:
                config=deepcopy(parent['config']);config['local'].update(enabled=True,core=variant)
                task=train_task(root,f'GDL_C_{variant}_{seed}',config,seed,phase='local',
                    parent=parent['trial'],parent_checkpoint=parent['checkpoint'],
                    reference_metrics=write_reference(root,parent),selection_local=True)
            tasks.append(task)
    return tasks


def expand_parallel_pilots(root,state):
    """Activation under the single-writer lock may add seeds; existing tasks remain immutable."""
    if state['phase'] not in ('decoder_pilot','local_pilot'):return []
    known={t['id'] for t in state['all_tasks']}
    additions=[t for t in stage_pilot_tasks(root,state,state['phase']) if t['id'] not in known]
    state['tasks'].extend(additions);state['all_tasks'].extend(additions)
    return [t['id'] for t in additions]


def review_stage_pilots(root,state,tasks,phase):
    """Apply the existing comparator independently to every seed before aggregate ranking."""
    seeds=pilot_seeds(state);decoder=phase=='decoder_pilot'
    def variant(task):
        config=task['training']['config']
        return config['decoder']['kind'] if decoder else config['local']['core']
    indexed={(variant(t),t['training']['seed']):t for t in tasks}
    controls={seed:selected(root,indexed['d0',seed]) for seed in seeds} if decoder else {}
    variants=('d1','d2') if decoder else ('conv','ssm','hybrid')
    valid=[];decisions=[]
    for name in variants:
        parents=[]
        for seed in seeds:
            task=indexed.get((name,seed))
            parent=(admissible_pilot(root,task,controls[seed]) if decoder else selected(root,task)) if task else None
            decisions.append(dict(variant=name,seed=seed,trial=task['id'] if task else None,
                control=controls[seed]['trial'] if decoder else state['gd_parents'][str(seed)]['trial'],
                gate_passed=parent is not None))
            if parent:parents.append(parent)
        if len(parents)==len(seeds):
            valid.append(dict(config=parents[0]['config'],parents=parents,
                rank=(sum(score(p)[0] for p in parents)/len(parents),
                      sum(score(p)[1] for p in parents)/len(parents),name)))
    state['decoder_pilot_decisions' if decoder else 'local_pilot_decisions']=decisions
    return sorted(valid,key=lambda candidate:candidate['rank'])


def queue_decoders(root,state):
    parent=state['global_parents']['20261025']
    configs=[base_config(root,parent['config']['global_ssm']['topology'],d) for d in ('d0','d1','d2')]
    queue(state,'decoder_profile',[dict(id='profile_decoder',kind='profile',gpu=True,configs=configs)])


def queue_evidence(root,state):
    index=state['finalist_index'];state['final_parents']=state['finalists'][index]
    queue(state,'evidence',[dict(id=f'evidence_{index}_{seed}',kind='evidence',gpu=True,parent=parent)
        for seed,parent in state['final_parents'].items()])


def reject_finalist(root,state,phase,reason):
    state.setdefault('rejected_finalists',[]).append(dict(index=state['finalist_index'],gate=phase,reason=reason))
    state['finalist_index']+=1
    if state['finalist_index']<len(state['finalists']):queue_evidence(root,state)
    else:state.update(phase=phase,reason=reason)


def queue_local(root,state):
    parent=state['gd_parents']['20261025'];configs=[]
    for core in ('conv','ssm','hybrid'):
        config=deepcopy(parent['config']);config['local'].update(enabled=True,core=core)
        configs.append(config)
    queue(state,'local_profile',[dict(id='profile_local',kind='profile',gpu=True,configs=configs)])


def local_confirmation(root,state,core):
    tasks=[]
    for seed,parent in state['gd_parents'].items():
        config=deepcopy(parent['config']);config['local'].update(enabled=True,core=core)
        tasks.append(train_task(root,f'GDL_Cconfirm_{core}_{seed}',config,int(seed),30000,None,'local',
            parent=parent['trial'],parent_checkpoint=parent['checkpoint'],reference_metrics=write_reference(root,parent),selection_local=True))
    return tasks


ACTIVE_PHASES={'profile','global_pilot','global_control_confirmation','global_challenger_confirmation','decoder_profile','local_profile',
    'decoder_pilot','decoder_confirmation','local_pilot','local_confirmation','evidence','evidence_cpu',
    'benchmark','verification','official_global','official_local','final'}


def report(root,state):
    out=root/'architecture_gdl'
    official=('COMPLETED' if state['phase'] in ('final','COMPLETED') else 'IN_PROGRESS'
              if state['phase'] in ('official_global','official_local') else 'FAILED'
              if state.get('failed_stage','').startswith('official') else 'NOT_STARTED')
    verification=out/'tasks/verify_frozen/verification.json'
    verification_status='PASSED' if verification.exists() and read(verification).get('passed') else 'NOT_STARTED'
    registry=dict(phase=state['phase'],controller_job_id=state.get('controller_job_id'),
        controller_health=state.get('controller_health'),tasks=state['all_tasks'],
        jobs=state['ledger'],task_status=state.get('task_status',{}),failed_tasks=state.get('failed_tasks',{}),
        decisions=state['stage_history'],global_winner=state.get('global_parents'),
        decoder_winner=state.get('gd_parents'),local_winner=state.get('final_parents'),
        verification=verification_status,official_training=official,
        final_evaluation='COMPLETED' if state['phase']=='COMPLETED' else 'IN_PROGRESS' if state['phase']=='final' else 'NOT_STARTED',
        gpu_hour_limit=None,max_campaign_gpus=campaign_gpu_limit(state),pilot_seeds=list(pilot_seeds(state)),reason=state.get('reason'))
    registry['selection']=dict(global_pilot_decisions=state.get('global_pilot_decisions'),
        global_candidates=state.get('global_candidates'),decoder_candidates=state.get('decoder_candidates'),
        decoder_pilot_decisions=state.get('decoder_pilot_decisions'),local_pilot_decisions=state.get('local_pilot_decisions'),
        local_choices=state.get('local_choices'),rejected_finalists=state.get('rejected_finalists'))
    registry['progress']={}
    for task in state['all_tasks']:
        path=root/'trials'/task['id']/'progress.json'
        if path.exists():registry['progress'][task['id']]=read(path)
    dump(out/'registry.json',registry)
    lines=['<!-- gdl-search -->','## G–D–L search — thực thi',
        f"Trạng thái: **{state['phase']}**. Default baseline giữ nguyên.",
        f"GPU-giờ đã cấp (gồm legacy): {state.get('allocated_gpu_hours',0):.4f}; trần GPU-giờ đã được người dùng gỡ bỏ. Tối đa {campaign_gpu_limit(state)} GPU đồng thời.",
        'Decoder/Local pilot seeds: '+', '.join(map(str,pilot_seeds(state)))+'; gate từng seed, xếp hạng bằng trung bình hai nguồn rồi trung bình seed.',
        'Legacy một hướng: STOPPED/PRESERVED; checkpoint đã được xác minh trước cancellation.',
        'Selection chỉ dùng validation; synthetic test/D455 chỉ mở sau final lock.',
        'Metrics trong bảng thuộc checkpoint đã chọn/bản validation gần nhất; checkpoint hash và updates gốc nằm trong artifact_integrity.json.',
        f"Verification: **{verification_status}**; official: **{official}**; final evaluation: **{registry['final_evaluation']}**.",
        f"Controller: {state.get('controller_job_id','pending')}; health: {state.get('controller_health',{}).get('status','STARTING')}.",
        '', '| Trial | Trạng thái | Updates | Balanced mean | OLD P95/PCK05 | NEW P95/PCK05 |', '|---|---|---:|---:|---|---|']
    for task in state.get('all_tasks',[]):
        if task['kind']!='train':continue
        completed=root/'trials'/task['id']/'completed.json'
        history=root/'trials'/task['id']/'history.jsonl'
        if completed.exists():
            value=read(completed);m=value.get('best_admissible') or value['best'];updates=value['updates']
        elif history.exists():
            rows=history_rows(history)
            if not rows:continue
            m=rows[-1]['validation'];updates=rows[-1]['updates']
        else:
            progress=root/'trials'/task['id']/'progress.json'
            updates=read(progress).get('updates',0) if progress.exists() else 0
            status=state.get('task_status',{}).get(task['id'],'PENDING')
            lines.append(f"| {task['id']} | {status} | {updates} | pending validation | — | — |")
            continue
        f=m['final'];old=f['OLD'];new=f['NEW']
        status=state.get('task_status',{}).get(task['id'],'COMPLETED' if completed.exists() else 'RUNNING')
        if task['id'] in state.get('failed_tasks',{}):status='FAILED'
        lines.append(f"| {task['id']} | {status} | {updates} | {f['balanced_mean']:.4f} | {old['p95']:.4f}/{old['PCK05']:.6f} | {new['p95']:.4f}/{new['PCK05']:.6f} |")
    lines.extend(['','| Job | Task | Slurm state | Elapsed seconds |','|---|---|---|---:|'])
    for job in state['ledger']:
        if job.get('legacy'):continue
        lines.append(f"| {job.get('job_id','reconciling')} | {job['task']} | {job['state']} | {job.get('elapsed_seconds',0)} |")
    for label,key in (('Global winner','global_parents'),('Decoder parent','gd_parents'),('G–D–L finalist','final_parents')):
        parents=state.get(key)
        if parents:lines.extend(['',label+': '+', '.join(f"{seed}: `{parent['checkpoint']}`" for seed,parent in parents.items())])
    if state.get('reason'):lines.extend(['',str(state['reason'])])
    lines.extend(['','Artifact: `architecture_gdl/`; legacy source/checkpoints giữ nguyên.','<!-- /gdl-search -->'])
    section='\n'.join(lines)+'\n';dump(out/'summary.json',dict(phase=state['phase'],reason=state.get('reason'),gpu_hours=state.get('allocated_gpu_hours')))
    p=root/'REPORT.md';text=p.read_text();start=text.find('<!-- gdl-search -->')
    if start>=0:text=text[:start]+section
    else:text+='\n'+section
    paragraph='Architecture search RGB-D: **IN PROGRESS**. Control/variants dùng cùng trainer/evaluator; chưa freeze và chưa official train. Bảo vệ 136 GPU-giờ cho official/verification/final/recovery; không train YOLO, không dùng D455/test để chọn cấu hình. Kết quả đợt mới được bổ sung tại đây sau khi artifact được xác minh.'
    current=f"Campaign G–D–L: **{state['phase']}**. Trần 288 GPU-giờ đã được người dùng gỡ bỏ; giữ tối đa {campaign_gpu_limit(state)} GPU đồng thời. Trạng thái hiện hành và job ledger xem phần G–D–L cuối báo cáo. Các bảng phía trên là lịch sử; không dùng test/D455 để chọn cấu hình."
    text=text.replace(paragraph,current)
    text=re.sub(r'^Campaign G–D–L: .*$',current,text,count=1,flags=re.MULTILINE)
    p.write_text(text)
    status=read(root/'status.json');status.update(phase='gdl_'+state['phase'],details='architecture_gdl/state.json',
        gpu_hour_cap_removed_by_user=True,budget=dict(allocated_gpu_hours=state.get('allocated_gpu_hours'),
        charged_or_reserved_gpu_hours=state.get('reserved_or_charged_gpu_hours'),limit_gpu_hours=None,
        active=sum(r['gpu'] and r['state'] not in TERMINAL for r in state['ledger']),max_campaign_gpus=campaign_gpu_limit(state)),
        gdl=dict(phase=state['phase'],reason=state.get('reason'),official_training='COMPLETED' if state['phase'] in ('final','COMPLETED') else
        'IN_PROGRESS' if state['phase'] in ('official_global','official_local') else 'NOT_STARTED'),default_model_changed=False)
    dump(root/'status.json',status)


def mandatory_failure(state,task):
    phase=state['phase'];training=task.get('training',{})
    if phase=='global_pilot':return training['config']['global_ssm']['topology']=='g0'
    if phase=='decoder_pilot':return training['config']['decoder']['kind']=='d0'
    return phase in ('profile','global_control_confirmation','decoder_profile','local_profile',
                     'verification','official_global','official_local','final')


def ensure_successor(root,state,current):
    if current in state.setdefault('controller_successors',{}):
        job=state['controller_successors'][current]
        record=next((r for r in state['ledger'] if r.get('job_id')==job),None)
        if record is None or record['state'] not in TERMINAL:return
        del state['controller_successors'][current]
    identifier='controller_successor_'+current
    records=[r for r in state['ledger'] if r['task']==identifier]
    if records and records[-1].get('job_id') and records[-1]['state'] not in TERMINAL:
        state['controller_successors'][current]=records[-1]['job_id'];return
    if records and records[-1]['state'] not in TERMINAL:return
    record=dict(task=identifier,gpu=False,seconds=43200,state='RESERVED',elapsed_seconds=0,
        created_epoch=time.time(),submission_token=token(root,identifier,len(records)+1))
    state['ledger'].append(record);dump(root/'architecture_gdl/state.json',state)
    out=root/'architecture_gdl'
    command=['sbatch','--parsable','--export=NONE','--partition=compute','--cpus-per-task=4','--mem=16G','--time=12:00:00',
        '--job-name=gdl-controller','--comment='+record['submission_token'],'--dependency=afterany:'+current,
        '--output='+str(out/'controller-%j.out'),'--error='+str(out/'controller-%j.err'),
        str(Path(state['source'])/'scripts/slurm/gdl.slurm'),str(root),'controller','unused','43200',state['source']]
    try:
        record['job_id']=slurm_output(command).strip().split(';')[0];record['state']='PENDING'
        state['controller_successors'][current]=record['job_id']
    except SlurmUnavailable:
        record['state']='SUBMISSION_UNKNOWN';dump(out/'state.json',state);raise
    dump(out/'state.json',state)


def tick(root,state):
    recover_submissions(root,state);reconcile(state)
    statuses={t['id']:task_status(root,state,t) for t in state['tasks']}
    state['task_status']=statuses
    failed=[t for t in state['tasks'] if statuses[t['id']]=='FAILED']
    for task in failed:
        state.setdefault('failed_tasks',{})[task['id']]=dict(kind=task['kind'],
            reason=state.get('artifact_errors',{}).get(task['id'],'Worker failed; see Slurm exit and failure.json'),
            source=task.get('source_directory'))
    fatal=[t for t in failed if mandatory_failure(state,t)]
    if fatal:
        state.update(failed_stage=state['phase'],phase='TECHNICAL_BLOCKER',
            reason={t['id']:state['failed_tasks'][t['id']] for t in fatal});return
    if all(value in ('VERIFIED','FAILED') for value in statuses.values()):
        if failed and state['phase'] in ('evidence','evidence_cpu','benchmark'):
            reject_finalist(root,state,'FINALIST_TECHNICAL_FAILURE',[t['id'] for t in failed])
        else:advance(root,state)
    else:
        for task in state['tasks']:
            if statuses[task['id']]=='READY':submit(root,state,task)
    heartbeat(state)


def controller(root):
    out=root/'architecture_gdl'
    with (root/'active_controller.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return  # A recovered duplicate cannot become a second writer.
        state=read(out/'state.json');assert_source(root,state)
        if state['phase'] not in ACTIVE_PHASES:return
        current=os.environ['SLURM_JOB_ID'];state['controller_job_id']=current
        while state['phase'] in ACTIVE_PHASES:
            try:
                recover_submissions(root,state)
                ensure_successor(root,state,current)
                tick(root,state)
            except SlurmUnavailable as error:
                heartbeat(state,'WAITING_SLURM',str(error))
            dump(out/'state.json',state);report(root,state)
            if state['phase'] in ACTIVE_PHASES:time.sleep(15)
        heartbeat(state,'TERMINAL');dump(out/'state.json',state);report(root,state)
        successor=state.get('controller_successors',{}).get(current)
        if successor:
            # Only the dependent CPU controller created by this campaign is canceled.
            info=slurm_output(['scontrol','show','job',successor])
            if 'Partition=compute' not in info or str(root) not in info or ' controller ' not in info:
                raise ValueError('Refusing to cancel unverified successor controller')
            slurm_output(['scancel',successor])


def initialize(root):
    out=root/'architecture_gdl';out.mkdir(exist_ok=True)
    if (out/'state.json').exists():raise ValueError('Already initialized')
    import xml.etree.ElementTree as ET
    suite=ET.parse(out/'correctness.xml').getroot()
    if any(int(s.get('failures',0)) or int(s.get('errors',0)) for s in suite.iter('testsuite')):raise ValueError('CPU correctness failed')
    if len(read(out/'legacy/preservation.json'))!=4:raise ValueError('Legacy preservation incomplete')
    old=read(root/'architecture/state.json')
    if any(r['state'].split()[0] not in TERMINAL for r in old['reservations']):raise ValueError('Legacy allocations still active')
    ids=','.join(r['job_id'] for r in old['reservations'] if r.get('job_id'))
    accounting=subprocess.check_output(['sacct','-X','-j',ids,'-n','-P',
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True)
    rows={row[0]:row for row in (line.split('|') for line in accounting.splitlines())}
    for record in old['reservations']:
        row=rows.get(record['job_id'])
        if not row or 'gres/gpu=1' not in row[3]:raise ValueError('Historical GPU accounting unresolved')
        record['state']=row[1].split()[0].split('+')[0];record['elapsed']=int(row[2])
    snapshot(out)
    policy=dict(total_gpu_hour_limit=None,max_gpus=4,removes_previous_288_hour_cap=True,selection='validation only',
        pilot_updates=15000,confirmation_target_updates=100000,local_confirmation_updates=30000,
        convergence_extension_factor=1.5,convergence_extensions=1,local_required=True,global_only_fallback=False,
        g2_initialization='zero output projection; exact G0 identity before learning',
        decoder_roi_coordinates='fine heatmap -> image pixel centers -> feature normalized sampling; never reuse coarse indices',
        legacy_references=['F_baseline','F_candidate'],final_test_requires_checkpoint_lock=True)
    dump(out/'policy.json',policy)
    ledger=[dict(task=r['trial'],job_id=r['job_id'],state=r['state'].split()[0],seconds=r['seconds'],
        elapsed_seconds=r['elapsed'],gpu=True,legacy=True) for r in old['reservations']]
    task=dict(id='profile',kind='profile',gpu=True)
    state=dict(phase='profile',tasks=[task],all_tasks=[task],ledger=ledger,stage_history=[],
        source=str(out/'source'),manifest_sha256=sha(root/'manifest50k.json'))
    dump(out/'state.json',state);report(root,state)


def worker(root,path):
    task=read(path);out=root/'architecture_gdl/tasks'/task['id'];out.mkdir(parents=True,exist_ok=True)
    try:
        if task.get('source_directory'):
            source=Path(task['source_directory']);manifest=source.parent/'source_manifest.json'
            if sha(manifest)!=task['source_manifest_sha256']:raise ValueError('Worker source manifest mismatch')
            if Path(__file__).resolve().parents[2]!=source.resolve():raise ValueError('Worker executing wrong snapshot')
            for name,digest in read(manifest)['files'].items():
                if sha(source/name)!=digest:raise ValueError('Worker source content mismatch '+name)
        if task['kind']=='train':
            from .trial_runner import train
            train(task['training'],root)
            complete=root/'trials'/task['id']/'completed.json'
            if not complete.exists():return
            result(root,task)
        else:
            from . import gdl_workers
            getattr(gdl_workers,task['kind'])(root,task,out)
        dump(out/'completed.json',dict(passed=True,task_sha256=sha(path)))
    except Exception:
        dump(out/'failure.json',dict(error=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('role',choices=('initialize','controller','worker'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--task',type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('All workloads require Slurm compute')
    try:
        if args.role=='worker':worker(args.root,args.task)
        else:globals()[args.role](args.root)
    except Exception:
        path=args.root/'architecture_gdl/state.json'
        if args.role=='controller' and path.exists():
            state=read(path);state.update(phase='FAILED',reason=traceback.format_exc());dump(path,state);report(args.root,state)
        raise

if __name__=='__main__':main()
