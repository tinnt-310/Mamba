"""Architecture-stage orchestration using the existing trainer, evaluator and ledger.

Only synthetic validation is accessible to research workers. Official training
and final evaluation require explicit freeze/verification artifacts, not a
successful Slurm exit alone.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .campaign import read,dump,sha,snapshot,update_accounting
from .campaign_budget import Budget

PROTECTED={'official':96,'verification':8,'final':20,'recovery':12}
TERMINAL_OK={'complete','early_stopped'}


def variants(root):
    """One primary change per isolated experiment; all candidates remain RGB-D."""
    base=dict(global_ssm=dict(backend='mamba'),local=dict(enabled=False,backend='mamba'),
              pretrained_path=str(root/'pretrained/seresnet50_a1_0-ffa00869.pth'))
    specs=[('control',{},'Existing RGB-D architecture with the matched pretrained/augmentation recipe'),
           ('concat',{'decoder':{'merge':'concat'}},'Learn scale/channel combination instead of a fixed mean'),
           ('silu',{'decoder':{'activation':'silu'}},'Change decoder activation without changing topology'),
           ('residual',{'decoder':{'residual':True}},'Preserve multiscale localization through decoder refinement'),
           ('validity',{'fusion_mode':'validity'},'Condition self/cross reliability on measured depth quality'),
           ('detail_before',{'detail_decoder':True,'decoder':{'detail_position':'before'}},
            'Refine actual RGB stride-two detail jointly with fused semantic features'),
           ('scan4',{'global_ssm':{'directions':4}},'Test spatial ordering with shared four-direction scans')]
    result=[]
    def merge(a,b):
        out=deepcopy(a)
        for key,value in b.items():
            out[key]=merge(out.get(key,{}),value) if isinstance(value,dict) else value
        return out
    for name,change,hypothesis in specs:
        result.append(dict(id=name,config=merge(base,change),hypothesis=hypothesis,
                           control='control',initialization='official_local_RGB_pretrained; other modules seeded'))
    for core in ('ssm','conv'):
        config=merge(base,dict(local=dict(enabled=True,mode='residual',layout='one',embedding=True,
                                         feature_source='f1',core=core)))
        result.append(dict(id='local_'+core,config=config,
            hypothesis='Compare '+core+' refinement with zero-init identity and semantic conditioning',
            control='historical_F_candidate_20261025_global',initialization='same frozen Global parent',
            parent='F_candidate_20261025',phase='local'))
    return result


def protected_seconds(budget,protected=PROTECTED):
    return sum(max(0,int(hours*3600)-sum(r.charge for r in budget.records
                                       if r.phase=='architecture_'+phase))
               for phase,hours in protected.items())


def can_research(budget,seconds):
    return sum(r.charge for r in budget.records)+seconds+protected_seconds(budget)<=budget.total_seconds


def initialize(root):
    out=root/'architecture'; out.mkdir(exist_ok=True)
    if (out/'state.json').exists():
        raise ValueError('Architecture stage already initialized; resume existing controller')
    tests=ET.parse(out/'correctness.xml').getroot()
    suites=[tests] if tests.tag=='testsuite' else list(tests)
    if not suites or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):
        raise ValueError('CPU correctness must pass before architecture initialization')
    old=read(root/'continuation/state.json')
    budget=Budget(old['reservations'])
    if any(not r.terminal for r in budget.records):
        raise ValueError('Reconcile all historical active reservations before initialization')
    # Re-query every historical row, including terminal jobs, to verify the handoff.
    for r in budget.records:
        text=subprocess.check_output(['sacct','-X','-j',str(r.job_id),'-n','-P',
            '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True)
        matches=[line.split('|') for line in text.splitlines() if line.split('|')[0]==str(r.job_id)]
        if len(matches)!=1: raise ValueError('Missing/ambiguous Slurm accounting: '+str(r.job_id))
        row=matches[0]
        if 'gres/gpu=1' not in row[3]: raise ValueError('Expected single-GPU historical allocation')
        r.state=row[1].split()[0].split('+')[0]; r.elapsed=int(row[2])
    snapshot(out)
    registry=variants(root)
    dump(out/'registry.json',registry)
    policy=dict(total_gpu_hours=288,concurrency=4,protected_gpu_hours=PROTECTED,
        selection='synthetic validation only; research workers never evaluate test/D455',
        manifest_sha256=sha(root/'manifest50k.json'),
        seeds=[20261025,20261026,20261027],subset_seed=20261025,
        modality='learned RGB + learned geometry + adaptive fusion + selective SSM mandatory',
        pilot_updates=5000,pilot_train_limit=8000,official_training='requires freeze and full verification',
        supersedes_continuation_limit_162=True)
    dump(out/'policy.json',policy)
    dump(out/'state.json',dict(phase='profile',reservations=budget.serialize(),
        tasks=[dict(id='profile',kind='profile',seconds=1800,phase='research')],
        failed_tasks=[],budget=budget.summary()))


def task_status(root,t,budget):
    records=[r for r in budget.records if r.trial=='architecture_'+t['id']]
    if any(not r.terminal for r in records): return 'IN PROGRESS'
    if records and records[-1].state!='COMPLETED': return 'FAILED'
    path=root/'architecture/tasks'/t['id']/'completed.json'
    if path.exists():
        status=read(path).get('status')
        if status=='incomplete' and t.get('kind')=='train':
            slice_path=root/'trials'/t['training']['id']/'slice_complete.json'
            if slice_path.exists() and read(slice_path).get('status')=='resume_required': return 'RESUME REQUIRED'
        return 'VERIFIED' if status in TERMINAL_OK else 'FAILED'
    if records: return 'FAILED'
    return 'NOT STARTED'


def submit(root,t,budget,state):
    if t['phase']=='official':
        freeze_path=root/'architecture/freeze.json'
        proof_path=root/'architecture/tasks/verify_frozen/verification.json'
        if not freeze_path.exists() or not proof_path.exists():
            raise ValueError('Official training requires architecture freeze and GPU verification')
        proof=read(proof_path); freeze=read(freeze_path)
        if not proof.get('passed') or proof.get('freeze_sha256')!=sha(freeze_path):
            raise ValueError('Official architecture verification is stale or failed')
        if t['training']['config'] not in (freeze['config'],freeze['global_config']):
            raise ValueError('Official config differs from architecture freeze')
    if t['phase']=='research' and not can_research(budget,t['seconds']): return False
    prior=[r for r in budget.records if r.trial=='architecture_'+t['id']]
    if len(prior)>=2: return False  # At most two resumable slices; never unbounded retry.
    reservation=budget.reserve('architecture_'+t['id'],'architecture_'+t['phase'],t['seconds'],
                               trial_cap_seconds=2*t['seconds'])
    if reservation is None: return False
    out=root/'architecture'; folder=out/'tasks'/t['id']; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'task.json'
    if path.exists() and read(path)!=t: raise ValueError('Immutable task changed')
    dump(path,t); state['reservations']=budget.serialize(); dump(out/'state.json',state)
    source=Path(t.get('source_directory',out/'source'))
    if not (source/'models/keypoints/architecture_search.py').is_file():
        raise ValueError('Frozen task source missing')
    command=['sbatch','--parsable','--export=NONE','--partition=gpu',
        '--gres=gpu:nvidia_rtx_a5000:1','--cpus-per-task=8','--mem=64G',
        '--time='+str(t['seconds']//60),'--job-name=mb-arch-'+t['id'][:7],
        '--output='+str(folder/'%j.out'),'--error='+str(folder/'%j.err'),
        str(source/'scripts/slurm/improve_continue.slurm'),str(root),'architecture-worker',
        str(path),str(t['seconds']),str(source)]
    try:
        reservation.job_id=subprocess.check_output(command,text=True).strip().split(';')[0]
        reservation.state='PENDING'
    except subprocess.CalledProcessError:
        reservation.state='FAILED'; reservation.elapsed=0
        raise
    finally:
        state['reservations']=budget.serialize(); dump(out/'state.json',state)
    print(json.dumps(dict(submitted=reservation.job_id,task=t['id'],seconds=t['seconds'])),flush=True)
    return True


def make_pilots(root,profiles):
    tasks=[]
    for spec in read(root/'architecture/registry.json'):
        if spec['id'] not in profiles or not profiles[spec['id']].get('correctness'): continue
        # Four-direction scans enter only if cost leaves the official reserve feasible.
        if spec['id']=='scan4' and profiles[spec['id']]['step_p50_ms']>3*profiles['control']['step_p50_ms']:
            continue
        training=dict(id='A_'+spec['id']+'_20261025',config=spec['config'],seed=20261025,
            subset_seed=20261025,batch=8,workers=4,effective_batch=8,train_limit=8000,
            max_updates=5000,augment=True,precision='fp32',stage='exploratory',
            hypothesis=spec['hypothesis'],architecture_id=spec['id'])
        if spec.get('parent'):
            training.update(parent=spec['parent'],phase=spec['phase'],augment=False)
        # Estimate includes full validation every epoch plus a conservative I/O margin.
        estimate=5000*profiles[spec['id']]['step_p50_ms']/1000*2+600
        seconds=max(1800,min(7200,int((estimate+59)//60)*60))
        tasks.append(dict(id=training['id'],kind='train',phase='research',seconds=seconds,training=training))
    return tasks


def controller(root):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('CPU compute allocation required')
    out=root/'architecture'
    with (root/'active_controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(out/'state.json')
        budget=Budget(state['reservations'],phase_hours={'architecture_'+k:v for k,v in PROTECTED.items()})
        if state['phase'] in ('FAILED','BLOCKED','COMPLETED','RESEARCH_REVIEW'): return
        # At most three CPU slices; dependency survives a controller walltime.
        # A successor exits immediately if campaign has already reached a terminal gate.
        if state['phase']!='pilot':
            successors=state.setdefault('controller_successors',{})
            current_job=os.environ['SLURM_JOB_ID']
            if current_job not in successors and len(successors)<2:
                source=Path(__file__).resolve().parents[2]
                command=['sbatch','--parsable','--export=NONE','--partition=compute',
                    '--cpus-per-task=2','--mem=8G','--time=12:00:00',
                    '--dependency=afterany:'+current_job,'--job-name=mb-arch-control',
                    '--output='+str(out/'logs/controller-%j.out'),'--error='+str(out/'logs/controller-%j.err'),
                    str(source/'scripts/slurm/improve_continue.slurm'),str(root),'architecture-controller',
                    'unused','43200',str(source)]
                successors[current_job]=subprocess.check_output(command,text=True).strip().split(';')[0]
                dump(out/'state.json',state)
        if any(not r.job_id and not r.terminal for r in budget.records):
            raise RuntimeError('Unresolved reservation; reconcile Slurm before resuming')
        while True:
            update_accounting(budget)
            statuses={t['id']:task_status(root,t,budget) for t in state['tasks']}
            for t in state['tasks']:
                if statuses[t['id']] not in ('NOT STARTED','RESUME REQUIRED'): continue
                required=[statuses.get(parent,'NOT STARTED') for parent in t.get('requires',[])]
                if 'FAILED' in required:
                    dump(out/'tasks'/t['id']/'completed.json',dict(status='failed',reason='Required parent failed'))
                elif all(value=='VERIFIED' for value in required):
                    submit(root,t,budget,state)
            statuses={t['id']:task_status(root,t,budget) for t in state['tasks']}
            state['task_status']=statuses
            done=all(v in ('VERIFIED','FAILED') for v in statuses.values())
            if done and state['phase']=='profile':
                profile=out/'profile.json'
                if any(v=='FAILED' for v in statuses.values()) or not profile.exists():
                    state['phase']='FAILED'; state['reason']='GPU profile/correctness did not pass'
                else:
                    state['profile_tasks']=state['tasks']
                    state['tasks']=make_pilots(root,read(profile)['variants'])
                    state['phase']='pilot'
                    if not any(t['training']['architecture_id']=='control' for t in state['tasks']):
                        state['phase']='FAILED'; state['reason']='Matched control missing'
            elif done and state['phase']=='pilot':
                # Explicit review barrier: no automatic official training from pilot results.
                state['phase']='PILOT_REVIEW'
                state['reason']='Pilot artifacts ready for validation and modality analysis; no freeze yet'
            elif done and state['phase'] in ('research_followup','confirmation','recipe','local_confirmation',
                                             'combination','local_second_seed','modality_confirmation','verification','official','final'):
                advance(root,state,budget)
            state['reservations']=budget.serialize(); state['budget']=budget.summary()
            state['protected_remaining_gpu_hours']=protected_seconds(budget)/3600
            dump(out/'state.json',state)
            status=read(root/'status.json'); status.update(phase='architecture_'+state['phase'],
                budget=budget.summary(),architecture=dict(details='architecture/state.json',
                official_training=state.get('official_status','NOT STARTED'),task_status=statuses,
                protected_remaining_gpu_hours=state['protected_remaining_gpu_hours']),
                details='architecture/state.json',default_model_changed=False)
            dump(root/'status.json',status)
            if time.time()-state.get('_last_report_epoch',0)>=60:
                summarize(root); state['_last_report_epoch']=time.time()
            if state['phase'] in ('FAILED','PILOT_REVIEW','RESEARCH_REVIEW','COMPLETED','BLOCKED'): return
            ready=[t for t in state['tasks'] if statuses.get(t['id']) in ('NOT STARTED','RESUME REQUIRED')
                   and all(statuses.get(p)=='VERIFIED' for p in t.get('requires',[]))]
            if not any(not r.terminal for r in budget.records) and ready:
                state.update(phase='BLOCKED',reason='Budget cannot admit tasks while protecting official/final allocations')
                dump(out/'state.json',state)
                status.update(phase='architecture_BLOCKED'); dump(root/'status.json',status)
                return
            time.sleep(30)


def worker(root,path):
    import torch
    t=read(path); output=path.parent
    torch.set_num_threads(1)
    if t['kind']=='train':
        from .trial_runner import train
        if t.get('profile_required'):
            measured=read(root/'architecture/tasks'/t['profile_required']/'profile.json')['variants']
            if not measured[t['training']['architecture_id']].get('correctness'):
                raise ValueError('Required GPU architecture correctness did not pass')
        train(t['training'],root)
        result=root/'trials'/t['training']['id']/'completed.json'
        if not result.exists():
            dump(output/'completed.json',dict(status='incomplete',reason='Training slice needs explicitly budgeted resume'))
            return
        dump(output/'completed.json',read(result)); return
    if t['kind']=='profile':
        profile(root,specs=t.get('variants'),output=output/'profile.json' if t.get('variants') else None)
    elif t['kind']=='diagnostic':
        from .modality_diagnostics import run
        run(root,root/'trials'/t['checkpoint_trial']/'best.pt',output/'modality.json')
    elif t['kind']=='verify':
        verify_frozen(root,output)
    elif t['kind']=='final':
        final_evaluation(root,t,output)
    else: raise ValueError('Unknown architecture worker kind')
    dump(output/'completed.json',dict(status='complete'))


def training_task(root,identifier,config,*,seed,source,seconds=43200,phase='research',**extra):
    source=Path(source)
    training=dict(id=identifier,config=deepcopy(config),seed=seed,subset_seed=20261025,
        batch=8,workers=4,effective_batch=8,train_limit=None,max_updates=100000,
        augment=True,precision='fp32',stage='official' if phase=='official' else 'exploratory',
        architecture_id=identifier,source_sha256=sha(source/'models/keypoints/architecture_search.py'),
        hypothesis='Full-manifest confirmation of the selected config')
    training.update(extra)
    if phase=='official': training['selection_reference']=f'F_baseline_{seed}'
    return dict(id=identifier,kind='train',phase=phase,seconds=seconds,training=training,source_directory=str(source))


def successful_training(root,t,budget):
    if t is None: return None
    if t.get('kind')!='train' or task_status(root,t,budget)!='VERIFIED': return None
    path=root/'trials'/t['training']['id']/'completed.json'
    value=read(path)
    return value if value.get('best') is not None else None


def archive_stage(state):
    state.setdefault('stages',[]).append(dict(phase=state['phase'],tasks=state['tasks']))


def modality_benefit(value):
    """Predeclared diagnostic evidence, never a replacement for release gates."""
    comparisons=value.get('paired_comparisons',{})
    # Require depth benefit beyond zero using two interventions, not just nonzero gates.
    supported=[]
    for condition in ('depth_invalid','depth_features_zero','rgb_features_zero'):
        item=comparisons.get(condition,{})
        if any(s not in item or 'ci95' not in item[s] for s in ('OLD','NEW')): return False
        positive=any(item[s]['ci95']['mean'][0]>0 for s in ('OLD','NEW'))
        opposite=any(item[s]['ci95']['mean'][1]<0 for s in ('OLD','NEW'))
        supported.append(positive and not opposite)
    gradients=value.get('train_gradient_norms',{})
    return all(supported) and all(gradients.get(k,{}).get('finite') and
                                  gradients.get(k,{}).get('norm',0)>0 for k in ('rgb','depth'))


def advance(root,state,budget):
    """Evidence-gated transitions; final real-domain evaluation cannot feed search."""
    from .acceptance import acceptable
    out=root/'architecture'
    source=Path(state.get('execution_source',out/'followup/source'))
    current=state['phase']
    verified=[(t,successful_training(root,t,budget)) for t in state['tasks'] if t.get('kind')=='train']
    verified=[(t,r) for t,r in verified if r is not None]
    archive_stage(state)
    if current=='research_followup':
        # Each eligible finalist gets full 40K confirmation on two seeds, from RGB pretraining.
        low=[(t,r) for t,r in verified if t['training']['id'].startswith('A15_') and
             t['training']['architecture_id']!='control']
        if not low:
            state.update(phase='BLOCKED',reason='No successful architecture follow-up; official not started'); return
        low.sort(key=lambda item:item[1]['best']['final']['balanced_mean'])
        choices=[('selected',low[0][0]['training']['config'])]
        high=next(((t,r) for t,r in verified if t['training']['architecture_id']=='highres'),None)
        if high is not None:
            choices.append(('highres',high[0]['training']['config']))
        elif len(low)>1:
            choices.append(('alternate',low[1][0]['training']['config']))
        control=next(((t,r) for t,r in verified if t['training'].get('architecture_id')=='control'),None)
        if control is None:
            state.update(phase='RESEARCH_REVIEW',reason='Matched 15K control did not complete'); return
        promising=[(t,r) for t,r in low if r['best']['final']['balanced_mean']<control[1]['best']['final']['balanced_mean']]
        # Test interaction, not the assumption that individual benefits add.
        if len(promising)>=2:
            combo=deepcopy(control[0]['training']['config'])
            for t,r in promising[:2]:
                combo.setdefault('decoder',{}).update(t['training']['config'].get('decoder',{}))
            spec=dict(id='combined_decoder',config=combo,
                hypothesis='Interaction of the two isolated decoder changes at the same 15K update horizon')
            profile_task=dict(id='profile_combination',kind='profile',phase='research',seconds=1800,
                              source_directory=str(source),variants=[spec])
            trial=training_task(root,'A_combined_decoder_20261025',combo,seed=20261025,source=source,seconds=7200)
            trial['training'].update(train_limit=8000,max_updates=15000,architecture_id=spec['id'],hypothesis=spec['hypothesis'])
            trial.update(requires=[profile_task['id']],profile_required=profile_task['id'])
            state.update(phase='combination',tasks=[profile_task,trial],
                isolated_confirmation_choices=dict(choices),isolated_best=low[0][1]['best'],
                combination_control=control[1]['best'])
        else:
            queue_confirmation(root,state,choices,source)
    elif current=='combination':
        choices=list(state['isolated_confirmation_choices'].items())
        if verified:
            trial,result=verified[0]
            won=acceptable(result['best'],state['isolated_best']) and acceptable(result['best'],state['combination_control'])
            state['combination_accepted']=bool(won)
            if won: choices[0]=('combined',trial['training']['config'])
        else:
            state.update(phase='RESEARCH_REVIEW',reason='Combination trial failed; no inference about interaction benefit'); return
        queue_confirmation(root,state,choices,source)
    elif current=='confirmation':
        groups={}
        for t,r in verified:
            label=t['id'].rsplit('_',1)[0]
            groups.setdefault(label,[]).append((t,r))
        groups={k:v for k,v in groups.items() if {t['training']['seed'] for t,r in v}=={20261025,20261026}}
        if not groups:
            state.update(phase='BLOCKED',reason='No architecture completed both 40K exploratory seeds'); return
        chosen=min(groups.values(),key=lambda group:sum(r['best']['final']['balanced_mean'] for t,r in group)/2)
        if state.get('modality_repair_reference'):
            references=state['modality_repair_reference']
            if not all(acceptable(r['best'],read(root/'trials'/references[str(t['training']['seed'])]/'completed.json')['best'])
                       for t,r in chosen):
                state.update(phase='BLOCKED',reason='Validity-aware modality repair did not preserve per-source accuracy gates; no official training'); return
        state['selected_global_config']=chosen[0][0]['training']['config']
        state['selected_confirmation']={str(t['training']['seed']):t['training']['id'] for t,r in chosen}
        tasks=[]
        for seed in (20261025,20261026):
            parent=state['selected_confirmation'][str(seed)]
            for schedule in ('constant','cosine'):
                t=training_task(root,f'AR_{schedule}_{seed}',state['selected_global_config'],seed=seed,
                               source=source,seconds=7200,parent=parent,learning_rate=3e-5)
                t['training'].update(max_updates=10000,cosine_learning_rate=schedule=='cosine',
                                     min_learning_rate=3e-7,hypothesis='Matched low-LR continuation: constant versus cosine')
                tasks.append(t)
        state.update(phase='recipe',tasks=tasks)
    elif current=='recipe':
        if len(verified)!=len(state['tasks']):
            state.update(phase='RESEARCH_REVIEW',reason='Recipe comparison incomplete; failed jobs are not negative accuracy evidence'); return
        metrics={t['training']['id']:r['best'] for t,r in verified}
        cosine_wins=all(acceptable(metrics.get(f'AR_cosine_{seed}'),metrics.get(f'AR_constant_{seed}')) and
            acceptable(metrics.get(f'AR_cosine_{seed}'),read(root/'trials'/state['selected_confirmation'][str(seed)]/'completed.json')['best'])
            for seed in (20261025,20261026))
        state['recipe_cosine_finish']=bool(cosine_wins)
        state['global_parents']={str(seed):(f'AR_cosine_{seed}' if cosine_wins else state['selected_confirmation'][str(seed)])
                                 for seed in (20261025,20261026)}
        tasks=[]
        for core in ('ssm','conv'):
            config=deepcopy(state['selected_global_config'])
            config['local'].update(enabled=True,mode='residual',layout='one',embedding=True,
                                   feature_source='f1',core=core)
            t=training_task(root,'AL_'+core+'_20261025',config,seed=20261025,source=source,seconds=7200,
                            parent=state['global_parents']['20261025'])
            t['training'].update(phase='local',augment=False,max_updates=10000)
            tasks.append(t)
        state.update(phase='local_confirmation',tasks=tasks)
    elif current=='local_confirmation':
        if len(verified)!=len(state['tasks']):
            state.update(phase='RESEARCH_REVIEW',reason='Local comparison incomplete; cannot infer that Local is ineffective'); return
        reference=read(root/'trials'/state['global_parents']['20261025']/'completed.json')['best']
        options=[(t,r) for t,r in verified if acceptable(r['best'],reference,local=True)]
        if options:
            t,r=min(options,key=lambda x:x[1]['best']['final']['balanced_mean'])
            state['selected_local_config']=t['training']['config']
            state['local_parent_25']=t['id']
            second=training_task(root,'AL_selected_20261026',state['selected_local_config'],seed=20261026,
                source=source,seconds=7200,parent=state['global_parents']['20261026'])
            second['training'].update(phase='local',augment=False,max_updates=10000)
            state.update(phase='local_second_seed',tasks=[second])
        else:
            state['selected_local_config']=None
            queue_finalist_diagnostics(root,state,source)
    elif current=='local_second_seed':
        if not verified:
            state.update(phase='RESEARCH_REVIEW',reason='Second Local seed failed; not an accuracy-gate decision'); return
        reference=read(root/'trials'/state['global_parents']['20261026']/'completed.json')['best']
        if not verified or not acceptable(verified[0][1]['best'],reference,local=True):
            state['selected_local_config']=None
        queue_finalist_diagnostics(root,state,source)
    elif current=='modality_confirmation':
        from .modality_diagnostics import add_intervals
        proofs=[]
        for t in state['tasks']:
            path=out/'tasks'/t['id']/'modality.json'
            if task_status(root,t,budget)!='VERIFIED' or not path.exists():
                state.update(phase='BLOCKED',reason='Finalist modality diagnostic incomplete'); return
            add_intervals(path); proofs.append(modality_benefit(read(path)))
        if not all(proofs):
            if not state.get('modality_repair_attempted') and state['selected_global_config'].get('fusion_mode','legacy')!='validity':
                config=deepcopy(state['selected_global_config']); config['fusion_mode']='validity'
                state['modality_repair_reference']=state['selected_confirmation']
                state['modality_repair_attempted']=True
                state['selected_local_config']=None
                state.update(phase='confirmation',tasks=[training_task(root,f'AC_reliability_{seed}',config,
                    seed=seed,source=source,hypothesis='Repair weak depth utility with measured validity and explicit RGB self fallback')
                    for seed in (20261025,20261026)],modality_evidence=proofs)
                return
            state.update(phase='BLOCKED',reason='Depth benefit not established on both exploratory seeds; no official training',
                         modality_evidence=proofs); return
        config=state.get('selected_local_config') or state['selected_global_config']
        def elapsed(trial):
            return sum(r.elapsed for r in budget.records if r.trial=='architecture_'+trial)
        estimate=3*max(elapsed(parent) for parent in state['selected_confirmation'].values())
        if state['recipe_cosine_finish']:
            estimate+=3*max(elapsed(f'AR_cosine_{seed}') for seed in (20261025,20261026))
        if state.get('selected_local_config'):
            estimate+=3*max(elapsed(state['local_parent_25']),elapsed('AL_selected_20261026'))
        estimate*=1.25
        if estimate>PROTECTED['official']*3600:
            state.update(phase='BLOCKED',reason='Measured official-training estimate exceeds protected budget; reduce research and reallocate before freeze',
                         official_gpu_hours_estimate=estimate/3600); return
        freeze=dict(config=config,global_config=state['selected_global_config'],
            local=state.get('selected_local_config') is not None,cosine_finish=state['recipe_cosine_finish'],
            manifest_sha256=sha(root/'manifest50k.json'),source_sha256=read(source.parent/'source_manifest.json')['sha256'],
            source_directory=str(source),
            seeds=[20261025,20261026,20261027],global_updates=100000,finish_updates=10000,
            local_updates=10000,initialization='Same verified RGB pretrained; independently seeded remaining modules',
            local_fallback='If any official seed fails Local validation gate, all final seeds use their Global parents',
            selection='synthetic validation only',benchmark=dict(precision='fp32',batch=1,workers=2),
            eligibility='Learned RGB+geometry, adaptive fusion, SSM and demonstrated depth contribution',
            recipe='AdamW legacy, effective batch 8, locked augmentation; optional verified cosine finishing stage')
        freeze['official_gpu_hours_estimate']=estimate/3600
        dump(out/'freeze.json',freeze)
        # CPU suite runs on the allocated compute controller before GPU smoke.
        result=subprocess.run([sys.executable,'-m','pytest','-q',str(source/'tests'),
            '--junitxml='+str(out/'verification_cpu.xml')],cwd=source,
            stdout=(out/'verification_cpu.out').open('w'),stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(phase='BLOCKED',reason='Frozen-source full CPU correctness failed'); return
        state.update(phase='verification',tasks=[dict(id='verify_frozen',kind='verify',phase='verification',
                     seconds=1800,source_directory=str(source))])
    elif current=='verification':
        proof=out/'tasks/verify_frozen/verification.json'
        if not proof.exists() or task_status(root,state['tasks'][0],budget)!='VERIFIED' or not read(proof).get('passed'):
            state.update(phase='BLOCKED',reason='GPU-native verification did not pass'); return
        freeze=read(out/'freeze.json'); tasks=[]
        state['official_final_trials']={}
        for seed in freeze['seeds']:
            t=training_task(root,f'AO_global_{seed}',freeze['global_config'],seed=seed,source=source,phase='official')
            tasks.append(t); parent=t['id']
            if freeze['cosine_finish']:
                finish=training_task(root,f'AO_finish_{seed}',freeze['global_config'],seed=seed,source=source,
                    phase='official',seconds=7200,parent=parent,learning_rate=3e-5)
                finish['training'].update(max_updates=10000,cosine_learning_rate=True,min_learning_rate=3e-7)
                finish['requires']=[parent]; tasks.append(finish); parent=finish['id']
            if freeze['local']:
                local=training_task(root,f'AO_local_{seed}',freeze['config'],seed=seed,source=source,
                    phase='official',seconds=7200,parent=parent)
                local['training'].update(phase='local',augment=False,max_updates=10000)
                local['requires']=[parent]; tasks.append(local); parent=local['id']
            state['official_final_trials'][str(seed)]=parent
        state.update(phase='official',tasks=tasks,official_status='IN PROGRESS')
    elif current=='official':
        tasks=[]; locks={}
        frozen=read(out/'freeze.json'); use_local=frozen['local']
        if use_local:
            for seed,trial in state['official_final_trials'].items():
                candidate=next((t for t in state['tasks'] if t['id']==trial),None)
                result=successful_training(root,candidate,budget) if candidate is not None else None
                if result is None:
                    state.update(phase='BLOCKED',reason='Official Local training incomplete',official_status='FAILED'); return
                parent=candidate['training']['parent']
                if not acceptable(result['best'],read(root/'trials'/parent/'completed.json')['best'],local=True):
                    use_local=False
        state['official_local_accepted']=use_local
        for seed,trial in state['official_final_trials'].items():
            candidate=next((t for t in state['tasks'] if t['id']==trial),None)
            if candidate is None or successful_training(root,candidate,budget) is None:
                state.update(phase='BLOCKED',reason='Official training incomplete',official_status='FAILED'); return
            if frozen['local'] and not use_local:
                trial=candidate['training']['parent']
            locks[seed]=dict(trial=trial,checkpoint_sha256=sha(root/'trials'/trial/'best.pt'))
            tasks.append(dict(id='final_'+seed,kind='final',phase='final',seconds=3600,
                              checkpoint_trial=trial,seed=int(seed),source_directory=str(source)))
        dump(out/'final_lock.json',dict(checkpoints=locks,freeze_sha256=sha(out/'freeze.json'),
                                      manifest_sha256=sha(root/'manifest50k.json')))
        state.update(phase='final',tasks=tasks,official_status='COMPLETED')
    elif current=='final':
        if any(task_status(root,t,budget)!='VERIFIED' for t in state['tasks']):
            state.update(phase='BLOCKED',reason='Final evaluation incomplete'); return
        write_release(root,state)
        state.update(phase='COMPLETED',reason='Official training and locked final evaluation verified')


def queue_finalist_diagnostics(root,state,source):
    parents=(dict(zip(('20261025','20261026'),(state.get('local_parent_25'),'AL_selected_20261026')))
             if state.get('selected_local_config') else state['global_parents'])
    state.update(phase='modality_confirmation',tasks=[dict(id='finalist_modality_'+seed,kind='diagnostic',
        phase='research',seconds=1800,checkpoint_trial=parent,source_directory=str(source))
        for seed,parent in parents.items()])


def queue_confirmation(root,state,choices,source):
    state['tasks']=[training_task(root,f'AC_{name}_{seed}',config,seed=seed,source=source)
                    for name,config in choices for seed in (20261025,20261026)]
    state.update(phase='confirmation',confirmation_configs=dict(choices))


def verify_frozen(root,output):
    import torch
    from .model import RGBDKeypointModel
    from .checkpoints import save_checkpoint,load_checkpoint
    from .multisource import load_manifest,MultiSourceDataset
    from .trial_runner import loader,transfer,optimizer,final_maps
    from .local_losses import weighted_heatmap_loss
    freeze=read(root/'architecture/freeze.json')
    if sha(root/'manifest50k.json')!=freeze['manifest_sha256']: raise ValueError('Frozen manifest changed')
    torch.manual_seed(20261025); torch.cuda.manual_seed_all(20261025)
    manifest=load_manifest(root/'manifest50k.json')
    model=RGBDKeypointModel(freeze['config']).cuda()
    phase='joint' if model.local_enabled else 'global'
    model.set_phase(phase).train(); opt=optimizer(model,phase)
    batch=transfer(next(iter(loader(MultiSourceDataset(manifest,'train',model.config),2,2))))
    def step(net,optim):
        optim.zero_grad(set_to_none=True)
        stage=net.forward_stages(*(batch[k] for k in ('rgb','depth','K')))
        loss=weighted_heatmap_loss(final_maps(stage),batch['target'],batch['weights'])
        if not torch.isfinite(loss): raise ValueError('Nonfinite frozen smoke loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True); optim.step()
        return float(loss.detach())
    first=[step(model,opt) for _ in range(2)]
    path=output/'resume_smoke.pt'
    kwargs=dict(epoch=0,phase=phase,keypoint_names=manifest['keypoint_names'],
                split_ids={s:[r['uid'] for r in manifest[s]] for s in ('train','val','test')},metric=None,
                training={'stage':'verification'},extra={'manifest_sha256':freeze['manifest_sha256']})
    save_checkpoint(path,model,opt,**kwargs)
    expected_loss=step(model,opt)
    expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    restored,payload=load_checkpoint(path,restore_rng=True); restored.cuda().set_phase(phase).train()
    restored_opt=optimizer(restored,phase); restored_opt.load_state_dict(payload['optimizer'])
    actual_loss=step(restored,restored_opt)
    maximum=0.
    for name,value in restored.state_dict().items():
        reference=expected[name].to(value.device)
        if value.is_floating_point():
            maximum=max(maximum,float((value-reference).abs().max()))
            torch.testing.assert_close(value,reference,atol=1e-5,rtol=1e-4)
        elif not torch.equal(value,reference): raise ValueError('Resume integer-state mismatch')
    restored.eval(); final_path=output/'roundtrip_smoke.pt'
    save_checkpoint(final_path,restored,restored_opt,**kwargs)
    again,_=load_checkpoint(final_path); again.cuda().eval()
    with torch.no_grad():
        args=tuple(batch[k] for k in ('rgb','depth','K'))
        left=final_maps(restored.forward_stages(*args)); right=final_maps(again.forward_stages(*args))
        torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
        bad=[]
        for value in (0.,float('nan'),float('inf')):
            pred=final_maps(again.forward_stages(args[0],torch.full_like(args[1],value),args[2]))
            if not torch.isfinite(pred).all(): raise ValueError('Invalid-depth frozen prediction is not finite')
            bad.append(pred)
        if not all(torch.equal(bad[0],value) for value in bad[1:]):
            raise ValueError('Invalid depth encodings do not share a deterministic fallback')
    dump(output/'verification.json',dict(passed=True,freeze_sha256=sha(root/'architecture/freeze.json'),
        smoke_losses=first,uninterrupted_loss=expected_loss,resumed_loss=actual_loss,
        resume_max_parameter_difference=maximum,resume_tolerance={'atol':1e-5,'rtol':1e-4},
        roundtrip_max_heatmap_difference=float((left-right).abs().max()),
        invalid_depth_finite=True,invalid_encoding_parity=True,gpu=torch.cuda.get_device_name(),
        note='Frozen full architecture, real synthetic train batch, native CUDA; no historical weights changed'))


def final_evaluation(root,t,output):
    import torch
    from .checkpoints import load_checkpoint
    from .multisource import MultiSourceDataset,load_manifest
    from .trial_runner import evaluate
    from .trial_validation import real_evaluation,benchmark
    out=root/'architecture'; lock=read(out/'final_lock.json')
    checkpoint=root/'trials'/t['checkpoint_trial']/'best.pt'
    if (lock['manifest_sha256']!=sha(root/'manifest50k.json') or
        lock['checkpoints'][str(t['seed'])]['checkpoint_sha256']!=sha(checkpoint)):
        raise ValueError('Final checkpoint/manifest differs from lock')
    model,_=load_checkpoint(checkpoint); model.cuda().eval()
    manifest=load_manifest(root/'manifest50k.json')
    synthetic=evaluate(model,MultiSourceDataset(manifest,'test',model.config,diagnostics=True),
                       records_path=output/'synthetic_predictions.json')
    from .regression import geometry_metrics
    predictions=read(output/'synthetic_predictions.json')['records']
    synthetic['mm']={source:geometry_metrics([r for r in predictions if r['source']==source]) for source in ('OLD','NEW')}
    real=real_evaluation(model,root)
    torch.cuda.empty_cache()
    timing=benchmark(model,MultiSourceDataset(manifest,'val',model.config,inference_only=True),1,2)
    gpu=torch.cuda.get_device_name()
    del model; torch.cuda.empty_cache()
    baseline_path=root/'trials'/f'F_baseline_{t["seed"]}'/'best.pt'
    baseline,_=load_checkpoint(baseline_path); baseline.cuda().eval()
    baseline_timing=benchmark(baseline,MultiSourceDataset(manifest,'val',baseline.config,inference_only=True),1,2)
    historical=read(root/'trials'/f'E_F_baseline_{t["seed"]}'/'metrics.json')
    dump(output/'metrics.json',dict(synthetic_test=synthetic,D455=real,checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),gpu=gpu,baseline_checkpoint_sha256=sha(baseline_path),
        baseline_metrics=historical,benchmark=timing,baseline_benchmark=baseline_timing,
        protocol='Final only after lock; D455 observed session; inference reader/prefetch applied to both models'))


def write_release(root,state):
    import statistics
    out=root/'architecture'; reasons=[]; rows=[]
    for t in state['tasks']:
        value=read(out/'tasks'/t['id']/'metrics.json')
        old=value['baseline_metrics']; new=value
        for source in ('OLD','NEW'):
            c,b=new['synthetic_test']['final'][source],old['synthetic_test']['final'][source]
            if c['PCK05']<b['PCK05'] or c['p95']>b['p95']:
                reasons.append(f'{t["seed"]}: synthetic {source} PCK/P95 gate failed')
        c,b=new['D455']['pixel'],old['D455']['pixel']
        if not (c['mean']<b['mean'] and c['p95']<=b['p95'] and c['PCK05']>=b['PCK05']):
            reasons.append(f'{t["seed"]}: final D455 mean/P95/PCK gate failed')
        latency=lambda bench:statistics.median(r['batch_p50_ms'] for r in bench['gpu_ready'])
        memory=lambda bench:max(r['allocated_bytes'] for r in bench['gpu_ready'])
        if old['gpu']!=new['gpu']: reasons.append(f'{t["seed"]}: historical accuracy reference hardware differs')
        if latency(new['benchmark'])>latency(new['baseline_benchmark']) or memory(new['benchmark'])>memory(new['baseline_benchmark']):
            reasons.append(f'{t["seed"]}: matched GPU latency/VRAM release gate failed')
        rows.append(dict(seed=t['seed'],checkpoint=new['checkpoint'],D455=new['D455']['pixel'],
                         synthetic_test=new['synthetic_test']['final'],benchmark=new['benchmark']))
    release=dict(accepted=not reasons,reasons=reasons,rows=rows,default_model_changed=False,
        official_training='COMPLETED',policy='Locked legacy release gates; final outcomes never feed architecture search')
    dump(out/'release.json',release)
    if not (out/'historical_release.json').exists(): dump(out/'historical_release.json',read(root/'release.json'))
    dump(root/'release.json',release)
    state['release_accepted']=release['accepted']


def prepare_followup(root):
    """Evidence-based next stage; retains initial tasks and their frozen source."""
    out=root/'architecture'; state=read(out/'state.json')
    if state['phase']!='PILOT_REVIEW': raise ValueError('Initial pilots must finish before follow-up')
    suites=ET.parse(out/'pre_followup_correctness.xml').getroot()
    suites=[suites] if suites.tag=='testsuite' else list(suites)
    if not suites or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):
        raise ValueError('Current full CPU correctness required before follow-up snapshot')
    budget=Budget(state['reservations']); update_accounting(budget)
    if any(not r.terminal for r in budget.records): raise ValueError('Active GPU jobs remain')
    summarize(root)
    summary=read(out/'pilot_summary.json')
    valid=[r for r in summary['rows'] if r['status'] in TERMINAL_OK and r['best'] is not None]
    control=next((r for r in valid if r['architecture_id']=='control'),None)
    if control is None: raise ValueError('No successful matched control')
    globals_=sorted((r for r in valid if r['phase']=='global' and r['architecture_id']!='control'),
                    key=lambda r:(r['best']['final']['balanced_mean'],r['best']['final']['balanced_p95']))
    selected=globals_[:2]
    follow=out/'followup'; follow.mkdir(exist_ok=True)
    snapshot(follow)
    source=str(follow/'source'); source_hash=read(follow/'source_manifest.json')['sha256']
    by_id={t['id']:t for t in state['tasks']}; tasks=[]
    for row in [control,*selected]:
        original=by_id[row['id']]['training']
        training={**original,'id':original['id'].replace('A_','A15_',1),
                  'resume_trial':original['id'],'max_updates':15000,'source_sha256':source_hash}
        tasks.append(dict(id=training['id'],kind='train',phase='research',seconds=3600,
                          source_directory=source,training=training))
    checkpoints=['F_candidate_20261025',*[r['id'] for r in selected]]
    local_success=[r for r in valid if r['phase']=='local' and r['admissible']]
    if local_success:
        local=min(local_success,key=lambda r:r['best']['final']['balanced_mean'])
        original=by_id[local['id']]['training']
        training={**original,'id':'A_local_second_20261026','seed':20261026,
                  'parent':'F_candidate_20261026','source_sha256':source_hash}
        tasks.append(dict(id=training['id'],kind='train',phase='research',seconds=1800,
                          source_directory=source,training=training))
        checkpoints.append(local['id'])
    for checkpoint in checkpoints:
        tasks.append(dict(id='modality_'+checkpoint,kind='diagnostic',phase='research',seconds=1200,
                          checkpoint_trial=checkpoint,source_directory=source))
    base=next(v for v in read(out/'registry.json') if v['id']=='control')
    high=deepcopy(base); high.update(id='highres',hypothesis='Double input and heatmap resolution with the same input-space Gaussian width')
    high['config'].update(input_size=[360,640],heatmap_size=[180,320],target_sigma=4.)
    tasks.append(dict(id='profile_highres',kind='profile',phase='research',seconds=1800,
                      source_directory=source,variants=[high]))
    training=dict(id='A_highres_20261025',config=high['config'],seed=20261025,subset_seed=20261025,
        batch=8,workers=4,effective_batch=8,train_limit=8000,max_updates=5000,augment=True,
        precision='fp32',stage='exploratory',hypothesis=high['hypothesis'],architecture_id='highres',source_sha256=source_hash)
    tasks.append(dict(id=training['id'],kind='train',phase='research',seconds=3600,
        source_directory=source,training=training,requires=['profile_highres'],profile_required='profile_highres'))
    state['initial_tasks']=state['tasks']; state['tasks']=tasks; state['phase']='research_followup'
    state['execution_source']=source
    state['followup_selection']=dict(control=control['id'],selected=[r['id'] for r in selected],
        rule='Top two global pilot balanced validation mean, P95 tie-break; research only, not release acceptance',
        source_sha256=source_hash)
    state['reservations']=budget.serialize(); dump(out/'state.json',state)
    dump(follow/'selection.json',state['followup_selection'])


def summarize(root):
    """CPU aggregation of synthetic worker artifacts, without loading weights."""
    from .acceptance import acceptable
    out=root/'architecture'; state=read(out/'state.json')
    rows=[]; task_index={}
    for group in [state.get('initial_tasks',[]),*[s['tasks'] for s in state.get('stages',[])],state['tasks']]:
        for t in group: task_index[t['id']]=t
    for t in task_index.values():
        if t.get('kind')!='train': continue
        trial=t['training']; folder=root/'trials'/trial['id']
        result=read(folder/'completed.json') if (folder/'completed.json').exists() else {}
        progress=read(folder/'progress.json') if (folder/'progress.json').exists() else {}
        history=folder/'history.jsonl'
        values=[json.loads(line) for line in history.read_text().splitlines() if line.strip()] if history.exists() else []
        best=result.get('best') or (min((r['validation'] for r in values),key=lambda r:r['final']['balanced_mean']) if values else None)
        rows.append(dict(id=trial['id'],architecture_id=trial['architecture_id'],
                         status=result.get('status',state.get('task_status',{}).get(t['id'],'NOT STARTED')),
                         updates=result.get('updates',progress.get('updates',0)),
                         samples_seen=result.get('updates',progress.get('updates',0))*trial['effective_batch'],
                         best=best,best_epoch=min(values,key=lambda r:r['validation']['final']['balanced_mean'])['epoch'] if values else None,
                         phase=trial.get('phase','global'),hypothesis=trial['hypothesis']))
    for row in rows:
        task=task_index[row['id']]['training']; seed=task.get('seed',20261025)
        if row['phase']=='local': control=task['parent']
        elif task.get('selection_reference'): control=task['selection_reference']
        elif row['id'].startswith('AR_cosine_'): control='AR_constant_'+str(seed)
        elif row['id'].startswith('AR_constant_'): control=task['parent']
        elif task.get('train_limit')==8000:
            control='A15_control_20261025' if task['max_updates']==15000 else 'A_control_20261025'
        else: control='F_candidate_'+str(seed)
        path=root/'trials'/control/'completed.json'
        reference=read(path).get('best') if path.exists() else None
        row['control_trial']=control
        finished=row['status'] in TERMINAL_OK and reference is not None
        row['admissible']=acceptable(row['best'],reference,local=row['phase']=='local') if finished else None
    dump(out/'pilot_summary.json',dict(rows=rows,policy='synthetic validation only; no architecture freeze from single-seed pilot'))
    lines=['<!-- architecture-search-20261026 -->','## Architecture search RGB-D — kết quả thực thi',
           '',f'Trạng thái: **{state["phase"]}**. Official training: **{state.get("official_status","NOT STARTED")}**; default baseline giữ nguyên.',
           '',f'GPU-giờ đã cấp: {state["budget"]["allocated_gpu_hours"]:.4f}/288; '
           f'đã dùng + reservation: {state["budget"]["charged_or_reserved_gpu_hours"]:.4f}. '
           'Bảo vệ official/verification/final/recovery: 136 GPU-giờ.',
           '', '| Trial | Updates | OLD mean/P95/PCK05 | NEW mean/P95/PCK05 | Gate validation / reference |',
           '|---|---:|---|---|---|']
    for row in rows:
        def cell(source):
            if row['best'] is None:return 'chưa có'
            m=row['best']['final'][source]
            return f'{m["mean"]:.4f} / {m["p95"]:.4f} / {100*m["PCK05"]:.4f}%'
        gate=('control' if row['id']==row['control_trial'] else 'pending' if row['admissible'] is None else str(row['admissible']))
        gate+=' / '+row['control_trial']
        lines.append(f'| {row["id"]} | {row["updates"]} | {cell("OLD")} | {cell("NEW")} | {gate} |')
    lines.extend(['','Stage/initialization/train subset đầy đủ trong immutable task JSON; pilot 8K, confirmation/official 40K. '
        'Local so với đúng Global parent đóng băng; không so trực tiếp Local parent 100K với control pilot 5K.',
        'GPU profile trong `architecture/profile.json`: inclusive component timing, không cộng các block lồng nhau. '
        'Các bước profile tạm thời trên một batch không chứng minh accuracy hoặc time-to-convergence.',
        'Registry/config/source snapshot và GPU accounting nằm trong `architecture/`; không dùng test/D455 để chọn biến thể.'])
    report=root/'REPORT.md'; old=report.read_text(); marker='<!-- architecture-search-20261026 -->'
    if marker in old: old=old.split(marker)[0].rstrip()+'\n'
    report.write_text(old.rstrip()+'\n\n'+'\n'.join(lines)+'\n')


def refresh_controller(root):
    """Replace only future orchestration, preserving every active task snapshot."""
    out=root/'architecture'
    with (root/'active_controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        suites=ET.parse(out/'controller_correctness.xml').getroot()
        suites=[suites] if suites.tag=='testsuite' else list(suites)
        if not suites or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):
            raise ValueError('Current controller correctness is not verified')
        state=read(out/'state.json'); budget=Budget(state['reservations'])
        update_accounting(budget)
        if any(not r.job_id and not r.terminal for r in budget.records):
            raise ValueError('Resolve pending submission before source migration')
        review=out/('controller_review_'+os.environ['SLURM_JOB_ID'])
        review.mkdir(exist_ok=True); snapshot(review)
        state.setdefault('controller_source_history',[]).append(state.get('execution_source',str(out/'followup/source')))
        state['execution_source']=str(review/'source')
        state['reservations']=budget.serialize(); state['budget']=budget.summary()
        # Prior dependent CPU successors were explicitly canceled before migration.
        state['previous_controller_successors']=state.pop('controller_successors',{})
        dump(out/'state.json',state)


def profile(root,specs=None,output=None):
    import numpy as np
    import torch
    from .config import make_config
    from .model import RGBDKeypointModel
    from .checkpoints import load_checkpoint
    from .multisource import load_manifest,MultiSourceDataset
    from .trial_runner import loader,transfer,optimizer,final_maps
    from .local_losses import weighted_heatmap_loss
    out=root/'architecture'; results={}
    manifest=load_manifest(root/'manifest50k.json')
    for spec in (read(out/'registry.json') if specs is None else specs):
        torch.manual_seed(20261025); torch.cuda.manual_seed_all(20261025)
        config=make_config(spec['config']); model=RGBDKeypointModel(config)
        if spec.get('parent'):
            parent,payload=load_checkpoint(root/'trials'/spec['parent']/'best.pt')
            incoming=payload['model_state']; expected=model.state_dict()
            missing=set(expected)-set(incoming)
            if set(incoming)-set(expected) or any(not k.startswith('residual_local.') for k in missing):
                raise ValueError('Unsafe Local profile parent transfer')
            model.load_state_dict({**expected,**incoming},strict=True); del parent,payload
        model.cuda().set_phase(spec.get('phase','global')).train()
        data=MultiSourceDataset(manifest,'train',config)
        batch=transfer(next(iter(loader(data,8,2))))
        opt=optimizer(model,spec.get('phase','global'))
        times=[]; gradient={}; torch.cuda.reset_peak_memory_stats()
        for step in range(25):
            torch.cuda.synchronize(); start=time.perf_counter(); opt.zero_grad(set_to_none=True)
            stage=model.forward_stages(*(batch[k] for k in ('rgb','depth','K')))
            loss=weighted_heatmap_loss(final_maps(stage),batch['target'],batch['weights'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite profile loss '+spec['id'])
            loss.backward()
            if step==24:
                for name,module in [('rgb',model.rgb_encoder),('depth',model.geometry_encoder),
                                    ('fusion',model.fusion),('decoder',model.decoder),('local',model.residual_local)]:
                    if module is None: continue
                    grads=[p.grad.detach().float() for p in module.parameters() if p.grad is not None]
                    if any(not torch.isfinite(g).all() for g in grads): raise ValueError('Nonfinite gradients')
                    gradient[name]=float(torch.stack([g.norm().square() for g in grads]).sum().sqrt()) if grads else 0.
            opt.step(); torch.cuda.synchronize()
            if step>=5: times.append((time.perf_counter()-start)*1000)
        step_peak=torch.cuda.max_memory_allocated()
        # Inclusive event timings identify costly blocks, not an additive cost partition.
        model.eval(); events={}; handles=[]; component={}
        modules={'rgb_encoder':model.rgb_encoder,'geometry':model.geometry_encoder,
                 'decoder':model.decoder,'head':model.head}
        if hasattr(model.fusion,'blocks'):
            modules.update({f'fusion_{i}':block for i,block in enumerate(model.fusion.blocks)})
        for name,module in modules.items():
            events[name]=[]
            def begin(m,args,n=name):
                start=torch.cuda.Event(enable_timing=True); start.record(); events[n].append([start,None])
            def end(m,args,value,n=name):
                stop=torch.cuda.Event(enable_timing=True); stop.record(); events[n][-1][1]=stop
            handles.extend((module.register_forward_pre_hook(begin),module.register_forward_hook(end)))
        with torch.no_grad():
            for _ in range(10): model.forward_stages(*(batch[k][:1] for k in ('rgb','depth','K')))
        torch.cuda.synchronize()
        for name,pairs in events.items(): component[name]=float(np.median([a.elapsed_time(b) for a,b in pairs]))
        for handle in handles: handle.remove()
        with torch.no_grad():
            bad=tuple(batch[k][:1] for k in ('rgb','depth','K'))
            fallback=[]
            for invalid in (0.,float('nan'),float('inf')):
                stage=model.forward_stages(bad[0],torch.full_like(bad[1],invalid),bad[2])
                fallback.append(final_maps(stage).detach())
            finite=all(bool(torch.isfinite(x).all()) for x in fallback)
            identical=all(torch.equal(fallback[0],x) for x in fallback[1:])
        results[spec['id']]=dict(correctness=finite and identical,gradient_norms=gradient,
            step_p50_ms=float(np.median(times)),step_p95_ms=float(np.percentile(times,95)),
            train_images_per_second=8/(float(np.mean(times))/1000),step_allocated_bytes=step_peak,
            parameters=sum(p.numel() for p in model.parameters()),component_p50_ms=component,
            all_invalid_finite=finite,invalid_encoding_parity=identical,
            flops=None,flops_note='Selective-scan operation count not provided by this timing profile',
            phase=spec.get('phase','global'),note='25 temporary optimization steps on one train batch; no accuracy claim')
        dump(output or out/'profile.json',dict(variants=results,gpu=torch.cuda.get_device_name(),
            manifest_sha256=sha(root/'manifest50k.json')))
        del model,opt,loss,stage,batch,fallback
        torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('role',choices=['initialize','controller','worker','summarize','prepare-followup','refresh-controller'])
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--task',type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Architecture jobs require Slurm')
    if args.role=='initialize': initialize(args.root)
    elif args.role=='controller': controller(args.root)
    elif args.role=='summarize': summarize(args.root)
    elif args.role=='prepare-followup': prepare_followup(args.root)
    elif args.role=='refresh-controller': refresh_controller(args.root)
    else: worker(args.root,args.task)


if __name__=='__main__': main()
