"""Controller restart/fault cases and the complete dependency graph, without a scheduler."""
from copy import deepcopy
import json
from pathlib import Path
import time
import pytest
import torch
from torch import nn
from models.keypoints import gdl_search as search
from models.keypoints import gdl_recovery as recovery
from models.keypoints.campaign import dump,read,sha


def confirmed_parents(root):
    parents={}
    for seed in search.SEEDS:
        path=root/f'parent_{seed}.pt';path.write_text('verified weights')
        parents[str(seed)]=dict(trial=f'parent_{seed}',checkpoint=str(path),seed=seed,
            config=search.base_config(root),metrics=pilot_metric(1.))
    return parents


def pilot_metric(mean,p95=2.,pck=1.):
    return dict(final={**{s:dict(mean=mean,p95=p95,PCK05=pck) for s in ('OLD','NEW')},
                       'balanced_mean':mean,'balanced_p95':p95})


@pytest.mark.parametrize('phase',['decoder_pilot','local_pilot'])
def test_parallel_pilots_use_own_seed_parent_and_preserve_existing_tasks(tmp_path,phase):
    parents=confirmed_parents(tmp_path)
    state=dict(phase=phase,global_parents=parents,gd_parents=parents,pilot_seeds=[search.SEEDS[0]])
    original=search.stage_pilot_tasks(tmp_path,state,phase)
    original[0]['source_directory']='original immutable snapshot'
    before=deepcopy(original)
    state.update(tasks=original,all_tasks=list(original),pilot_seeds=list(search.SEEDS))
    added=search.expand_parallel_pilots(tmp_path,state)
    assert len(added)==3 and len(state['tasks'])==6
    assert state['tasks'][:3]==before
    assert search.expand_parallel_pilots(tmp_path,state)==[]
    assert len({t['id'] for t in state['all_tasks']})==6
    for task in state['tasks']:
        tr=task['training'];parent=parents[str(tr['seed'])]
        assert tr['parent_checkpoint']==parent['checkpoint'] and tr['parent']==parent['trial']
        assert tr['max_updates']==15000 and tr['train_limit']==8000
        assert tr['subset_seed']==20261025 and tr['validation_interval']==1000
        assert tr['parent_checkpoint_sha256']==sha(parent['checkpoint'])
        if phase=='local_pilot':
            assert read(tr['reference_metrics'])==parent['metrics'] and tr['selection_local']


def test_decoder_pilot_requires_each_seed_gate_before_average_ranking(tmp_path,monkeypatch):
    state=dict(global_parents=confirmed_parents(tmp_path),pilot_seeds=list(search.SEEDS))
    tasks=search.stage_pilot_tasks(tmp_path,state,'decoder_pilot')
    def selected(root,task):
        tr=task['training'];d=tr['config']['decoder']['kind'];seed=tr['seed']
        # Different control metrics ensure comparison never mixes seeds.
        mean={('d0',20261025):1.,('d0',20261026):2.,('d1',20261025):.8,
              ('d1',20261026):1.8,('d2',20261025):.5,('d2',20261026):1.}[d,seed]
        p95=3. if (d,seed)==('d2',20261026) else 2.
        return dict(trial=task['id'],config=tr['config'],metrics=pilot_metric(mean,p95))
    def admissible(root,task,control):
        assert control['trial']==f"GDL_B_d0_{task['training']['seed']}"
        parent=selected(root,task)
        return parent if search.acceptable(parent['metrics'],control['metrics']) else None
    monkeypatch.setattr(search,'selected',selected);monkeypatch.setattr(search,'admissible_pilot',admissible)
    valid=search.review_stage_pilots(tmp_path,state,tasks,'decoder_pilot')
    assert [p['config']['decoder']['kind'] for p in valid]==['d1']
    assert valid[0]['rank'][0]==pytest.approx(1.3)
    assert [d['gate_passed'] for d in state['decoder_pilot_decisions']]==[True,True,True,False]


def test_local_pilots_require_both_seeds_and_ignore_missing_optional_candidate(tmp_path,monkeypatch):
    state=dict(gd_parents=confirmed_parents(tmp_path),pilot_seeds=list(search.SEEDS))
    tasks=search.stage_pilot_tasks(tmp_path,state,'local_pilot')
    # Missing optional seed is a failed candidate, never single-seed admission.
    tasks=[t for t in tasks if t['id']!='GDL_C_ssm_20261026']
    def selected(root,task):
        tr=task['training'];core=tr['config']['local']['core']
        if core=='hybrid' and tr['seed']==20261026:return None
        return dict(trial=task['id'],config=tr['config'],metrics=pilot_metric(.9))
    monkeypatch.setattr(search,'selected',selected)
    valid=search.review_stage_pilots(tmp_path,state,tasks,'local_pilot')
    assert [p['config']['local']['core'] for p in valid]==['conv']
    assert len(state['local_pilot_decisions'])==6


@pytest.mark.parametrize('count,expected',[(5,True),(6,False)])
def test_authorized_sixth_gpu_submits_but_seventh_is_blocked(tmp_path,monkeypatch,count,expected):
    source=tmp_path/'source';source.mkdir();dump(tmp_path/'source_manifest.json',dict(files={}))
    state=dict(max_campaign_gpus=6,source=str(source),ledger=[])
    commands=[]
    def output(command):
        commands.append(command)
        return ('1|gres/gpu:nvidia_rtx_a5000:'+str(count)+'|RUNNING\n' if command[0]=='squeue' else '9001\n')
    monkeypatch.setattr(search,'slurm_output',output)
    monkeypatch.setattr(search,'calibration_seconds',lambda *a:900)
    search.submit(tmp_path,state,dict(id='ready',kind='profile',gpu=True))
    assert any(c[0]=='sbatch' for c in commands)==expected
    assert len(state['ledger'])==int(expected)


def test_ledger_reservation_prevents_oversubmission_during_queue_lag(tmp_path,monkeypatch):
    state=dict(max_campaign_gpus=6,ledger=[dict(gpu=True,state='PENDING',job_id=str(i)) for i in range(6)])
    calls=[]
    monkeypatch.setattr(search,'slurm_output',lambda command:calls.append(command) or '')
    search.submit(tmp_path,state,dict(id='seventh',gpu=True))
    assert len(calls)==1 and calls[0][0]=='squeue'


def test_gpu_device_count_includes_typed_untyped_and_pending_jobs():
    queue='1|gres/gpu:nvidia_rtx_a5000:2|RUNNING\n2|gpu:3|PENDING\n3|N/A|RUNNING\n4|gres/gpu:1(S:0)|RUNNING\n'
    assert search.requested_gpus(queue)==6
    assert search.campaign_gpu_limit({})==4
    with pytest.raises(ValueError):search.campaign_gpu_limit(dict(max_campaign_gpus=7))


def test_admissible_checkpoint_selection_survives_multiple_validation_steps(tmp_path,monkeypatch):
    from models.keypoints import trial_runner as runner
    from models.keypoints.model import StageOutput
    from models.keypoints.config import make_config
    config=make_config(dict(input_size=(4,4),heatmap_size=(2,2),num_keypoints=1,local=dict(enabled=False)))
    class Model(nn.Module):
        def __init__(self,cfg):
            super().__init__();self.config=cfg;self.head=nn.Conv2d(1,1,1)
        def cuda(self):return self
        def set_phase(self,phase):return self
        def _global_modules(self):return (self.head,)
        def forward_stages(self,rgb,depth,K):
            maps=self.head(torch.ones(len(rgb),1,2,2));return StageOutput(maps,None,maps)
    class Dataset:
        def __init__(self,*a,**kw):self.epoch=0
    class Process:
        def __init__(self,*a,**kw):pass
        def terminate(self):pass
        def wait(self,**kw):pass
    def metrics(mean):
        return dict(final={**{s:dict(mean=mean,p95=3.,PCK05=1.) for s in ('OLD','NEW')},
                           'balanced_mean':mean,'balanced_p95':3.})
    reference=tmp_path/'reference.json';dump(reference,metrics(2.))
    batch=dict(rgb=torch.ones(1,3,4,4),depth=torch.ones(1,1,4,4),K=torch.eye(3)[None],
        target=torch.ones(1,1,2,2),weights=torch.ones(1,1),points=torch.zeros(1,1,2))
    scores=iter([metrics(2.),metrics(1.9),metrics(1.8),metrics(1.7)])
    monkeypatch.setattr(runner,'RGBDKeypointModel',Model)
    monkeypatch.setattr(runner,'MultiSourceDataset',Dataset)
    monkeypatch.setattr(runner,'load_manifest',lambda *a:dict(train=[{}],val=[],test=[]))
    monkeypatch.setattr(runner,'digest',lambda *a:'manifest')
    monkeypatch.setattr(runner,'loader',lambda *a,**kw:[batch,batch])
    monkeypatch.setattr(runner,'transfer',lambda value:value)
    monkeypatch.setattr(runner,'evaluate',lambda *a,**kw:next(scores))
    monkeypatch.setattr(runner,'checkpoint',lambda path,*a:path.write_text('saved'))
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    runner.train(dict(id='repeat_reference',config=config,seed=1,max_updates=3,batch=1,effective_batch=1,
                     workers=0,validation_interval=1,reference_metrics=str(reference)),tmp_path)
    value=read(tmp_path/'trials/repeat_reference/completed.json')
    assert value['updates']==3 and value['best_admissible']['final']['balanced_mean']==1.7


def test_recover_accepted_submission_without_sending_duplicate(tmp_path,monkeypatch):
    value=recovery.token(tmp_path,'trial',1)
    state=dict(ledger=[dict(task='trial',submission_token=value,state='RESERVED',created_epoch=time.time())])
    calls=[]
    def output(command):
        calls.append(command)
        return '8123|'+value+'\n'
    monkeypatch.setattr(recovery,'output',output)
    recovery.recover_submissions(tmp_path,state)
    recovery.recover_submissions(tmp_path,state)
    assert state['ledger'][0]['job_id']=='8123'
    assert state['ledger'][0]['recovered_submission']
    assert not any(command[0]=='sbatch' for command in calls)


def test_unknown_submission_waits_for_accounting_before_retry(tmp_path,monkeypatch):
    row=dict(task='trial',submission_token='gdl-receipt',state='SUBMISSION_UNKNOWN',created_epoch=time.time())
    state=dict(ledger=[row]);monkeypatch.setattr(recovery,'output',lambda command:'')
    recovery.recover_submissions(tmp_path,state);assert row['state']=='SUBMISSION_UNKNOWN'
    row['created_epoch']-=181
    recovery.recover_submissions(tmp_path,state);assert row['state']=='SUBMIT_FAILED'


@pytest.mark.parametrize('slurm_state',['COMPLETED','TIMEOUT','PREEMPTED','NODE_FAIL'])
def test_slices_resume_from_verified_progress_without_shortening_horizon(tmp_path,monkeypatch,slurm_state):
    trial=tmp_path/'trials/long';trial.mkdir(parents=True);(trial/'last.pt').write_text('checkpoint')
    task=dict(id='long',kind='train',training=dict(max_updates=100000))
    state=dict(ledger=[dict(task='long',state=slurm_state,start_updates=2000)]*5)
    monkeypatch.setattr('models.keypoints.gdl_integrity.audit_training',
        lambda *a,**kw:dict(checkpoints={'last':{'updates':4000}}))
    assert search.task_status(tmp_path,state,task)=='READY'
    assert task['training']['max_updates']==100000 and task['resume_verified_updates']==4000


def test_corrupt_resume_checkpoint_fails_closed(tmp_path,monkeypatch):
    folder=tmp_path/'trials/bad';folder.mkdir(parents=True);(folder/'last.pt').write_text('truncated')
    state=dict(ledger=[dict(task='bad',state='TIMEOUT')])
    def corrupt(*a,**kw):raise ValueError('truncated checkpoint')
    monkeypatch.setattr('models.keypoints.gdl_integrity.audit_training',corrupt)
    assert search.task_status(tmp_path,state,dict(id='bad',kind='train'))=='FAILED'
    assert 'truncated' in state['artifact_errors']['bad']


def test_manual_cancellation_is_not_automatically_restarted(tmp_path):
    state=dict(ledger=[dict(task='manual',state='CANCELLED')])
    assert search.task_status(tmp_path,state,dict(id='manual',kind='train'))=='FAILED'


def test_pending_gpu_has_no_allocated_tres_yet(monkeypatch):
    state=dict(ledger=[dict(task='queued',job_id='1',state='PENDING',gpu=True,seconds=3600)])
    monkeypatch.setattr(search,'slurm_output',lambda command:'1|PENDING|0||0:0\n')
    search.reconcile(state)
    assert state['ledger'][0]['state']=='PENDING' and state['allocated_gpu_hours']==0


def test_controller_recovers_after_transient_slurm_outage(tmp_path,monkeypatch):
    root=tmp_path/'architecture_gdl';root.mkdir()
    dump(root/'state.json',dict(phase='global_pilot',ledger=[]))
    class EndTick(Exception):pass
    monkeypatch.setenv('SLURM_JOB_ID','8124')
    monkeypatch.setattr(search,'assert_source',lambda *a:None)
    monkeypatch.setattr(search,'recover_submissions',lambda *a:None)
    monkeypatch.setattr(search,'ensure_successor',lambda *a:None)
    def outage(*a):raise recovery.SlurmUnavailable('accounting unavailable')
    monkeypatch.setattr(search,'tick',outage)
    monkeypatch.setattr(search,'report',lambda *a:None)
    monkeypatch.setattr(search.time,'sleep',lambda *a:(_ for _ in ()).throw(EndTick()))
    with pytest.raises(EndTick):search.controller(tmp_path)
    state=read(root/'state.json')
    assert state['phase']=='global_pilot' and state['controller_health']['status']=='WAITING_SLURM'


def test_failed_candidate_does_not_abort_independent_successful_candidates(tmp_path,monkeypatch):
    tasks=[dict(id=g,kind='train',training=dict(config=search.base_config(tmp_path,g))) for g in ('g0','g1','g2')]
    state=dict(phase='global_pilot',tasks=tasks,ledger=[])
    monkeypatch.setattr(search,'recover_submissions',lambda *a:None)
    monkeypatch.setattr(search,'reconcile',lambda *a:None)
    monkeypatch.setattr(search,'task_status',lambda root,state,task:'FAILED' if task['id']=='g1' else 'VERIFIED')
    def advance(root,state):state['phase']='global_control_confirmation'
    monkeypatch.setattr(search,'advance',advance)
    search.tick(tmp_path,state)
    assert state['phase']=='global_control_confirmation' and 'g1' in state['failed_tasks']


def test_mandatory_control_failure_stops_before_official(tmp_path,monkeypatch):
    task=dict(id='g0',kind='train',training=dict(config=search.base_config(tmp_path)))
    state=dict(phase='global_pilot',tasks=[task],ledger=[])
    monkeypatch.setattr(search,'recover_submissions',lambda *a:None)
    monkeypatch.setattr(search,'reconcile',lambda *a:None)
    monkeypatch.setattr(search,'task_status',lambda *a:'FAILED')
    search.tick(tmp_path,state)
    assert state['phase']=='TECHNICAL_BLOCKER' and not (tmp_path/'architecture_gdl/freeze.json').exists()


def test_history_reader_ignores_only_inflight_tail(tmp_path):
    path=tmp_path/'history.jsonl';path.write_text('{"updates": 1}\n{"updates":')
    assert search.history_rows(path)==[{'updates':1}]
    path.write_text('broken\n')
    with pytest.raises(json.JSONDecodeError):search.history_rows(path)


def test_complete_automatic_dependency_graph_to_locked_release(tmp_path,monkeypatch):
    root=tmp_path/'architecture_gdl';root.mkdir();source=root/'source';source.mkdir()
    dump(root/'source_manifest.json',dict(files={}))
    dump(root/'legacy_reference_lock.json',dict(checkpoints={}))
    dump(tmp_path/'manifest50k.json',{})
    tasks=[search.train_task(tmp_path,'pilot_'+g,search.base_config(tmp_path,g)) for g in ('g0','g1','g2')]
    state=dict(phase='global_pilot',tasks=tasks,all_tasks=list(tasks),ledger=[],stage_history=[],calibration={},source=str(source),
        manifest_sha256=sha(tmp_path/'manifest50k.json'))
    def selected(root,task):
        tr=task['training'];cfg=tr['config'];g=cfg['global_ssm']['topology'];d=cfg['decoder']['kind']
        mean={'g0':1.,'g1':.9,'g2':.8}[g]-{'d0':0.,'d1':.1,'d2':.2}[d]
        if cfg['local']['enabled']:mean-=dict(conv=.08,ssm=.01,hybrid=.12)[cfg['local']['core']]
        metric=dict(final={**{s:dict(mean=mean,p95=2.,PCK05=1.) for s in ('OLD','NEW')},
                           'balanced_mean':mean,'balanced_p95':2.})
        path=root/'trials'/task['id']/'best.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('weights')
        if tr.get('reference_metrics') and not search.acceptable(metric,read(tr['reference_metrics']),local=tr.get('selection_local',False)):
            return None
        return dict(trial=task['id'],metrics=metric,checkpoint=str(path),checkpoint_sha256=sha(path),config=cfg,seed=tr['seed'])
    monkeypatch.setattr(search,'selected',selected)
    monkeypatch.setattr(search,'admissible_pilot',lambda root,task,control:selected(root,task))
    monkeypatch.setattr(search,'converged',lambda *a:True)
    phases=[]
    for _ in range(35):
        phase=state['phase'];phases.append(phase)
        for task in state['tasks']:
            folder=root/'tasks'/task['id'];folder.mkdir(parents=True,exist_ok=True)
            if task['kind']=='profile':dump(folder/'profile.json',dict(variants={}))
            if task['kind']=='evidence':
                comparisons={c:{s:dict(ci95={'mean':[.1,.2]}) for s in ('OLD','NEW')}
                             for c in ('depth_invalid','depth_features_zero','rgb_features_zero')}
                dump(folder/'modality.json',dict(paired_comparisons=comparisons));dump(folder/'branches.json',dict(passed=True))
            if task['kind']=='benchmark':dump(folder/'benchmark.json',dict(passed=True))
            if task['kind']=='verify':dump(folder/'verification.json',dict(passed=True,freeze_sha256=sha(root/'freeze.json')))
            if task['kind']=='final':
                assert (root/'final_lock.json').exists()
                dump(folder/'metrics.json',dict(accepted=True,seed=task['seed']))
        state['task_status']={t['id']:'VERIFIED' for t in state['tasks']}
        if phase=='COMPLETED':break
        search.advance(tmp_path,state)
    assert state['phase']=='COMPLETED'
    assert {'global_control_confirmation','global_challenger_confirmation','decoder_pilot','decoder_confirmation',
            'local_pilot','local_confirmation','evidence','benchmark','verification','official_global','official_local','final'}<=set(phases)
    assert set(read(root/'final_lock.json')['checkpoints'])=={'20261025','20261026','20261027'}
    assert read(root/'release.json')['accepted']


def test_inherited_best_checkpoint_requires_exact_immutable_lineage(tmp_path,monkeypatch):
    import shutil
    from models.keypoints.gdl_integrity import audit_checkpoint
    from models.keypoints.config import make_config
    config=make_config(dict(local=dict(enabled=False)))
    original=dict(id='parent',config=config,max_updates=100000)
    extended=dict(id='extended',config=config,max_updates=150000,resume_trial='parent')
    dump(tmp_path/'manifest50k.json',{})
    dump(tmp_path/'architecture_gdl/tasks/parent/task.json',dict(training=original))
    folder=tmp_path/'trials/parent';folder.mkdir(parents=True)
    payload=dict(config=config,training=original,model_state={'weight':torch.ones(1)},
        optimizer={'state':{}},rng=dict(torch=torch.zeros(2,dtype=torch.uint8),numpy=(1,),python=(1,),cuda=[]),
        extra=dict(updates=50000,epoch=10,batch_cursor=0,manifest_sha256=sha(tmp_path/'manifest50k.json')))
    torch.save(payload,folder/'best.pt')
    target=tmp_path/'trials/extended/best.pt';target.parent.mkdir(parents=True)
    shutil.copyfile(folder/'best.pt',target)
    monkeypatch.setattr('models.keypoints.gdl_integrity.expected_shapes',lambda cfg:{'weight':(1,)})
    proof=audit_checkpoint(tmp_path,dict(training=extended),target)
    assert proof['origin_trial']=='parent'
    payload['model_state']['weight'].fill_(2);torch.save(payload,target)
    with pytest.raises(ValueError,match='immutable ancestor'):
        audit_checkpoint(tmp_path,dict(training=extended),target)
