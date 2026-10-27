"""Prepare, test and activate a persistent controller; running GPU workers are preserved."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
from models.keypoints.campaign import read,dump,sha,snapshot
from models.keypoints.gdl_integrity import audit_training
from models.keypoints.gdl_search import reconcile,report,campaign_gpu_limit,pilot_seeds,expand_parallel_pilots
from models.keypoints.gdl_recovery import heartbeat

assert os.environ.get('SLURM_JOB_ID'), 'CPU Slurm compute allocation required'
parser=argparse.ArgumentParser()
parser.add_argument('--max-gpus',type=int,default=4)
parser.add_argument('--pilot-seeds',type=int,nargs='+',default=[20261025])
args=parser.parse_args()
campaign_gpu_limit(dict(max_campaign_gpus=args.max_gpus))
pilot_seeds(dict(pilot_seeds=args.pilot_seeds))
root=Path('/home/tinnt/mamba/cross-modal/runs/improve_20261025')
out=root/'architecture_gdl';project=Path('/home/tinnt/mamba/cross-modal')
activation_lock=(out/'activation.lock').open('a')
try:fcntl.flock(activation_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise RuntimeError('Another activation owns the campaign deployment lock; no changes made')
current_job=os.environ['SLURM_JOB_ID']
proof=out/'automation_correctness.xml'
subprocess.run([os.sys.executable,'-m','pytest','-q','--junitxml='+str(proof)],cwd=project,check=True)
assert all(not int(s.get('failures',0)) and not int(s.get('errors',0))
           for s in ET.parse(proof).getroot().iter('testsuite'))
state=read(out/'state.json')
if not set(pilot_seeds(state))<=set(args.pilot_seeds):raise ValueError('Activation cannot remove existing pilot seeds')
# All already completed pilots receive the same strict artifact audit as future workers.
for task in state['all_tasks']:
    if task['kind']=='train' and (root/'trials'/task['id']/'completed.json').exists():
        audit_training(root,task)
revision=out/('revision_20261027_automation_'+current_job);revision.mkdir(exist_ok=True)
snapshot(revision)
policy=dict(gpu_hour_limit=None,max_campaign_gpus=args.max_gpus,pilot_seeds=sorted(args.pilot_seeds),persistent_state=True,
    parallel_pilots='Decoder and Local: three variants on each confirmed seed; unchanged gate per seed; aggregate ranking',
    transient_slurm_errors='retain phase and retry; heartbeat WAITING_SLURM',
    duplicate_prevention='persist submission token before sbatch; recover job ID from squeue/sacct Comment',
    slices='atomic optimizer-boundary checkpoint; verified resume to unchanged horizon; signal USR1 180 seconds before walltime',
    retry='progressing slices continue; infrastructure failures without progress retry up to 3 allocations; cancellation is not retried',
    validation='full synthetic validation; unchanged acceptance comparators; invalid candidates excluded',
    official='freeze + source-bound GPU verification required; all three seeds and mandatory Local',
    final='three checkpoint hashes locked before synthetic test/D455 access')
dump(revision/'automation_policy.json',policy)
# Prepare the replacement with an activation dependency BEFORE stopping any controller.
command=['sbatch','--parsable','--export=NONE','--partition=compute','--cpus-per-task=4','--mem=16G','--time=12:00:00',
    '--job-name=gdl-controller','--dependency=afterok:'+current_job,'--comment=gdl-activation-'+current_job,
    '--output='+str(out/'controller-%j.out'),'--error='+str(out/'controller-%j.err'),
    str(revision/'source/scripts/slurm/gdl.slurm'),str(root),'controller','unused','43200',str(revision/'source')]
new_controller=subprocess.check_output(command,text=True).strip().split(';')[0]
dump(out/'automation_activation.json',dict(status='PREPARED',activation_job=current_job,
    controller_job=new_controller,source=str(revision/'source'),correctness=str(proof)))
state=read(out/'state.json')
known=set(state.get('controller_successors',{}))|set(state.get('controller_successors',{}).values())
queue=subprocess.check_output(['squeue','--me','--noheader','--format=%i|%T'],text=True)
active={line.split('|')[0].strip():line.split('|')[1].strip() for line in queue.splitlines()}
controllers=[]
for job in known:
    if job not in active:continue
    info=subprocess.check_output(['scontrol','show','job',job],text=True)
    assert 'Partition=compute' in info and 'gres/gpu' not in info and str(root) in info and ' controller ' in info
    controllers.append((job,active[job]))
# The only canceled allocations are verified CPU controllers from this campaign.
controllers.sort(key=lambda pair:pair[1]!='PENDING')
for job,_ in controllers:subprocess.run(['scancel',job],check=True)
with (root/'active_controller.lock').open('w') as lock:
    deadline=time.monotonic()+120
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:raise RuntimeError('Old CPU controller did not release campaign lock')
            time.sleep(2)
    state=read(out/'state.json');dump(revision/'pre_activation_state.json',state);reconcile(state)
    previous_limit=campaign_gpu_limit(state)
    if not set(pilot_seeds(state))<=set(args.pilot_seeds):raise ValueError('Activation cannot remove existing pilot seeds')
    state['max_campaign_gpus']=args.max_gpus;state['pilot_seeds']=sorted(args.pilot_seeds)
    added_pilots=expand_parallel_pilots(root,state)
    state.setdefault('runtime_policy_revisions',[]).append(dict(activation_job=current_job,
        previous_max_campaign_gpus=previous_limit,max_campaign_gpus=args.max_gpus,
        pilot_seeds=state['pilot_seeds'],added_pilots=added_pilots))
    initial_policy=read(out/'policy.json');dump(revision/'pre_activation_policy.json',initial_policy)
    dump(out/'policy.json',dict(initial_policy,max_gpus=args.max_gpus,pilot_seeds=state['pilot_seeds'],
        runtime_policy_revision=str(revision/'automation_policy.json')))
    state.setdefault('source_history',[]).append(dict(source=state['source'],
        manifest_sha256=sha(Path(state['source']).parent/'source_manifest.json'),reason='Authorized concurrency and parallel pilot update'))
    state['source']=str(revision/'source')
    state['previous_controller_successors']=state.pop('controller_successors',{})
    state['controller_successors']={};state['controller_job_id']=new_controller
    state['automation_enabled']=True;state['automation_activation_job']=current_job
    state['automation_policy']=str(revision/'automation_policy.json')
    state['ledger'].append(dict(task='controller_activation_'+current_job,job_id=new_controller,gpu=False,
        seconds=43200,state='PENDING',elapsed_seconds=0,source_directory=state['source']))
    heartbeat(state,'STARTING');dump(out/'state.json',state);report(root,state)
    receipt=dict(status='ACTIVE',activation_job=current_job,
        controller_job=new_controller,source=state['source'],correctness=str(proof),
        canceled_cpu_controllers=[job for job,_ in controllers],gpu_workers_preserved=True,
        max_campaign_gpus=args.max_gpus,pilot_seeds=state['pilot_seeds'],added_pilots=added_pilots)
    dump(out/'automation_activation.json',receipt);dump(revision/'activation_receipt.json',receipt)
print('AUTOMATION ACTIVE; controller '+new_controller,flush=True)
