"""CPU Slurm migration: test canonical updates, lock historical references, resume controller."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from models.keypoints.campaign import read,dump,sha,snapshot
from models.keypoints.gdl_search import reconcile,report

assert os.environ.get('SLURM_JOB_ID'), 'CPU Slurm allocation required'
root=Path('/home/tinnt/mamba/cross-modal/runs/improve_20261025')
out=root/'architecture_gdl'
project=Path('/home/tinnt/mamba/cross-modal')
proof=out/'correctness_v2.xml'
subprocess.run([os.sys.executable,'-m','pytest','-q','--junitxml='+str(proof)],cwd=project,check=True)
suite=ET.parse(proof).getroot()
assert all(not int(s.get('failures',0)) and not int(s.get('errors',0)) for s in suite.iter('testsuite'))
with (root/'active_controller.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(out/'state.json')
    if any('gdl-controller' in line for line in subprocess.check_output(['squeue','--me','--noheader','--format=%j'],text=True).splitlines()):
        raise ValueError('A controller is still scheduled; no concurrent migration')
    dump(out/'pre_v2_state.json',state)
    reconcile(state)
    references={}
    for path in sorted((root/'continuation/diagnostics').glob('*.json')):
        # Read only the saved provenance field; never reuse observed test/D455 metrics for selection.
        digest=None
        with path.open() as stream:
            for line in stream:
                if line.strip().startswith('"checkpoint_sha256":'):
                    digest=json.loads('{'+line.strip().rstrip(',')+'}')['checkpoint_sha256'];break
        if digest is None:raise ValueError('Missing historical checkpoint digest '+str(path))
        name=path.stem;parts=name.split('_');kind=parts[0];seed=parts[1];which=parts[2]
        checkpoint=root/'trials'/f'F_{kind}_{seed}'/(which+'.pt')
        if sha(checkpoint)!=digest:raise ValueError('Historical checkpoint changed '+str(checkpoint))
        references[str(checkpoint.resolve())]=digest
    if len(references)!=9:raise ValueError('Expected all nine protected historical checkpoints')
    dump(out/'legacy_reference_lock.json',dict(checkpoints=references,verified_job=os.environ['SLURM_JOB_ID'],
        policy='Established historical diagnostics hashes; metrics not used for architecture search'))
    revision=out/'revision_20261027_v2';revision.mkdir(exist_ok=True)
    snapshot(revision)
    state.setdefault('source_history',[]).append(dict(source=state['source'],
        manifest_sha256=sha(Path(state['source']).parent/'source_manifest.json'),reason='Completed native calibration'))
    for record in state['ledger']:
        if record['task']=='profile':record['source_directory']=state['source']
    state['source']=str(revision/'source')
    state['previous_controller_successors']=state.pop('controller_successors',{})
    state['controller_successors']={}
    state['migration_job']=os.environ['SLURM_JOB_ID']
    state['correctness_proof']=str(proof)
    state['task_status']['profile']='VERIFIED'
    dump(out/'state.json',state);report(root,state)
    text=(root/'REPORT.md').read_text()
    start=text.find('<!-- architecture-search-20261026 -->');end=text.find('<!-- gdl-search -->')
    if start>=0 and end>start:
        previous=text[start:end].replace('Trạng thái: **confirmation**.',
            'Trạng thái: **LEGACY_ONE_DIRECTION — STOPPED/PRESERVED**.')
        text=text[:start]+previous+text[end:];(root/'REPORT.md').write_text(text)
    command=['sbatch','--parsable','--export=NONE','--partition=compute','--cpus-per-task=2','--mem=8G','--time=12:00:00',
        '--job-name=gdl-controller','--output='+str(out/'controller-%j.out'),'--error='+str(out/'controller-%j.err'),
        str(revision/'source/scripts/slurm/gdl.slurm'),str(root),'controller','unused','43200',state['source']]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    dump(out/'migration_v2.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],controller_job=job,
        source=state['source'],legacy_checkpoints_verified=len(references),correctness=str(proof)))
    print('MIGRATION COMPLETE; controller job '+job,flush=True)
