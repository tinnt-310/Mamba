"""Run only in a Slurm CPU allocation; pin atomic checkpoints before cancellation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

assert os.environ.get('SLURM_JOB_ID'), 'Slurm compute allocation required'
root = Path('/home/tinnt/mamba/cross-modal/runs/improve_20261025')
out = root / 'architecture_gdl/legacy'
jobs = dict(zip(('6177','6178','6179','6180'), (
    'AC_selected_20261025','AC_selected_20261026',
    'AC_highres_20261025','AC_highres_20261026')))
proof = {}
for job, trial in jobs.items():
    info = subprocess.check_output(['scontrol','show','job',job], text=True)
    assert 'UserId=tinnt(' in info and str(root) in info and f'tasks/{trial}/' in info
    folder = out / trial
    folder.mkdir(exist_ok=True)
    records = {}
    for name in ('last.pt','best.pt'):
        source = root / 'trials' / trial / name
        target = folder / name
        assert not target.exists(), 'Preservation is immutable'
        # The open fd pins a complete inode even when the worker replaces last.pt.
        with source.open('rb') as incoming, target.open('xb') as dest:
            stat = os.fstat(incoming.fileno())
            shutil.copyfileobj(incoming, dest)
            dest.flush(); os.fsync(dest.fileno())
        assert target.stat().st_size == stat.st_size
        payload = torch.load(target, map_location='cpu', weights_only=False)
        assert payload['config']['global_ssm']['directions'] == 1
        assert payload['model_state'] and payload['optimizer'] and payload['rng']
        assert all(torch.isfinite(t).all() for t in payload['model_state'].values() if t.is_floating_point())
        extra = payload['extra']
        assert all(k in extra for k in ('epoch','batch_cursor','updates','manifest_sha256'))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        records[name] = dict(sha256=digest, bytes=stat.st_size, updates=extra['updates'],
                             epoch=extra['epoch'], batch_cursor=extra['batch_cursor'])
        del payload
    records['observed_temporary_files'] = [p.name for p in source.parent.glob('*.tmp')]
    records['job_before_cancel'] = info
    proof[job] = dict(trial=trial, artifacts=records, status='VERIFIED_PRESERVED')
    (out / 'preservation.json').write_text(json.dumps(proof, indent=2)+'\n')
    # Verified immutable copies survive cancellation even if a fresh write is in flight.
    subprocess.run(['scancel',job], check=True)
    proof[job]['status'] = 'LEGACY_ONE_DIRECTION — STOPPED/PRESERVED'
    (out / 'preservation.json').write_text(json.dumps(proof, indent=2)+'\n')
print(json.dumps({k:dict(trial=v['trial'],status=v['status']) for k,v in proof.items()}),flush=True)
