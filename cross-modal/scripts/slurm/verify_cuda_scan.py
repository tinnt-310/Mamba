"""Fail before training unless audit 2585's native CUDA forward/backward works."""
import hashlib
import json
import os
from pathlib import Path
import sys

if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST'):
    raise RuntimeError('Submit this check through Slurm; no numerical work on login nodes')

ROOT = Path('/home/tinnt/mamba/cross-modal')
AUDIT = ROOT / 'runs/cuda_scan_audit/2585'
audit = json.loads((AUDIT / 'report.json').read_text())
if not audit.get('CUDA_USABLE') or not all(audit['gates'].values()):
    raise RuntimeError('Audit 2585 did not pass all CUDA gates')
expected = json.loads((AUDIT / 'environment.json').read_text())

import torch
import selective_scan_cuda as extension
from mamba_ssm.ops import selective_scan_interface as interface
from models.fusion.ssm.simple_ssm import SimpleSelectiveSSM

path = Path(extension.__file__).resolve()
if not path.is_relative_to(AUDIT / 'environment'):
    raise RuntimeError(f'Extension came from an unexpected environment: {path}')
if hashlib.sha256(path.read_bytes()).hexdigest() != expected['extension_sha256']:
    raise RuntimeError('Native extension differs from audit 2585')
if interface.selective_scan_cuda is not extension:
    raise RuntimeError('Mamba is not using the audited native extension')
if not torch.cuda.is_available() or 'RTX A5000' not in torch.cuda.get_device_name(0):
    raise RuntimeError('An allocated RTX A5000 is required')
if torch.__version__ != expected['torch'] or torch.version.cuda != expected['torch_cuda']:
    raise RuntimeError('PyTorch/CUDA version differs from the audit')

counts = {'fwd': 0, 'bwd': 0}
originals = {name: getattr(extension, name) for name in counts}
def counted(name):
    def call(*args, **kwargs):
        if args[0].device.type != 'cuda' or args[0].dtype != torch.float32:
            raise RuntimeError('Expected CUDA FP32 input to the native extension')
        counts[name] += 1
        return originals[name](*args, **kwargs)
    return call

try:
    for name in counts:
        setattr(extension, name, counted(name))
    for state_dim in (4, 16):
        model = SimpleSelectiveSSM(32, state_dim, backend='mamba').cuda()
        x = torch.randn(1, 32, 127, device='cuda', requires_grad=True)
        y = model(x)
        y.square().mean().backward()
        if not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(x.grad).all()):
            raise RuntimeError('Nonfinite native forward/backward result')
    torch.cuda.synchronize()
finally:
    for name, function in originals.items():
        setattr(extension, name, function)
if not all(counts.values()):
    raise RuntimeError(f'Native forward/backward was not used: {counts}')

result = {'status': 'PASS', 'audit_job_id': '2585',
          'slurm_job_id': os.environ['SLURM_JOB_ID'], 'python': sys.executable,
          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
          'extension': str(path), 'extension_sha256': expected['extension_sha256'],
          'native_calls': counts, 'backend': 'mamba'}
destination = ROOT / 'runs/slurm_logs' / f"cuda-backend-{os.environ['SLURM_JOB_ID']}.json"
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
