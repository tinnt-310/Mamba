"""Slurm-only audit coordinator. No writes to campaign, source, or base environment."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST'):
    raise SystemExit('Submit scripts/slurm/cuda_scan_audit.slurm; compute-node execution required')
if os.environ['SLURM_JOB_ID'] == '2082' or platform.node().split('.')[0] == 'hpc01':
    raise SystemExit('This audit must use a separate allocation and node')

PROJECT = Path('/home/tinnt/mamba/cross-modal')
BASE = Path('/opt/hpc/miniforge3/envs/ai-pytorch')
ROOT = PROJECT / 'runs/cuda_scan_audit' / os.environ['SLURM_JOB_ID']
ROOT.mkdir(parents=True, exist_ok=False)
(ROOT / 'logs').mkdir()
ENV = ROOT / 'environment'
STAGE = PROJECT / 'runs/cuda_scan_audit/offline-inputs-20261005'
PYTHON = ENV / 'bin/python'
WORKER = ROOT / 'source/scripts/cuda_scan_audit/worker.py'
state = {'status': 'INCOMPLETE', 'CUDA_USABLE': False, 'phase': 'initializing',
         'job_id': os.environ['SLURM_JOB_ID'], 'node': platform.node(),
         'scope': 'A1 trained checkpoint and controlled, untrained local/A4; no campaign changes',
         'failures': [], 'completed': [], 'gates': {}}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save():
    write_json(ROOT / 'report.json', state)
    lines = ['# CUDA selective-scan audit', '',
             f"Status: **{state['status']}**; **CUDA_USABLE={str(state['CUDA_USABLE']).lower()}**", '',
             f"Slurm job {state['job_id']} on {state['node']}; phase: {state['phase']}.",
             'Job 2082, its allocation, campaign, checkpoint metadata and base environment are unchanged.', '',
             'A1 uses a trained checkpoint. A4 reuses that global checkpoint with deterministic untrained local weights;',
             'the A4 measurements do not establish trained A4 accuracy.', '',
             'FP32, autocast/TF32 disabled. Inference: 10 warmup + 100 validation images; training: 16 warmup + 32 train images.',
             'Three rounds, alternating backend order; full scopes include disk reads, CPU preprocessing and GPU transfers.',
             'Inference full scope also includes subpixel decode and CPU coordinates; training full scope includes optimizer updates.',
             'Disk reads use normal OS caching; no cache flushing. Forward scopes exclude I/O/transfers.', '',
             '## Gates', '']
    lines += [f'- {name}: `{value}`' for name, value in state['gates'].items()]
    if state['failures']:
        lines += ['', '## Failures', ''] + [f'- {item}' for item in state['failures']]
    if 'speedups' in state:
        lines += ['', '## Mean latency speedups', '', '| Model | Round | Scope | torch/CUDA | Pass |',
                  '|---|---:|---|---:|---|']
        lines += [f"| {v['model']} | {v['round']} | {v['scope']} | {v['speedup']:.3f}x | {v['pass']} |"
                  for v in state['speedups']]
    if 'benchmarks' in state:
        lines += ['', '## Timings (seconds; FPS = 1 / mean)', '',
                  '| Model | Backend | Round | Scope | N | Mean | p50 | p95 | FPS | Allocated MiB | Reserved MiB |',
                  '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for entry in state['benchmarks']:
            lines.append(f"| {entry['model']} | {entry['backend']} | {entry['round']} | {entry['scope']} | "
                         f"{entry['count']} | {entry['mean_seconds']:.6f} | {entry['p50_seconds']:.6f} | "
                         f"{entry['p95_seconds']:.6f} | {entry['fps']:.3f} | "
                         f"{entry['peak_allocated_bytes']/2**20:.1f} | {entry['peak_reserved_bytes']/2**20:.1f} |")
    lines += ['', '## Artifacts', '',
              '- `environment.json`, `pip-freeze.txt`, `conda-explicit.txt`: exact environment and extension details.',
              '- `snapshot.json`, `source/`, `checkpoint.pt`, `manifest.json`: immutable audit inputs and SHA256.',
              '- `correctness.json`, `validation-comparison.json`, `bench-*.json`: tests and raw per-image timings.',
              '- `logs/`, Slurm stdout/stderr links, `selective-scan-trace.json`: installation and execution evidence.', '',
              'Reproduce from the project root: `sbatch scripts/slurm/cuda_scan_audit.slurm`.',
              'A result is usable only if extension, correctness, validation and every 2x full-pipeline speed gate pass.',
              'Incomplete or failed stages never authorize a backend change.', '']
    temporary = ROOT / 'RESULTS.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(ROOT / 'RESULTS.md')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def command(args, label, *, env=None, timeout=7200, check=True):
    state['phase'] = label
    save()
    print(f'[{time.strftime("%FT%T%z")}] {label}', flush=True)
    with (ROOT / 'logs' / f'{label}.log').open('w') as output:
        output.write('Command: ' + repr([str(arg) for arg in args]) + '\n')
        output.flush()
        result = subprocess.run([str(arg) for arg in args], cwd=ROOT / 'source', env=env,
                                stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        message = f'{label}: exit {result.returncode}; see logs/{label}.log'
        state['failures'].append(message)
        save()
        if check:
            raise RuntimeError(message)
        return False
    state['completed'].append(label)
    save()
    return True


def worker(action, backend='torch', model='A1', round_number=0):
    label = f'{action}-{model}-{backend}-{round_number}'
    return command([PYTHON, WORKER, '--root', ROOT, '--action', action, '--backend', backend,
                    '--model', model, '--round', str(round_number)], label, env=runtime_env, check=False)


try:
    save()
    for suffix in ('out', 'err'):
        (ROOT / f'slurm.{suffix}').symlink_to(ROOT.parent / f'slurm-{state["job_id"]}.{suffix}')
    source = ROOT / 'source'
    source.mkdir()
    shutil.copytree(PROJECT / 'models', source / 'models', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(PROJECT / 'scripts/cuda_scan_audit', source / 'scripts/cuda_scan_audit',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT / 'scripts/slurm/cuda_scan_audit.slurm', ROOT / 'audit.slurm')
    shutil.copy2('/home/tinnt/slurm.md', ROOT / 'slurm.md')
    # best.pt is published atomically; one opened descriptor yields a complete old or new checkpoint.
    checkpoint_source = PROJECT / 'runs/ablation5000/A1/global/best.pt'
    with checkpoint_source.open('rb') as incoming, (ROOT / 'checkpoint.pt').open('wb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    shutil.copy2(PROJECT / 'runs/subset5000.json', ROOT / 'manifest.json')
    snapshot = {'checkpoint_source': str(checkpoint_source), 'checkpoint_sha256': sha(ROOT / 'checkpoint.pt'),
                'manifest_sha256': sha(ROOT / 'manifest.json'),
                'slurm_script_sha256': sha(ROOT / 'audit.slurm'),
                'source_sha256': {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
                                  if p.is_file() and p.suffix in ('.py', '.sh')}}
    write_json(ROOT / 'snapshot.json', snapshot)
    command([BASE / 'bin/python', '-m', 'compileall', '-q', source / 'scripts/cuda_scan_audit'],
            'compile-harness', timeout=120)
    command(['scontrol', 'show', 'job', '-d', state['job_id']], 'slurm-allocation', timeout=30)
    command(['nvidia-smi', '-q'], 'gpu-before', timeout=30)
    base_info_code = ('import json,torch,sys; from pathlib import Path; '
                      f'Path({str(ROOT / "environment-base.json")!r}).write_text(json.dumps(dict('
                      'python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,'
                      'gpu=torch.cuda.get_device_name(0),executable=sys.executable,'
                      'abi=torch._C._GLIBCXX_USE_CXX11_ABI)))')
    command([BASE / 'bin/python', '-c', base_info_code], 'base-environment-info', timeout=120)
    state['phase'] = 'verify-offline-inputs'; save()
    shutil.copytree(STAGE / 'metadata', ROOT / 'offline-metadata')
    verified = {}
    for line in (STAGE / 'metadata/conda-packages.tsv').read_text().splitlines():
        url, filename, expected_sha, expected_md5 = line.split('\t')
        archive = STAGE / 'conda-cache' / filename
        if len(expected_sha) == 64 and sha(archive) != expected_sha:
            raise RuntimeError(f'Conda archive checksum mismatch: {filename}')
        if len(expected_sha) != 64 and len(expected_md5) == 32:
            with archive.open('rb') as content:
                if hashlib.file_digest(content, 'md5').hexdigest() != expected_md5:
                    raise RuntimeError(f'Conda archive MD5 mismatch: {filename}')
        verified[filename] = sha(archive)
    for metadata in sorted((STAGE / 'metadata').glob('*.json')):
        for entry in json.loads(metadata.read_text()).get('urls', []):
            path = STAGE / 'wheels' / entry['filename']
            if path.exists():
                if sha(path) != entry['digests']['sha256']:
                    raise RuntimeError(f'Wheel checksum mismatch: {path.name}')
                verified[path.name] = sha(path)
    write_json(ROOT / 'offline-inputs-sha256.json', verified)
    clone_env = os.environ.copy()
    clone_env.update(CONDA_PKGS_DIRS=f'{ROOT}/conda-pkgs,{STAGE}/conda-cache,/opt/hpc/miniforge3/pkgs',
                     CONDA_ENVS_PATH=str(ROOT), PIP_CACHE_DIR=str(ROOT / 'pip-cache'),
                     CONDA_OFFLINE='true', PIP_NO_INDEX='1', PIP_FIND_LINKS=str(STAGE / 'wheels'))
    command(['/opt/hpc/miniforge3/bin/conda', 'config', '--show', 'pkgs_dirs', '--json'],
            'conda-cache-config', env=clone_env, timeout=60)
    command(['/opt/hpc/miniforge3/bin/conda', 'create', '-y', '--offline', '--copy', '--prefix', ENV,
             '--clone', BASE], 'clone-environment', env=clone_env, timeout=3600)
    runtime_env = clone_env.copy()
    runtime_env.update(PATH=f'{ENV}/bin:' + os.environ['PATH'],
                       LD_LIBRARY_PATH=str(ENV / 'lib'), PYTHONPATH=str(source),
                       MAX_JOBS='2', CUDA_HOME=str(ENV / 'lib/python3.13/site-packages/nvidia/cu13'),
                       TORCH_EXTENSIONS_DIR=str(ROOT / 'torch-extensions'),
                       TRITON_CACHE_DIR=str(ROOT / 'triton-cache'),
                       TMPDIR=str(ROOT / 'tmp'), PIP_DISABLE_PIP_VERSION_CHECK='1')
    (ROOT / 'tmp').mkdir()
    command([PYTHON, '-m', 'pip', 'install', 'cuda-toolkit[nvcc,crt,cccl,nvvm,cudart]==13.0.2',
             'ninja==1.11.1.4', 'packaging', 'wheel==0.45.1'], 'install-cuda-build-tools', env=runtime_env, timeout=1800)
    toolkit = Path(runtime_env['CUDA_HOME'])
    # NVIDIA's pip runtime supplies the versioned library; the extension linker uses -lcudart.
    cudart_link = toolkit / 'lib/libcudart.so'
    if not cudart_link.exists() and (toolkit / 'lib/libcudart.so.13').exists():
        cudart_link.symlink_to('libcudart.so.13')
    command([toolkit / 'bin/nvcc', '--version'], 'nvcc-version', env=runtime_env, timeout=30)
    # Metadata is computed by the cloned Python on the allocated compute node.
    info_code = ('import json,torch,sys; from pathlib import Path; '
                 f'Path({str(ROOT / "build-tuple.json")!r}).write_text(json.dumps(dict('
                 'python=list(sys.version_info[:3]),torch=torch.__version__,cuda=torch.version.cuda,'
                 'abi=torch._C._GLIBCXX_USE_CXX11_ABI)))')
    command([PYTHON, '-c', info_code], 'build-tuple', env=runtime_env, timeout=120)
    build_tuple = json.loads((ROOT / 'build-tuple.json').read_text())
    write_json(ROOT / 'environment-before-mamba.json', {**build_tuple,
               'prefix': str(ENV), 'CUDA_HOME': runtime_env['CUDA_HOME'], 'node': state['node']})
    command([PYTHON, '-m', 'pip', 'freeze'], 'pip-freeze-before-mamba', env=runtime_env, timeout=120)
    if build_tuple['python'][:2] != [3, 13] or build_tuple['torch'].split('+')[0] != '2.11.0' or build_tuple['cuda'] != '13.0':
        raise RuntimeError(f'Original Python/PyTorch/CUDA tuple changed: {build_tuple}')
    wheel_name = ('mamba_ssm-2.3.2.post1+cu13torch2.11cxx11abi'
                  f'{str(build_tuple["abi"]).upper()}-cp313-cp313-linux_x86_64.whl')
    wheel_url = f'https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/{wheel_name}'
    state['phase'] = 'select-offline-mamba-distribution'; save()
    if (STAGE / 'wheels' / wheel_name).exists():
        shutil.copy2(STAGE / 'wheels' / wheel_name, ROOT / wheel_name)
        state['mamba_distribution'] = {'url': wheel_url, 'sha256': sha(ROOT / wheel_name), 'kind': 'wheel'}
        install_target = ROOT / wheel_name
    else:
        source_url = 'https://codeload.github.com/state-spaces/mamba/tar.gz/refs/tags/v2.3.2.post1'
        archive = ROOT / 'mamba-v2.3.2.post1.tar.gz'
        shutil.copy2(STAGE / archive.name, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(ROOT / 'upstream', filter='data')
        install_target = ROOT / 'upstream/mamba-2.3.2.post1'
        state['mamba_distribution'] = {'url': source_url, 'sha256': sha(archive), 'kind': 'source'}
        runtime_env['MAMBA_FORCE_BUILD'] = 'TRUE'
    save()
    command([PYTHON, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-v', install_target],
            'install-mamba', env=runtime_env, timeout=7200)
    command([PYTHON, '-m', 'pip', 'freeze'], 'pip-freeze', env=runtime_env, timeout=120)
    shutil.copy2(ROOT / 'logs/pip-freeze.log', ROOT / 'pip-freeze.txt')
    command(['/opt/hpc/miniforge3/bin/conda', 'list', '--prefix', ENV, '--explicit'],
            'conda-explicit', env=runtime_env, timeout=120)
    shutil.copy2(ROOT / 'logs/conda-explicit.log', ROOT / 'conda-explicit.txt')
    if not worker('environment', 'mamba'):
        raise RuntimeError('CUDA extension load or native forward/backward verification failed')
    state['gates']['extension'] = True
    if not worker('selftest'):
        raise RuntimeError('Audit harness self-check failed')
    checks = [worker('scan', b) for b in ('torch', 'mamba')]
    for model in ('A1', 'A4'):
        checks.extend(worker('correctness', b, model) for b in ('torch', 'mamba'))
    checks.append(worker('compare'))
    state['gates']['correctness'] = all(checks)
    vals = [worker('validation', b) for b in ('torch', 'mamba')]
    vals.append(worker('compare-validation'))
    state['gates']['validation'] = all(vals)
    for repeat in range(3):
        order = ('torch', 'mamba') if repeat % 2 == 0 else ('mamba', 'torch')
        for model in ('A1', 'A4'):
            for backend in order:
                worker('benchmark', backend, model, repeat)
    speeds = []
    state['benchmarks'] = []
    for model in ('A1', 'A4'):
        for repeat in range(3):
            paths = [ROOT / f'bench-{model}-{b}-{repeat}.json' for b in ('torch', 'mamba')]
            if not all(p.exists() for p in paths):
                state['failures'].append(f'Missing benchmark: {model}, round {repeat}')
                continue
            reference, cuda = (json.loads(p.read_text()) for p in paths)
            if (reference['initial_state_sha256'] != cuda['initial_state_sha256'] or
                    reference['checkpoint_sha256'] != cuda['checkpoint_sha256']):
                state['failures'].append(f'Benchmark weights/checkpoint differ: {model}, round {repeat}')
                continue
            for benchmark_result in (reference, cuda):
                for scope in ('inference_forward', 'inference_full', 'training_compute', 'training_full'):
                    summary = {key: value for key, value in benchmark_result[scope].items()
                               if key not in ('samples', 'optimizer_steps')}
                    state['benchmarks'].append(dict(summary, model=model, backend=benchmark_result['backend'],
                                                    round=repeat, scope=scope))
            for scope in ('inference_full', 'training_full'):
                if ([row['frame_id'] for row in reference[scope]['samples']] !=
                        [row['frame_id'] for row in cuda[scope]['samples']]):
                    state['failures'].append(f'Benchmark input IDs differ: {model}, round {repeat}, {scope}')
                    continue
                speedup = reference[scope]['mean_seconds'] / cuda[scope]['mean_seconds']
                speeds.append({'model': model, 'round': repeat, 'scope': scope,
                               'speedup': speedup, 'pass': speedup >= 2.})
    state['speedups'] = speeds
    state['gates']['speedup'] = len(speeds) == 12 and all(row['pass'] for row in speeds)
    for gate, passed in state['gates'].items():
        if not passed:
            state['failures'].append(f'Acceptance gate failed: {gate}')
    state['CUDA_USABLE'] = all(state['gates'].values()) and not state['failures']
    state['status'] = 'PASS' if state['CUDA_USABLE'] else 'FAIL'
    state['phase'] = 'finished'
    save()
except BaseException as error:
    state['status'] = 'INCOMPLETE' if isinstance(error, (KeyboardInterrupt, subprocess.TimeoutExpired)) else 'FAIL'
    state['CUDA_USABLE'] = False
    state['failures'].append(f'{type(error).__name__}: {error}')
    (ROOT / 'failure.txt').write_text(traceback.format_exc())
    save()
    traceback.print_exc()
finally:
    print(f"Audit {state['status']}; CUDA_USABLE={state['CUDA_USABLE']}; {ROOT / 'RESULTS.md'}", flush=True)
sys.exit(0 if state['CUDA_USABLE'] else 1)
