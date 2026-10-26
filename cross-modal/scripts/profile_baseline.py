"""Measure frozen historical models before changing their implementation."""
import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from models.keypoints.checkpoints import load_checkpoint
from models.keypoints.data import RGBDKeypointDataset, load_manifest
from models.keypoints.heatmaps import decode_heatmaps


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm required'
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, _ = load_checkpoint(args.checkpoint)
    model.cuda().eval()
    manifest = load_manifest(args.manifest)
    data = RGBDKeypointDataset(manifest['dataset_root'], manifest['val'][:500],
                              manifest['keypoint_names'], model.config['input_size'], model.config['heatmap_size'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = [data[i] for i in range(min(10, len(data)))]

    def transfer(sample):
        return tuple(sample[k][None].cuda() for k in ('rgb', 'depth', 'K'))

    def forward(inputs):
        stage = model.forward_stages(*inputs)
        maps = stage.local.heatmaps if stage.local else stage.global_heatmaps
        return decode_heatmaps(maps, support=stage.local.support if stage.local else None).coords.cpu()

    inputs = transfer(samples[0])
    results = {}
    with torch.inference_mode():
        for _ in range(50):
            forward(inputs)
        for mode in ('gpu_ready', 'file_to_coordinates'):
            laps = []
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(3):
                durations, read_times, h2d_times = [], [], []
                for i in range(500):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if mode == 'file_to_coordinates':
                        sample = data[i % len(data)]
                        read_end = time.perf_counter()
                        current = transfer(sample)
                        torch.cuda.synchronize()
                        transferred = time.perf_counter()
                    else:
                        current = inputs
                        read_end = transferred = start
                    forward(current)
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - start)
                    read_times.append(read_end - start)
                    h2d_times.append(transferred - read_end)
                laps.append(dict(p50_ms=float(np.median(durations)*1000),
                                 p95_ms=float(np.percentile(durations, 95)*1000),
                                 images_per_second=500/sum(durations),
                                 mean_read_ms=float(np.mean(read_times)*1000),
                                 mean_h2d_ms=float(np.mean(h2d_times)*1000)))
            results[mode] = dict(repeats=laps, allocated_bytes=torch.cuda.max_memory_allocated(),
                                 reserved_bytes=torch.cuda.max_memory_reserved())
        # Hooks are enabled only in this separate profiling pass.
        timings, starts, handles = {}, {}, []
        modules = {'rgb': model.rgb_encoder, 'geometry': model.geometry_encoder,
                   'decoder': model.decoder, 'head': model.head}
        modules.update({f'fusion_{i}': block for i, block in enumerate(model.fusion.blocks)})
        if model.local_enabled:
            modules['local'] = model.local_refinement
            model.local_refinement.profile = True
        def pre(name):
            def hook(*_):
                torch.cuda.synchronize()
                starts[name] = time.perf_counter()
            return hook
        def post(name):
            def hook(*_):
                torch.cuda.synchronize()
                timings.setdefault(name, []).append(time.perf_counter()-starts[name])
            return hook
        for name, module in modules.items():
            handles.extend([module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as prof:
            for sample in samples:
                forward(transfer(sample))
                prof.step()
        prof.export_chrome_trace(str(output/'trace.json'))
        (output/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
        for handle in handles:
            handle.remove()
        results['components_mean_ms'] = {k:float(np.mean(v)*1000) for k,v in timings.items()}
        if model.local_enabled:
            results['local_components_last'] = model.local_refinement.last_timings
    results.update(job_id=os.environ['SLURM_JOB_ID'], checkpoint=args.checkpoint,
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv']).decode())
    (output/'profile.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
