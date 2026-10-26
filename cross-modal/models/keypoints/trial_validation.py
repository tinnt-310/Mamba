"""Post-lock synthetic/D455 evaluation and measured execution configurations."""
from contextlib import nullcontext
import csv
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image
import torch

from .checkpoints import load_checkpoint
from .heatmaps import decode_heatmaps,transform_points
from .multisource import MultiSourceDataset,read,dump,preprocess_rgbd,digest
from .trial_runner import loader,transfer,final_maps,evaluate,metrics


def stats(durations,images):
    return dict(batch_p50_ms=float(np.median(durations)*1000),batch_p95_ms=float(np.percentile(durations,95)*1000),
                images_per_second=images/sum(durations),allocated_bytes=torch.cuda.max_memory_allocated(),
                reserved_bytes=torch.cuda.max_memory_reserved())


@torch.no_grad()
def benchmark(model,data,batch_size,workers,precision='fp32',samples=500):
    model.eval(); amp=precision=='bf16'
    inputs=next(iter(loader(data,batch_size,workers)))
    current=transfer(inputs)
    def forward(batch):
        context=torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext()
        with context:
            stage=model.forward_stages(*(batch[k] for k in ('rgb','depth','K')))
        return decode_heatmaps(final_maps(stage).float(),support=stage.local.support if stage.local else None).coords.cpu()
    for _ in range(50): forward(current)
    result={}
    for mode in ('gpu_ready','file_to_coordinates'):
        repeats=[]
        for _ in range(3):
            torch.cuda.reset_peak_memory_stats(); durations=[]; seen=0
            batches=iter(loader(data,batch_size,workers))
            while seen<samples:
                torch.cuda.synchronize(); start=time.perf_counter()
                if mode=='file_to_coordinates':
                    try: batch=next(batches)
                    except StopIteration:
                        batches=iter(loader(data,batch_size,workers)); batch=next(batches)
                    batch=transfer(batch)
                else: batch=current
                forward(batch); torch.cuda.synchronize()
                durations.append(time.perf_counter()-start); seen+=batch['rgb'].shape[0]
            repeats.append(stats(durations,seen))
        result[mode]=repeats
    result.update(batch=batch_size,workers=workers,precision=precision)
    return result


def real_evaluation(model,root):
    # Import the fixed calibration implementation used by the historical report.
    comparison=Path('/home/tinnt/model_comparison_20261025')
    sys.path.insert(0,str(comparison))
    from evaluate import rectified_to_original
    samples=[s for s in read(comparison/'samples.json') if s['dataset']=='D455']
    errors,normalized,rows,mm_rows=[],[],[],[]
    alignment={}
    for row in csv.DictReader((comparison/'metric_mm/depth_audit.csv').open()):
        if row['dataset']=='D455' and row['mm_evaluable']=='True':
            alignment.setdefault(row['image'],[]).append(float(row['z_m']))
    calibration=read(comparison/'real_alignment_audit.json')['color']
    fx,fy=calibration['fx'],calibration['fy']
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            image=Image.open(sample['mamba_image']).convert('RGB')
            depth=np.load(sample['depth'],allow_pickle=False)
            rgb,z,K=preprocess_rgbd(image,depth,sample['K'],model.config['input_size'])
            torch.cuda.synchronize(); start=time.perf_counter()
            stage=model.forward_stages(rgb[None].cuda(),z[None].cuda(),K[None].cuda())
            maps=final_maps(stage)
            decoded=decode_heatmaps(maps.float(),support=stage.local.support if stage.local else None).coords[0]
            rectified=transform_points(decoded,maps.shape[-2:],sample['original_hw']).cpu().numpy()
            original=rectified_to_original(rectified)
            torch.cuda.synchronize(); duration=time.perf_counter()-start
            gt=np.asarray(sample['points']); valid=np.asarray(sample['visibility'])>0
            e=np.linalg.norm(original-gt,axis=-1)
            diag=np.linalg.norm(np.asarray(sample['bbox'])[2:]-np.asarray(sample['bbox'])[:2])
            errors.extend(e[valid]); normalized.extend(e[valid]/diag)
            rows.append(dict(id=sample['id'],coords=original.tolist(),mean_px=float(e[valid].mean()),
                             runtime_ms=duration*1000))
            # Match the existing mm protocol: lateral camera error at GT depth,
            # all nine labels must have evaluable sensor depth.
            depths=alignment.get(sample['id'],[])
            if len(depths)==9:
                # Undistort both UV sets using the established camera model.
                from prepare import undistort_color
                delta=undistort_color(original,calibration)-undistort_color(gt,calibration)
                mm=np.linalg.norm(delta,axis=-1)*np.asarray(depths)*1000
                mm_rows.append(dict(id=sample['id'],errors_mm=mm.tolist(),mean_mm=float(mm.mean())))
    output=dict(pixel=metrics(errors,normalized),records=rows,observed_session=True,
                alignment_ms_included=False,limitation='Prepared alignment excluded from model timing; '
                'single observed real session; mm is lateral error at labeled sensor depth, not independent 3D truth')
    if mm_rows:
        all_mm=np.array([r['errors_mm'] for r in mm_rows])
        output['mm']=dict(images=len(mm_rows),mean=float(all_mm.mean()),median=float(np.median(all_mm)),
                          p95=float(np.percentile(all_mm,95)),image_mean_below_5mm=float((all_mm.mean(1)<5).mean()),records=mm_rows)
    return output


def run(task,campaign):
    output=campaign/'trials'/task['id']; output.mkdir(parents=True,exist_ok=True)
    manifest=read(campaign/'manifest50k.json')
    checkpoint=campaign/'trials'/task['checkpoint_trial']/'best.pt'
    if not checkpoint.exists():
        dump(output/'completed.json',dict(status='skipped',reason='Checkpoint trial did not produce a model'))
        return
    model,_=load_checkpoint(checkpoint); model.cuda().eval()
    if task['kind']=='evaluate_final':
        lock=read(campaign/'final_lock.json')
        if lock['manifest_sha256']!=digest(campaign/'manifest50k.json'):
            raise ValueError('Final evaluation manifest changed after lock')
        result=dict(synthetic_test=evaluate(model,MultiSourceDataset(manifest,'test',model.config)),
                    D455=real_evaluation(model,campaign),checkpoint=str(checkpoint),gpu=torch.cuda.get_device_name())
        dump(output/'metrics.json',result)
        # Batch-one deployment benchmark remains comparable to historical runs.
        bench=benchmark(model,MultiSourceDataset(manifest,'val',model.config),1,0)
        dump(output/'benchmark.json',bench)
    elif task['kind']=='runtime':
        data=MultiSourceDataset(manifest,'val',model.config)
        existing=read(output/'runtime.json') if (output/'runtime.json').exists() else {}
        reference=existing.get('reference') or evaluate(model,data,batch_size=1,workers=4)
        results=existing.get('configurations',[])
        started=time.monotonic(); limit=float(os.environ.get('MAMBA_JOB_SECONDS','3600'))-180
        reference_batch=transfer(next(iter(loader(data,8,0))))
        with torch.inference_mode():
            reference_coords=[]
            for i in range(reference_batch['rgb'].shape[0]):
                stage=model.forward_stages(*(reference_batch[k][i:i+1] for k in ('rgb','depth','K')))
                reference_coords.append(decode_heatmaps(final_maps(stage).float(),
                     support=stage.local.support if stage.local else None).coords)
            reference_coords=torch.cat(reference_coords)
        for precision in ('fp32','bf16'):
            for batch in (1,2,4,8):
                for workers in (0,2,4,8):
                    if time.monotonic()-started>=limit: break
                    if any(r.get('precision')==precision and r['batch']==batch and r['workers']==workers for r in results):
                        continue
                    try:
                        item=benchmark(model,data,batch,workers,precision)
                        coordinates=[]
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                            for i in range(0,reference_batch['rgb'].shape[0],batch):
                                stage=model.forward_stages(*(reference_batch[k][i:i+batch] for k in ('rgb','depth','K')))
                                coordinates.append(decode_heatmaps(final_maps(stage).float(),
                                    support=stage.local.support if stage.local else None).coords)
                        difference=float((torch.cat(coordinates)-reference_coords).abs().max())
                        original_scale=max(model.config['input_size'][0]/model.config['heatmap_size'][0],
                                           model.config['input_size'][1]/model.config['heatmap_size'][1])
                        original_scale*=float(reference_batch['original_scale'].max())
                        item['max_coordinate_difference_original_px']=difference*original_scale
                        if precision=='bf16':
                            # Candidate AMP is retained only after full validation;
                            # original-coordinate tolerance is conservative.
                            score=evaluate_precision(model,data,precision,batch,workers)
                            item['validation']=score
                            item['accepted']=all(score['final'][s]['mean']<=reference['final'][s]['mean']+1e-3 and
                                score['final'][s]['p95']<=reference['final'][s]['p95']+1e-3 and
                                score['final'][s]['PCK05']>=reference['final'][s]['PCK05'] for s in ('OLD','NEW'))
                        else:
                            item['accepted']=item['max_coordinate_difference_original_px']<=1e-3
                        results.append(item)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        results.append(dict(batch=batch,workers=workers,precision=precision,accepted=False,error='OOM'))
                    except (ValueError,RuntimeError) as error:
                        results.append(dict(batch=batch,workers=workers,precision=precision,accepted=False,
                                            error=f'{type(error).__name__}: {error}'))
                    dump(output/'runtime.json',dict(reference=reference,configurations=results,
                         gpu=torch.cuda.get_device_name(),cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        accepted=[r for r in results if r.get('accepted')]
        if accepted:
            best=max(accepted,key=lambda r:np.median([x['images_per_second'] for x in r['file_to_coordinates']]))
            dump(output/'recommended_runtime.json',best)
        expected={(p,b,w) for p in ('fp32','bf16') for b in (1,2,4,8) for w in (0,2,4,8)}
        done={(r['precision'],r['batch'],r['workers']) for r in results}
        if done != expected:
            dump(output/'slice_complete.json',dict(status='resume_required',configurations=len(done)))
            return
        errors=[r for r in results if r.get('error')]
        dump(output/'completed.json',dict(status='completed_with_errors' if errors else 'complete',
                                         kind=task['kind'],errors=len(errors)))
        return
    dump(output/'completed.json',dict(status='complete',kind=task['kind']))


def evaluate_precision(model,data,precision,batch,workers):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
        return evaluate(model,data,batch_size=batch,workers=workers)
