"""Training step and I/O profile before selecting loader/batch optimizations."""
import os
import argparse
import resource
from pathlib import Path
import time
import torch
import numpy as np
from models.keypoints.checkpoints import load_checkpoint
from models.keypoints.multisource import load_manifest,MultiSourceDataset,dump
from models.keypoints.trial_runner import loader,transfer,optimizer,coordinate_loss
from models.keypoints.local_losses import weighted_heatmap_loss


def compare_unused_coordinate(root,checkpoint,output):
    """Paired step benchmark, no dataset/checkpoint updates on disk."""
    torch.set_num_threads(1)
    manifest=load_manifest(root/'manifest50k.json')
    model,_=load_checkpoint(checkpoint)
    data=MultiSourceDataset(manifest,'train',model.config)
    batch=transfer(next(iter(loader(data,8,0))))
    rows=[]; reference_state=None; max_difference=0.
    for repeat in range(3):
        # Reverse order to reduce a systematic first-run timing advantage.
        for unused in ((True,False) if repeat%2==0 else (False,True)):
            model,_=load_checkpoint(checkpoint); model.cuda().set_phase('global').train()
            opt=optimizer(model,'global'); times=[]
            torch.cuda.reset_peak_memory_stats()
            for step in range(35):
                torch.cuda.synchronize(); start=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                maps=model(*(batch[k] for k in ('rgb','depth','K')))
                loss=weighted_heatmap_loss(maps,batch['target'],batch['weights'])
                if unused:
                    loss=loss+0.*coordinate_loss(maps,batch['points'],model.config['input_size'],batch['weights'])
                loss.backward(); opt.step(); torch.cuda.synchronize()
                if step>=10: times.append(time.perf_counter()-start)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if reference_state is None: reference_state=state
            else:
                max_difference=max(max_difference,max(float((v-reference_state[k]).abs().max())
                    for k,v in state.items() if v.is_floating_point()))
            rows.append(dict(repeat=repeat,unused_coordinate=unused,p50_ms=float(np.median(times)*1000),
                p95_ms=float(np.percentile(times,95)*1000),images_per_second=8/float(np.mean(times)),
                allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()))
            del model,opt,maps,loss
            torch.cuda.empty_cache()
    old=np.median([r['p50_ms'] for r in rows if r['unused_coordinate']])
    new=np.median([r['p50_ms'] for r in rows if not r['unused_coordinate']])
    dump(output,dict(rows=rows,max_weight_difference=max_difference,
         speedup_fraction=float(1-new/old),effective_batch=8,gpu=torch.cuda.get_device_name(),
         cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         allocated_cpus=int(os.environ.get('SLURM_CPUS_PER_TASK','1')),
         note='Paired GPU-resident training steps, not time-to-convergence; checkpoint files unchanged'))


def main():
    assert os.environ.get('SLURM_JOB_ID')
    parser=argparse.ArgumentParser()
    parser.add_argument('--compare-unused-coordinate',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    root=Path('/home/tinnt/mamba/cross-modal/runs/improve_20261025')
    if args.compare_unused_coordinate:
        compare_unused_coordinate(root,args.checkpoint,args.output)
        return
    manifest=load_manifest(root/'manifest50k.json')
    path=Path('/home/tinnt/mamba/cross-modal/runs/ablation5000_new50k_cuda_20261005/A1/global/best.pt')
    rows=[]
    for batch,workers in ((1,0),(1,4),(8,0),(8,4)):
        model,_=load_checkpoint(path); model.cuda().set_phase('global').train()
        opt=optimizer(model,'global')
        data=MultiSourceDataset(manifest,'train',model.config)
        batches=iter(loader(data,batch,workers))
        measured={k:[] for k in ('data_wait','h2d','forward','backward','optimizer','total')}
        torch.cuda.reset_peak_memory_stats()
        for index in range(55):
            torch.cuda.synchronize(); began=time.perf_counter()
            sample=next(batches); loaded=time.perf_counter()
            sample=transfer(sample); torch.cuda.synchronize(); transferred=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            maps=model(*(sample[k] for k in ('rgb','depth','K')))
            loss=weighted_heatmap_loss(maps,sample['target'],sample['weights'])
            torch.cuda.synchronize(); forwarded=time.perf_counter()
            loss.backward(); torch.cuda.synchronize(); backwarded=time.perf_counter()
            opt.step(); torch.cuda.synchronize(); ended=time.perf_counter()
            if index>=5:
                for key,value in [('data_wait',loaded-began),('h2d',transferred-loaded),
                      ('forward',forwarded-transferred),('backward',backwarded-forwarded),
                      ('optimizer',ended-backwarded),('total',ended-began)]:
                    measured[key].append(value)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
             torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
            opt.zero_grad(set_to_none=True)
            weighted_heatmap_loss(model(*(sample[k] for k in ('rgb','depth','K'))),sample['target'],sample['weights']).backward()
            opt.step()
        prof.export_chrome_trace(str(root/'baseline'/f'train-b{batch}-w{workers}.json'))
        rows.append(dict(batch=batch,workers=workers,mean_ms={k:float(np.mean(v)*1000) for k,v in measured.items()},
             images_per_second=batch/np.mean(measured['total']),allocated_bytes=torch.cuda.max_memory_allocated(),
             reserved_bytes=torch.cuda.max_memory_reserved(),
             note='Step benchmark only: different microbatches change BatchNorm; no accuracy equivalence claim'))
        del batches,model,opt,loss,maps
        torch.cuda.empty_cache()
        dump(root/'baseline/training_profile.json',dict(configurations=rows,gpu=torch.cuda.get_device_name()))


if __name__=='__main__': main()
