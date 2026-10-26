"""Training step and I/O profile before selecting loader/batch optimizations."""
import os
from pathlib import Path
import time
import torch
import numpy as np
from models.keypoints.checkpoints import load_checkpoint
from models.keypoints.multisource import load_manifest,MultiSourceDataset,dump
from models.keypoints.trial_runner import loader,transfer,optimizer
from models.keypoints.local_losses import weighted_heatmap_loss


def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(1)
    root=Path('/home/tinnt/mamba/cross-modal/runs/improve_20261025')
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
