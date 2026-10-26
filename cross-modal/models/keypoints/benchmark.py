"""Batch-one runtime/VRAM benchmark, including local stage components."""
from __future__ import annotations
import time
import torch


def benchmark_model(model, dataset, *, device='cuda'):
    from .heatmaps import transform_points
    from .local_losses import local_patch_loss, weighted_heatmap_loss
    from .regions import extract_regions
    rows = []
    is_cuda = str(device).startswith('cuda')
    def synchronize():
        if is_cuda: torch.cuda.synchronize()
    for sample in dataset:
        model.train(); model.set_phase('joint' if model.local_enabled else 'global')
        # Batch one SE-ResNet requires eval BatchNorm for ultra-small smoke images.
        rgb,depth,K = (sample[key].unsqueeze(0).to(device) for key in ('rgb','depth','K'))
        target,weights = sample['target'].unsqueeze(0).to(device),sample['weights'].unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        if is_cuda: torch.cuda.reset_peak_memory_stats()
        if model.local_enabled: model.local_refinement.profile = True
        synchronize(); started=time.perf_counter()
        fused = model.encode_fused(rgb,depth,K)
        global_maps = model.global_from_fused(fused)
        synchronize(); global_seconds=time.perf_counter()-started
        started=time.perf_counter()
        local = model.refine_cached(fused[0],global_maps,return_patches=True).local if model.local_enabled else None
        synchronize(); local_seconds=time.perf_counter()-started
        loss=weighted_heatmap_loss(global_maps,target,weights,positive_weight=100.)
        if local is not None:
            points=transform_points(sample['points'].unsqueeze(0).to(device),rgb.shape[-2:],fused[0].shape[-2:])
            loss=loss+local_patch_loss(local.patches,points,weights,fused[0].shape[-2:])
        synchronize(); started=time.perf_counter()
        loss.backward()
        synchronize(); backward_seconds=time.perf_counter()-started
        row={'frame_id':sample['frame_id'],'global_forward_seconds':global_seconds,
             'local_forward_seconds':local_seconds,'backward_seconds':backward_seconds,
             'total_train_seconds':global_seconds+local_seconds+backward_seconds,
             'peak_vram_bytes':int(torch.cuda.max_memory_allocated()) if is_cuda else 0,
             'fused_shapes':[list(t.shape) for t in fused],'global_shape':list(global_maps.shape),
             'loss':float(loss.detach())}
        if local is not None:
            row['local_shape']=list(local.heatmaps.shape)
            row['patch_count']=len(local.patches)
            lengths=[p.box.width*p.box.height for p in local.patches]
            row['mean_sequence_length']=sum(lengths)/len(lengths)
            row['max_sequence_length']=max(lengths)
            row['local_components']=getattr(model.local_refinement,'last_timings',{})
            model.local_refinement.profile=False
        row['all_gradients_finite']=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        rows.append(row)
        print(f'benchmark {sample["frame_id"]}: {row["total_train_seconds"]:.3f}s, VRAM={row["peak_vram_bytes"]}',flush=True)
        model.zero_grad(set_to_none=True)
    mean = sum(row['total_train_seconds'] for row in rows)/len(rows)
    return {'device':str(device),'batch_size':1,'backend':model.config['global_ssm']['backend'],
            'config':model.config,'images':rows,'mean_train_seconds_per_image':mean,
            'estimated_4000_image_epoch_hours':mean*4000/3600,
            'peak_vram_bytes':max(row['peak_vram_bytes'] for row in rows),
            'note':'Single measured pass includes first-use overhead; estimate excludes validation and disk I/O.'}
