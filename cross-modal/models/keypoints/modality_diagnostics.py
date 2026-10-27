"""Validation-only modality interventions; not independently trained RGB baselines."""
from __future__ import annotations
import hashlib
from contextlib import contextmanager

import torch
from torch.utils.data import Dataset

from .local_losses import weighted_heatmap_loss
from .trial_runner import evaluate,loader,transfer,final_maps
from .multisource import MultiSourceDataset,load_manifest,dump,digest
from .checkpoints import load_checkpoint


CONDITIONS=('clean','depth_invalid','depth_noise_2mm','depth_hole_25pct',
            'rgb_half_brightness','rgb_occlusion_25pct','depth_features_zero','rgb_features_zero')


class InterventionDataset(Dataset):
    def __init__(self,data,condition):
        if condition not in CONDITIONS: raise ValueError('Unknown diagnostic intervention')
        self.data,self.condition=data,condition
        self.input_size=data.input_size

    def __len__(self): return len(self.data)

    def __getitem__(self,index):
        value=self.data[index]
        seed=int(hashlib.sha256(('modality-v1:'+value['frame_id']).encode()).hexdigest()[:15],16)
        generator=torch.Generator().manual_seed(seed)
        z=value['depth'].clone(); rgb=value['rgb'].clone()
        valid=torch.isfinite(z)&(z>0)
        if self.condition=='depth_invalid': z.zero_()
        elif self.condition=='depth_noise_2mm':
            z=torch.where(valid,(z+torch.randn(z.shape,generator=generator)*.002).clamp_min(0),0)
        elif self.condition in ('depth_hole_25pct','rgb_occlusion_25pct'):
            h,w=z.shape[-2:]; ch,cw=max(1,h//2),max(1,w//2)
            y=int(torch.randint(h-ch+1,(),generator=generator)); x=int(torch.randint(w-cw+1,(),generator=generator))
            if self.condition=='depth_hole_25pct': z[:,y:y+ch,x:x+cw]=0
            else: rgb[:,y:y+ch,x:x+cw]=0  # ImageNet mean in normalized space.
        elif self.condition=='rgb_half_brightness':
            mean=rgb.new_tensor([.485,.456,.406])[:,None,None]
            std=rgb.new_tensor([.229,.224,.225])[:,None,None]
            rgb=((rgb*std+mean)*.5-mean)/std
        value={**value,'rgb':rgb,'depth':z}
        return value


@contextmanager
def feature_intervention(model,condition):
    handle=None
    if condition in ('rgb_features_zero','depth_features_zero'):
        module=model.rgb_encoder if condition=='rgb_features_zero' else model.geometry_encoder
        def zero(module,args,outputs):
            if not isinstance(outputs,(tuple,list)): raise ValueError('Expected multiscale features')
            return tuple(torch.zeros_like(x) for x in outputs)
        handle=module.register_forward_hook(zero)
    try: yield
    finally:
        if handle is not None: handle.remove()


def run(root,checkpoint,output,per_source=256):
    """Fixed source-balanced subset, defined by UID hash rather than observed errors."""
    from .multisource import read
    from .regression import paired_rows,summarize,geometry_metrics
    manifest=load_manifest(root/'manifest50k.json')
    records=[]
    for source in ('OLD','NEW'):
        pool=[r for r in manifest['val'] if r['source']==source]
        records.extend(sorted(pool,key=lambda r:hashlib.sha256(('modality-v1:'+r['uid']).encode()).hexdigest())[:per_source])
    model,payload=load_checkpoint(checkpoint); model.cuda().eval()
    if model.config['modality']!='rgbd': raise ValueError('Diagnostics require a trained RGB-D model')
    data=MultiSourceDataset(manifest,'val',model.config,records=records,diagnostics=True)
    rows={}; gates={}; handles=[]; clean_records=None; comparisons={}
    # Capture learned gates only on clean inputs, without retaining feature maps.
    def capture(name):
        def hook(module,args,value):
            if isinstance(value,torch.Tensor):
                flat=value.detach().float()
                gates.setdefault(name,[]).append((float(flat.mean()),float(flat.std(unbiased=False)),
                    float((flat<.05).float().mean()),float((flat>.95).float().mean())))
        return hook
    for name,module in model.named_modules():
        if name.endswith('.d2r') or name.endswith('.r2d'):
            handles.append(module.register_forward_hook(capture(name)))
    for condition in CONDITIONS:
        if condition!='clean':
            for h in handles: h.remove()
            handles=[]
        with feature_intervention(model,condition):
            prediction_path=output.parent/(output.stem+'_'+condition+'_predictions.json')
            rows[condition]=evaluate(model,InterventionDataset(data,condition),batch_size=8,workers=2,
                records_path=prediction_path)
            current=read(prediction_path)['records']
            rows[condition]['mm']={source:geometry_metrics([r for r in current if r['source']==source])
                                   for source in ('OLD','NEW')}
            if condition=='clean': clean_records=current
            else:
                pairs=paired_rows(clean_records,current)
                comparisons[condition]={source:summarize(subset)
                    for source in ('OLD','NEW')
                    if (subset:=[p for p in pairs if p[0]['source']==source])}
    # Attribution uses TRAIN only, eval BN, no parameter updates.
    calibration=MultiSourceDataset(manifest,'train',model.config)
    batch=transfer(next(iter(loader(calibration,8,0))))
    # A frozen-Global Local checkpoint still needs attribution to both encoders.
    # Enable autograd only; retain eval BatchNorm and never step an optimizer.
    model.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    scale_features={}; attribution_handles=[]
    def capture_scales(name):
        def hook(module,args,values):
            scale_features[name]=list(values)
            for value in values:
                if value.requires_grad: value.retain_grad()
        return hook
    for name,module in [('rgb',model.rgb_encoder),('depth',model.geometry_encoder)]:
        attribution_handles.append(module.register_forward_hook(capture_scales(name)))
    rgb=batch['rgb'].detach().requires_grad_(True)
    depth=batch['depth'].detach().requires_grad_(True)
    stage=model.forward_stages(rgb,depth,batch['K'])
    loss=weighted_heatmap_loss(final_maps(stage),batch['target'],batch['weights'])
    loss.backward()
    for handle in attribution_handles: handle.remove()
    scale_attribution={name:[dict(shape=list(value.shape),gradient_norm=float(value.grad.norm()) if value.grad is not None else None,
                                 finite=bool(torch.isfinite(value.grad).all()) if value.grad is not None else None)
                            for value in values] for name,values in scale_features.items()}
    norms={}
    for name,module in [('rgb',model.rgb_encoder),('depth',model.geometry_encoder),('fusion',model.fusion)]:
        gradients=[p.grad.detach().float() for p in module.parameters() if p.grad is not None]
        norms[name]=dict(norm=float(torch.stack([g.norm().square() for g in gradients]).sum().sqrt()) if gradients else 0.,
            finite=all(bool(torch.isfinite(g).all()) for g in gradients),tensors=len(gradients))
    input_attribution={name:dict(mean_absolute=float(value.grad.abs().mean()),finite=bool(torch.isfinite(value.grad).all()))
                       for name,value in [('rgb',rgb),('depth',depth)] if value.grad is not None}
    summaries={name:dict(mean=sum(v[0] for v in values)/len(values),std_batch_mean=sum(v[1] for v in values)/len(values),
                        fraction_below_005=sum(v[2] for v in values)/len(values),
                        fraction_above_095=sum(v[3] for v in values)/len(values)) for name,values in gates.items()}
    dump(output,dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        manifest_sha256=digest(root/'manifest50k.json'),subset_uids=[r['uid'] for r in records],
        conditions=rows,paired_comparisons=comparisons,gate_summary=summaries,train_gradient_norms=norms,
        train_scale_attribution=scale_attribution,input_gradient=input_attribution,
        interpretation='Inference interventions are diagnostic, not trained RGB-only/Depth-only baselines; '
                       'nonzero gates or sensitivity alone do not establish localization benefit'))


def add_intervals(output):
    """Run on CPU compute after the GPU export, using cached paired predictions."""
    from .multisource import read
    from .regression import paired_rows,bootstrap
    value=read(output)
    if value.get('interval_policy') and all('ci95' in value['paired_comparisons'].get(condition,{}).get(source,{})
        for condition in CONDITIONS if condition!='clean' for source in ('OLD','NEW')):
        return
    clean=read(output.parent/(output.stem+'_clean_predictions.json'))['records']
    for condition in CONDITIONS:
        if condition=='clean': continue
        changed=read(output.parent/(output.stem+'_'+condition+'_predictions.json'))['records']
        pairs=paired_rows(clean,changed)
        for source in ('OLD','NEW'):
            value['paired_comparisons'][condition][source].update(bootstrap([p for p in pairs if p[0]['source']==source]))
    value['interval_policy']='2000 paired group bootstrap on fixed synthetic validation subset; exploratory interventions'
    dump(output,value)


def summarize_cached(root):
    """CPU-only paired analysis of already exported intervention predictions."""
    from .multisource import read
    from .regression import paired_rows,summarize,bootstrap
    out=root/'architecture'; summaries={}
    for path in sorted((out/'tasks').glob('modality_*/modality.json')):
        add_intervals(path)
        value=read(path)
        clean=read(path.parent/'modality_clean_predictions.json')['records']
        for condition in CONDITIONS:
            if condition=='clean': continue
            changed=read(path.parent/('modality_'+condition+'_predictions.json'))['records']
            pairs=paired_rows(clean,changed)
            for source in ('OLD','NEW'):
                paired=summarize([p for p in pairs if p[0]['source']==source])
                value['paired_comparisons'][condition][source]['mm']=paired['mm']
        summaries[path.parent.name]=dict(checkpoint=value['checkpoint'],checkpoint_sha256=value['checkpoint_sha256'],
            clean=value['conditions']['clean'],paired_comparisons=value['paired_comparisons'],
            gradients=value['train_gradient_norms'])
    parent=out/'tasks/modality_F_candidate_20261025/modality_clean_predictions.json'
    local=out/'tasks/modality_A_local_conv_20261025/modality_clean_predictions.json'
    paired=None
    if parent.exists() and local.exists():
        pairs=paired_rows(read(parent)['records'],read(local)['records'])
        paired={source:{**summarize(subset),**bootstrap(subset)} for source in ('OLD','NEW')
                if (subset:=[p for p in pairs if p[0]['source']==source])}
    dump(out/'modality_summary.json',dict(diagnostics=summaries,local_vs_parent=paired,
        limitation='UID-hash subset of validation; interventions are diagnostic and not trained single-modality baselines'))
    for name,value in summaries.items():
        print(name,flush=True)
        for source in ('OLD','NEW'):
            for condition in ('depth_invalid','depth_features_zero','rgb_features_zero'):
                row=value['paired_comparisons'][condition][source]
                print(source,condition,'delta mean',row['delta']['mean'],'CI95',row['ci95']['mean'],flush=True)


if __name__=='__main__':
    import argparse
    import os
    from pathlib import Path
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('CPU compute allocation required')
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); summarize_cached(args.root)
