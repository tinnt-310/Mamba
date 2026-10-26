"""GPU smoke/parity/gradient checks and immutable provenance preflight."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from models.keypoints.multisource import read,dump,digest,load_manifest,MultiSourceDataset
from models.keypoints.config import make_config
from models.keypoints.model import RGBDKeypointModel
from models.keypoints.checkpoints import load_checkpoint
from models.keypoints.heatmaps import decode_heatmaps,transform_points
from models.keypoints.local_losses import weighted_heatmap_loss
from models.keypoints.campaign import pilot_tasks


def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(1)
    root=Path(sys.argv[1]); manifest=load_manifest(root/'manifest50k.json')
    pretrained=root/'pretrained/seresnet50_a1_0-ffa00869.pth'
    hash_value=digest(pretrained)
    if not hash_value.startswith('ffa00869'): raise ValueError('Official pretrained hash prefix mismatch')
    dump(root/'pretrained/provenance.json',dict(sha256=hash_value,
         url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a1_0-ffa00869.pth',
         architecture='seresnet50.a1_in1k',verified_filename_hash=True))
    profile=read(root/'baseline/A1_NEW/profile.json')
    outcomes=[]
    for t in pilot_tasks(root,profile):
        config=make_config(t['config']); config['input_size']=(48,64); config['heatmap_size']=(24,32)
        dataset=MultiSourceDataset(manifest,'train',config,augment=t.get('augment',False))
        sample=dataset[0]
        model=RGBDKeypointModel(config).cuda().eval()
        model.set_phase(t.get('phase','global'))
        inputs=tuple(sample[k][None].cuda() for k in ('rgb','depth','K'))
        stage=model.forward_stages(*inputs)
        maps=stage.final_heatmaps
        if model.residual_local is not None and t.get('phase')=='local':
            assert torch.equal(maps,stage.global_heatmaps)
        loss=weighted_heatmap_loss(maps,sample['target'][None].cuda(),sample['weights'][None].cuda())
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite gradients in '+t['id'])
        outcomes.append(dict(trial=t['id'],finite_loss=float(loss),parameters=sum(p.numel() for p in model.parameters())))
        del model,stage,maps,loss
        torch.cuda.empty_cache()
    # Compare new legacy loading/forward to retained source artifact coordinates.
    historical=Path('/home/tinnt/mamba/cross-modal/runs/ablation5000_new50k_cuda_20261005')
    subset=read('/work/tinnt/MAMBA_NEW50K_SUBSET5000_20261005/subset5000.json')
    from models.keypoints.data import RGBDKeypointDataset
    data_root=Path(subset['dataset_root'])
    max_error=0.
    for variant,phase in [('A1','global'),('A4','joint')]:
        model,_=load_checkpoint(historical/variant/phase/'best.pt'); model.cuda().eval()
        dataset=RGBDKeypointDataset(data_root,subset['test'][:10],subset['keypoint_names'])
        with torch.inference_mode():
            for sample in dataset:
                old=read(historical/variant/'test_artifacts'/f'{sample["frame_id"]}.json')
                stage=model.forward_stages(*(sample[k][None].cuda() for k in ('rgb','depth','K')))
                maps=stage.local.heatmaps if stage.local else stage.global_heatmaps
                coords=decode_heatmaps(maps,support=stage.local.support if stage.local else None).coords[0]
                coords=transform_points(coords,maps.shape[-2:],sample['original_size'].tolist())
                prefix='local' if stage.local else 'global'
                expected=torch.tensor(old['coordinates'][prefix]['subpixel']['original'],device='cuda')
                max_error=max(max_error,float((coords-expected).abs().max()))
        del model
    if max_error>1e-3: raise ValueError(f'Legacy parity failed: {max_error}')
    environment=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (root/'environment.txt').write_text(environment)
    dump(root/'gpu_preflight_passed.json',dict(job_id=os.environ['SLURM_JOB_ID'],trials=outcomes,
         legacy_max_abs_original_px=max_error,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
         environment_sha256=digest(root/'environment.txt')))


if __name__=='__main__': main()
