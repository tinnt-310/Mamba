"""Decode and validate every authorized frame before a long GPU campaign."""
from pathlib import Path
import time
import torch
from .data import RGBDKeypointDataset


def validate_subset(root, manifest):
    from .experiment import check_deadline
    root = Path(root).resolve()
    if root != Path(manifest['dataset_root']).resolve():
        raise ValueError('validation dataset_root differs from manifest')
    total = sum(len(manifest[split]) for split in ('train','val','test'))
    if total != manifest['limit'] or total > 5000:
        raise ValueError('validation must use only the persisted subset of at most5000')
    names = manifest['keypoint_names']
    valid_labels = torch.zeros(len(names),dtype=torch.int64)
    original_sizes = set()
    depth_valid_sum = 0.
    zero_depth_frames = 0
    split_counts = {}
    checked = 0
    started = time.perf_counter()
    for split in ('train','val','test'):
        dataset = RGBDKeypointDataset(root,manifest[split],names)
        split_counts[split] = len(dataset)
        split_valid_labels = 0
        for sample in dataset:
            check_deadline()
            if not torch.isfinite(sample['rgb']).all() or not torch.isfinite(sample['target']).all():
                raise ValueError(f'nonfinite RGB/target in frame {sample["frame_id"]}')
            K=sample['K']
            if K[0,1] != 0 or K[1,0] != 0:
                raise ValueError(f'unsupported camera skew in frame {sample["frame_id"]}')
            labels=sample['weights']>0
            valid_labels+=labels.to(torch.int64)
            split_valid_labels+=int(labels.sum())
            original_sizes.add(tuple(int(v) for v in sample['original_size']))
            depth_valid=torch.isfinite(sample['depth']) & (sample['depth']>0)
            fraction=float(depth_valid.float().mean())
            depth_valid_sum+=fraction
            zero_depth_frames+=int(fraction==0)
            checked+=1
            if checked%500==0 or checked==total:
                print(f'dataset validation: {checked}/{total}, {time.perf_counter()-started:.1f}s',flush=True)
        if not split_valid_labels:
            raise ValueError(f'{split} split has no valid keypoint labels')
    return {'passed':True,'dataset_root':str(root),'images_checked':checked,
            'counts':split_counts,'source_split_sha256':manifest.get('source_split_sha256'),
            'seed':manifest['seed'],'input_size':[180,320],'original_sizes':[list(v) for v in sorted(original_sizes)],
            'valid_labels_per_keypoint':dict(zip(names,valid_labels.tolist())),
            'mean_resized_depth_valid_fraction':depth_valid_sum/checked,
            'frames_with_no_valid_depth':zero_depth_frames,'seconds':time.perf_counter()-started}
