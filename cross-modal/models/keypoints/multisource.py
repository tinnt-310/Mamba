"""Versioned, namespaced synthetic manifests and shared RGB-D preprocessing."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.nn import functional as F
from torch.utils.data import Dataset

from .heatmaps import gaussian_heatmaps, transform_points

NAMES = ['signal_top_left', 'signal_top_right', 'signal_middle_left',
         'signal_middle_center', 'signal_middle_right', 'signal_bottom_left',
         'signal_bottom_right', 'power_left', 'power_right']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def preprocess_rgbd(image, depth, K, input_hw):
    """The common PIL RGB/nearest depth, integer-center pinhole transform."""
    original_hw = (image.height, image.width)
    if depth.ndim != 2 or depth.shape != original_hw or depth.dtype != np.float32:
        raise ValueError('Paired depth must be float32 metric Z with original RGB shape')
    K = torch.as_tensor(K, dtype=torch.float32)
    if (K.shape != (3,3) or not torch.isfinite(K).all() or
            K[0,0] <= 0 or K[1,1] <= 0 or
            not torch.equal(K[2], torch.tensor([0.,0.,1.])) or K[0,1] != 0 or K[1,0] != 0):
        raise ValueError('Finite positive zero-skew pinhole K required')
    rgb = torch.from_numpy(np.asarray(image.convert('RGB').resize(
        (input_hw[1], input_hw[0]), Image.Resampling.BILINEAR), dtype=np.float32).copy()).permute(2,0,1)/255
    rgb = (rgb-rgb.new_tensor([.485,.456,.406])[:,None,None])/rgb.new_tensor([.229,.224,.225])[:,None,None]
    depth = torch.from_numpy(np.array(depth, copy=True))[None,None]
    depth = F.interpolate(depth, size=input_hw, mode='nearest')[0]
    depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, 0)
    sy, sx = input_hw[0]/original_hw[0], input_hw[1]/original_hw[1]
    affine = K.new_tensor([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]])
    return rgb, depth, affine@K


def load_manifest(path):
    manifest = read(path)
    if manifest.get('manifest_version') != 2 or manifest.get('keypoint_names') != NAMES:
        raise ValueError('Unsupported multi-source manifest or semantic order')
    ids, groups, hashes = set(), {}, {}
    for split in ('train','val','test'):
        for record in manifest[split]:
            if record['source'] not in ('OLD','NEW') or record['uid'] in ids:
                raise ValueError('Duplicate identity or non-synthetic training source')
            ids.add(record['uid'])
            for table, key in ((groups, record['group']), (hashes, record['image_sha256'])):
                if key in table and table[key] != split:
                    raise ValueError('Group/image content leaks across splits')
                table[key] = split
    return manifest


class MultiSourceDataset(Dataset):
    def __init__(self, manifest, split, config, *, augment=False, records=None, diagnostics=False, inference_only=False):
        self.records = list(manifest[split] if records is None else records)
        self.keypoint_names = manifest['keypoint_names']
        self.input_size, self.heatmap_size = tuple(config['input_size']), tuple(config['heatmap_size'])
        self.target_sigma = config.get('target_sigma',2.)
        self.augment = bool(augment)
        self.epoch = 0
        self.seed = int(manifest.get('seed', 20261025))
        self.diagnostics = diagnostics
        self.inference_only = inference_only
        if inference_only and (augment or diagnostics):
            raise ValueError('Inference reader cannot augment or consume label diagnostics')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        annotation = None if self.inference_only else read(record['annotation'])
        root = Path(record['root'])
        frame = record['frame_id']
        expected = {'rgb':f'rgb/{frame}.png', 'depth_sensor':f'depth_sensor/{frame}.npy',
                    'poses':f'poses/{frame}.json'}
        if annotation is not None and (annotation.get('frame_id') != frame or any(annotation['paths'].get(k) != v for k,v in expected.items())):
            raise ValueError('Paired annotation paths/frame ID changed')
        with Image.open(root/expected['rgb']) as handle:
            image = handle.convert('RGB')
        depth = np.load(root/expected['depth_sensor'], allow_pickle=False)
        K = torch.tensor(read(root/expected['poses'])['K'], dtype=torch.float32)
        original_hw = (image.height, image.width)
        if self.inference_only:
            rgb,depth_tensor,K=preprocess_rgbd(image,depth,K,self.input_size)
            return dict(frame_id=record['uid'],source=record['source'],rgb=rgb,depth=depth_tensor,
                        K=K,original_size=torch.tensor(original_hw))
        labels = {p['name']:p for p in annotation['keypoints_2d']}
        if len(labels) != len(annotation['keypoints_2d']) or set(labels) != set(NAMES):
            raise ValueError('Semantic labels missing/duplicated')
        points = torch.tensor([labels[name]['uv'] for name in NAMES], dtype=torch.float32)
        original_points = points.clone()
        diagnostic = None
        if self.diagnostics:
            depths = []
            for u,v in points.tolist():
                x,y = round(u),round(v)
                z = float(depth[y,x]) if 0<=x<image.width and 0<=y<image.height else 0.
                depths.append(z if np.isfinite(z) and z>0 else 0.)
            depth_kinds=[]
            for i,name in enumerate(NAMES):
                if 'depth_m' in labels[name]:
                    z=float(labels[name]['depth_m'])
                    depths[i]=z if np.isfinite(z) and z>0 else 0.
                    depth_kinds.append('synthetic_keypoint_GT')
                else:
                    depth_kinds.append('sensor_at_label')
            diagnostic = dict(group=record['group'],bbox=annotation['object_bbox_xyxy'],
                              visibility=[labels[n]['visibility'] for n in NAMES],
                              depth_fraction=float((np.isfinite(depth)&(depth>0)).mean()),
                              gt_depth=depths,depth_reference=depth_kinds,fx=float(K[0,0]),fy=float(K[1,1]),
                              gt=points.tolist())
        weights = torch.tensor([float(labels[name]['visibility'] in (1,2)) for name in NAMES])
        weights *= (torch.isfinite(points).all(-1) & (points[:,0]>=-.5) &
                    (points[:,0]<image.width-.5) & (points[:,1]>=-.5) & (points[:,1]<image.height-.5))
        points = torch.nan_to_num(points)
        if self.augment:
            seed = int(hashlib.sha256(f'{self.seed}:{self.epoch}:{record["uid"]}'.encode()).hexdigest()[:16],16)
            rng = random.Random(seed)
            # Crop/resize changes RGB, depth, keypoints and K with one affine.
            ratio = rng.uniform(.85,1.)
            ch, cw = max(1,round(image.height*ratio)), max(1,round(image.width*ratio))
            left, top = rng.randint(0,image.width-cw), rng.randint(0,image.height-ch)
            image = image.crop((left,top,left+cw,top+ch))
            depth = depth[top:top+ch,left:left+cw].copy()
            points = points-points.new_tensor([left,top])
            K = K.clone(); K[0,2] -= left; K[1,2] -= top
            weights *= ((points[:,0]>=-.5)&(points[:,0]<cw-.5)&(points[:,1]>=-.5)&(points[:,1]<ch-.5))
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enhancer(image).enhance(rng.uniform(.7,1.3))
            generator = np.random.default_rng(seed)
            valid = np.isfinite(depth) & (depth>0)
            depth[valid] += generator.normal(0,.002,int(valid.sum())).astype(np.float32)
            depth[generator.random(depth.shape)<.05] = 0
            if rng.random()<.15:
                depth[:] = 0
        rgb, depth_tensor, K = preprocess_rgbd(image,depth,K,self.input_size)
        transformed = transform_points(points,(image.height,image.width),self.input_size)
        target_points = transform_points(points,(image.height,image.width),self.heatmap_size)
        scale = points.new_tensor([original_hw[1]/self.input_size[1],original_hw[0]/self.input_size[0]])
        box = annotation['object_bbox_xyxy']
        diagonal = float(np.linalg.norm(np.asarray(box[2:])-np.asarray(box[:2])))
        result = dict(frame_id=record['uid'], source=record['source'], rgb=rgb, depth=depth_tensor,
                    K=K, points=transformed, target=gaussian_heatmaps(target_points,self.heatmap_size,self.target_sigma,weights),
                    weights=weights, original_points=original_points, original_size=torch.tensor(original_hw),
                    original_scale=scale, bbox_diagonal=torch.tensor(diagonal),
                    cache_key=f'{record["uid"]}:{self.input_size}:{self.epoch if self.augment else "fixed"}')
        if diagnostic is not None:
            result['diagnostics'] = json.dumps(diagnostic)
        return result


def prepare(output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Manifest hashing/processing requires Slurm compute')
    output = Path(output)
    if output.exists():
        load_manifest(output)
        return
    old = Path('/work/tinnt/BlenderProc/output/keypoints_bbox_50000_job_1634')
    new_index = read('/work/tinnt/CCS2_20261005_50K_ABLATION/manifest.json')
    historical_new = read('/work/tinnt/MAMBA_NEW50K_SUBSET5000_20261005/selection.json')['selected']
    protected = {(str(Path(r['source_root']).resolve()),r['frame_id']):r['split'] for r in historical_new}
    rng = random.Random(20261025)
    pools = {source:{split:[] for split in ('train','val','test')} for source in ('OLD','NEW')}
    for split in ('train','val','test'):
        for frame in (old/f'{split}.txt').read_text().split():
            pools['OLD'][split].append((old,frame))
    for item in new_index:
        if item['group'] != 'synthetic':
            continue
        root = Path(item['source_image']).parent.parent
        frame = Path(item['source_image']).stem
        key = (str(root.resolve()),frame)
        split = protected.get(key, 'val' if item['split']=='val' else 'unassigned')
        if split == 'unassigned':
            # Reserve a test candidate pool before sampling train. Historical
            # Mamba train frames keep train membership through `protected`.
            split = 'test' if int(hashlib.sha256(f'{root}:{frame}'.encode()).hexdigest()[:8],16)%10 == 0 else 'train'
        pools['NEW'][split].append((root,frame))
    manifest = dict(manifest_version=2,seed=20261025,keypoint_names=NAMES,
                    policy='25k/source; source splits and historical Mamba holdouts preserved; '
                           'grouped by render seed when scene/parent metadata absent',
                    independence_limitation='No verified physical scene/session IDs in synthetic metadata; '
                                            'render seed grouping is not proof of scene independence',
                    train=[],val=[],test=[])
    used_hashes, used_groups, observed = set(), {}, set()
    for sample in read('/home/tinnt/model_comparison_20261025/samples.json'):
        if sample['dataset'] != 'D455':
            observed.add(sample['image_sha256'])
    for source in ('OLD','NEW'):
        # Holdouts get first claim if a render/content duplicate crosses source splits.
        for split, count in (('test',2500),('val',2500),('train',20000)):
            candidates = sorted(pools[source][split],key=lambda v:(str(v[0]),v[1]))
            rng.shuffle(candidates)
            selected = 0
            for root,frame in candidates:
                annotation_path = root/'annotations'/f'{frame}.json'
                metadata = read(root/'metadata'/f'{frame}.json')
                group = f'{source}:render:{metadata["seed"]}'
                if group in used_groups and used_groups[group] != split:
                    continue
                image_path = root/'rgb'/f'{frame}.png'
                image_hash = digest(image_path)
                if image_hash in used_hashes:
                    continue
                annotation = read(annotation_path)
                if {p['name'] for p in annotation['keypoints_2d']} != set(NAMES):
                    raise ValueError(f'Semantic order mismatch: {annotation_path}')
                record = dict(uid=f'{source}_{root.parent.name}_{frame}',source=source,root=str(root),
                              frame_id=frame,group=group,annotation=str(annotation_path),
                              image_sha256=image_hash,annotation_sha256=digest(annotation_path),
                              depth_sha256=digest(root/'depth_sensor'/f'{frame}.npy'),
                              calibration_sha256=digest(root/'poses'/f'{frame}.json'),
                              observed=image_hash in observed)
                manifest[split].append(record)
                used_hashes.add(image_hash); used_groups[group] = split
                selected += 1
                if selected%500 == 0:
                    print(json.dumps(dict(source=source,split=split,selected=selected)),flush=True)
                if selected == count:
                    break
            if selected != count:
                raise ValueError(f'Insufficient unique {source}/{split}: {selected}/{count}; refusing silent leakage')
    manifest['counts'] = {s:dict(Counter(r['source'] for r in manifest[s])) for s in ('train','val','test')}
    dump(output,manifest)
    load_manifest(output)
    dump(output.with_name('manifest_audit.json'),dict(sha256=digest(output),counts=manifest['counts'],
         observed={s:sum(r['observed'] for r in manifest[s]) for s in ('train','val','test')},
         independence_limitation=manifest['independence_limitation']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    prepare(parser.parse_args().output)
