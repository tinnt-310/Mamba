"""Behavioral checks for the new campaign, run through Slurm."""
import copy
import json
import numpy as np
import pytest
import torch
from PIL import Image

from models.keypoints.config import make_config
from models.keypoints.residual_local import ResidualLocal
from models.keypoints.quality_fusion import QualityFusion
from models.keypoints.multisource import preprocess_rgbd,load_manifest,NAMES


def test_residual_identity_support_and_semantic_conditioning():
    torch.manual_seed(42)
    model = ResidualLocal(4,4,2,roi_size=5,state_dim=2,conditioning=True)
    maps = torch.zeros(1,2,9,11)
    maps[:,:,4,5] = 1
    feature = torch.randn(1,4,5,6,requires_grad=True)
    result = model(maps,feature)
    assert torch.equal(result.heatmaps,maps)
    assert result.support.sum()==50
    result.heatmaps.square().mean().backward()
    assert model.delta_head.weight.grad.abs().sum()>0
    torch.nn.init.normal_(model.delta_head.weight,std=.1)
    changed = model(maps,feature)
    assert not torch.equal(changed.delta[:,0],changed.delta[:,1])
    assert torch.equal(changed.delta[~changed.support],torch.zeros_like(changed.delta[~changed.support]))
    assert torch.equal(model(maps,feature,gate_override=0).heatmaps,maps)
    model.zero_grad(); feature.grad=None
    changed.heatmaps.square().mean().backward()
    assert feature.grad.abs().sum()>0


@pytest.mark.parametrize('peak',[(0,0),(8,10),(4,5)])
def test_residual_border_odd_shape_and_gradient(peak):
    model = ResidualLocal(2,2,1,roi_size=5,state_dim=1)
    maps = torch.zeros(2,1,9,11)
    maps[:,:,peak[0],peak[1]]=1
    torch.nn.init.constant_(model.delta_head.bias,1.)
    output = model(maps,torch.randn(2,2,5,6))
    assert torch.isfinite(output.heatmaps).all()
    assert output.support.sum(dim=(-2,-1)).min()>=9
    torch.testing.assert_close(output.delta[output.support],torch.full_like(output.delta[output.support],.5))


def test_quality_depth_invalid_has_exact_rgb_identity_even_with_cross_bias():
    fusion = QualityFusion(3,2,4,state_dim=2,expand=1,backend='torch')
    fusion.scale.data.fill_(2)
    rgb,depth = torch.randn(2,3,3,4),torch.randn(2,2,3,4)
    actual = fusion(rgb,depth,torch.zeros(2,1,6,8,dtype=torch.bool))
    assert torch.equal(actual,fusion.rgb(rgb))
    valid = fusion(rgb,depth,torch.ones(2,1,6,8,dtype=torch.bool))
    valid.square().mean().backward()
    assert fusion.quality[0].weight.grad.abs().sum()>0


def test_common_preprocessing_metric_depth_intrinsics_and_invalid_values():
    image = Image.fromarray(np.zeros((4,8,3),dtype=np.uint8))
    depth = np.arange(1,33,dtype=np.float32).reshape(4,8)
    depth[0,0]=np.nan
    K = [[10,0,3.5],[0,12,1.5],[0,0,1]]
    rgb,z,k = preprocess_rgbd(image,depth,K,(2,4))
    assert rgb.shape==(3,2,4)
    assert z[0,0,0]==0 and z[0,1,1]==19
    torch.testing.assert_close(k,torch.tensor([[5.,0,1.5],[0,6,.5],[0,0,1]]))


@pytest.mark.parametrize('conflict',['group','hash','uid','real','semantic'])
def test_manifest_rejects_cross_split_leakage_and_real_training(tmp_path,conflict):
    first=dict(uid='OLD_a',source='OLD',group='OLD_seed1',image_sha256='one')
    second=dict(uid='NEW_b',source='NEW',group='NEW_seed2',image_sha256='two')
    manifest=dict(manifest_version=2,keypoint_names=NAMES,train=[first],val=[second],test=[])
    if conflict=='group': second['group']=first['group']
    elif conflict=='hash': second['image_sha256']=first['image_sha256']
    elif conflict=='uid': second['uid']=first['uid']
    elif conflict=='real': first['source']='D455'
    else: manifest['keypoint_names']=NAMES[::-1]
    path=tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): load_manifest(path)


def test_v3_checkpoint_roundtrip_and_global_forward_contract(tmp_path):
    from models.keypoints.model import RGBDKeypointModel
    from models.keypoints.checkpoints import load_checkpoint,save_checkpoint
    config=make_config(dict(input_size=(17,25),heatmap_size=(9,13),out_indices=(1,2),
        fusion_channels=(4,4),geometry_channels=(4,4,4,4,4),decoder_width=4,num_keypoints=2,
        global_ssm=dict(state_dim=1),local=dict(enabled=True,mode='residual',layout='one',
                embedding=True,feature_source='detail',state_dim=1,roi_size=5)))
    model=RGBDKeypointModel(config).eval()
    inputs=(torch.randn(1,3,17,25),torch.ones(1,1,17,25),torch.tensor([[[20.,0,12],[0,20,8],[0,0,1]]]))
    before=model.forward_stages(*inputs)
    assert torch.equal(before.final_heatmaps,before.global_heatmaps)
    torch.testing.assert_close(model(*inputs),before.global_heatmaps)
    save_checkpoint(tmp_path/'v3.pt',model,None,epoch=0,phase='joint',keypoint_names=['a','b'],
                    split_ids={'train':['a'],'val':['b'],'test':['c']},metric={})
    restored,payload=load_checkpoint(tmp_path/'v3.pt'); restored.eval()
    assert payload['config_version']==3
    torch.testing.assert_close(restored.forward_stages(*inputs).final_heatmaps,before.final_heatmaps)


def test_final_coordinate_loss_has_finite_gradient():
    from models.keypoints.trial_runner import coordinate_loss
    maps=torch.ones(2,2,7,9,requires_grad=True)
    loss=coordinate_loss(maps,torch.tensor([[[4.,3.],[5.,4.]]]*2),(7,9),torch.ones(2,2))
    loss.backward()
    assert torch.isfinite(maps.grad).all() and maps.grad.abs().sum()>0


def test_augmented_points_and_intrinsics_project_same_camera_geometry(tmp_path):
    from models.keypoints.multisource import MultiSourceDataset
    for folder in ('rgb','depth_sensor','poses','annotations'):
        (tmp_path/folder).mkdir()
    Image.fromarray(np.full((40,60,3),128,dtype=np.uint8)).save(tmp_path/'rgb/000000.png')
    np.save(tmp_path/'depth_sensor/000000.npy',np.ones((40,60),dtype=np.float32))
    K=torch.tensor([[80.,0,29.5],[0,80,19.5],[0,0,1.]])
    (tmp_path/'poses/000000.json').write_text(json.dumps({'K':K.tolist()}))
    points=torch.tensor([[25.+i,18.+i*.1] for i in range(9)])
    annotation=dict(frame_id='000000',paths=dict(rgb='rgb/000000.png',depth_sensor='depth_sensor/000000.npy',poses='poses/000000.json'),
                    object_bbox_xyxy=[20,10,40,30],keypoints_2d=[dict(name=n,uv=p.tolist(),visibility=2) for n,p in zip(NAMES,points)])
    (tmp_path/'annotations/000000.json').write_text(json.dumps(annotation))
    record=dict(uid='OLD_same',source='OLD',root=str(tmp_path),frame_id='000000',annotation=str(tmp_path/'annotations/000000.json'))
    manifest=dict(seed=42,keypoint_names=NAMES,train=[record])
    data=MultiSourceDataset(manifest,'train',{'input_size':(21,31),'heatmap_size':(11,16)},augment=True)
    sample=data[0]
    homogeneous=torch.cat((points,torch.ones(9,1)),1)
    rays=(torch.linalg.inv(K)@homogeneous.T).T
    projected=(sample['K']@rays.T).T
    torch.testing.assert_close(projected[:,:2]/projected[:,2:],sample['points'],atol=1e-5,rtol=1e-5)
    assert torch.equal(sample['rgb'],data[0]['rgb'])
    data.epoch=1
    assert sample['cache_key']!=data[0]['cache_key']
