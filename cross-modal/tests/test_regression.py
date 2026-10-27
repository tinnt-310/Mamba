import numpy as np
import pytest
import torch
from models.keypoints.regression import paired_rows,summarize,bootstrap
from models.keypoints.continuation import outcome
from models.keypoints.campaign_budget import Budget,Reservation


def row(uid,group,error):
    return dict(uid=uid,group=group,source='OLD',valid=[True]*9,gt=[[0.,0.]]*9,
        coords=[[error,0.]]*9,errors=[error]*9,normalized=[error/100]*9,
        bbox=[0.,0.,10.,10.],gt_depth=[1.]*9,fx=100.,fy=100.)


def test_paired_bootstrap_preserves_groups_and_constant_shift():
    left=[row('a','parent1',1.),row('b','parent1',2.),row('c','parent2',3.)]
    right=[row('c','parent2',4.),row('a','parent1',2.),row('b','parent1',3.)]
    pairs=paired_rows(left,right)
    result=bootstrap(pairs,repeats=30)
    assert result['groups']==2
    assert result['ci95']['mean']==pytest.approx([1.,1.])
    assert summarize(pairs)['delta']['mean']==pytest.approx(1.)
    with pytest.raises(ValueError): paired_rows(left,right[:-1])
    right[0]['valid'][0]=False
    with pytest.raises(ValueError): paired_rows(left,right)


def test_pck_transition_counts_and_mm_common_mask():
    a=row('a','g',4.); b=row('a','g',6.)
    a['gt_depth'][0]=0.; b['gt_depth'][0]=0.
    result=summarize([(a,b)])
    assert result['lost_pck_points']==9 and result['lost_pck_images']==1
    assert result['mm']['points']==8
    assert result['mm']['baseline_mean']==pytest.approx(40.)


def test_failed_trial_is_not_success_even_if_queue_ended(tmp_path):
    budget=Budget([Reservation('next_x','next_diagnostic',60,job_id='1',state='FAILED',elapsed=10)])
    assert outcome(tmp_path,{'id':'x'},budget)=='failed'
    import json
    folder=tmp_path/'continuation/tasks/x'; folder.mkdir(parents=True)
    (folder/'completed.json').write_text(json.dumps({'status':'complete'}))
    assert outcome(tmp_path,{'id':'x'},budget)=='failed'


def test_historical_charges_and_active_reservations_count_towards_total():
    budget=Budget([Reservation('historical','old',3600,job_id='1',state='COMPLETED',elapsed=287*3600)])
    assert budget.reserve('next_x','next_diagnostic',3600) is not None
    assert budget.reserve('next_y','next_diagnostic',60) is None
    assert budget.summary()['charged_or_reserved_gpu_hours']==288


def test_rgbd_cpu_autocast_boundary_is_finite():
    from models.keypoints.model import RGBDKeypointModel
    model=RGBDKeypointModel({'input_size':(32,48),'heatmap_size':(16,24),
                            'local':{'enabled':False}}).eval()
    rgb=torch.randn(1,3,32,48); depth=torch.ones(1,1,32,48)
    K=torch.tensor([[[50.,0.,24.],[0.,50.,16.],[0.,0.,1.]]])
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        output=model(rgb,depth,K)
    assert torch.isfinite(output).all()


def test_unused_coordinate_term_has_identical_gradient():
    from models.keypoints.trial_runner import coordinate_loss
    maps=torch.randn(2,9,8,12,requires_grad=True)
    points=torch.rand(2,9,2)*maps.new_tensor([24.,16.])
    weights=torch.ones(2,9)
    heat=maps.square().mean()
    old=heat+0.*coordinate_loss(maps,points,(16,24),weights)
    a=torch.autograd.grad(old,maps,retain_graph=True)[0]
    b=torch.autograd.grad(heat,maps)[0]
    assert torch.equal(a,b)


def test_runtime_requires_five_percent_gain_and_prefers_fewer_workers():
    from models.keypoints.trial_validation import choose_batch_one_runtime
    def sample(workers,fps,accepted=True):
        return dict(batch=1,workers=workers,precision='fp32',accepted=accepted,
                    file_to_coordinates=[dict(images_per_second=fps)]*3)
    assert choose_batch_one_runtime([sample(0,30),sample(2,80),sample(4,81)])['workers']==2
    assert choose_batch_one_runtime([sample(0,30),sample(2,31)]) is None
    assert choose_batch_one_runtime([sample(0,30),sample(2,80,False)]) is None


def test_inference_reader_matches_preprocessing_and_does_not_require_labels(tmp_path):
    import json
    from PIL import Image
    from models.keypoints.multisource import MultiSourceDataset,NAMES
    for name in ('rgb','depth_sensor','poses'): (tmp_path/name).mkdir()
    Image.fromarray(np.full((32,48,3),127,np.uint8)).save(tmp_path/'rgb/0.png')
    np.save(tmp_path/'depth_sensor/0.npy',np.ones((32,48),np.float32))
    (tmp_path/'poses/0.json').write_text(json.dumps({'K':[[50,0,24],[0,50,16],[0,0,1]]}))
    annotation=tmp_path/'annotation.json'
    annotation.write_text(json.dumps(dict(frame_id='0',paths={'rgb':'rgb/0.png','depth_sensor':'depth_sensor/0.npy','poses':'poses/0.json'},
        object_bbox_xyxy=[0,0,47,31],keypoints_2d=[dict(name=n,uv=[20,15],visibility=2) for n in NAMES])))
    record=dict(uid='OLD_0',source='OLD',root=str(tmp_path),frame_id='0',annotation=str(annotation))
    manifest=dict(val=[record],keypoint_names=NAMES)
    config=dict(input_size=(17,25),heatmap_size=(9,13))
    labeled=MultiSourceDataset(manifest,'val',config)[0]
    annotation.unlink()
    pure=MultiSourceDataset(manifest,'val',config,inference_only=True)[0]
    assert 'target' not in pure and 'points' not in pure
    for key in ('rgb','depth','K','original_size'): assert torch.equal(pure[key],labeled[key])
    with pytest.raises(ValueError): MultiSourceDataset(manifest,'val',config,inference_only=True,augment=True)


def test_worker_passes_snapshot_path_explicitly_for_slurm_spooled_script(tmp_path,monkeypatch):
    from models.keypoints import continuation as module
    commands=[]
    monkeypatch.setattr(module.subprocess,'check_output',lambda cmd,text:commands.append(cmd) or '123\n')
    state={}; budget=Budget()
    task=dict(id='test',kind='efficiency',seconds=600,phase='next_runtime',cpus=2,memory='16G',node='hpc06')
    assert module.submit(tmp_path,task,budget,state)
    assert commands[0][-1]==str(__import__('pathlib').Path(module.__file__).resolve().parents[2])
    assert commands[0][-2]=='600'
    assert commands[0][-4]=='worker'
    assert '--nodelist=hpc06' in commands[0]


def test_admissible_checkpoint_serializes_its_own_metric(tmp_path,monkeypatch):
    from models.keypoints import trial_runner as module
    captured={}
    monkeypatch.setattr(module,'save_checkpoint',lambda path,model,opt,**kwargs:captured.update(kwargs))
    manifest={'keypoint_names':['one'],'train':[{'uid':'a'}],'val':[{'uid':'b'}],'test':[]}
    extra=dict(manifest_sha256='hash',best={'mean':1.},best_admissible={'mean':2.})
    module.checkpoint(tmp_path/'best_admissible.pt',None,None,manifest,{},0,extra)
    assert captured['metric']==extra['best_admissible']
    module.checkpoint(tmp_path/'best.pt',None,None,manifest,{},0,extra)
    assert captured['metric']==extra['best']
