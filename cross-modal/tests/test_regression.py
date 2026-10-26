import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from models.keypoints.regression import paired_rows,summarize,bootstrap
from models.keypoints.residual_local import ResidualLocal
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


def test_packed_roi_matches_legacy_crop_and_feature_gradient():
    torch.manual_seed(1)
    feature=torch.randn(2,4,7,11,requires_grad=True)
    grid=torch.rand(2*9,5,5,2)*2-1
    expanded=feature[:,None].expand(-1,9,-1,-1,-1).reshape(18,4,7,11)
    expected=F.grid_sample(expanded,grid,align_corners=False)
    packed=F.grid_sample(feature,grid.reshape(2,45,5,2),align_corners=False)
    actual=packed.reshape(2,4,9,5,5).permute(0,2,1,3,4).reshape(18,4,5,5)
    assert torch.equal(actual,expected)
    a=torch.autograd.grad(expected.square().sum(),feature,retain_graph=True)[0]
    b=torch.autograd.grad(actual.square().sum(),feature)[0]
    assert torch.allclose(a,b,atol=1e-5)


def test_failed_trial_is_not_success_even_if_queue_ended(tmp_path):
    budget=Budget([Reservation('next_x','next_diagnostic',60,job_id='1',state='FAILED',elapsed=10)])
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
