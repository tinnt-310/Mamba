"""Behavioral gates for config-driven RGB-D architecture experiments."""
import copy
import pytest
import torch

from models.keypoints.config import make_config
from models.keypoints.model import MeanMultiScaleDecoder
from models.keypoints.residual_local import ResidualLocal
from models.fusion.rgbd_compensation import RGBDCompensation


def test_concat_starts_as_mean_and_can_learn_scale_selection():
    channels=(3,5)
    baseline=MeanMultiScaleDecoder(channels,4)
    variant=MeanMultiScaleDecoder(channels,4,dict(merge='concat',activation='gelu',depth=2,residual=False))
    incoming=baseline.state_dict()
    variant.load_state_dict({**variant.state_dict(),**incoming},strict=True)
    inputs=[torch.randn(2,3,7,11),torch.randn(2,5,4,6)]
    assert torch.allclose(baseline(inputs,(9,15)),variant(inputs,(9,15)),atol=2e-6,rtol=2e-5)
    variant(inputs,(9,15)).square().mean().backward()
    assert variant.merge_projection.weight.grad.abs().sum()>0


@pytest.mark.parametrize('activation',['gelu','silu','relu'])
def test_decoder_rectangular_odd_detail_gradient(activation):
    model=MeanMultiScaleDecoder((3,5),4,dict(merge='mean',activation=activation,depth=3,residual=True))
    inputs=[torch.randn(2,3,7,11,requires_grad=True),torch.randn(2,5,4,6,requires_grad=True)]
    detail=torch.randn(2,4,17,29,requires_grad=True)
    output=model(inputs,(9,15),detail)
    assert output.shape==(2,4,9,15)
    output.square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs+[detail])


def test_quality_gate_receives_gradients_and_all_invalid_is_depth_independent():
    torch.manual_seed(13)
    block=RGBDCompensation(3,4,4,state_dim=2,expand=1,validity_aware=True)
    rgb=torch.randn(2,3,5,7,requires_grad=True)
    depth=torch.randn(2,4,5,7,requires_grad=True)
    validity=torch.ones(2,1,19,27,dtype=torch.bool)
    value,aux=block(rgb,depth,validity)
    value.square().mean().backward()
    assert rgb.grad.abs().sum()>0 and depth.grad.abs().sum()>0
    assert block.compensation.d2r.net[0].weight.grad.abs().sum()>0
    invalid=torch.zeros_like(validity)
    left,_=block(rgb.detach(),depth.detach(),invalid)
    right,_=block(rgb.detach(),torch.randn_like(depth)*20,invalid)
    assert torch.equal(left,right) and torch.isfinite(left).all()
    with pytest.raises(ValueError): block(rgb,depth)


@pytest.mark.parametrize('core',['ssm','conv'])
def test_local_core_identity_support_and_learning(core):
    model=ResidualLocal(4,4,2,roi_size=5,state_dim=2,core=core,conditioning=True)
    maps=torch.randn(2,2,9,15)
    feature=torch.randn(2,4,5,8,requires_grad=True)
    result=model(maps,feature)
    assert torch.equal(maps,result.heatmaps)
    result.heatmaps.square().mean().backward()
    assert model.delta_head.weight.grad.abs().sum()>0
    with torch.no_grad(): model.delta_head.weight.normal_(std=.1)
    result=model(maps,feature)
    assert torch.equal(result.delta[~result.support],torch.zeros_like(result.delta[~result.support]))
    assert torch.equal(model(maps,feature,gate_override=0).heatmaps,maps)
    result.heatmaps.square().mean().backward()
    assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum()>0


def test_config_rejects_ambiguous_architecture():
    with pytest.raises(ValueError): make_config({'decoder':{'depth':True}})
    with pytest.raises(ValueError): make_config({'local':{'core':'conv','mode':'legacy'}})
    original=make_config({'local':{'enabled':False}})
    modified=copy.deepcopy(original); modified['decoder']['activation']='silu'
    assert make_config(modified)['decoder']['activation']=='silu'
    assert original['decoder']['activation']=='gelu'


def test_research_cannot_spend_official_final_or_recovery_reserve():
    from models.keypoints.architecture_search import can_research,protected_seconds
    from models.keypoints.campaign_budget import Budget,Reservation
    budget=Budget([Reservation('past','historical',3600,state='COMPLETED',elapsed=3600)])
    assert protected_seconds(budget)==136*3600
    assert can_research(budget,151*3600)
    assert not can_research(budget,151*3600+1)
    budget.records.append(Reservation('official_seed','architecture_official',3600,state='COMPLETED',elapsed=3600))
    assert protected_seconds(budget)==135*3600


def test_all_registered_candidates_are_rgbd_and_no_official_job_is_created():
    from pathlib import Path
    from models.keypoints.architecture_search import variants,make_pilots
    specs=variants(Path('/tmp/campaign'))
    profiles={s['id']:dict(correctness=True,step_p50_ms=100) for s in specs}
    assert len({s['id'] for s in specs})==len(specs)
    for spec in specs:
        config=make_config(spec['config'])
        assert config['modality']=='rgbd' and config['global_ssm']['backend']=='mamba'


def test_modality_corruption_is_uid_deterministic_and_preserves_labels():
    from models.keypoints.modality_diagnostics import InterventionDataset
    class Sample:
        input_size=(9,15)
        def __len__(self): return 1
        def __getitem__(self,index):
            return dict(frame_id='OLD:example',depth=torch.ones(1,9,15),rgb=torch.ones(3,9,15),
                        K=torch.eye(3),points=torch.tensor([[4.,5.]]),weights=torch.ones(1))
    clean=Sample()[0]
    for condition in ('depth_invalid','depth_noise_2mm','depth_hole_25pct','rgb_half_brightness','rgb_occlusion_25pct'):
        data=InterventionDataset(Sample(),condition)
        left,right=data[0],data[0]
        assert torch.equal(left['depth'],right['depth']) and torch.equal(left['rgb'],right['rgb'])
        assert torch.equal(left['K'],clean['K']) and torch.equal(left['points'],clean['points'])
        assert torch.isfinite(left['depth']).all() and torch.isfinite(left['rgb']).all()
    assert InterventionDataset(Sample(),'depth_invalid')[0]['depth'].count_nonzero()==0


def test_feature_ablation_removes_hook_after_context():
    from models.keypoints.modality_diagnostics import feature_intervention
    class Encoder(torch.nn.Module):
        def forward(self,x): return (x*2,x*3)
    class Model:
        rgb_encoder=Encoder()
        geometry_encoder=Encoder()
    model=Model(); value=torch.ones(1,3,4,5)
    with feature_intervention(model,'rgb_features_zero'):
        assert all(x.count_nonzero()==0 for x in model.rgb_encoder(value))
    assert all(x.count_nonzero()>0 for x in model.rgb_encoder(value))


def test_successive_training_refuses_recipe_change_before_any_cuda_work(tmp_path,monkeypatch):
    import types
    from models.keypoints import trial_runner
    config=make_config({'local':{'enabled':False}})
    model=types.SimpleNamespace(config=config)
    monkeypatch.setattr(trial_runner,'load_manifest',lambda path:{'train':[],'val':[],'test':[]})
    monkeypatch.setattr(trial_runner,'digest',lambda path:'same-manifest')
    payload=dict(extra={'manifest_sha256':'same-manifest'},training={'augment':True},optimizer={})
    monkeypatch.setattr(trial_runner,'load_checkpoint',lambda *args,**kwargs:(model,payload))
    with pytest.raises(ValueError,match='Unsafe successive-training change: augment'):
        trial_runner.train(dict(id='next',config=config,resume_trial='parent',augment=False),tmp_path)


def test_target_sigma_changes_only_supervision_not_model_parameters():
    from models.keypoints.heatmaps import gaussian_heatmaps
    points=torch.tensor([[4.,5.]])
    narrow=gaussian_heatmaps(points,(11,13),2.,torch.ones(1))
    wide=gaussian_heatmaps(points,(11,13),4.,torch.ones(1))
    assert torch.equal(narrow.argmax(),wide.argmax()) and wide.sum()>narrow.sum()
    with pytest.raises(ValueError): make_config({'target_sigma':float('nan')})


def test_mm_success_denominator_includes_missing_prediction_and_tracks_depth_coverage():
    from models.keypoints.regression import geometry_metrics
    row=dict(gt=[[0.,0.]]*9,coords=[[1.,0.]]*9,gt_depth=[1.]*9,valid=[True]*9,fx=1000.,fy=1000.)
    missing=copy.deepcopy(row); missing['coords'][0]=[float('nan'),0.]
    no_depth=copy.deepcopy(row); no_depth['gt_depth'][0]=0.
    value=geometry_metrics([row,missing,no_depth])
    assert value['mean']==pytest.approx(1.)
    assert value['complete_depth_images']==2 and value['predicted_complete_images']==1
    assert value['image_mean_below_5mm']==.5 and value['all_nine_below_5mm']==.5
    assert value['reference_points']==26 and value['points']==25


def test_modality_gate_requires_accuracy_evidence_and_both_gradients():
    from models.keypoints.architecture_search import modality_benefit
    def comparison(low,high): return {'ci95':{'mean':[low,high]}}
    proof=dict(paired_comparisons={condition:{'OLD':comparison(0.,1.),'NEW':comparison(.01,.1)}
        for condition in ('depth_invalid','depth_features_zero','rgb_features_zero')},
        train_gradient_norms={name:dict(finite=True,norm=1.) for name in ('rgb','depth')})
    assert modality_benefit(proof)
    proof['paired_comparisons']['depth_features_zero']['NEW']=comparison(-.1,.1)
    assert not modality_benefit(proof)
    proof['paired_comparisons']['depth_features_zero']['NEW']=comparison(.01,.1)
    proof['train_gradient_norms']['depth']['norm']=0.
    assert not modality_benefit(proof)


def test_official_submission_rejects_missing_freeze_without_spending_budget(tmp_path):
    from models.keypoints.architecture_search import submit
    from models.keypoints.campaign_budget import Budget
    budget=Budget(); state={}
    with pytest.raises(ValueError,match='freeze and GPU verification'):
        submit(tmp_path,dict(phase='official'),budget,state)
    assert not budget.records


def test_missing_recipe_or_local_artifacts_cannot_open_official_training(tmp_path):
    from models.keypoints.architecture_search import advance
    from models.keypoints.campaign_budget import Budget
    for phase in ('recipe','local_confirmation','local_second_seed'):
        state=dict(phase=phase,tasks=[dict(id='missing',kind='train',training={'id':'missing'})])
        advance(tmp_path,state,Budget())
        assert state['phase']=='RESEARCH_REVIEW'
        assert not (tmp_path/'architecture/freeze.json').exists()


def test_rgb_attribution_alone_does_not_establish_rgb_localization_benefit():
    from models.keypoints.architecture_search import modality_benefit
    comparison={'ci95':{'mean':[.01,.1]}}
    proof=dict(paired_comparisons={condition:{'OLD':comparison,'NEW':comparison}
        for condition in ('depth_invalid','depth_features_zero')},
        train_gradient_norms={name:dict(finite=True,norm=1.) for name in ('rgb','depth')})
    assert not modality_benefit(proof)


def test_architecture_confirmation_requires_two_complete_seeds(tmp_path,monkeypatch):
    from models.keypoints import architecture_search as search
    from models.keypoints.campaign_budget import Budget
    task=dict(id='AC_selected_20261025',kind='train',training={'id':'AC_selected_20261025','seed':20261025})
    result={'best':{'final':{'balanced_mean':1.}}}
    monkeypatch.setattr(search,'successful_training',lambda *args:result)
    state=dict(phase='confirmation',tasks=[task])
    search.advance(tmp_path,state,Budget())
    assert state['phase']=='BLOCKED'
    assert not (tmp_path/'architecture/freeze.json').exists()


def test_combined_decoder_only_selected_when_it_beats_isolated_and_control(tmp_path,monkeypatch):
    from models.keypoints import architecture_search as search
    from models.keypoints.campaign_budget import Budget
    def score(mean):
        return {'final':{'balanced_mean':mean,
            'OLD':{'PCK05':1.,'p95':3.},'NEW':{'PCK05':1.,'p95':3.}}}
    config={'decoder':{'merge':'concat','activation':'silu'}}
    task=dict(id='combo',kind='train',training={'id':'combo','config':config})
    captured=[]
    monkeypatch.setattr(search,'queue_confirmation',lambda root,state,choices,source:captured.append(choices))
    for mean,expected in ((1.1,'selected'),(.9,'combined')):
        monkeypatch.setattr(search,'successful_training',lambda *args,m=mean:{'best':score(m)})
        state=dict(phase='combination',tasks=[task],isolated_confirmation_choices={'selected':{'decoder':{'activation':'silu'}}},
                   isolated_best=score(1.),combination_control=score(1.2))
        search.advance(tmp_path,state,Budget())
        assert captured[-1][0][0]==expected


def test_failed_frozen_verification_does_not_start_official(tmp_path,monkeypatch):
    from models.keypoints import architecture_search as search
    from models.keypoints.campaign_budget import Budget
    out=tmp_path/'architecture/tasks/verify_frozen'; out.mkdir(parents=True)
    search.dump(out/'verification.json',{'passed':False})
    state=dict(phase='verification',tasks=[dict(id='verify_frozen',kind='verify')])
    monkeypatch.setattr(search,'task_status',lambda *args:'VERIFIED')
    search.advance(tmp_path,state,Budget())
    assert state['phase']=='BLOCKED'
    assert state['tasks'][0]['kind']=='verify'
