import copy
import pytest
import torch
from torch import nn
from models.fusion.gdl import SearchRGBDCompensation, WindowCrossC
from models.fusion.rgbd_compensation import RGBDCompensation
from models.keypoints.gdl_decoders import CoarseToFineDecoder,FPNDecoder
from models.keypoints.residual_local import ResidualLocal
from models.keypoints.heatmaps import transform_points


def test_window_partition_odd_rectangles_no_padding_or_batch_leak():
    class Identity(nn.Module):
        def forward(self,r,d): return {'rgb_self':r,'depth_self':d,'rgb_cross':d,'depth_cross':r}
    r=torch.randn(2,3,11,19,requires_grad=True); d=torch.randn_like(r)
    result=WindowCrossC(Identity(),8)(r,d)
    assert torch.equal(result['rgb_self'],r) and torch.equal(result['rgb_cross'],d)
    result['rgb_self'].sum().backward(); assert torch.equal(r.grad,torch.ones_like(r))
    class SizeSensitive(nn.Module):
        def forward(self,r,d):
            assert r.shape[-2:] in ((8,8),(8,3),(3,8),(3,3))
            return {'value':r+r.mean((-2,-1),keepdim=True)}
    output=WindowCrossC(SizeSensitive(),8)(r,d)['value']
    assert torch.allclose(output[:,:,-3:,-3:],r[:,:,-3:,-3:]+r[:,:,-3:,-3:].mean((-2,-1),keepdim=True))


def test_g2_starts_identical_to_g0_then_learns_cnn():
    torch.manual_seed(10)
    base=RGBDCompensation(3,4,4,state_dim=2,expand=1,directions=4)
    hybrid=SearchRGBDCompensation(3,4,4,state_dim=2,expand=1,directions=4,topology='g2')
    hybrid.load_state_dict({**hybrid.state_dict(),**base.state_dict()},strict=True)
    r,d=torch.randn(2,3,5,7),torch.randn(2,4,5,7)
    assert torch.equal(base(r,d)[0],hybrid(r,d)[0])
    opt=torch.optim.AdamW(hybrid.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad(); hybrid(r,d)[0].square().mean().backward(); opt.step()
    assert hybrid.cnn[0].weight.grad.abs().sum()>0
    assert hybrid.cnn_output.weight.grad.abs().sum()>0


def test_d2_both_shallow_modalities_affect_fine_prediction():
    decoder=CoarseToFineDecoder((4,4),4,(3,2),9)
    features=[torch.randn(2,4,5,8),torch.randn(2,4,3,4)]
    rgb=torch.randn(2,3,9,15,requires_grad=True)
    depth=torch.randn(2,2,9,15,requires_grad=True)
    fine,coarse=decoder(features,(18,30),(rgb,depth))
    assert fine.shape==(2,4,18,30) and coarse.shape==(2,9,9,15)
    fine.square().mean().backward()
    assert rgb.grad.abs().sum()>0 and depth.grad.abs().sum()>0
    for field in ('mask_rgb','mask_depth'):
        setattr(decoder,field,True)
        assert not torch.equal(fine,decoder(features,(18,30),(rgb,depth))[0])
        setattr(decoder,field,False)


def test_d2_coordinate_bridge_preserves_pixel_centers_and_semantic_order():
    coarse=torch.tensor([[[0.,0.],[31.25,27.5],[159.,89.]]])
    original=transform_points(coarse,(90,160),(720,1280))
    fine=transform_points(original,(720,1280),(180,320))
    assert torch.allclose(transform_points(fine,(180,320),(720,1280)),original)
    assert torch.allclose(fine,(coarse+.5)*2-.5)
    assert not torch.equal(fine,coarse)


def test_hybrid_local_identity_both_gradients_and_branch_masks():
    model=ResidualLocal(4,4,3,roi_size=5,state_dim=2,core='hybrid')
    maps=torch.randn(2,3,9,15); feature=torch.randn(2,4,5,8)
    assert torch.equal(model(maps,feature).heatmaps,maps)
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    target=maps+torch.randn_like(maps)*.1
    for _ in range(4):
        opt.zero_grad(); (model(maps,feature).heatmaps-target).square().mean().backward(); opt.step()
    for branch in (model.mamba.cnn,model.mamba.ssm):
        assert sum(p.grad.abs().sum() for p in branch.parameters() if p.grad is not None)>0
    assert model.mamba.gate.weight.grad.abs().sum()>0
    clean=model(maps,feature).heatmaps
    for branch in ('cnn','ssm'):
        model.mamba.mask_branch=branch
        assert not torch.equal(model(maps,feature).heatmaps,clean)
    model.mamba.mask_branch=None
    assert torch.isfinite(model.mamba.last_gate).all()


def test_fpn_receives_all_scales():
    features=[torch.randn(2,4,h,w,requires_grad=True) for h,w in ((9,15),(5,8),(3,4),(2,2))]
    out=FPNDecoder((4,)*4,4)(features,(18,30))
    out.square().mean().backward()
    assert all(x.grad.abs().sum()>0 for x in features)


@pytest.mark.parametrize('decoder',['d0','d1','d2'])
def test_full_model_decoder_phase_freezes_encoder_buffers_and_roundtrip(tmp_path,decoder):
    from models.keypoints.config import make_config
    from models.keypoints.model import RGBDKeypointModel
    from models.keypoints.checkpoints import save_checkpoint,load_checkpoint
    config=make_config(dict(input_size=(17,25),heatmap_size=(17,25) if decoder=='d2' else (9,13),
        out_indices=(1,2),fusion_channels=(4,4),geometry_channels=(4,4,4,4,4),decoder_width=4,num_keypoints=2,
        global_ssm=dict(topology='g0',directions=4,state_dim=1),decoder=dict(kind=decoder),local=dict(enabled=False)))
    model=RGBDKeypointModel(config).set_phase('decoder').train()
    before={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('rgb_encoder.','geometry_encoder.','fusion.'))}
    inputs=(torch.randn(2,3,17,25),torch.ones(2,1,17,25),torch.tensor([[[20.,0,12],[0,20,8],[0,0,1.]]]*2))
    output=model.forward_stages(*inputs)
    output.final_heatmaps.square().mean().backward()
    assert model.head.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.rgb_encoder.parameters())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    save_checkpoint(tmp_path/'new.pt',model,None,epoch=0,phase='decoder',keypoint_names=['a','b'],split_ids={},metric={})
    restored,payload=load_checkpoint(tmp_path/'new.pt');restored.eval();model.eval()
    assert payload['config_version']==4
    torch.testing.assert_close(restored.forward_stages(*inputs).final_heatmaps,model.forward_stages(*inputs).final_heatmaps)


def test_gdl_registry_enforces_four_direction_rgbd(tmp_path):
    from models.keypoints.gdl_search import base_config
    from models.keypoints.config import make_config
    for g in ('g0','g1','g2'):
        for d in ('d0','d1','d2'):
            cfg=make_config(base_config(tmp_path,g,d,'hybrid'))
            assert cfg['global_ssm']['directions']==4 and cfg['modality']=='rgbd'


def test_official_cannot_submit_without_frozen_proof(tmp_path,monkeypatch):
    from models.keypoints.gdl_search import submit
    monkeypatch.setattr('subprocess.check_output',lambda *a,**kw:'')
    state=dict(ledger=[])
    with pytest.raises(ValueError,match='frozen GPU verification'):
        submit(tmp_path,state,dict(id='official',kind='train',gpu=True,official=True))
    assert not state['ledger']


def test_selected_local_rejects_unconstrained_best(tmp_path):
    import json
    from models.keypoints.gdl_search import selected
    folder=tmp_path/'trials'/'local';folder.mkdir(parents=True)
    (folder/'completed.json').write_text(json.dumps(dict(status='complete',updates=30000,best={'final':{}},best_admissible=None)))
    task=dict(id='local',training=dict(max_updates=30000,reference_metrics='parent.json'))
    assert selected(tmp_path,task) is None


def test_artifact_exports_final_and_coarse_predictions_in_original_coordinates(tmp_path):
    import json
    import numpy as np
    from models.keypoints.outputs import save_prediction_artifact
    from models.keypoints.model import StageOutput
    coarse=torch.zeros(1,2,9,15);coarse[:,:,2,3]=1
    fine=torch.zeros(1,2,18,30);fine[:,0,5,7]=1;fine[:,1,13,21]=1
    stage=StageOutput(torch.zeros_like(fine),None,fine,None,coarse)
    sample=dict(frame_id='test',rgb=torch.zeros(3,18,30),original_size=torch.tensor([180,300]),
                points=torch.zeros(2,2),original_points=torch.zeros(2,2),weights=torch.ones(2))
    paths=save_prediction_artifact(tmp_path/'pred',sample,stage,{},['a','b'])
    saved=np.load(paths['npz']);metadata=json.loads(open(paths['json']).read())
    np.testing.assert_array_equal(saved['final_heatmaps'],fine[0].numpy())
    np.testing.assert_allclose(saved['final_argmax_original'],[[74.5,54.5],[214.5,134.5]])
    assert 'coarse' in metadata['coordinates'] and 'final' in metadata['coordinates']
