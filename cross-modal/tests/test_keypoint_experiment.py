"""Training/checkpoint contracts; run on a Slurm compute node."""
from pathlib import Path
import torch
import pytest


def small_config(local=False):
    from models.keypoints.config import make_config
    return make_config({
        'input_size': [16, 24], 'heatmap_size': [8, 12],
        'out_indices': [1, 2], 'fusion_channels': [4, 4],
        'decoder_width': 4, 'geometry_channels': [4, 4, 4, 4, 4],
        'global_ssm': {'state_dim': 2},
        'local': {'enabled': local, 'state_dim': 2, 'embedding': True},
    })


def test_cli_contract_and_requires_compute_node(monkeypatch):
    from models.keypoints.experiment import build_parser, require_compute_node
    parser = build_parser()
    args = parser.parse_args(['train', '--phase', 'local', '--local', 'on',
                              '--patches', 'one', '--embedding', 'off'])
    assert args.phase == 'local' and args.patches == 'one'
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('SLURM_JOB_NODELIST', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        require_compute_node()


def test_optimizer_respects_no_weight_decay():
    from models.fusion.ssm.simple_ssm import SimpleSelectiveSSM
    from models.keypoints.experiment import make_optimizer
    model = SimpleSelectiveSSM(4, state_dim=2)
    optimizer = make_optimizer(model, lr=3e-4, weight_decay=1e-4)
    decay_by_id = {id(p): group['weight_decay'] for group in optimizer.param_groups for p in group['params']}
    assert decay_by_id[id(model.A_log)] == 0
    assert decay_by_id[id(model.D)] == 0
    assert decay_by_id[id(model.x_proj.weight)] == 1e-4


def test_checkpoint_strict_roundtrip_and_local_only_warmstart(tmp_path):
    from models.keypoints.model import RGBDKeypointModel
    from models.keypoints.checkpoints import save_checkpoint, load_checkpoint
    model = RGBDKeypointModel(small_config()).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / 'global.pt'
    names = [f'k{i}' for i in range(9)]
    splits = {'train': ['000001'], 'val': ['000002'], 'test': ['000003']}
    save_checkpoint(path, model, optimizer, epoch=3, phase='global',
                    keypoint_names=names, split_ids=splits,
                    metric={'mean': 4., 'p95': 8.})
    restored, metadata = load_checkpoint(path)
    assert metadata['config_version'] == 2 and metadata['split_ids'] == splits
    assert metadata['phase'] == 'global' and metadata['epoch'] == 3
    rgb = torch.rand(1, 3, 16, 24)
    depth = torch.ones(1, 1, 16, 24)
    K = torch.tensor([[[20., 0., 11.5], [0., 20., 7.5], [0., 0., 1.]]])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(rgb, depth, K), restored(rgb, depth, K))
    local = RGBDKeypointModel(small_config(True))
    load_checkpoint(path, model=local, allow_local_missing=True)
    corrupted = torch.load(path, weights_only=False)
    corrupted['model_state'].pop('head.weight')
    bad = tmp_path / 'corrupt.pt'
    torch.save(corrupted, bad)
    with pytest.raises(RuntimeError, match='(global|Missing|missing)'):
        load_checkpoint(bad, model=local, allow_local_missing=True)
    # Existing local keys are mandatory on reload and never silently ignored.
    local_path = tmp_path / 'local.pt'
    save_checkpoint(local_path, local, None, epoch=1, phase='joint',
                    keypoint_names=names, split_ids=splits, metric={'mean': 3., 'p95': 6.})
    loaded, _ = load_checkpoint(local_path)
    loaded.eval(); local.eval()
    with torch.no_grad():
        before, after = local.forward_stages(rgb, depth, K), loaded.forward_stages(rgb, depth, K)
    torch.testing.assert_close(before.local.heatmaps, after.local.heatmaps)
    assert torch.equal(before.local.support, after.local.support)


def test_checkpoint_rejects_config_mismatch_even_if_shapes_same(tmp_path):
    from models.keypoints.model import RGBDKeypointModel
    from models.keypoints.checkpoints import save_checkpoint, load_checkpoint
    first = RGBDKeypointModel(small_config(True))
    path = tmp_path / 'a.pt'
    save_checkpoint(path, first, None, epoch=1, phase='local', keypoint_names=list('abcdefghi'),
                    split_ids={'train': ['1'], 'val': ['2'], 'test': ['3']}, metric={'mean': 1., 'p95': 2.})
    second = RGBDKeypointModel({**small_config(True), 'heatmap_size': [9, 13]})
    with pytest.raises(ValueError, match='config'):
        load_checkpoint(path, model=second)


def test_metric_selection_tiebreak_and_partial_accumulation():
    from models.keypoints.experiment import metric_improved, accumulation_window
    assert metric_improved({'mean': 1., 'p95': 2.}, {'mean': 1., 'p95': 3.})
    assert not metric_improved({'mean': 2., 'p95': 1.}, {'mean': 1., 'p95': 3.})
    assert [accumulation_window(i, 10, 8) for i in range(10)] == [8]*8 + [2]*2


def test_ablation_configs_share_region_and_common_local_initialization():
    from models.keypoints.ablation import ablation_configs, initialize_local
    from models.keypoints.model import RGBDKeypointModel
    configs = ablation_configs(small_config(), level=.15, margin=.5)
    assert len(configs['A0']['out_indices']) == 2
    assert tuple(configs['A1']['out_indices']) == (1, 2)  # custom smoke sizes retained
    assert configs['A2']['local']['layout'] == 'one'
    assert not configs['A2']['local']['embedding']
    assert configs['A3']['local']['layout'] == 'five'
    assert configs['A4']['local']['embedding']
    a, b = RGBDKeypointModel(configs['A3']), RGBDKeypointModel(configs['A4'])
    initialize_local(a, 42); initialize_local(b, 42)
    for key, value in a.local_refinement.state_dict().items():
        assert torch.equal(value, b.local_refinement.state_dict()[key])


def test_training_partial_accumulation_has_correct_gradient():
    from models.keypoints.experiment import train_epoch
    class ScalarPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.))
        def set_phase(self, phase):
            assert phase == 'global'
        def forward(self, rgb, depth, K):
            return self.weight.expand(rgb.shape[0], 1, 2, 2)
    samples = [{'frame_id':str(i), 'rgb':torch.ones(3,2,2), 'depth':torch.ones(1,2,2),
                'K':torch.eye(3), 'weights':torch.ones(1), 'target':torch.zeros(1,2,2)} for i in range(10)]
    model = ScalarPredictor()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    train_epoch(model, samples, optimizer, phase='global', device='cpu', accumulation=8, shuffle=False)
    torch.testing.assert_close(model.weight, torch.tensor(.64))


def test_feature_cache_preserves_outputs_without_reencoding():
    from models.keypoints.experiment import FeatureCache
    from models.keypoints.model import RGBDKeypointModel
    model = RGBDKeypointModel(small_config(True)); model.set_phase('local')
    batch = {'frame_id':['x'], 'rgb':torch.rand(1,3,16,24), 'depth':torch.ones(1,1,16,24),
             'K':torch.tensor([[[20.,0.,11.5],[0.,20.,7.5],[0.,0.,1.]]])}
    cache = FeatureCache(1024*1024)
    first = cache.get_or_compute(model, batch, 'cpu')
    second = cache.get_or_compute(model, batch, 'cpu')
    assert cache.hits == 1 and cache.misses == 1 and cache.bytes > 0
    assert all(not t.requires_grad and t.device.type == 'cpu' for t in first)
    for a,b in zip(first,second): torch.testing.assert_close(a,b)


def test_global_phase_selects_global_validation_with_local_enabled():
    from models.keypoints.experiment import evaluate_model
    from models.keypoints.model import RGBDKeypointModel
    class SingleFrameDataset(torch.utils.data.Dataset):
        keypoint_names = [f'k{i}' for i in range(9)]
        def __len__(self): return 1
        def __getitem__(self, index):
            return {'frame_id':'x','rgb':torch.rand(3,16,24),'depth':torch.ones(1,16,24),
                    'K':torch.tensor([[20.,0.,11.5],[0.,20.,7.5],[0.,0.,1.]]),
                    'points':torch.ones(9,2), 'original_points':torch.ones(9,2),
                    'original_size':torch.tensor([16,24]), 'weights':torch.ones(9)}
    model = RGBDKeypointModel(small_config(True)); model.set_phase('global')
    metrics = evaluate_model(model, SingleFrameDataset(), device='cpu')
    assert metrics['selected'] == metrics['global_subpixel']
    assert 'local_subpixel' not in metrics


def test_checkpoint_manifest_rejects_semantic_reordering():
    from models.keypoints.experiment import validate_checkpoint_manifest
    manifest = {'train':['1'],'val':['2'],'test':['3'], 'keypoint_names':['left','right']}
    payload = {'split_ids':{s:manifest[s] for s in ('train','val','test')},'keypoint_names':['right','left']}
    with pytest.raises(ValueError,match='keypoint'):
        validate_checkpoint_manifest(payload,manifest)


def test_resume_to_new_output_preserves_selected_best(tmp_path):
    from models.keypoints.model import RGBDKeypointModel
    from models.keypoints.checkpoints import save_checkpoint,load_checkpoint
    from models.keypoints.experiment import train_run,make_optimizer
    original, destination = tmp_path/'original', tmp_path/'continued'
    model=RGBDKeypointModel(small_config()); model.set_phase('global')
    optimizer=make_optimizer(model,lr=3e-4)
    manifest={'train':['1'],'val':['2'],'test':['3'],'keypoint_names':[f'k{i}' for i in range(9)]}
    kwargs={'keypoint_names':manifest['keypoint_names'],'split_ids':{s:manifest[s] for s in ('train','val','test')},
            'metric':{'mean':1.,'p95':2.},'phase':'global','training':{'epochs':3},
            'extra':{'best_metric':{'mean':1.,'p95':2.},'best_epoch':2}}
    save_checkpoint(original/'best.pt',model,optimizer,epoch=2,**kwargs)
    save_checkpoint(original/'last.pt',model,optimizer,epoch=3,**kwargs)
    restored,payload=load_checkpoint(original/'last.pt')
    best=train_run(restored,tmp_path,manifest,destination,phase='global',epochs=3,resume_payload=payload)
    assert best.exists()
    _,best_payload=load_checkpoint(best)
    assert best_payload['epoch']==2


def test_evaluation_statistics_ignore_artifacts_from_previous_invocations(tmp_path):
    from models.keypoints.experiment import evaluate_model
    from models.keypoints.model import RGBDKeypointModel
    class NamedSamples(list):
        keypoint_names = [f'k{i}' for i in range(9)]
    samples=NamedSamples([{'frame_id':'current','rgb':torch.rand(3,16,24),'depth':torch.ones(1,16,24),
                           'K':torch.tensor([[20.,0.,11.5],[0.,20.,7.5],[0.,0.,1.]]),
                           'points':torch.ones(9,2),'original_points':torch.ones(9,2),
                           'original_size':torch.tensor([16,24]),'weights':torch.ones(9)}])
    # A stale file must never be parsed or counted by the current invocation.
    (tmp_path/'old.json').write_text('{}')
    model=RGBDKeypointModel(small_config()); model.set_phase('global')
    metrics=evaluate_model(model,samples,device='cpu',artifact_dir=tmp_path)
    assert metrics['images']==1 and metrics['artifact_statistics']['image_count']==1


@pytest.mark.parametrize('change', ['config', 'state', 'phase', 'metric'])
def test_resume_rejects_incompatible_companion_best(tmp_path, change):
    from models.keypoints.model import RGBDKeypointModel
    from models.keypoints.checkpoints import save_checkpoint,load_checkpoint
    from models.keypoints.experiment import train_run,make_optimizer
    model=RGBDKeypointModel(small_config()); model.set_phase('global')
    optimizer=make_optimizer(model,lr=3e-4)
    manifest={'train':['1'],'val':['2'],'test':['3'],'keypoint_names':[f'k{i}' for i in range(9)]}
    kwargs={'keypoint_names':manifest['keypoint_names'],'split_ids':{s:manifest[s] for s in ('train','val','test')},
            'metric':{'mean':1.,'p95':2.},'phase':'global','training':{'epochs':3},
            'extra':{'best_metric':{'mean':1.,'p95':2.},'best_epoch':2}}
    save_checkpoint(tmp_path/'best.pt',model,optimizer,epoch=2,**kwargs)
    save_checkpoint(tmp_path/'last.pt',model,optimizer,epoch=3,**kwargs)
    corrupted=torch.load(tmp_path/'best.pt',weights_only=False)
    if change=='config': corrupted['config']['heatmap_size']=(9,13)
    elif change=='state': corrupted['model_state'].pop('head.weight')
    elif change=='phase': corrupted['phase']='joint'
    else: corrupted['metric']={'mean':100.,'p95':200.}
    torch.save(corrupted,tmp_path/'best.pt')
    restored,payload=load_checkpoint(tmp_path/'last.pt')
    with pytest.raises((ValueError,RuntimeError),match='(best|config|Missing|missing|phase)'):
        train_run(restored,tmp_path,manifest,tmp_path/'continued',phase='global',epochs=3,resume_payload=payload)


def test_compute_deadline_requests_checkpoint_resume(monkeypatch):
    from models.keypoints.experiment import check_deadline,ComputeBudgetExpired
    monkeypatch.setenv('MAMBA_STOP_AT','1')
    with pytest.raises(ComputeBudgetExpired): check_deadline()
    monkeypatch.setenv('MAMBA_STOP_AT','99999999999')
    check_deadline()


def test_relative_allocation_budget_is_independent_of_wall_clock(monkeypatch):
    from models.keypoints.experiment import check_deadline,ComputeBudgetExpired
    monkeypatch.delenv('MAMBA_STOP_AT',raising=False)
    monkeypatch.setenv('MAMBA_BUDGET_SECONDS','0')
    with pytest.raises(ComputeBudgetExpired): check_deadline()
    monkeypatch.setenv('MAMBA_BUDGET_SECONDS','99999999999')
    check_deadline()
