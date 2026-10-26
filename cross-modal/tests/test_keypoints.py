import pytest
import torch

from models.keypoints.config import DEFAULT_CONFIG, make_config
from models.keypoints.model import RGBDKeypointModel


def small_config(**overrides):
    values = {"input_size": (24, 32), "heatmap_size": (12, 16),
              "out_indices": (1, 2), "geometry_channels": (4, 4, 4, 4, 4),
              "fusion_channels": (4, 4), "decoder_width": 4,
              "num_keypoints": 2, "pretrained": False,
              "global_ssm": {"state_dim": 2, "expand": 1},
              "local": {"enabled": True, "state_dim": 2, "microbatch_size": 2}}
    values.update(overrides)
    return make_config(values)


def inputs(hw=(24, 32)):
    h, w = hw
    return (torch.randn(1, 3, h, w), torch.ones(1, 1, h, w),
            torch.tensor([[[60., 0., (w-1)/2], [0., 60., (h-1)/2], [0., 0., 1.]]]))


def test_config_deep_merge_does_not_mutate_defaults_or_override_dict():
    overrides = {"local": {"enabled": True, "layout": "one"}}
    config = make_config(overrides)
    config["local"]["level"] = .8
    assert DEFAULT_CONFIG["local"]["level"] == .30
    assert overrides == {"local": {"enabled": True, "layout": "one"}}
    assert config["local"]["state_dim"] == 16
    assert tuple(config["out_indices"]) == (1, 2, 3, 4)


@pytest.mark.parametrize("overrides", [
    {"out_indices": (1, 2), "fusion_channels": (32,)},
    {"out_indices": (2, 1)}, {"geometry_channels": (32, 48)},
    {"heatmap_size": (0, 160)}, {"num_keypoints": 0},
    {"local": {"directions": 1}}, {"local": {"level": 0}}, {"misspelled": True},
])
def test_invalid_config_is_rejected_before_allocating_model(overrides):
    with pytest.raises(ValueError):
        make_config(overrides)


def test_default_four_scale_fusion_and_heatmap_shapes():
    model = RGBDKeypointModel().eval()
    with torch.no_grad():
        fused = model.encode_fused(*inputs((180, 320)))
        maps = model.global_from_fused(fused)
    assert [tuple(t.shape) for t in fused] == [
        (1, 32, 45, 80), (1, 32, 23, 40),
        (1, 32, 12, 20), (1, 32, 6, 10)]
    assert maps.shape == (1, 9, 90, 160) and torch.isfinite(maps).all()
    assert model.rgb_encoder.out_indices == (1, 2, 3, 4)
    assert model.geometry_encoder.out_indices == (1, 2, 3, 4)


def test_odd_shape_stages_share_encoder_pass_and_forward_remains_global():
    model = RGBDKeypointModel(small_config(input_size=(25, 33))).eval()
    rgb, depth, K = inputs((25, 33))
    calls = []
    handle = model.rgb_encoder.register_forward_hook(lambda *args: calls.append(1))
    try:
        with torch.no_grad():
            result = model.forward_stages(rgb, depth, K, return_patches=True)
    finally:
        handle.remove()
    assert calls == [1]
    assert result.global_heatmaps.shape == (1, 2, 12, 16)
    assert result.local.heatmaps.shape == (1, 2, 7, 9)
    with torch.no_grad():
        torch.testing.assert_close(model(rgb, depth, K), result.global_heatmaps)
        fused = model.encode_fused(rgb, depth, K)
        cached = model.refine_cached(fused[0], result.global_heatmaps, return_patches=True)
    assert cached.global_heatmaps is result.global_heatmaps
    torch.testing.assert_close(cached.local.heatmaps, result.local.heatmaps)


def test_local_only_loss_reaches_fusion_without_gradient_to_global_head():
    torch.manual_seed(17)
    model = RGBDKeypointModel(small_config()).eval()
    model.set_phase("joint")
    rgb, depth, K = inputs()
    fused = model.encode_fused(rgb, depth, K)
    fused[0].retain_grad()
    maps = model.global_from_fused(fused)
    stage = model.refine_cached(fused[0], maps, return_patches=True)
    loss = sum(p.prediction.square().mean() for p in stage.local.patches)
    loss.backward()
    assert fused[0].grad is not None and fused[0].grad.abs().sum() > 0
    assert model.head.weight.grad is None
    assert all(p.grad is None for p in model.decoder.parameters())
    assert model.fusion.blocks[0].rgb_project.weight.grad.abs().sum() > 0
    assert model.rgb_encoder.encoder.conv1.weight.grad.abs().sum() > 0
    assert model.geometry_encoder.encoder.stem[0].weight.grad.abs().sum() > 0
    assert model.local_refinement.local_mamba.ssm.A_log.grad.abs().sum() > 0


def test_phase_local_freezes_global_batchnorm_and_only_trains_local():
    model = RGBDKeypointModel(small_config())
    model.set_phase("local")
    model.train()
    global_modules = [model.rgb_encoder, model.geometry_encoder, model.fusion,
                      model.decoder, model.head]
    assert all(not module.training for module in global_modules)
    assert model.local_refinement.training
    assert all(not p.requires_grad for module in global_modules for p in module.parameters())
    assert all(p.requires_grad for p in model.local_refinement.parameters())
    bn = next(m for m in model.rgb_encoder.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    stage = model.forward_stages(*inputs(), return_patches=True)
    stage.local.heatmaps.square().mean().backward()
    torch.testing.assert_close(before, bn.running_mean)
    assert all(p.grad is None for module in global_modules for p in module.parameters())
    assert model.local_refinement.head.weight.grad.abs().sum() > 0
    model.set_phase("joint")
    assert all(p.requires_grad for p in model.parameters())
    model.train()
    assert all(module.training for module in global_modules)
    model.set_phase("global")
    assert all(not p.requires_grad for p in model.local_refinement.parameters())


def test_disabled_local_and_decoder_feature_count_validation():
    model = RGBDKeypointModel(small_config(local={"enabled": False})).eval()
    with torch.no_grad():
        fused = model.encode_fused(*inputs())
        out = model.forward_stages(*inputs())
    assert out.local is None and model.local_refinement is None
    with pytest.raises(ValueError, match="scale|feature|count"):
        model.global_from_fused(fused[:-1])
    with pytest.raises(ValueError):
        model.set_phase("local")


def test_explicit_legacy_config_without_local_remains_global_only():
    config = small_config()
    del config["local"]
    model = RGBDKeypointModel(config).eval()
    assert not model.local_enabled
    with torch.no_grad():
        assert model.forward_stages(*inputs()).local is None
