from pathlib import Path
import pytest
from models.keypoints.campaign import pilot_tasks,acceptable
from models.keypoints.config import make_config


def test_pilot_allocation_configs_and_deterministic_bottleneck_choice():
    tasks=pilot_tasks(Path('/tmp/example'),{'components_mean_ms':{f'fusion_{i}':float(i+1) for i in range(4)}})
    assert len({t['id'] for t in tasks})==len(tasks)
    assert sum(t['hours'] for t in tasks)==69  # 3 hours reserved for conditional joint
    assert all(sum(t['hours'] for t in tasks if t['branch']==b)<=18 for b in ('A','B','C','D'))
    assert next(t for t in tasks if t['id']=='P_skip_fusion')['config']['skip_fusion_scales']==[3]
    for t in tasks:
        make_config(t['config'])
        assert t['effective_batch']==8 and t['seed']==20261025
        if t.get('phase')=='local':
            assert t['parent']=='P_base' and t['config']['local']['embedding']


def score(mean=10.,p95=20.,pck=.8):
    return {'final':{'OLD':{'mean':mean,'p95':p95,'PCK05':pck},
                     'NEW':{'mean':mean,'p95':p95,'PCK05':pck},'balanced_mean':mean}}


def test_selection_requires_per_source_pck_p95_and_local_five_percent():
    assert acceptable(score(9),score(),True)
    assert not acceptable(score(9.8),score(),True)
    assert not acceptable(score(9,21),score())
    assert not acceptable(score(9,19,.79),score())
    assert not acceptable(None,score())
