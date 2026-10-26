import pytest
from models.keypoints.campaign_budget import Budget


def test_global_four_gpu_cap_across_phases_and_retry():
    ledger=Budget(phase_hours={'pilot':72,'runtime':36})
    for i in range(4):
        assert ledger.reserve(str(i),'pilot' if i<3 else 'runtime',3600)
    assert ledger.reserve('fifth','runtime',3600) is None
    ledger.records[0].state='FAILED'; ledger.records[0].elapsed=900
    assert ledger.reserve('retry','runtime',3600)
    assert ledger.summary()['active']==4
    assert ledger.summary()['allocated_gpu_hours']==.25


def test_budget_reserves_full_walltime_and_charges_failed_allocations():
    ledger=Budget(total_hours=2,concurrency=4)
    first=ledger.reserve('a','pilot',7200)
    assert ledger.reserve('b','pilot',60) is None
    first.state='TIMEOUT'; first.elapsed=7200
    assert ledger.reserve('b','pilot',60) is None


def test_phase_trial_cap_and_early_completion_releases_unused_reservation():
    ledger=Budget(phase_hours={'pilot':1})
    first=ledger.reserve('a','pilot',3600,trial_cap_seconds=3600)
    first.state='COMPLETED'; first.elapsed=1800
    assert ledger.reserve('a','pilot',1860,trial_cap_seconds=3600) is None
    assert ledger.reserve('a','pilot',1800,trial_cap_seconds=3600)
    recovered=Budget(ledger.serialize(),phase_hours={'pilot':1})
    assert recovered.summary()==ledger.summary()


@pytest.mark.parametrize('seconds',[True,0,-1,59,259201,3.5])
def test_invalid_job_time_is_rejected(seconds):
    with pytest.raises(ValueError): Budget().reserve('a','pilot',seconds)
