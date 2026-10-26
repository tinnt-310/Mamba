"""Exercise the whole controller graph without Slurm submission or tensors."""
import json
from pathlib import Path
from models.keypoints import campaign as module


def test_full_controller_graph_respects_budget_and_finishes(tmp_path,monkeypatch):
    root=tmp_path
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    (root/'source').mkdir()
    for name in ('manifest50k.json','preflight_passed.json','gpu_preflight_passed.json','pretrained/provenance.json'):
        put(root/name,{})
    put(root/'source_manifest.json',{'sha256':'source'})
    for variant in ('A1_NEW','A4_NEW'):
        put(root/'baseline'/variant/'profile.json',{'components_mean_ms':{f'fusion_{i}':i+1 for i in range(4)}})
    put(root/'baseline/training_profile.json',{'configurations':[{'batch':8,'workers':4,'images_per_second':100}]})
    monkeypatch.setenv('SLURM_JOB_ID','cpu-test')
    submitted=[]; next_id=[100]
    def check_output(command,text=True):
        assert command[0]=='sbatch'
        spec=json.loads(Path(command[-2]).read_text())
        submitted.append(spec)
        folder=root/'trials'/spec['id']; folder.mkdir(parents=True,exist_ok=True)
        # Every trained candidate wins the validation guards in this simulation.
        value=10 if spec['id']=='P_base' else 5
        metric={'final':{'OLD':{'mean':value,'p95':value*2,'PCK05':.9},
                         'NEW':{'mean':value,'p95':value*2,'PCK05':.9},
                         'balanced_mean':value,'balanced_p95':value*2}}
        (folder/'history.jsonl').write_text(json.dumps({'validation':metric})+'\n')
        (folder/'best.pt').write_text('fake-only-controller-never-loads-tensors')
        put(folder/'completed.json',{'status':'complete'})
        next_id[0]+=1
        return str(next_id[0])+'\n'
    monkeypatch.setattr(module.subprocess,'check_output',check_output)
    peaks=[]
    def account(budget):
        peaks.append(sum(not r.terminal for r in budget.records))
        for reservation in budget.records:
            reservation.state='COMPLETED'; reservation.elapsed=60
    monkeypatch.setattr(module,'update_accounting',account)
    iterations=[0]
    def sleep(_):
        iterations[0]+=1
        assert iterations[0]<100,'Controller stalled'
    monkeypatch.setattr(module.time,'sleep',sleep)
    module.controller(root,'5705')
    state=json.loads((root/'state.json').read_text())
    assert state['phase']=='complete'
    assert max(peaks)<=4
    assert state['budget']['charged_or_reserved_gpu_hours']<=288
    assert len([t for t in submitted if t['id'].startswith('F_')])==6
    assert (root/'final_lock.json').exists()
    assert not json.loads((root/'release.json').read_text())['accepted']
