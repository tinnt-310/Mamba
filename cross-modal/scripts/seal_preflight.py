"""Seal CPU correctness/provenance only after immutable snapshot tests pass."""
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from models.keypoints.multisource import read,dump,digest,load_manifest

assert os.environ.get('SLURM_JOB_ID')
root=Path(sys.argv[1])
suite=ET.parse(root/'correctness.xml').getroot()
counts={key:sum(int(s.get(key,0)) for s in suite.iter('testsuite')) for key in ('tests','failures','errors','skipped')}
assert not counts['failures'] and not counts['errors']
load_manifest(root/'manifest50k.json')
snapshot=read(root/'source_manifest.json')
for relative,expected in snapshot['files'].items():
    if digest(root/'source'/relative)!=expected:
        raise ValueError('Source changed after snapshot: '+relative)
for trial in ('S_global','S_local'):
    assert read(root/'trials'/trial/'completed.json')['status']=='complete'
dump(root/'preflight_passed.json',dict(job_id=os.environ['SLURM_JOB_ID'],cpu_tests=counts,
     source_sha256=snapshot['sha256'],manifest_sha256=digest(root/'manifest50k.json'),
     training_smoke=['S_global','S_local']))
