"""Durable Slurm submission receipts and transient-error handling."""
from __future__ import annotations
from datetime import datetime,timezone,timedelta
import hashlib
import subprocess
import time
from .campaign import dump


class SlurmUnavailable(RuntimeError):pass


def output(command):
    try:return subprocess.check_output(command,text=True,timeout=30)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired,OSError) as error:
        raise SlurmUnavailable(str(error)) from error


def token(root,identifier,attempt):
    return 'gdl-'+hashlib.sha256(f'{root}:{identifier}:{attempt}'.encode()).hexdigest()[:32]


def recover_submissions(root,state):
    unresolved=[r for r in state['ledger'] if not r.get('job_id') and r.get('state') in ('RESERVED','SUBMISSION_UNKNOWN')]
    if not unresolved:return
    queue=output(['squeue','--me','--noheader','--format=%i|%k'])
    accounting=output(['sacct','-X','-u','tinnt','-S',state.get('accounting_start_date','2026-10-26'),
        '-n','-P','--format=JobIDRaw,Comment%120'])
    rows=[line.split('|') for line in (queue+'\n'+accounting).splitlines() if line.strip()]
    for record in unresolved:
        if not record.get('submission_token'):raise ValueError('Legacy unresolved reservation without durable token')
        matches={fields[0].strip() for fields in rows if len(fields)>1 and fields[1].strip()==record['submission_token']}
        if len(matches)>1:raise ValueError('Duplicate Slurm submission token: '+record['submission_token'])
        if matches:
            record['job_id']=matches.pop();record['state']='PENDING';record['recovered_submission']=True
        elif time.time()-record.get('created_epoch',time.time())>=180:
            # Both live queue and accounting were checked after the grace interval.
            record.update(state='SUBMIT_FAILED',elapsed_seconds=0,reason='No matching Slurm receipt after reconciliation')
    dump(root/'architecture_gdl/state.json',state)


def heartbeat(state,status='RUNNING',error=None):
    state['controller_health']=dict(status=status,updated_at=datetime.now(timezone(timedelta(hours=7))).isoformat(),
        error=error)
