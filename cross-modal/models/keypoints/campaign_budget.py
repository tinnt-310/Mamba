"""Conservative admission: allocated GPU time, active reservations, global cap."""
from dataclasses import dataclass,asdict
import math

TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}


@dataclass
class Reservation:
    trial: str
    phase: str
    seconds: int
    job_id: str | None = None
    state: str = 'RESERVED'
    elapsed: int = 0

    @property
    def terminal(self):
        return self.state.split()[0].split('+')[0] in TERMINAL

    @property
    def charge(self):
        return self.elapsed if self.terminal else self.seconds


class Budget:
    def __init__(self,records=(),total_hours=288,concurrency=4,phase_hours=None):
        self.records = [Reservation(**r) if isinstance(r,dict) else r for r in records]
        self.total_seconds = int(total_hours*3600)
        self.concurrency = concurrency
        self.phase_seconds = {k:int(v*3600) for k,v in (phase_hours or {}).items()}

    def reserve(self,trial,phase,seconds,trial_cap_seconds=None):
        if isinstance(seconds,bool) or not isinstance(seconds,int) or seconds<60 or seconds>72*3600:
            raise ValueError('Reservation must be 60 seconds to 72 hours')
        active = [r for r in self.records if not r.terminal]
        if len(active)>=self.concurrency:
            return None
        if sum(r.charge for r in self.records)+seconds>self.total_seconds:
            return None
        if phase in self.phase_seconds and sum(r.charge for r in self.records if r.phase==phase)+seconds>self.phase_seconds[phase]:
            return None
        if trial_cap_seconds is not None and sum(r.charge for r in self.records if r.trial==trial)+seconds>trial_cap_seconds:
            return None
        reservation=Reservation(trial,phase,seconds)
        self.records.append(reservation)
        return reservation

    def remaining_trial(self,trial,cap):
        return max(0,cap-sum(r.charge for r in self.records if r.trial==trial))

    def serialize(self):
        return [asdict(r) for r in self.records]

    def summary(self):
        return dict(allocated_gpu_hours=sum(r.elapsed for r in self.records)/3600,
                    charged_or_reserved_gpu_hours=sum(r.charge for r in self.records)/3600,
                    active=sum(not r.terminal for r in self.records),limit_gpu_hours=self.total_seconds/3600)
