"""Shared runtime controls, including execution as python -m experiment."""
import os
import time
ALLOCATION_STARTED = time.monotonic()


class ComputeBudgetExpired(RuntimeError):
    """End this allocation gracefully; the Slurm wrapper resumes saved epochs."""


def check_deadline():
    budget = os.environ.get('MAMBA_BUDGET_SECONDS')
    if budget is not None and time.monotonic()-ALLOCATION_STARTED >= float(budget):
        raise ComputeBudgetExpired('Slurm allocation budget reached; resume from completed checkpoint')
    deadline = os.environ.get('MAMBA_STOP_AT')
    if deadline is not None and time.time() >= float(deadline):
        raise ComputeBudgetExpired('Slurm allocation budget reached; resume from completed checkpoint')
