"""Locked validation gates shared by training and orchestration."""


def acceptable(candidate, baseline, local=False):
    if candidate is None or baseline is None:
        return False
    for source in ('OLD', 'NEW'):
        c, b = candidate['final'][source], baseline['final'][source]
        if c['PCK05'] + 1e-12 < b['PCK05'] or c['p95'] > b['p95']:
            return False
    return candidate['final']['balanced_mean'] < (.95 if local else 1.) * baseline['final']['balanced_mean']
