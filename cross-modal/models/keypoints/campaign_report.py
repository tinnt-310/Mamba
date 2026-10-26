"""Human-readable, fail-closed release report from immutable trial outputs."""
import json
from pathlib import Path
import statistics


def read(path): return json.loads(Path(path).read_text())


def report(root,reservations):
    from .campaign_budget import Budget
    state=read(root/'state.json')
    rows=[]
    for seed in (20261025,20261026,20261027):
        for label in ('baseline','candidate'):
            path=root/'trials'/f'E_F_{label}_{seed}'/'metrics.json'
            bench=path.with_name('benchmark.json')
            if not path.exists() or not bench.exists(): continue
            metrics=read(path); benchmark=read(bench)
            real=metrics['D455']['pixel']; synthetic=metrics['synthetic_test']['final']
            rows.append(dict(label=label,seed=seed,real=real,synthetic=synthetic,
                             latency_ms=statistics.median(x['batch_p50_ms'] for x in benchmark['gpu_ready']),
                             end_to_end_ms=statistics.median(x['batch_p50_ms'] for x in benchmark['file_to_coordinates']),
                             vram=max(x['allocated_bytes'] for x in benchmark['gpu_ready']),
                             gpu=metrics['gpu'],checkpoint=metrics['checkpoint']))
    valid=len(rows)==6
    reasons=[]
    if not valid: reasons.append('Thiếu kết quả kiểm chứng cho đủ hai cấu hình × ba seed.')
    if valid:
        for seed in (20261025,20261026,20261027):
            baseline=next(r for r in rows if r['label']=='baseline' and r['seed']==seed)
            candidate=next(r for r in rows if r['label']=='candidate' and r['seed']==seed)
            if candidate['gpu']!=baseline['gpu']:
                valid=False; reasons.append(f'{seed}: phần cứng khác nhau, không chấp nhận so tốc độ trực tiếp.')
            if not (candidate['real']['mean']<baseline['real']['mean'] and candidate['real']['p95']<=baseline['real']['p95']
                    and candidate['real']['PCK05']>=baseline['real']['PCK05']):
                valid=False; reasons.append(f'{seed}: D455 chưa vượt baseline về mean/P95/PCK.')
            if candidate['latency_ms']>baseline['latency_ms'] or candidate['vram']>baseline['vram']:
                valid=False; reasons.append(f'{seed}: candidate tăng latency hoặc VRAM.')
            if any(candidate['synthetic'][s]['PCK05']<baseline['synthetic'][s]['PCK05'] or
                   candidate['synthetic'][s]['p95']>baseline['synthetic'][s]['p95'] for s in ('OLD','NEW')):
                valid=False; reasons.append(f'{seed}: synthetic test có regression PCK hoặc P95.')
    budget=Budget(reservations)
    release=dict(accepted=valid,reasons=reasons,rows=rows,budget=budget.summary(),
                 policy='No code default changes unless all acceptance gates pass; '
                        'single observed D455 session is not an independent generalization test')
    (root/'release.json').write_text(json.dumps(release,indent=2)+'\n')
    lines=['# Mamba RGB-D — kết quả campaign 50K',
           '',f'Chấp nhận thay mặc định: **{"Có" if valid else "Chưa"}**.',
           '',f'GPU-giờ đã cấp: {budget.summary()["allocated_gpu_hours"]:.3f} / 288.',
           '', '50K synthetic: 25K cũ + 25K mới; D455 không dùng train hoặc chọn hyperparameter.',
           '', '| Model | Seed | D455 mean px | P95 px | PCK5 | GPU P50 ms | File→coords P50 ms | VRAM MiB |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["label"]} | {r["seed"]} | {r["real"]["mean"]:.3f} | {r["real"]["p95"]:.3f} | '
                     f'{100*r["real"]["PCK05"]:.2f}% | {r["latency_ms"]:.3f} | {r["end_to_end_ms"]:.3f} | {r["vram"]/2**20:.1f} |')
    lines+=['','## Giới hạn và lý do chưa chấp nhận','']+[f'- {r}' for r in reasons]
    lines+=['- D455 thuộc một session đã được quan sát; cần session mới để chứng minh chuyển miền độc lập.',
            '- Test synthetic có ảnh đã quan sát, được đánh dấu trong manifest; chia theo render seed chưa chứng minh độc lập theo scene vật lý.',
            '- Timing file→coords D455 không bao gồm alignment đã chuẩn bị; không gọi đây là timing có alignment.',
            '', '## Artifact','',
            '- `pilot_selection.json`: mọi pilot, kể cả phương án bị loại.',
            '- `final_lock.json`: cấu hình khóa trước test/D455.',
            '- `trials/*/history.jsonl`, `gpu-*.csv`, `runtime.json`: learning curves, utilization và sweep.',
            '- `state.json`, `source_manifest.json`, `environment.txt`: ngân sách, snapshot và environment.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
