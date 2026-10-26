"""Paired validation diagnostics; cluster bootstrap never reads test or real labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from .multisource import read, dump, NAMES
from .acceptance import acceptable


def paired_rows(left, right):
    a={r['uid']:r for r in left}; b={r['uid']:r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():
        raise ValueError('Paired UID mismatch or duplicates')
    pairs=[]
    for uid in sorted(a):
        x,y=a[uid],b[uid]
        for key in ('group','source','valid','gt','bbox'):
            if x[key]!=y[key]: raise ValueError(f'Paired {key} differs: {uid}')
        pairs.append((x,y))
    return pairs


def values(pairs):
    valid=np.array([a['valid'] for a,b in pairs],bool)
    ea,eb=(np.array([row[j]['errors'] for row in pairs],float) for j in (0,1))
    na,nb=(np.array([row[j]['normalized'] for row in pairs],float) for j in (0,1))
    return ea,eb,na,nb,valid


def summarize(pairs):
    ea,eb,na,nb,valid=values(pairs)
    def score(e,n):
        return dict(mean=float(e[valid].mean()),median=float(np.median(e[valid])),
                    p95=float(np.percentile(e[valid],95)),PCK05=float((n[valid]<=.05).mean()))
    a,b=score(ea,na),score(eb,nb)
    lost=valid&(na<=.05)&(nb>.05); gained=valid&(na>.05)&(nb<=.05)
    result=dict(images=len(pairs),points=int(valid.sum()),baseline=a,candidate=b,
                delta={k:b[k]-a[k] for k in a},worse_points=int(((eb>ea)&valid).sum()),
                lost_pck_points=int(lost.sum()),lost_pck_images=int(lost.any(1).sum()),
                gained_pck_points=int(gained.sum()),gained_pck_images=int(gained.any(1).sum()))
    mm_a=[]; mm_b=[]
    for x,y in pairs:
        z=np.array(x['gt_depth']); mask=(z>0)&np.array(x['valid'])
        for row,out in ((x,mm_a),(y,mm_b)):
            displacement=(np.array(row['coords'])-np.array(x['gt']))/np.array([x['fx'],x['fy']])
            out.extend((np.linalg.norm(displacement,axis=-1)*z*1000)[mask].tolist())
    result['mm']=dict(points=len(mm_a),coverage=len(mm_a)/max(1,int(valid.sum())),
                      baseline_mean=float(np.mean(mm_a)) if mm_a else None,
                      candidate_mean=float(np.mean(mm_b)) if mm_b else None,
                      baseline_median=float(np.median(mm_a)) if mm_a else None,
                      candidate_median=float(np.median(mm_b)) if mm_b else None,
                      baseline_p95=float(np.percentile(mm_a,95)) if mm_a else None,
                      candidate_p95=float(np.percentile(mm_b,95)) if mm_b else None,
                      definition='Lateral error at paired valid GT sensor Z, pinhole synthetic camera')
    return result


def bootstrap(pairs,repeats=2000,seed=20261026):
    # Sample whole render/parent groups, preserving all images/keypoints and pairing.
    groups={}
    for i,(a,b) in enumerate(pairs): groups.setdefault(a['group'],[]).append(i)
    blocks=list(groups.values()); rng=np.random.default_rng(seed)
    ea,eb,na,nb,valid=values(pairs); deltas=[]
    for _ in range(repeats):
        index=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])
        mask=valid[index]; x,y=ea[index][mask],eb[index][mask]
        u,v=na[index][mask],nb[index][mask]
        deltas.append((y.mean()-x.mean(),np.median(y)-np.median(x),
                       np.percentile(y,95)-np.percentile(x,95),(v<=.05).mean()-(u<=.05).mean()))
    intervals=np.percentile(np.asarray(deltas),[2.5,97.5],axis=0)
    return dict(groups=len(groups),repeats=repeats,seed=seed,
                ci95={k:intervals[:,i].tolist() for i,k in enumerate(('mean','median','p95','PCK05'))})


def analyze(root):
    from .multisource import digest
    manifest=read(root/'manifest50k.json'); out=root/'continuation'
    # Derive scale bins exclusively from train annotations, once on CPU compute.
    scales=[]
    for r in manifest['train']:
        box=read(r['annotation'])['object_bbox_xyxy']
        scales.append(float(np.linalg.norm(np.array(box[2:])-box[:2])))
    edges=np.quantile(scales,[1/3,2/3]).tolist()
    results={}; priority=False; selection={}
    for seed in (20261025,20261026,20261027):
        base=read(out/'diagnostics'/f'baseline_{seed}_best.json')
        cand=read(out/'diagnostics'/f'candidate_{seed}_best.json')
        pairs=paired_rows(base['records'],cand['records']); per={}
        for source in ('OLD','NEW'):
            subset=[p for p in pairs if p[0]['source']==source]
            item=summarize(subset); item.update(bootstrap(subset))
            # Practical priority threshold is not an acceptance tolerance.
            meaningful=any(item['ci95'][k][0]>0 and item['delta'][k]>=max(.05,.05*item['baseline'][k])
                           for k in ('mean','p95'))
            meaningful |= item['ci95']['PCK05'][1]<0 and item['delta']['PCK05']<=-.001
            item['training_priority']=bool(meaningful); priority |= bool(meaningful)
            item['keypoints']={name:summarize([(dict(a,valid=[v and j==i for j,v in enumerate(a['valid'])]),
                                               dict(b,valid=[v and j==i for j,v in enumerate(b['valid'])]))
                                              for a,b in subset]) for i,name in enumerate(NAMES)}
            for label,condition in (
                ('small',lambda a:np.linalg.norm(np.array(a['bbox'][2:])-a['bbox'][:2])<edges[0]),
                ('medium',lambda a:edges[0]<=np.linalg.norm(np.array(a['bbox'][2:])-a['bbox'][:2])<edges[1]),
                ('large',lambda a:np.linalg.norm(np.array(a['bbox'][2:])-a['bbox'][:2])>=edges[1]),
                ('depth_below_half',lambda a:a['depth_fraction']<.5),
                ('depth_at_least_half',lambda a:a['depth_fraction']>=.5)):
                selected=[p for p in subset if condition(p[0])]
                if selected: item[label]=summarize(selected)
            item['worst_images']=[dict(uid=a['uid'],group=a['group'],
                 delta_mean=float(np.mean(np.array(b['errors'])-a['errors'])))
                 for a,b in sorted(subset,key=lambda p:np.mean(np.array(p[1]['errors'])-p[0]['errors']),reverse=True)[:20]]
            groups={a['group'] for a,b in subset}
            item['worst_groups']=sorted([dict(group=g,**summarize([p for p in subset if p[0]['group']==g]))
                                          for g in groups],key=lambda r:r['delta']['mean'],reverse=True)[:20]
            per[source]=item
        results[str(seed)]=per
        last=read(out/'diagnostics'/f'candidate_{seed}_last.json')
        candidates=[('best',cand),('last',last)]
        admitted=[(name,data) for name,data in candidates if acceptable(data['metrics'],base['metrics'])]
        selection[str(seed)]=dict(admissible=[name for name,data in admitted],
                                 selected=min(admitted,key=lambda t:t[1]['metrics']['final']['balanced_mean'])[0] if admitted else None)
        history=root/'trials'/f'F_candidate_{seed}'/'history.jsonl'
        eligible=[read_value for line in history.read_text().splitlines()
                  if acceptable((read_value:=json.loads(line))['validation'],base['metrics'])]
        selection[str(seed)]['historical_admissible_epochs']=[r['epoch'] for r in eligible]
        selection[str(seed)]['historical_epoch_weights_available']=False
    result=dict(manifest_sha256=digest(root/'manifest50k.json'),scale_edges_train=edges,
                results=results,checkpoint_selection=selection,training_priority=bool(priority),
                group_limitation='Render/parent groups are not verified independent physical scenes',
                subgroup_policy='Keypoint/scale/depth findings are descriptive, unadjusted for multiple comparisons; not causal claims',
                priority_policy='CI excludes zero plus >=5% and >=0.05px mean/P95 increase, or >=0.1pp PCK loss; locked acceptance unchanged')
    dump(out/'regression.json',result)
    return result


def write_report(root,state):
    """Update the canonical report; retain historical rows and release decision."""
    out=root/'continuation'; diagnosis=read(out/'regression.json') if (out/'regression.json').exists() else None
    marker='\n<!-- continuation-20261026 -->\n'
    original=(root/'REPORT.md').read_text().split(marker)[0]
    lines=['','## Tiếp nối ngày 26/10/2026',
        f'Trạng thái: **{state["phase"]}**. Default baseline được giữ; candidate lịch sử được bảo toàn.',
        f'GPU-giờ tổng: {state["budget"]["allocated_gpu_hours"]:.4f}/288; '
        f'đợt tiếp nối: {state["continuation_budget"]["allocated_gpu_hours"]:.4f}/162.',
        'Không dùng D455 hoặc synthetic test để chọn thí nghiệm. CI lấy mẫu lại theo render/parent, giữ nguyên cặp model.',
        'Gate không được nới; tiêu chí độ lớn thực tế chỉ quyết định ưu tiên training.']
    if diagnosis:
        lines += ['', '| Seed | Nguồn | Δmean px [CI95] | ΔP95 px [CI95] | ΔPCK điểm % [CI95] | PCK mất/được | Nhóm |',
                  '|---|---|---|---|---|---|---|']
        for seed,sources in diagnosis['results'].items():
            for source,r in sources.items():
                def fmt(key,scale=1):
                    a,b=r['ci95'][key]
                    return f'{r["delta"][key]*scale:+.5f} [{a*scale:+.5f}, {b*scale:+.5f}]'
                lines.append(f'| {seed} | {source} | {fmt("mean")} | {fmt("p95")} | {fmt("PCK05",100)} | '
                             f'{r["lost_pck_points"]}/{r["gained_pck_points"]} | {r["groups"]} |')
        lines+=['',f'Ưu tiên retrain theo diagnostic: **{diagnosis["training_priority"]}**.',
                'Chi tiết paired/keypoint/scale/depth/mm và checkpoint selection: `continuation/regression.json`.',
                'Các nhóm render chưa chứng minh độc lập theo scene vật lý; CI không giải quyết hạn chế này.']
    runtime=root/'trials/N_runtime/runtime.json'
    if runtime.exists():
        data=read(runtime); configs=data.get('configurations',[])
        lines+=['','### Runtime trên cùng checkpoint candidate',
                '| Precision | Batch | Workers | File→coords ảnh/s | GPU P50 ms/batch | VRAM MiB | Parity/accuracy gate |',
                '|---|---:|---:|---:|---:|---:|---|']
        for row in configs:
            if 'gpu_ready' not in row:
                lines.append(f'| {row["precision"]} | {row["batch"]} | {row["workers"]} | — | — | — | {row.get("error","failed")} |')
                continue
            throughput=np.median([r['images_per_second'] for r in row['file_to_coordinates']])
            latency=np.median([r['batch_p50_ms'] for r in row['gpu_ready']])
            memory=max(r['allocated_bytes'] for r in row['gpu_ready'])/2**20
            lines.append(f'| {row["precision"]} | {row["batch"]} | {row["workers"]} | {throughput:.2f} | {latency:.3f} | {memory:.1f} | {row["accepted"]} |')
        lines+=['','Batch throughput và batch latency không được gọi là cải thiện latency batch-one.',
                'Cấu hình runtime được giữ chỉ khi qua accuracy/parity gate; cần lợi ích >=5% vượt dao động đo để khuyến nghị thay đổi.']
    lines+=['','Default chưa thay đổi. Đánh giá D455 lịch sử vẫn là benchmark real-domain đã quan sát.',
            'Artifact mới có snapshot và hash riêng tại `continuation/`; source/checkpoint lịch sử không bị ghi đè.']
    (root/'REPORT.md').write_text(original+marker+'\n'.join(lines)+'\n')
    release=read(root/'release.json')
    if not (out/'historical_release.json').exists(): dump(out/'historical_release.json',release)
    release.update(accepted=False,budget=state['budget'],continuation_budget=state['continuation_budget'],
        continuation_status=state['phase'],continuation_reason='Locked default gates remain required; historical test regressions not retuned',
        validation_diagnostics='continuation/regression.json')
    dump(root/'release.json',release)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); analyze(args.root)
