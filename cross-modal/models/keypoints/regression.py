"""Paired validation diagnostics; cluster bootstrap never reads test or real labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from .multisource import read, dump, NAMES
from .acceptance import acceptable


def geometry_metrics(rows):
    """Lateral camera error at labeled Z; missing predictions remain failures."""
    if not rows: return dict(points=0,images=0,mean=None,median=None,p95=None)
    gt=np.asarray([r['gt'] for r in rows],float)
    prediction=np.asarray([r['coords'] for r in rows],float)
    z=np.asarray([r['gt_depth'] for r in rows],float)
    labeled=np.asarray([r['valid'] for r in rows],bool)
    focal=np.asarray([[r['fx'],r['fy']] for r in rows],float)[:,None,:]
    reference=labeled&np.isfinite(z)&(z>0)&np.isfinite(gt).all(-1)
    predicted=np.isfinite(prediction).all(-1)
    evaluable=reference&predicted
    mm=np.linalg.norm((prediction-gt)/focal,axis=-1)*z*1000
    values=mm[evaluable]
    complete=reference.all(-1); available=predicted.all(-1)
    mean9=mm.mean(-1)
    denominator=int(complete.sum())
    result=dict(points=int(evaluable.sum()),reference_points=int(reference.sum()),images=len(rows),
        point_coverage=float(evaluable.sum()/max(1,reference.sum())),
        complete_depth_images=denominator,predicted_complete_images=int((complete&available).sum()),
        mean=float(values.mean()) if values.size else None,
        median=float(np.median(values)) if values.size else None,
        p95=float(np.percentile(values,95)) if values.size else None,
        image_mean_below_5mm=float((complete&available&(mean9<5)).sum()/denominator) if denominator else None,
        all_nine_below_5mm=float((complete&available&(mm<5).all(-1)).sum()/denominator) if denominator else None,
        definition='Lateral camera displacement at labeled Z, not independent 3D pose error; depth provenance in records')
    result['keypoints']={name:dict(points=int(evaluable[:,i].sum()),
        mean=float(mm[:,i][evaluable[:,i]].mean()) if evaluable[:,i].any() else None,
        p95=float(np.percentile(mm[:,i][evaluable[:,i]],95)) if evaluable[:,i].any() else None)
        for i,name in enumerate(NAMES)}
    return result


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
                      reference_types=sorted({kind for x,y in pairs for kind in x.get('depth_reference',['sensor_at_label'])}),
                      definition='Lateral error at paired labeled Z, pinhole camera; reference_types distinguish synthetic keypoint GT from sensor samples')
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


def subgroup_intervals(root):
    """Exploratory follow-up on the small OLD stratum; does not change priority/gates."""
    out=root/'continuation'; result=read(out/'regression.json')
    edge=result['scale_edges_train'][0]
    for seed in (20261025,20261026,20261027):
        a=read(out/'diagnostics'/f'baseline_{seed}_best.json')['records']
        b=read(out/'diagnostics'/f'candidate_{seed}_best.json')['records']
        pairs=[p for p in paired_rows(a,b) if p[0]['source']=='OLD' and
               np.linalg.norm(np.array(p[0]['bbox'][2:])-p[0]['bbox'][:2])<edge]
        result['results'][str(seed)]['OLD']['small'].update(bootstrap(pairs))
    result['subgroup_intervals_policy']='Exploratory small-OLD follow-up, unadjusted for multiplicity; priority and locked gates unchanged'
    result['interval_limitation']='Conditional on checkpoints selected using this validation; not independent evidence of unseen-domain generalization'
    dump(out/'regression.json',result)


def render_failures(root):
    """Render paired validation failures from cached predictions, without model inference."""
    from PIL import Image,ImageDraw
    out=root/'continuation'; diagnosis=read(out/'regression.json')
    manifest=read(root/'manifest50k.json'); records={r['uid']:r for r in manifest['val']}
    destination=out/'failures'; destination.mkdir(exist_ok=True)
    index=[]
    for seed in (20261025,20261026,20261027):
        baseline={r['uid']:r for r in read(out/'diagnostics'/f'baseline_{seed}_best.json')['records']}
        candidate={r['uid']:r for r in read(out/'diagnostics'/f'candidate_{seed}_best.json')['records']}
        for source in ('OLD','NEW'):
            for item in diagnosis['results'][str(seed)][source]['worst_images'][:3]:
                uid=item['uid']; record=records[uid]; a,b=baseline[uid],candidate[uid]
                with Image.open(Path(record['root'])/'rgb'/(record['frame_id']+'.png')) as handle:
                    image=handle.convert('RGB')
                arrays=[np.array(a[key],float) for key in ('gt','coords')]+[np.array(b['coords'],float)]
                points=np.concatenate(arrays)
                left=max(0,int(np.floor(points[:,0].min()))-35)
                top=max(0,int(np.floor(points[:,1].min()))-35)
                right=min(image.width,int(np.ceil(points[:,0].max()))+36)
                bottom=min(image.height,int(np.ceil(points[:,1].max()))+36)
                if right<=left or bottom<=top: raise ValueError('Invalid failure crop')
                canvas=image.crop((left,top,right,bottom)); draw=ImageDraw.Draw(canvas)
                for coords,color in zip(arrays,('lime','cyan','magenta')):
                    for i,(x,y) in enumerate(coords):
                        if not a['valid'][i]: continue
                        x,y=x-left,y-top
                        draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
                        draw.text((x+4,y-6),str(i),fill=color)
                board=Image.new('RGB',(max(520,canvas.width),canvas.height+90),'black')
                board.paste(canvas,(0,65)); text=ImageDraw.Draw(board)
                text.text((8,5),f'{seed} {uid}',fill='white')
                text.text((8,22),'GT green; baseline cyan; candidate magenta; labels 0..8',fill='white')
                text.text((8,39),f'paired mean delta {item["delta_mean"]:+.4f} original px',fill='white')
                name=f'{seed}_{uid}.png'; board.save(destination/name)
                index.append(dict(seed=seed,uid=uid,group=a['group'],source=source,file=name,
                                  baseline_errors=a['errors'],candidate_errors=b['errors'],
                                  valid=a['valid'],crop_xyxy=[left,top,right,bottom]))
    dump(destination/'index.json',dict(validation_only=True,keypoint_names=NAMES,records=index))


def write_report(root,state):
    """Update the canonical report; retain historical rows and release decision."""
    out=root/'continuation'; diagnosis=read(out/'regression.json') if (out/'regression.json').exists() else None
    marker='\n<!-- continuation-20261026 -->\n'
    original=(root/'REPORT.md').read_text().split(marker)[0]
    lines=['','## Tiếp nối ngày 26/10/2026','',
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
        lines+=['CI điều kiện trên checkpoint đã chọn bằng validation, không phải chứng cứ độc lập về unseen-domain generalization.']
        lines+=['Candidate full-run thay cả pretrained và augmentation; paired diagnostics không đủ để gán quan hệ nhân quả riêng cho một thành phần.']
        lines+=['','Nhóm OLD nhỏ gồm 31 ảnh: regression mean +0,212/+0,233 px ở seed 20261025/20261027, '
                'CI bootstrap bổ sung được lưu trong `regression.json`; seed 20261026 không cho thấy cùng xu hướng. '
                'Đây là phân tích subgroup thăm dò, không nới gate hoặc tự mở training để khớp 31 ảnh validation.']
    runtime=root/'trials/N_runtime/runtime.json'
    if runtime.exists():
        data=read(runtime); configs=data.get('configurations',[])
        lines+=['','### Runtime trên cùng checkpoint candidate','',
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
    efficiency=out/'training_efficiency.json'
    confirmation=out/'runtime_confirmation.json'
    if confirmation.exists():
        confirmed=read(confirmation)
        lines+=['','### Kiểm chứng runtime không giữ buffer của sweep','',
                '| Cấu hình batch-one | File→coords P50/P95 ms | Throughput ảnh/s | GPU P50 ms | Pipeline VRAM allocated/reserved MiB |',
                '|---|---:|---:|---:|---:|']
        for label in ('before','reader_only','after'):
            b=confirmed[label]
            lines.append(f'| {label}: {b["precision"]}, workers={b["workers"]} | '
                f'{np.median([r["batch_p50_ms"] for r in b["file_to_coordinates"]]):.3f}/'
                f'{np.median([r["batch_p95_ms"] for r in b["file_to_coordinates"]]):.3f} | '
                f'{np.median([r["images_per_second"] for r in b["file_to_coordinates"]]):.2f} | '
                f'{np.median([r["batch_p50_ms"] for r in b["gpu_ready"]]):.3f} | '
                f'{max(r["allocated_bytes"] for r in b["file_to_coordinates"])/2**20:.3f}/'
                f'{max(r["reserved_bytes"] for r in b["file_to_coordinates"])/2**20:.3f} |')
        lines+=['Bảng sweep ở trên bao gồm reference buffers của harness; dùng phép đo độc lập này để so VRAM triển khai.']
        lines+=['`reader_only` dùng cùng RGB/depth/K preprocessing nhưng bỏ đọc nhãn, tạo Gaussian target và chuyển GT sang GPU; '
                '`after` chỉ thêm prefetch lên reader đó. Không dùng nhãn GT trong đường inference tối ưu.']
    if efficiency.exists():
        profile=read(efficiency)
        lines+=['','### Bỏ coordinate loss có hệ số 0','',
            f'Thay đổi thời gian step trung vị: {100*profile["speedup_fraction"]:+.2f}%; '
            f'max weight difference sau paired steps: {profile["max_weight_difference"]:.9g}.',
            '**Bị loại và rollback**: lợi ích dưới 5%; benchmark không thiết lập được weight parity GPU. '
            'Max difference được lấy trên các lượt của cả hai cách tính, không đủ để quy nguyên nhân cho shortcut.',
            'Đây là đo compute với input đã ở GPU và effective batch 8; không phải bằng chứng thời gian hội tụ.',
            'Chi tiết ba lượt, P50/P95, throughput và VRAM: `continuation/training_efficiency.json`.']
    lines+=['','Tối ưu packed Local ROI chỉ qua correctness CPU, chưa có benchmark Local phù hợp nên được rollback; '
            'không chọn Local hoặc kiến trúc mới cho default.']
    failures=out/'failures/index.json'
    if failures.exists():
        lines+=['',f'Overlay paired failure cases: {len(read(failures)["records"])} ảnh validation tại `continuation/failures/`, '
                'kèm UID/group, crop và sai số chín keypoint; dùng prediction đã lưu, không chạy lại model.']
    lines+=['','Default chưa thay đổi. Đánh giá D455 lịch sử vẫn là benchmark real-domain đã quan sát.',
            'Artifact mới có snapshot và hash riêng tại `continuation/`; source/checkpoint lịch sử không bị ghi đè.']
    (root/'REPORT.md').write_text(original+marker+'\n'.join(lines)+'\n')
    release=read(root/'release.json')
    if not (out/'historical_release.json').exists(): dump(out/'historical_release.json',release)
    release.update(accepted=False,budget=state['budget'],continuation_budget=state['continuation_budget'],
        continuation_status=state['phase'],continuation_reason='Locked default gates remain required; historical test regressions not retuned',
        validation_diagnostics='continuation/regression.json')
    runtime_recommendation=out/'recommended_runtime.json'
    if runtime_recommendation.exists():
        release['runtime_optimization']=read(runtime_recommendation)
    release['rejected_optimizations']=['BF16 validation regression','Batch>1 failed locked coordinate parity',
        'Zero-weight coordinate shortcut below 5% and GPU parity not established; rolled back',
        'Packed Local ROI lacks appropriate runtime evidence; rolled back']
    dump(root/'release.json',release)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); analyze(args.root)
