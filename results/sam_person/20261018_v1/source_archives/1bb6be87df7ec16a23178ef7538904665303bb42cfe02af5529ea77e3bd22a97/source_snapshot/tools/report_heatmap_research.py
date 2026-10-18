"""Summarize recorded runs, audit paired controls and render fixed diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageDraw


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def histogram(path,series,title):
    """Dependency-free diagnostic SVG; raw samples remain in JSON."""
    all_values=[value for _,values in series for value in values]
    if not all_values: return
    lo,hi=min(all_values),max(all_values)
    if hi<=lo: hi=lo+1e-6
    edges=np.linspace(lo,hi,41)
    histograms=[np.histogram(values,bins=edges)[0]/max(len(values),1) for _,values in series]
    peak=max(float(count.max()) for count in histograms) or 1
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="440">',
           '<rect width="800" height="440" fill="white"/>',
           '<text x="35" y="25" font-size="15">'+title+'</text>']
    for index,((name,values),counts) in enumerate(zip(series,histograms)):
        color=('#1565c0','#ef6c00')[index%2]
        for i,count in enumerate(counts):
            height=320*count/peak
            parts.append(f'<rect x="{45+i*18}" y="{370-height}" width="18" height="{height}" fill="{color}" opacity="0.5"/>')
        parts.append(f'<text x="{70+index*340}" y="420" fill="{color}">{name} (n={len(values)})</text>')
    parts.extend([f'<text x="45" y="395">{lo:.5g}</text>',f'<text x="720" y="395">{hi:.5g}</text>','</svg>'])
    Path(path).write_text('\n'.join(parts))


def colorize(values,size):
    values=np.asarray(Image.fromarray(values.astype(np.float32),mode='F').resize(size,Image.BILINEAR)).clip(0,1)
    rgb=np.stack((np.minimum(values*2,1),np.maximum(values*2-1,0),np.zeros_like(values)),-1)
    return Image.fromarray((rgb*255).astype(np.uint8))


def visualize(run):
    output=run/'visualizations'
    output.mkdir(exist_ok=True)
    config=json.loads((run/'resolved_config.json').read_text())
    for artifact in sorted(run.glob('heatmaps_*.npz')):
        data=np.load(artifact)
        if 'activities' not in data: continue
        vid,cid=data['clip'].tolist()
        fid=int(data['frame_ids'][0])
        image=Image.open(ROOT/config['data_path']/'cafe'/str(vid)/str(cid)/'images'/('frames_'+str(fid)+'.jpg'))
        boxes=data['boxes'][0,0]
        matches=dict(zip(data['matched_slots'].tolist(),data['matched_groups'].tolist()))
        size=(480,270)
        base=image.convert('RGB').resize(size,Image.BILINEAR)
        original=base.copy(); drawer=ImageDraw.Draw(original)
        for actor,box in enumerate(boxes):
            if not box[2] or not box[3]: continue
            cx,cy,w,h=box
            drawer.rectangle(((cx-w/2)*size[0],(cy-h/2)*size[1],(cx+w/2)*size[0],(cy+h/2)*size[1]),outline='lime')
            drawer.text((cx*size[0],cy*size[1]),str(actor+1),fill='yellow')
        panels=[(original,f'clip {vid}/{cid}, frame {fid}, actor IDs')]
        for slot in range(data['masks'].shape[1]):
            mask=data['masks'][0,slot,0]
            predicted=(np.where(data['membership'][0,slot]>.5)[0]+1).tolist()
            activity=int(data['activities'][0,slot].argmax())
            group=matches.get(slot)
            member=(np.where(data['gt_members'][0,group]>.5)[0]+1).tolist() if group is not None else []
            panel=Image.blend(base,colorize(mask,size),.55)
            panels.append((panel,f'slot {slot} -> GT {group}; activity={activity}\nGT={member}; predicted={predicted}'))
        for group in range(len(data['gt_heatmaps'])):
            panels.append((colorize(data['gt_heatmaps'][group,0],size),f'GT group {group}: Gaussian target only'))
        sheet=Image.new('RGB',(4*size[0],((len(panels)+3)//4)*(size[1]+40)),'white')
        draw=ImageDraw.Draw(sheet)
        for index,(panel,label) in enumerate(panels):
            x=(index%4)*size[0]; y=(index//4)*(size[1]+40)
            sheet.paste(panel,(x,y+40)); draw.text((x+3,y+3),label,fill='black')
        sheet.save(output/(artifact.stem+'.png'))
    diagnostics=[]
    for path in run.glob('heatmap_diagnostics_*.json'): diagnostics.extend(json.loads(path.read_text()))
    member=[x for row in diagnostics for x in row.get('member_responses',[])]
    nonmember=[x for row in diagnostics for x in row.get('nonmember_responses',[])]
    if member and nonmember:
        histogram(output/'response_histogram.svg',[('member',member),('nonmember',nonmember)],'Predicted mask ROI response')
    attention=run/'attention_probe.json'
    if attention.exists():
        samples=json.loads(attention.read_text())['samples']
        series=[(key,[value for row in samples for value in row[key]]) for key in ('member_response','nonmember_response')]
        histogram(output/'attention_probe_histogram.svg',series,'Native spatial softmax attention ROI response (not occupancy)')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='outputs/heatmap/place')
    parser.add_argument('--paired-root')
    parser.add_argument('--paired-variants',default='off,h1_sup,h1_match,h2_s',help='required completed variants, e.g. off,h2_s for an overfit control pair')
    parser.add_argument('--destination',default='results/heatmap',help='separate summaries preserve previous stage pairing evidence')
    parser.add_argument('--visualize',action='store_true')
    cli=parser.parse_args()
    root=ROOT/cli.root
    records=[]; commands=[]
    for path in sorted(root.rglob('run_record.json')):
        record=json.loads(path.read_text()); records.append(record)
        command_path=path.parent/'command.txt'
        if command_path.exists(): commands.append((record['run_id'],command_path.read_text().strip()))
        if cli.visualize: visualize(path.parent)
    destination=ROOT/cli.destination; destination.mkdir(parents=True,exist_ok=True)
    fields=['run_id','stage','variant','status','effective_batch','epochs_or_steps','group_map_1_0','group_map_0_5',
            'outlier_miou','wall_time','peak_memory','failure_reason']
    with open(destination/'comparison.csv','w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(records)
    (destination/'commands.md').write_text('# Actual executed commands\n\n'+'\n\n'.join(
        f'## {name}\n\n```bash\n{command}\n```' for name,command in commands)+'\n')
    if cli.paired_root:
        paired=ROOT/cli.paired_root
        by_variant={}
        for path in paired.glob('*/run_record.json'):
            record=json.loads(path.read_text())
            if record['status']=='completed': by_variant[record['variant']]=(record,path.parent)
        base=by_variant.get('off')
        if not base: raise RuntimeError('completed paired B0 missing')
        checks={}
        for name,(record,path) in by_variant.items():
            cache=path/'fixed_input_cache.json'; base_cache=base[1]/'fixed_input_cache.json'
            step_rows=[json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
            base_steps=[json.loads(line) for line in (base[1]/'steps.jsonl').read_text().splitlines()]
            lr_available=all('lr' in row for row in step_rows+base_steps)
            checks[name]=dict(common_initialization=record['init_hash']==base[0]['init_hash'],
                same_cpu_cuda_rng=record.get('initial_rng_hash') is not None and record['initial_rng_hash']==base[0].get('initial_rng_hash'),
                same_source_snapshot=record['source_manifest_hash']==base[0]['source_manifest_hash'],
                exact_sample_and_frame_order=sha(path/'sample_order.json')==sha(base[1]/'sample_order.json'),
                same_dataset=record['data_manifest_hash']==base[0]['data_manifest_hash'],
                same_evaluator=record['evaluator_hash']==base[0]['evaluator_hash'],
                same_steps=record['epochs_or_steps']==base[0]['epochs_or_steps'],
                same_batch=record['effective_batch']==base[0]['effective_batch'],
                input_content_check_available=cache.exists() and base_cache.exists(),
                same_cached_input_content=sha(cache)==sha(base_cache) if cache.exists() and base_cache.exists() else cache.exists()==base_cache.exists(),
                lr_trace_check_available=lr_available,
                same_lr_trace=[row.get('lr') for row in step_rows]==[row.get('lr') for row in base_steps],
                delta_group_map_1_0=record['group_map_1_0']-base[0]['group_map_1_0'],
                delta_group_map_0_5=record['group_map_0_5']-base[0]['group_map_0_5'])
        write_json(destination/'paired_checks.json',checks)
        if set(checks)!=set(cli.paired_variants.split(',')):
            raise AssertionError('required completed variants missing or unexpected: '+cli.paired_variants)
        if not all(all(row[key] for key in ('common_initialization','same_cpu_cuda_rng','same_source_snapshot','exact_sample_and_frame_order',
            'same_dataset','same_evaluator','same_steps','same_batch','same_cached_input_content','same_lr_trace')) for row in checks.values()):
            raise AssertionError('paired controls failed; see paired_checks.json')
    print('wrote',destination/'comparison.csv',flush=True)


if __name__=='__main__': main()
