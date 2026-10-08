"""Export recorded results only; all article rendering is performed in Julia."""
from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/protes_comparison/article_plots';OUT.mkdir(parents=True,exist_ok=True)
def save(name,header,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
full=json.loads((ROOT/'examples/protes_full_comparison_results.json').read_text())
save('speed.csv',['method','n','rep','rate'],
     [['U1',r['n'],r['repetition'],r['samples_per_second']] for r in full['full_u1']]+
     [[v['variant'],r['n'],r['repetition'],r['samples_per_second']] for v in full['full_protes'] for r in v['trials']])
old=json.loads((ROOT/'examples/protes_scaling_results.json').read_text())['results']
sens=json.loads((ROOT/'examples/protes_sensitivity_results.json').read_text())['results']
rows=[]
for r in old+sens:
    c=r['config'];phase=c['phase']
    if phase=='pilot' and c['repetition']!=0:continue
    if phase=='long' and c['n']!=8:continue
    name=phase.split('_')[1] if phase.startswith('sensitivity_') else 'default'
    rows.append([name,c['objective'],c['n'],c['repetition'],c['seconds'],r['novel_unique']])
rows += [['U1 EELS',r['objective'],r['n'],r['repetition'],r['budget'],r['novel_unique']] for r in full['data_u1']]
save('novelty.csv',['method','objective','n','rep','budget','novel'],rows)
with (ROOT/'examples/protes_scaling_memory.csv').open() as f: mem=list(csv.DictReader(f))
save('memory.csv',['n','bytes'],[[r['n'],r['julia_summarysize_bytes']] for r in mem if r['mode']=='full'])
print('Exported plot data; no experiments rerun.')
