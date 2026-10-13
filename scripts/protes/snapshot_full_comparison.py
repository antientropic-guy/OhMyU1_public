"""Save compact measured results and extend the existing actual-memory table."""
import csv,hashlib,json,math,subprocess
from datetime import date
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'data/protes_comparison/scaling'
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
u1=rows(out/'full_u1.csv')
assert len(u1)==39 and max(int(r['n']) for r in u1)==16
memory_audit=rows(out/'full_u1_initial.csv')
measured_memory={int(r['n']):r for r in memory_audit}
# Memory is taken by named columns from the actual Base.summarysize audit;
# throughput comes from the separate repeat after releasing temporary objects.
for r in u1:
    audit=measured_memory[int(r['n'])]
    assert r['numeric_bytes']==audit['numeric_bytes']
    r['object_bytes']=audit['object_bytes']
with (out/'full_u1.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(u1[0]));writer.writeheader();writer.writerows(u1)
dense=[json.loads(p.read_text()) for p in sorted(out.glob('full_protes_*.json'))]
assert len(dense)==14
data=rows(out/'data_u1_short.csv')+rows(out/'data_u1_long.csv')
assert len(data)==20
for r in u1:assert r['samples']==r['feasible']
for r in data:assert r['generated']==r['feasible'] and r['initial_training_complete']=='true'
def choose(n,k):return math.comb(n,k) if 0<=k<=n else 0
def rank(n,k):
    if k in [0,n*n]:return 1
    r,s=divmod(k,n)
    return choose(n,r)-choose(s,r-n+s)+choose(n,r+1)-choose(n-s,r+1)
memory=rows(out/'memory.csv' if (out/'memory.csv').exists() else root/'examples/protes_scaling_memory.csv')
memory=[r for r in memory if not (r['mode']=='full' and int(r['n'])>=4)]
for r in u1:
    if int(r['repetition'])!=0:continue
    n=int(r['n'])
    memory.append(dict(n=n,mode='full',total_bond_max=max(rank(n,k) for k in range(n*n+1)),
                       blocks=int(r['numeric_bytes'])//8,numeric_bytes=r['numeric_bytes'],julia_summarysize_bytes=r['object_bytes']))
memory.sort(key=lambda r:(int(r['n']),r['mode']))
for path in [out/'memory.csv',root/'examples/protes_scaling_memory.csv']:
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(memory[0]));writer.writeheader();writer.writerows(memory)
scripts=['assignment_memory.jl','benchmark_full_u1.jl','benchmark_full_protes.py','benchmark_data_u1.jl','export_scaling_inputs.py']
snapshot=dict(hardware=json.loads((out/'hardware.json').read_text()),
              full_u1=u1,full_u1_initial_memory_audit=memory_audit,full_protes=dense,data_u1=data,
              source_sha256={name:hashlib.sha256((root/'scripts/protes'/name).read_bytes()).hexdigest() for name in scripts})
if (out/'data_u1_long_table.csv').exists():
    snapshot['full_u1_table']=rows(out/'full_u1_table.csv')
    snapshot['data_u1_table']=rows(out/'data_u1_short_table.csv')+rows(out/'data_u1_long_table.csv')
    assert len(snapshot['full_u1_table'])==39 and len(snapshot['data_u1_table'])==20
    for r in snapshot['full_u1_table']: assert r['samples']==r['feasible']
    for r in snapshot['data_u1_table']: assert r['generated']==r['feasible']
    snapshot['backend_comparison_note']='Full table timings are the archived October 12 repeat (after releasing the memory audit); table EELS runs were measured separately with identical inputs and budgets. Legacy measurements are retained. Sampler code is unchanged.'
snapshot['u1_refresh'] = dict(
    date=date.today().isoformat(),
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    library_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((root/'src').glob('*.jl'))},
    protes_results_reused=True,
    sampling_backend='legacy',
    additional_sampling_backends=['table'] if snapshot.get('data_u1_table') else [],
    memory_audit_sampling_backend=json.loads((out/'full_u1_initial.metadata.json').read_text())['sampling_backend'] if (out/'full_u1_initial.metadata.json').exists() else 'unrecorded',
    memory_audit_note='Only object sizes are taken from the separate audit; its sampling timings are not used in figures or speed comparisons.',
    protocol='Unchanged inputs, seeds, CPU affinity, sample batches and time budgets; U1 memory remeasured; legacy and any additional backend records are stored separately.')
(root/'examples/protes_full_comparison_results.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print('Saved full models, all 20 matched U1 trials and memory through n=16.')
