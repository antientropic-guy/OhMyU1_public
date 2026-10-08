"""Cross-check saved measurements against independent counting identities."""
import json,math
from pathlib import Path
root=Path(__file__).resolve().parents[2]
s=json.loads((root/'examples/protes_full_comparison_results.json').read_text())
old=json.loads((root/'examples/protes_scaling_results.json').read_text())['results']
lookup={(r['config']['n'],r['config']['objective'],r['config']['repetition']):r for r in old}
def C(n,k):return math.comb(n,k) if 0<=k<=n else 0
def rank(n,k):
    if k in (0,n*n):return 1
    r,t=divmod(k,n);return C(n,r)-C(t,r-n+t)+C(n,r+1)-C(n-t,r+1)
for r in s['full_u1']:
    n=int(r['n'])
    blocks=sum(C(n-1,a)+C(n,a)-C(b+1,a-n+b+1)+C(n,a+1)-C(n-b,a+1) for a in range(n) for b in range(n))
    assert int(r['numeric_bytes'])==8*blocks
    assert float(r['support'])==float(math.factorial(n))
    assert int(r['feasible'])==int(r['samples'])
    assert int(r['novel_unique'])<=math.factorial(n)-lookup.get((n,'linear',0),{'initial_unique':0})['initial_unique']
for record in s['full_protes']:
    n=record['n'];rs=[rank(n,k) for k in range(n*n+1)];R=max(rs)
    expected=8*(2*(n*n-2)*R*R+4*R) if record['variant']=='fast' else 16*sum(a*b for a,b in zip(rs[:-1],rs[1:]))
    assert record['numeric_bytes']==expected and record['cpu_affinity']==[0]
    for r in record['trials']:
        assert r['feasible']==r['samples']
        assert r['novel_unique']<=math.factorial(n)-lookup[n,'linear',0]['initial_unique']
for r in s['data_u1']:
    ref=lookup[int(r['n']),r['objective'],int(r['repetition'])]
    assert all(r[key]==ref[key] for key in ['initial_sha256','objective_sha256'])
    assert math.isclose(float(r['initial_c_min']),ref['initial_c_min'],rel_tol=1e-12,abs_tol=1e-9)
    assert float(r['c_min'])<=float(r['initial_c_min'])+1e-9
    assert int(r['novel_unique'])<=ref['unseen_feasible']
    assert r['initial_training_complete']=='true' and r['generated']==r['feasible']
print('Verified exact core storage, factorial support, feasibility, novelty bounds, input hashes and initial objective values.')
