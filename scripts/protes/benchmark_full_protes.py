"""Actual dense full assignment indicators; unchanged upstream samplers."""
import argparse,gc,importlib.util,json,math,time
import assignment_scaling as exp
np,jax=exp.np,exp.jax
spec=importlib.util.spec_from_file_location('general',exp.ROOT/'PROTES/protes/protes_general.py')
general=importlib.util.module_from_spec(spec);spec.loader.exec_module(general)

def states(n,k):
    if k==n*n:return [(1<<n)-1]
    r,s=divmod(k,n); left=(1<<s)-1;right=((1<<n)-1)^left
    return [m for m in range(1<<n) if
            (m.bit_count()==r and (right & ~m)) or
            (m.bit_count()==r+1 and (left & m))]

def dense_indicator(n,variant):
    ss=[states(n,k) for k in range(n*n+1)]
    assert [len(s) for s in ss]==[exp.exact_rank(n,k) for k in range(n*n+1)]
    R=max(map(len,ss));cores=[];counts=np.ones(1,dtype=np.int64)
    for k in range(n*n):
        r,s=divmod(k,n);lookup={m:i for i,m in enumerate(ss[k+1])}
        shape=(len(ss[k]),2,len(ss[k+1])) if variant=='general' else (1 if k==0 else R,2,1 if k==n*n-1 else R)
        core=np.zeros(shape,dtype=np.float64);nextcounts=np.zeros(len(ss[k+1]),dtype=np.int64)
        for i,m in enumerate(ss[k]):
            for x in (0,1):
                if x and (m.bit_count()!=r or m & (1<<s)):continue
                dest=m | (1<<s) if x else m
                if dest in lookup:
                    j=lookup[dest];core[i,x,j]=1;nextcounts[j]+=counts[i]
        cores.append(core);counts=nextcounts
    assert counts.tolist()==[math.factorial(n)]
    if variant=='fast':return [jax.numpy.asarray(cores[0]),jax.numpy.asarray(np.stack(cores[1:-1])),jax.numpy.asarray(cores[-1])]
    return [jax.numpy.asarray(c) for c in cores]

def run(n,variant):
    path=exp.OUT/f'full_protes_{variant}_{n}.json'
    if path.exists():return
    t=time.perf_counter();P=dense_indicator(n,variant);jax.effects_barrier();build=time.perf_counter()-t
    if variant=='fast':
        module=exp.baseline._module
        # Same interface recomputation per batch as the public PROTES loop.
        interfaces=jax.jit(module._interface_matrices)
        sample=jax.jit(jax.vmap(module._sample,(None,None,None,None,0)))
        def batch(key):
            return sample(*P,interfaces(P[1],P[2]),jax.random.split(key,100))
    else:
        sample=jax.jit(jax.vmap(general._sample,(None,0)))
        def batch(key):return sample(P,jax.random.split(key,100))
    t=time.perf_counter();batch(jax.random.PRNGKey(0)).block_until_ready();warm=time.perf_counter()-t
    initial=np.fromfile(exp.OUT/'paired_inputs'/f'linear_{n}_0.i32',dtype=np.int32).reshape(400,n*n)
    initial_set={x.tobytes() for x in initial};trials=[]
    for rep in range(3):
        key=jax.random.PRNGKey(20261008+n*1000+rep);elapsed=0;generated=0;seen=set()
        while elapsed<1:
            t=time.perf_counter();key,subkey=jax.random.split(key);x=np.asarray(batch(subkey));elapsed+=time.perf_counter()-t
            x=x.astype(np.int32,copy=False) # canonical byte representation for novelty bookkeeping
            assert exp.baseline.feasible(x,n).all()
            seen.update(row.tobytes() for row in x);generated+=len(x)
        assert len(seen-initial_set)<=math.factorial(n)-len(initial_set)
        row=dict(n=n,repetition=rep,batch=100,samples=generated,seconds=elapsed,
                 samples_per_second=generated/elapsed,unique=len(seen),novel_unique=len(seen-initial_set),feasible=generated)
        trials.append(row);print(f'Full {variant} n={n} rep={rep}: {generated/elapsed:.1f}/s, {len(seen)} unique',flush=True)
    exp.baseline.atomic_json(path,dict(variant=variant,n=n,build_seconds=build,warmup_seconds=warm,
        numeric_bytes=sum(p.size*p.dtype.itemsize for p in P),cpu_affinity=exp.baseline.PROCESS.cpu_affinity(),
        upstream=exp.baseline.UPSTREAM,trials=trials))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,required=True);p.add_argument('--variant',choices=['fast','general'],required=True)
    a=p.parse_args();run(a.n,a.variant)
