"""Append reproducible exact-model and matched-data comparisons to the notebook."""
from pathlib import Path
import nbformat as nb
root=Path(__file__).resolve().parents[2]
path=root/'examples/PROTES_assignment_scaling.ipynb'
book=nb.read(path,as_version=4)
book.cells=[c for c in book.cells if not c.metadata.get('full_comparison_extension')]
for c in book.cells:
    if c.cell_type=='markdown' and 'actual U(1) object size' in c.source:
        c.source=c.source.replace(r'n\le12',r'n\le16')
def md(s):
    c=nb.v4.new_markdown_cell(s);c.metadata['full_comparison_extension']=True;book.cells.append(c)
def code(s):
    c=nb.v4.new_code_cell(s);c.metadata['full_comparison_extension']=True;book.cells.append(c)
md(r'''## 9. Constructed full models and sampling speed

**Clarification of the earlier experiment:** the previous large PROTES memory figures were analytical estimates of dense cores, not actual allocations. U(1) objects had been constructed up to $n=12$. Full U(1)-MPS have now been built up to $n=16$, and both dense PROTES implementations for $n=4,\ldots,10$. Larger PROTES models were not allocated on this laptop: at $n=16$, even variable-rank storage requires hundreds of GiB. Missing points are not replaced with invented speed measurements.

Both models represent **all** permutations in the same row-major binary order. Dynamic programming retains the set of occupied columns at each cut. Allowed transitions receive ones and other transitions zeros. Exact ranks and the path count $n!$ are verified.

For U(1), a backward pass counts completions $C_k(q)$. Each scalar block is replaced by $\sqrt{C_{k+1}(q')/C_k(q)}$. The product along any path is $1/\sqrt{n!}$, giving the exact right-canonical form of the uniform distribution. The project's unmodified `sample_nondeg!` is used. Preparing this form is excluded from sampling time and recorded separately.

PROTES indicator cores are actually allocated as dense arrays: padded to a common rank for fast `protes`, and using minimum variable ranks for `protes_general`. Their unmodified `_sample`, `_interface_matrices`, and JAX vectorization are used. This is the uniform distribution on the same $n!$ points. Training is disabled: the benchmark measures sampling from an exactly specified model.

**Timing protocol:** the same logical CPU 0; float64; batches of 100; three seeds; at least one second of cumulative generator-call time per seed. Construction, JIT warm-up, feasibility checks, and unique-point counting are excluded for both methods. PROTES timing includes transferring results to the CPU and recomputing right environments for each batch, as in its standard loop. Actual batch counts and durations are saved; a large batch can exceed one second. The primary metric is therefore samples/second, and unique-point counts are accompanied by actual durations.

"New" points are absent from the same reference dataset of 400 points (`linear`, seed index 0), which is not used to construct the full indicator. At $n=4$, that dataset already covers the entire space, so no new points can exist. At $n=5$, few unseen solutions remain. **Total sampling throughput and new-unique-point throughput are different metrics.**

Object sizes are measured in a separate full `Base.summarysize` traversal. Initial runs at $n=15,16$ showed memory pressure and paging on the 8 GB RAM machine. Speed runs were therefore repeated without the deep memory traversal, with `GC.gc()` before warm-up; initial results are preserved in the snapshot as `full_u1_initial_memory_audit`. The figure shows repeat measurements and their range. These are local measurements of this implementation and machine, not hardware-independent complexity estimates.
''')
code('''full_snapshot=json.loads((ROOT/'examples/protes_full_comparison_results.json').read_text())
u1_full=full_snapshot['full_u1']
dense_full=full_snapshot['full_protes']
full_trials=[dict(r,method='U1') for r in u1_full]
for result in dense_full:
    full_trials.extend(dict(r,method='PROTES '+result['variant']) for r in result['trials'])
methods=['U1','PROTES fast','PROTES general']
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for j,method in enumerate(methods):
    sizes=sorted({int(r['n']) for r in full_trials if r['method']==method})
    med=[];lo=[];hi=[];new=[]
    for n in sizes:
        g=[r for r in full_trials if r['method']==method and int(r['n'])==n]
        rates=[float(r['samples_per_second']) for r in g]
        med.append(np.median(rates));lo.append(min(rates));hi.append(max(rates))
        new.append(np.median([int(r['novel_unique'])/float(r['seconds']) for r in g]))
    axes[0].semilogy(sizes,med,'o-',label=method)
    axes[0].fill_between(sizes,lo,hi,alpha=.15)
    axes[1].plot(sizes,new,'o-',label=method)
axes[0].set(ylabel='Samples / second',title='Full uniform indicator: median and trial range')
axes[1].set(ylabel='New distinct feasible points / second',yscale='symlog',title='Novelty relative to the same 400 reference points')
for ax in axes:
    ax.set(xlabel='Assignment size n',xticks=range(4,17,2));ax.grid(alpha=.25);ax.legend()
fig.tight_layout();savefig(fig,'assignment_full_sampling_speed');plt.show()
table(['n','Sampler','Median samples/s','Feasible / generated','Unique per trial','New per trial','Measured seconds per trial'],
      [[n,method,f'{np.median([float(r["samples_per_second"]) for r in g]):.1f}',
        f'{sum(int(r["feasible"]) for r in g)} / {sum(int(r["samples"]) for r in g)}',
        str([int(r['unique']) for r in g]),str([int(r['novel_unique']) for r in g]),
        str([round(float(r['seconds']),3) for r in g])]
       for n in range(4,17) for method in methods
       if (g:=[r for r in full_trials if int(r['n'])==n and r['method']==method])])
table(['n','U1 / PROTES fast throughput','U1 / PROTES general throughput'],
      [[n,*[f'{np.median([float(r["samples_per_second"]) for r in full_trials if int(r["n"])==n and r["method"]=="U1"])/np.median([float(r["samples_per_second"]) for r in full_trials if int(r["n"])==n and r["method"]==method]):.2f}'
             for method in ['PROTES fast','PROTES general']]] for n in range(4,11)])
''')
md(r'''## 10. U(1)-EELS and PROTES with identical initial data and time budgets

U(1)-EELS is now constructed **only from the same 400 observations**, not the full feasible set. Observations and objective functions are exported from NumPy without regeneration by the Julia RNG; SHA-256 fingerprints of the original bytes are verified in both languages. A fundamental difference in structural access remains: U(1) uses $A,b$, while PROTES receives only data and an external evaluator. Thus this compares the specified settings, rather than isolating architecture effects under identical information.

EELS parameters: $\gamma=1$, sector degeneracy 1, one bidirectional training sweep, learning rate 0.05, up to 400 best points retained for the next iteration, and 10000 draws per iteration. No external feasible points are added after initialization. Training uses the same Boltzmann weights and updates as `run_EELS_optimization.jl`. To compare **time**, the number of outer iterations is limited by a deadline rather than the previous fixed value of 20; actual iteration counts are recorded. Sampling is divided into batches of 100 for deadline checks. A batch completed after the deadline is rejected, as in PROTES. Construction, training, objective evaluation, and novelty accounting are included in the budget. The deadline is cooperative: an ongoing construction/training operation can exceed it; actual durations are reported below.

One complete iteration at the same size is used for warm-up before measurement; neither the warm-up model nor its data is carried into the measured run. All runs are sequential, on one logical CPU, without competing benchmarks running concurrently. The scan uses $n=4,\ldots,10$, one seed, and 5 seconds. Long runs use $n=8$, three seeds, 50 seconds, and both objectives. These are the same diagnostic instances, not new held-out test files from the article.
''')
code('''u1_data=full_snapshot['data_u1']
for r in u1_data:
    ref=base[int(r['n']),r['objective'],int(r['repetition'])]
    assert r['initial_sha256']==ref['initial_sha256'] and r['objective_sha256']==ref['objective_sha256']
    assert int(r['generated'])==int(r['feasible'])
    assert int(r['novel_unique'])<=ref['unseen_feasible']
u1_long=[r for r in u1_data if float(r['budget'])==50]
comparison_names=names+['U1 EELS']
fig,axes=plt.subplots(1,2,figsize=(13,4.7),sharex=True,sharey=True)
for ax,obj in zip(axes,['linear','quadratic']):
    for j,name in enumerate(comparison_names):
        g=([r for r in u1_long if r['objective']==obj] if name=='U1 EELS' else
           [r for r in long_runs if config_name(r)==name and r['config']['objective']==obj])
        vals=[int(r['novel_unique']) for r in g]
        ax.scatter(j+np.linspace(-.13,.13,len(vals)),vals,color=colors[j],s=40)
        ax.plot([j-.23,j+.23],[np.mean(vals)]*2,color='black',lw=2,label='Group average' if j==0 else None)
        ax.plot(j,np.median(vals),'D',color='crimson',ms=5,label='Group median' if j==0 else None)
    ax.set(title=obj.capitalize(),xticks=range(len(comparison_names)),xticklabels=comparison_names,yscale='symlog')
    ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.25);ax.legend()
axes[0].set_ylabel('New distinct feasible points absent from initial data')
fig.suptitle('Same 400 input points and objectives, n=8, 50 seconds, one CPU')
fig.tight_layout();savefig(fig,'protes_u1_parameter_comparison_50s');plt.show()
fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True,sharey=True)
for ax,obj in zip(axes,['linear','quadratic']):
    for j,name in enumerate(comparison_names):
        if name=='U1 EELS':
            g=sorted([r for r in u1_data if r['objective']==obj and float(r['budget'])==5],key=lambda r:int(r['n']))
            xs=[int(r['n']) for r in g]
        else:
            g=sorted([r for r in short_runs if config_name(r)==name and r['config']['objective']==obj],key=lambda r:r['config']['n'])
            xs=[r['config']['n'] for r in g]
        ax.plot(xs,[int(r['novel_unique']) for r in g],'o-',label=name,color=colors[j])
    ax.set(title=obj.capitalize(),xlabel='Assignment size n',yscale='symlog',xticks=range(4,11));ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[0].set_ylabel('New distinct feasible points (one matched trial)')
fig.suptitle('Same initial data and objectives, 5 seconds, one CPU')
fig.tight_layout();savefig(fig,'protes_u1_parameter_comparison_5s');plt.show()
table(['n','Objective','Seed index','Budget','Actual seconds','Generated','New unique','Iterations started/completed','Initial fit complete','Initial best','Final best'],
      [[r['n'],r['objective'],r['repetition'],r['budget'],f'{float(r["elapsed_seconds"]):.3f}',r['generated'],r['novel_unique'],
        f'{r["iterations"]}/{r["completed_iterations"]}',r['initial_training_complete'],f'{float(r["initial_c_min"]):.3f}',f'{float(r["c_min"]):.3f}'] for r in u1_data])
''')
md('''## 11. Limits of interpretation

The full-model comparison measures storage efficiency and the specific native samplers under the same uniform distribution. It does not measure learned-model quality or time to reach a good objective value. Compact storage alone does not guarantee faster sampling: dictionaries, memory allocation, JIT, and vectorization affect performance.

The sample-based comparison separately measures new feasible points under a shared computational budget. U(1)-EELS uses constraints in its representation, while PROTES does not receive them in this protocol. In small spaces, novelty is limited by the remaining unseen solutions; in large spaces, novelty does not imply good cost. Best objective values and actual durations are therefore also retained.
''')
nb.write(book,path)
print('Updated full-model and matched-data comparison sections.')
