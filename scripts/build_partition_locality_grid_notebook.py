#!/usr/bin/env python3
"""Build the English benchmark report from Julia-generated verified tables."""
import base64
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'examples/partition_locality_grid'
rows=list(csv.DictReader((OUT/'rank_groups.csv').open()))
pooled=list(csv.DictReader((OUT/'rank_pooled.csv').open()))
cells=[]
def md(s):
    cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s):
    cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]})
md(r'''# Set partitioning: locality grid and training audit

This report describes a **new structural benchmark**, not completed optimization results. All 945 inputs were generated and validated locally. Existing optimization results are preserved. Their training implementation contained an indexing error, so they cannot establish the performance of the intended gradient-based method; rerun both arms with the corrected implementation before drawing conclusions.

## Gradient audit

For real amplitudes $\psi(x)$, $Z=\sum_x\psi(x)^2$, nonnegative target weights $p_t$, and $s=\sum_t p_t$, the normalized cross entropy is

$$L=s\log Z-2\sum_t p_t\log|\psi(x_t)|.$$

At a bond in mixed canonical form, merge its two cores into charge blocks $M_q$. The external environments are isometric, so $Z=\sum_q\|M_q\|_F^2$. With one-dimensional charge sectors, every sample follows one charge path. For its selected merged entry $(a_t,b_t)$, the external amplitude factors cancel in $\partial\log|\psi(x_t)|$:

$$\frac{\partial L}{\partial M_q[a,b]}=\frac{2sM_q[a,b]}{Z}-2\sum_{t:(q_t,a_t,b_t)=(q,a,b)}\frac{p_t}{M_q[a,b]}.$$

The denominator is the product of the two **old** scalar core entries. The code's analytical formula was correct for normalized weights. Its indexing was not: `SiteIndex.InvXindex[x][2]` is the physical degeneracy index **within a physical charge sector**. For nonzero columns of A it is always 1. Using this value to select `XboundsDict` therefore accumulated every likelihood term at the first row/column of a merged charge block, even when the sample required another entry. The fixed code caches the physical-value positions during core reshaping, including missing sectors and zero columns. Both sweep directions use this direct lookup, without searching physical-charge vectors for each training example.

On the original locality data (instance 1), 14,982 of 112,000 sample/bond likelihood terms at w=5 (13.4%) and 15,316 at w=18 (13.7%) selected a different entry from the old code. These are counts on the initial model, not weighted gradient magnitudes or estimates for every training iteration. Thus the defect is present on the actual benchmark, not only on toy examples.

Two other corrections: zero-column branches previously referenced undefined `left_shape`/`right_shape`; the shared lookup handles these cases too. `Loss` now includes $s\log Z$ and ignores zero-weight observations, making it valid for canonical MPS that are not normalized. The gradient normalization term also includes $s$ (unchanged for normalized weights).

Validation in `scripts/check_training_gradient.jl` exhaustively enumerates small feasible spaces, perturbs individual merged entries and computes the normalized loss from all amplitudes. It checks both sweep directions, nonuniform random cores, multiple constraints and zero columns; it also checks normalization identities, loss decrease for a small step, exact serial-sampling conditionals against the Born probabilities, RNG replay, and feasibility after training. Run the script for the current numerical error report. The corrected audit passed 583 assertions, with maximum finite-difference discrepancy below $3\times10^{-10}$; the old indexing gives discrepancies of order one.

`src/mps_core.jl`: collection/splitting and the canonical norm identities passed these tests; no change to that file was needed. After each gradient step, the SVD retains **one singular value per charge sector**, moves the singular value to the next center, and thus restores the canonical form. `MaxR` and `CutOff` do not tune this nondegenerate routine. This is a projected gradient step, not an exact two-site minimization; a fixed learning rate of 0.05 does not guarantee monotonic loss decrease after every step. The audit covers the real, sector-dimension-one path used by these experiments, not generic degenerate or complex MPS.

The optimizer never differentiates $f(x)$ itself: objective values determine normalized Boltzmann training weights. Consequently, replacing the nonlinear objective by a linear one isolates a change in optimization landscape and weights, not a different MPS differentiation formula. All previous experiments using the faulty training path, including assignment, need reassessment. Exact structural ranks, indicator storage estimates, and sampling benchmarks without training are not invalidated by this training bug.
''')
md(r'''## Dataset and uniform generator

Seven spans $w\in\{5,6,8,10,12,15,18\}$, three numbers of **equalities** $m\in\{18,19,20\}$, and three size parameters $n\in\{13,14,15\}$ define 63 groups. Each has 15 independently seeded objective/sample instances: **945 tasks, 135 per span**. There are $N=n^2$ binary variables (169, 196, 225), not 13–15 binary variables.

Each column of $A\in\{0,1\}^{m\times N}$ is a distinct tile containing 1–4 ordered elements, with $\max(t)-\min(t)+1\le w$, and $b=\mathbf 1_m$. All $m$ singleton tiles and all adjacent pairs are mandatory. The singleton columns contain an identity matrix, so all $m$ equality rows are linearly independent. The remaining $N-m$ tile counts are apportioned to sizes 2, 3, 4 by largest remainders in the original ratio 62:88:57. Optional tiles are drawn uniformly without replacement within each size. Columns are sorted by minimum element and lexicographic tile order, putting the singleton last within each minimum-element group. No column permutation is applied. Some small-span groups have identical constraint matrices because their catalogue is exhausted; their objectives and samples remain independently seeded.

For the exact sampler, let $C(S)$ count completions of the covered-element set $S$. Choose the first uncovered element $u$ and consider tiles $t$ containing $u$ and disjoint from $S$:

$$C(S)=\sum_{t\ni u,\ t\cap S=\varnothing}C(S\cup t),\qquad C(\{1,\ldots,m\})=1.$$

Choose a tile with probability $C(S\cup t)/C(S)$. Each exact cover has a unique path through this recursion and probability $1/C(\varnothing)$ by telescoping. Integer rejection sampling and BigInt counts avoid floating-point bias. Sampling is with replacement. Every input is checked against an independent column-frontier path counter, its initial 500 draws are replayed exactly, and another independent batch of 500 draws is checked for $Ax=b$. The small-case validation additionally enumerates every cover, verifies its probability using exact rational arithmetic, and compares the rank profile against numerical ranks of explicit indicator unfoldings.

Singleton-last ordering allows every retained prefix to finish using future singletons. Distinct prefix charges have disjoint suffix supports; each charge sector has rank one. Therefore the number of extendable frontier states equals the exact full-indicator unfolding rank. The sliding frontier retires covered elements after their singleton column. These ranks describe **full support**, not the model learned from the initial data. More solutions do not by themselves imply larger unfolding ranks.
''')
md(r'''## Experimental protocol and objectives

The same runner handles the old 280 inputs and the new 945 inputs. `--linear` or `--entropic` is mandatory. Both choices reuse exactly the same constraints, initial sample, linear coefficients and seeds. For each saved instance:

$$f_{\rm linear}(x)=\frac{c^Tx}{\sqrt m},\qquad f_{\rm entropic}(x)=\frac{c^Tx}{\sqrt m}+\tau\log\left[\frac1{32}\sum_{s=1}^{32}\exp\left(\frac{W_s^Tx}{\tau\sqrt m}\right)\right],\quad\tau=0.5.$$

The coefficients $c_j$ are uniform on $[-1,1)$; scenario entries are independent standard normals. Stable log-mean-exp evaluation is used. The linear objective removes only the risk term; it does not generate a different $c$.

* Initial data: 500 exact uniform draws shared within each pair. EELS removes duplicate columns as in the existing protocol; ADD retains its original handling.
* 15 global iterations, 5,000 MPS draws per iteration, one right-and-left training sweep, learning rate 0.05, one-dimensional charge sectors, elite fraction 0.05, no worst-sample retention.
* EELS uses gamma=1 and **no external draws after initialization**. Its baseline has the same restriction.
* ADD uses **Best Cost + GDF / best nonzero mixed**, filter weight 0.4, with 5,000 external uniform draws on each iteration. Its own BestCost baseline receives the identical external batch, verified by full-batch hashes. The EELS and ADD baselines are separate because their information budgets differ.
* Training weights are normalized exponentials using the standard deviation of current costs as temperature. The zero-variance case uses uniform weights.
* Each pair uses the same initial dataset and seeds, with alternating execution order. Runs are sequential within a job, Julia and BLAS have one thread, and a directory lock prevents concurrent writers. Atomic checkpoints contain source/input hashes and reject incompatible resumes.
* The incumbent is the best among **all evaluated candidates**, including the final external batch. Curves, feasible incumbents, paired initial hashes and paired external hashes are checked before saving.

Per objective: 1,120 solver runs on the old 280 tasks, or 3,780 on the new 945 tasks (two distinct baseline/strategy pairs per task). Optimization on the full benchmark is left to the cluster; only smoke checks run locally.

### Launch from the repository root

```bash
# Original 18-equality, n=15 dataset:
bash scripts/bash/run_PartitionLocality.sh --linear
bash scripts/bash/run_PartitionLocality.sh --entropic

# New 63-group grid:
bash scripts/bash/run_PartitionLocality.sh --grid --linear
bash scripts/bash/run_PartitionLocality.sh --grid --entropic
```

Use `bash` to invoke the self-submitting wrapper. It selects 7 Slurm jobs for the old dataset or 63 for the grid, at most four running concurrently, each with one CPU and 8 GB. The existing partition setting is retained. Copy the generated `data/set_partitioning/locality_grid_v1/instances/` directory to the cluster first. Missing inputs cause an error, never silent regeneration. A Julia environment with the project dependencies installed is required.

Results go to `data/set_partitioning/{locality_v1,locality_grid_v1}/results_gradient_v2/{linear,entropic}/`. Old `results/` files are preserved. Each grid result filename identifies span, equality count and n; dictionaries retain the usual nesting `res_dict["best_cost"][n][i]`, `res_dict["EELS"][n][i]`, or `res_dict["best nonzero mixed"][0.4][n][i]`. Separate files avoid collisions between groups with the same n and i. `diagnostics` includes timings, hashes and structural ranks. Do not pool old faulty-training results with the new version.
''')
md('''## Rank versus span

For each instance, maximum and mean ranks are computed over the **N−1 internal bonds**, excluding the two unit boundary ranks. Error bars show the **sample standard deviation (ddof=1) across all 135 instances at that span**. The vertical axes are logarithmic; error bars represent arithmetic mean plus/minus SD on the original rank scale. They include differences between m/n groups and within groups; they are not confidence intervals or standard errors. Every parameter combination has equal weight because all contain 15 tasks. Optimization is not used to obtain these ranks.
''')
image=base64.b64encode((OUT/'rank_vs_span.png').read_bytes()).decode()
cells.append({'cell_type':'markdown','metadata':{},'source':['![Exact indicator rank versus interaction span](attachment:rank_vs_span.png)'], 'attachments':{'rank_vs_span.png':{'image/png':image}}})
def table(data,group=True):
    s='| w | m | n | Tasks | Mean max rank ± SD | Mean mean rank ± SD |\n|---:|---:|---:|---:|---:|---:|\n' if group else '| w | Tasks | Mean max rank ± SD | Mean mean rank ± SD |\n|---:|---:|---:|---:|\n'
    for r in data:
        a=f"{float(r['mean_max']):.1f} ± {float(r['sd_max']):.1f}"
        b=f"{float(r['mean_mean']):.1f} ± {float(r['sd_mean']):.1f}"
        prefix=f"| {r['w']} | {r['m']} | {r['n']} | {r['count']} |" if group else f"| {r['w']} | {r['count']} |"
        s+=f'{prefix} {a} | {b} |\n'
    return s
md('### Pooled summaries (135 tasks per span)\n\n'+table(pooled,False))
md('### Full parameter table (15 tasks per row)\n\nSD is across the 15 instances of that group, not across its bonds.\n\n'+table(rows))
md('''## Reproduce

From the project root, run `julia --project=. --threads=1 scripts/check_training_gradient.jl` for the derivative audit and `julia --project=. --threads=1 scripts/check_PartitionLocality.jl` for the exact-generator, legacy-input and paired smoke checks. Generate or revalidate the new inputs with `julia --project=. --threads=1 scripts/generate_PartitionLocalityGrid.jl`.

The full tables are in `examples/partition_locality_grid/rank_groups.csv` and `rank_pooled.csv`, with unrounded values. The manifest and per-instance table are under `data/set_partitioning/locality_grid_v1/`. The PNG/PDF figures are generated by Julia/Plots. The following cell regenerates the structural report from the saved manifest; it does not run optimization.
''')
code('project_root = isfile("Project.toml") ? pwd() : dirname(pwd())\ninclude(joinpath(project_root, "scripts", "report_PartitionLocalityGrid.jl"))')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Julia','language':'julia','name':'julia-1.12'},'language_info':{'name':'julia'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells):c['id']=f'locality-grid-{i:02d}'
(ROOT/'examples/SetPartitioning_locality_grid.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Saved SetPartitioning_locality_grid.ipynb')
