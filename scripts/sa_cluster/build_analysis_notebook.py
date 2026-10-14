"""Build and optionally execute the cluster SA report; never run optimizers.

Prerequisites: collect.jl and analyze.jl. Python dependencies: numpy, nbformat,
nbclient, ipykernel. Tables use HTML, avoiding LaTeX tabular renderer failures.
"""
from pathlib import Path
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
import sys
import tempfile

import nbformat as nbf
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "examples" / "sa_cluster_analysis"
NOTEBOOK = ROOT / "examples" / "SA_cluster_analysis.ipynb"


def load(name):
    with (OUT / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(execute=False):
    pairs = load("paired_comparisons.csv")
    summaries = load("group_summary.csv")
    with (ROOT / "data/sa_cluster_v2/results/summary.csv").open() as f:
        raw = list(csv.DictReader(f))
    indexed = {(r["family"], r["objective"], int(r["n"]), int(r["budget"]), r["method"], int(r["instance"])): r for r in raw}
    histories = defaultdict(list)
    with (ROOT / "data/sa_cluster_v2/results/curves.csv").open() as f:
        for r in csv.DictReader(f):
            if int(r["budget"]) == 200:
                key = (r["family"], r["objective"], int(r["n"]), 200, r["method"], int(r["instance"]))
                histories[key].append((int(r["calls"]), float(r["best"])))
    query_rows = []
    q = 100000
    for family in ("assignment", "partition"):
        for objective in ("quadratic", "entropic"):
            for n in (10, 12, 14):
                g = (family, objective, n, 200)
                def at_q(method, i):
                    key = (*g, method, i)
                    h = histories[key]
                    assert int(indexed[key]["calls"]) >= q
                    return h[bisect_right([c for c, _ in h], q) - 1][1]
                sa = np.array([at_q("SA", i) for i in range(1, 41)])
                for method in ("U1_data_EELS", "U1_full"):
                    supported = all(int(indexed[(*g, method, i)]["calls"]) >= q for i in range(1, 41))
                    if supported:
                        values = np.array([at_q(method, i) for i in range(1, 41)])
                        effect = 100 * (sa - values) / np.abs(sa)
                        avg, med = float(effect.mean()), float(np.median(effect))
                        initial = np.array([float(indexed[(*g, method, i)]["initial"]) for i in range(1, 41)])
                        sa_gain = float(np.mean(100 * (initial - sa) / np.abs(initial)))
                        tt_gain = float(np.mean(100 * (initial - values) / np.abs(initial)))
                    else:
                        avg = med = ""
                        sa_gain = tt_gain = ""
                    query_rows.append(dict(family=family, objective=objective, n=n, method=method,
                                           query_budget=q, supported_tasks=40 if supported else 0,
                                           mean_pct=avg, median_pct=med,
                                           sa_mean_gain_pct=sa_gain,tt_mean_gain_pct=tt_gain))
    with (OUT / "query_diagnostic_100k.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(query_rows[0])); w.writeheader(); w.writerows(query_rows)
    rng = np.random.default_rng(20261013)
    for p in pairs:
        key = (p["family"], p["objective"], int(p["n"]), int(p["budget"]))
        differences = np.array([
            100 * (float(indexed[(*key, "SA", i)]["final"]) - float(indexed[(*key, p["method"], i)]["final"]))
            / abs(float(indexed[(*key, "SA", i)]["final"])) for i in range(1, 41)
        ])
        lo, hi = np.quantile(differences[rng.integers(0, 40, (10000, 40))].mean(axis=1), [.025, .975])
        p.update(mean_ci_low=lo, mean_ci_high=hi)
    with (OUT / "paired_comparisons_ci.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)

    positive = [p for p in pairs if float(p["mean_pct"]) > 1e-8]
    significant = [p for p in pairs if float(p["p_holm72"]) < .05]
    ratios = [float(p["calls_ratio"]) for p in pairs]
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md("""# SA and OhMyU1: complete cluster comparison

This report analyzes all **4,320 saved runs**, without rerunning any optimizer.
There are 40 paired instances in each family/size group, two objectives,
three independent time budgets (10, 50, 200 seconds), and three methods.
The 36 experimental groups are reported separately below. Earlier pilot results
and notebooks are retained. Figures are generated with Julia/Plots/StatsPlots.

**This is the original reset/rebuild diagnostic protocol. It is not the later
persistent-amplitude EELS experiment.** SA has feasibility-preserving moves;
there is no feasibility-oracle-only SA in this dataset.
""")
    code('''from pathlib import Path
import csv, html
from IPython.display import display, HTML, Markdown, Image
root = Path.cwd()
if not (root / "Project.toml").exists():
    root = root.parent
out = root / "examples" / "sa_cluster_analysis"
def read_csv(name):
    with (out / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
summary = read_csv("group_summary.csv")
paired = read_csv("paired_comparisons_ci.csv")
with (root / "data/sa_cluster_v2/results/summary.csv").open() as f:
    raw_summary = list(csv.DictReader(f))
names = {"SA": "SA", "U1_data_EELS": "Data + EELS", "U1_full": "Full MPS"}
def f1(x):
    value = float(x)
    return f"{0.0 if abs(value) < 0.05 else value:.1f}"
def sci(x): return f"{float(x):.3g}"
def integer(x): return f"{float(x):,.0f}"
def table(headers, rows):
    display(HTML('<div style="overflow-x:auto"><table><thead><tr>' +
                 ''.join('<th>'+html.escape(str(x))+'</th>' for x in headers) +
                 '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows) +
                 '</tbody></table></div>'))
def select(rows, family, objective, n, budget=None):
    return [r for r in rows if r['family']==family and r['objective']==objective and int(r['n'])==n and (budget is None or int(r['budget'])==budget)]
def image(name): display(Image(filename=str(out / name)))
display(Markdown((out / "audit.md").read_text(encoding="utf-8")))''')
    md("""## Protocol and interpretation of axes

| Parameter | Saved cluster setting |
|---|---|
| Sizes | n = 10, 12, 14; n squared binary variables |
| Sample size | 40 distinct tasks per family and size, one optimization seed per task |
| Initial data | The same 400 uniform feasible draws, with replacement, for all methods and budgets |
| SA | Optim.jl SimulatedAnnealing; 10 timed chains, starting at the ten best distinct initial points |
| SA moves | Assignment: transposition; Set Partitioning: symmetric patch-cover replacement, with possible self-loops |
| SA temperature | Geometric decay from std(initial costs) to 0.001 of that scale within each chain |
| Data + EELS | Construct from the current elite data and expand with gamma = 1 on every iteration |
| Full MPS | Construct the exact full indicator once inside the timed run; copy its uniform amplitudes on every iteration |
| MPS training | One sweep per iteration, learning rate 0.05, nondegenerate charge blocks |
| MPS temperature | std(current elite costs); normalized Boltzmann weights; uniform weights if spread is zero |
| Sampling | Legacy backend; up to 10,000 model samples per iteration, batches of 100 for deadline checks |
| Elite data | Best 400 distinct points from old elites plus new samples; no external injection |
| Time limits | Independent 10, 50, 200 second runs; number of iterations is determined by time |
| CPU | One Julia thread and one BLAS thread per process, with 32 processes per exclusive Slurm allocation |

Assignment retains its row-major order. Set Partitioning retains the fixed spatial
low-rank order. No variable order is optimized in this analysis. Full structural
ranks are 461, 1715, 6434 for assignment; mean maximum ranks are 27.7, 36.1, 45.4
for Set Partitioning. The nonpolynomial objective is the frozen 32-scenario
entropic objective; quadratic cost is x'Qx. Both families use matching objective
arrays for each (n, instance), but their feasible spaces differ.

Initialization, construction, training, sampling, and objective evaluations are
timed. Compilation, input loading, validation before the run, and the SA move
catalogue are excluded. Full-indicator construction can consume much of the
short budget. Stopping is cooperative; overruns are reported, while improvements
after the requested deadline are excluded. Budgets are independent runs, not
prefixes of a single 200-second trajectory.
Here "independent" means separate starts, not independent random seeds: each
task/method reuses its fixed seed across budgets, and the same tasks recur in
the three budget groups. No statistical independence across these groups is
assumed; Holm correction remains valid under this dependence.

**Learning curves:** compute `100 * (initial_best - incumbent) / abs(initial_best)`
per task, then average equally across the 40 tasks. The exact union of logged
improvement events is used with stepwise interpolation. Bands are **asymmetric
pointwise 95% percentile-bootstrap confidence intervals for the mean** (2,000
resamples of 40 tasks, fixed seed 20261014). The same task-resampling weights
are used across event times and methods. They quantify uncertainty of the mean,
not the spread of individual tasks, and are not simultaneous confidence bands.
Unlike the former mean +/- SD bands, they cannot imply a negative gain when all
observations are nonnegative. The mean curves themselves are unchanged.
Task SD is retained in the exported tables and curves. Before the first event, the common initial best is
used as a reference (not a claim that initial evaluation was instantaneous).

**Evaluation-axis curves:** use measured cumulative call counts at each
improvement. Each method's curve stops at its smallest final count across the
40 tasks, so every point is supported by all 40 tasks. End markers identify
these limits; compare methods only where their query ranges overlap.
This avoids carrying a stopped optimizer forward into an unobserved query
budget. It is a descriptive query-efficiency view of time-limited trajectories,
not a new equal-query experiment: in particular, SA cooling remains time-based.
If a method's supported range contains only the initial evaluations, it appears
as a single point and has no post-initial query comparison to offer.

**Call counts:** the final totals are exact measured objective invocations,
including duplicates, SA self-loops and Optim's repeated evaluations. Initial
SA evaluates all 400 draws; MPS deduplicates first, so its initial count can be
slightly smaller. Final correctness-check evaluations outside the measured
adapter count are excluded. The saved logs do not record a continuous call rate;
we therefore show exact final counts versus independent time budget, rather
than fabricate a calls-versus-time trajectory.

**Paired comparisons:** positive `100 * (SA - MPS) / abs(SA)` favors MPS.
Boxes use 1.5 IQR whiskers and include outliers; diamonds are group averages and
purple marks are group medians, calculated from the same 40 values. Percentages
are displayed to one decimal place; CSV data retain full precision.
""")
    md("""## Statistical procedure

As in the earlier A/B analyses, the paired one-sided Wilcoxon signed-rank test
uses the **raw objective difference** SA minus MPS (alternative: MPS is better).
Raw p-values are accompanied by Holm adjustment over all 72 comparisons
(36 groups times two MPS methods). Two-sided p-values and their separate Holm
adjustment are also exported, allowing detection of either direction.
Only differences below `1e-9 + 1e-11 * max(abs(SA), abs(MPS))` are treated as
numerical ties. This avoids interpreting floating-point summation noise as a win.
The test concerns paired location differences, not specifically the mean.

The tables also give percentile bootstrap intervals for the mean relative
improvement (10,000 paired task resamples, fixed seed). These intervals are
descriptive, **not adjusted for multiple comparisons**. Forty instances are
not forty random seeds on the same task, and no global optimum is certified.
Relative gain over the initial best can exceed 100% when an objective crosses
zero. Small initial magnitudes can produce large between-task SD; raw final
costs are therefore also tabulated. These ratios are not optimality gaps.
""")
    md(f"""## Main findings

At equal nominal time budgets, there are **{len(significant)} / 72**
Holm-significant improvements over SA.
Only **{len(positive)} / 72** group/method comparisons have a positive mean
beyond numerical noise. The ratio of mean SA calls to mean MPS calls ranges
from **{min(ratios):.1f} to {max(ratios):.1f}** across these comparisons.
The largest ratio includes full assignment runs that did not advance beyond
their initial data; it must not be interpreted as useful query efficiency.
No one-sided raw p-value is below 0.05 either.

- **Assignment:** neither MPS variant has a task-level win over SA in these
  runs. Full MPS approaches SA at n = 10 as time increases, but remains worse
  at larger sizes. Data + EELS commonly plateaus early. Poor wall-time results
  thus combine restricted/rebuilt support, training behavior, and computational
  throughput; this experiment does not isolate their causal contributions.
- **Set Partitioning:** low ranks make the full model much more competitive.
  There are individual wins and small positive means on the entropic objective,
  especially at intermediate budgets. These are not robust group-level evidence
  of superiority after multiplicity correction. Longer runs often reach the
  same observed incumbent. Matching SA does not certify a global optimum.
- **Objective calls:** MPS typically spends far fewer calls. SA's larger call
  count is part of its computational advantage on these cheap black-box
  objectives; call counts alone neither prove better solutions nor predict the
  outcome with an expensive oracle. Equal-query plots below help separate these
  issues descriptively, but do not replace an expensive-objective experiment.
- **Scope:** these results characterize an informed SA and the legacy
  reset/rebuild MPS protocol. They do not test persistent training, isolate the
  effect of EELS against a no-EELS MPS, or establish a universal solver ranking.
""")
    code('''table(["Family", "Objective", "n", "s", "Method", "Mean vs SA (%)", "Median (%)", "W/T/L", "Holm p", "SA/MPS calls"],
      [[r['family'], r['objective'], r['n'], r['budget'], names[r['method']], f1(r['mean_pct']), f1(r['median_pct']),
        f"{r['wins']}/{r['ties']}/{r['losses']}", sci(r['p_holm72']), f1(r['calls_ratio'])] for r in paired])''')
    md("""### Longest budget: query volume and stagnation

At 200 seconds all methods have time to perform search. The table below shows
whether further iterations translate into later improvements. Last improvement
time refers to the incumbent record, not a training-loss minimum; if no progress
is made it is the initial evaluation time. SA iterations in this table mean
completed chains, whereas MPS iterations mean completed outer iterations.
""")
    code('''table(["Family", "Objective", "n", "Method", "Mean calls", "Mean final cost", "Iterations / chains", "Mean last improvement (s)"],
      [[r['family'], r['objective'], r['n'], names[r['method']], integer(r['mean_calls']), f"{float(r['mean_final']):.4f}",
        f1(r['mean_iterations']), f1(r['mean_last_improvement_s'])] for r in summary if int(r['budget'])==200])''')
    md("""### Descriptive query-efficiency slice: 100,000 calls

For an interpretable view of the evaluation-axis curves, the following table
reads the incumbent at 100,000 calls from the **200-second trajectories**.
This round query threshold was chosen during analysis, not prespecified as an
experimental budget. All 40 tasks must reach it; full assignment at n = 14 is
therefore marked unavailable, without dropping slower tasks or extrapolating.

Both MPS variants are better on average at this query count on Set Partitioning;
full MPS is also better on assignment at n = 10 and 12. Thus the saved curves do
show **query-efficiency advantages**, despite the lack of a demonstrated
wall-time advantage. The distinction is essential: SA's temperature at 100,000
calls is inherited from its 200-second cooling schedule. Retuning SA for a fixed
query budget or changing objective cost could change the result. This table is
descriptive, with no additional confirmatory tests or expensive-oracle claims.
""")
    code('''query = read_csv("query_diagnostic_100k.csv")
table(["Family", "Objective", "n", "Method", "Mean vs SA (%)", "Median (%)", "Mean gain: SA / TT (%)"],
      [[r['family'], r['objective'], r['n'], names[r['method']],
        f1(r['mean_pct']) if r['mean_pct'] else "Not reached by all 40 tasks",
        f1(r['median_pct']) if r['median_pct'] else "—",
        f"{f1(r['sa_mean_gain_pct'])} / {f1(r['tt_mean_gain_pct'])}" if r['tt_mean_gain_pct'] else "—"] for r in query])''')
    md("""## All experimental groups

Each size/objective section first summarizes exact call totals at all three
independent budgets. Each budget then has its own four-panel diagnostic and
two tables. The first table reports quality, call volume, elapsed time, and
completed MPS iterations / SA chains. The second reports paired effects and
statistical evidence. A zero completed MPS iteration can still include useful
samples from a partially completed final iteration. Axes adapt per group;
comparisons between methods within each panel share the same axes.
""")
    for family in ("assignment", "partition"):
        for objective in ("quadratic", "entropic"):
            for n in (10, 12, 14):
                md(f"### {family.title()} — {objective}, n = {n}\n\n40 paired tasks; {n*n} binary variables.")
                code(f'image("{family}_{objective}_n{n}_budget_calls.png")')
                for budget in (10, 50, 200):
                    md(f"#### Independent budget: {budget} seconds")
                    code(f'''g = select(summary, "{family}", "{objective}", {n}, {budget})
image("{family}_{objective}_n{n}_{budget}s.png")
table(["Method", "Final cost: mean ± SD", "Gain: mean / median (%)", "Calls: mean ± SD", "Elapsed mean / max (s)", "Iterations / chains", "Zero iterations", "Overrun >5%", "Init mean (s)"],
      [[names[r['method']], f"{{float(r['mean_final']):.4f}} ± {{float(r['sd_final']):.4f}}", f"{{f1(r['mean_gain_pct'])}} / {{f1(r['median_gain_pct'])}}",
        f"{{integer(r['mean_calls'])}} ± {{integer(r['sd_calls'])}}", f"{{f1(r['mean_seconds'])}} / {{f1(r['max_seconds'])}}", f1(r['mean_iterations']), r['zero_iterations'], r['overrun_5pct'], f1(r['mean_initialization_s'])] for r in g])
p = select(paired, "{family}", "{objective}", {n}, {budget})
table(["Method", "Mean vs SA (%)", "Median (%)", "Mean 95% bootstrap CI (%)", "W/T/L", "Raw p", "Holm p", "SA/MPS calls"],
      [[names[r['method']], f1(r['mean_pct']), f1(r['median_pct']), f"[{{f1(r['mean_ci_low'])}}, {{f1(r['mean_ci_high'])}}]",
        f"{{r['wins']}}/{{r['ties']}}/{{r['losses']}}", sci(r['p_raw']), sci(r['p_holm72']), f1(r['calls_ratio'])] for r in p])''')
                    ps = [p for p in pairs if p["family"] == family and p["objective"] == objective and int(p["n"]) == n and int(p["budget"]) == budget]
                    fragments = []
                    for p in ps:
                        method = "Data + EELS" if p["method"] == "U1_data_EELS" else "Full MPS"
                        mean = float(p["mean_pct"])
                        fragments.append(f"{method}: mean relative difference {0.0 if abs(mean)<.05 else mean:.1f}%, with {p['wins']} wins, {p['ties']} ties, and {p['losses']} losses.")
                    md(" ".join(fragments) + " A positive number favors MPS. " + ("At least one advantage passes the Holm correction." if any(float(p["p_holm72"]) < .05 for p in ps) else "Neither method shows a Holm-significant advantage over SA in this group."))
    md("""## Timing, memory and limits of interpretation

Allocated bytes (exported in the CSV) accumulate all allocations and are not peak
live memory. Saved process MaxRSS is a high-water mark across sequential methods
and tasks, so it must not be compared as per-method peak memory. Shared-node
memory bandwidth and GC may influence timings even with exclusive allocations
and CPU binding. These runs use one seed per instance and one hardware platform.

The short full-assignment runs can be dominated by exact-indicator construction,
copying and training. The initial table and iteration counts quantify that cost.
By contrast, an early plateau with many additional calls indicates lack of
observed search progress under this protocol, rather than merely an inability
to execute any iterations. It does not alone establish a local minimum or its
cause. A persistent-amplitude control would be a different experiment.

In particular, at n = 14 with a 10-second budget, full assignment has zero
completed iterations on all 40 tasks for both objectives, and no incumbent
improvement. On quadratic assignment the mean actual duration is 22.8 seconds
and the maximum is 27.7 seconds: the cooperative check follows a costly sweep.
The requested budget alone is therefore insufficient to describe real resource
use. The tables retain the real durations rather than silently relabeling these
runs as exact 10-second executions.

## Reproduction and artifacts

From the repository root, without rerunning optimization:

```text
julia --project=. scripts/sa_cluster/collect.jl
julia --project=. --threads=1,0 scripts/sa_cluster/analyze.jl
python scripts/sa_cluster/build_analysis_notebook.py --execute
```

Raw results remain in `data/sa_cluster_v2/results/`. Aggregated tables,
full-precision curve coordinates, PNG figures and the audit are in
`examples/sa_cluster_analysis/`. The report does not modify the manuscript or
older notebooks. Notebook tables render as HTML rather than LaTeX tabular.
""")
    md("""## Manuscript layout proposal (not inserted into the article)

Use the 200-second trials for the manuscript. Keep families and objectives
separate, and do not pool n = 10, 12, 14 into one average: the rank contrast
would become harder to interpret.

- **Main text:** two figures at the largest size n = 14. One has gain versus
  time; the other has gain versus objective calls. Each is a 2-by-2 grid:
  assignment above Set Partitioning, quadratic left and entropic right. Each
  panel shows SA, EELS with data-initialized TT, and full-indicator TT.
- **Appendix:** the same two figures for n = 10 and again for n = 12: four more
  figures. Add the detailed protocol, endpoint tables, paired tests, call
  volumes and actual-time overruns. The remaining 10/50-second plots can remain
  in this notebook as supporting material rather than filling the article.

This gives **six figures / 24 panels** for the complete 200-second comparison:
two figures / eight panels in the main text, four figures / 16 panels in the
appendix. Selecting the largest size is a transparent size-based layout choice,
not selection of favorable outcomes. Each displayed point still averages 40
tasks. The six figure files below are also saved as vector SVGs.

Suggested claim: "In the recorded 200-second trajectories, the TT methods
often attained a larger mean gain at the same number of objective evaluations
on Set Partitioning, whereas informed SA was more effective under equal nominal
wall-time budgets. On assignment the full-indicator TT showed query-efficiency
advantages in some supported ranges at n = 10 and 12; the data-initialized EELS
variant did not share that advantage at the 100,000-call diagnostic slice."

Do not replace this by a universal fixed-query superiority claim. State the
query range, the common 40-task support, and time-based SA cooling. A single
curve crossing or a pointwise confidence band is not a paired significance test.
The counts are full oracle calls including repetitions, not unique states or
cost-delta operations. These results do not include an SA tuned to a fixed
query budget or to an expensive objective.
""")
    for n in (14, 10, 12):
        md(f"### {'Main-text candidate' if n == 14 else 'Appendix candidate'}: n = {n}, 200 seconds")
        for axis in ("time", "calls"):
            code(f'image("article_n{n}_{axis}.png")')
    # Keep algorithm identifiers intact while matching the manuscript's TT
    # terminology in all displayed prose, tables, and labels.
    for cell in cells:
        cell.source = cell.source.replace("MPS", "TT").replace("Full TT", "Full-indicator TT").replace("Data + EELS", "EELS (data TT)")
    nb.cells = cells
    nbf.write(nb, NOTEBOOK)
    if execute:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        scratch = ROOT / "tmp" / "sa_report_kernel"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            spec = Path(tmp) / "sa-report"; spec.mkdir()
            (spec / "kernel.json").write_text(json.dumps(dict(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], display_name="SA report", language="python")))
            km = KernelManager(kernel_name="sa-report", kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
            client = NotebookClient(nb, km=km, timeout=180, resources={"metadata": {"path": str(ROOT / "examples")}})
            try:
                client.execute()
                nbf.write(nb, NOTEBOOK)
            finally:
                if km.has_kernel:
                    km.shutdown_kernel(now=True)
    print(NOTEBOOK)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    main(parser.parse_args().execute)
