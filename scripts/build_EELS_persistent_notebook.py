"""Build the English, executed protocol-3 EELS report from audited Julia exports."""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "examples" / "EELS_persistent_report.ipynb"


def main(execute=False):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [md("""# EELS versus Best Cost: persistent TT (protocol 3)

This report analyzes `data/eels_optimization/gamma1_persistent_v3/` only.
It does not combine these data with the old reset/rebuild EELS or SA results.
There are 250 paired assignment tasks per objective: n = 12, 13, 14, 15, 16,
with 50 instances per size, and n squared binary variables. Objectives come
from the existing `test_random_vectors` and `test_random_bilinear_forms` files.
No hyperparameter selection is performed on these results.

## Saved protocol

Both arms receive the same 400 initial feasible draws. Each TT is constructed
**once**, orthogonalized and normalized. Best Cost uses Method (ii); the other
arm applies EELS with gamma = 1 **once at initialization**. Learned amplitudes
are then retained, and the structural support is fixed throughout optimization.

Each of 20 iterations first samples 10,000 points with the legacy backend,
deduplicates them, and trains on at most the best 400 distinct points **from
that new batch only**. Neither the initial data nor old elites are injected
into later training batches. The best-ever solution is recorded separately.
There are no restarts, no subsequent TT reinitializations and no external draws.
The first sampling round precedes the first training sweep.

Training uses one sweep, learning rate 0.05, nondegenerate charge blocks, and
normalized Boltzmann weights with temperature equal to the standard deviation
of the current elite costs. The stored UTILITY_FRACTION = 0.05 controls a logged
utility statistic; it does **not** reduce the training set to 20 points.
Iteration sampling seeds are paired; identical seeds do not imply identical
samples from different trained distributions.

## A/B procedure

As in the earlier reports, use a one-sided paired Wilcoxon signed-rank test on
raw costs, alternative `Best Cost - EELS > 0`. Display raw p-values plus Holm
adjustment for the two primary objective-level tests. The ten size-specific
tests are supplementary and receive their own Holm correction. The test does
not specifically test the mean percentage gain; its location interpretation
uses the usual symmetry assumption for paired differences.

Improvement is `100 * (Best Cost - EELS) / abs(Best Cost)`; positive values favor
EELS. Pooling all sizes follows the earlier analysis; separate size tables
are included because raw difference magnitudes can vary with size. One seed is
used per task. This compares equal sample/iteration settings, not equal time.
"""), code('''from pathlib import Path
import csv, html
from IPython.display import display, HTML, Markdown, Image
root = Path.cwd()
if not (root / "Project.toml").exists(): root = root.parent
out = root / "examples" / "eels_persistent_analysis"
def read(name):
    with (out / name).open(encoding="utf-8") as f: return list(csv.DictReader(f))
def table(headers, rows):
    display(HTML('<table><thead><tr>' + ''.join('<th>'+html.escape(str(x))+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'))
def pct(x): return f"{float(x):.1f}%"
def pv(x): return f"{float(x):.3g}"
summary, groups, pairs = read('summary.csv'), read('by_size.csv'), read('paired_costs.csv')
display(Markdown((out/'audit.md').read_text(encoding='utf-8')))
table(['Objective','Pairs','Mean improvement','Median improvement','Wins / ties / losses','Raw p','Holm p (2)'],
      [[r['objective'], r['N'], pct(r['mean_relative']), pct(r['median_relative']),
        f"{r['wins']} / {r['ties']} / {r['losses']}", pv(r['p']), pv(r['p_holm2'])] for r in summary])'''), md("""## Relative improvement by size

Exactly two boxplots are shown, one per objective, with identical y limits.
Each box contains all 50 paired tasks of that size. Whiskers extend to the
furthest observations within 1.5 IQR; outliers are shown. The two overlays are
the group average and group median of those same 50 observations. All displayed
percentages use one decimal place; exported CSV values retain full precision.
""")]
    for objective in ("linear", "quadratic"):
        nb.cells += [md(f"### {objective.title()} assignment"), code(f'''display(Image(filename=str(out/'EELS_{objective}.png')))
table(['n','Variables','Pairs','Mean improvement','Median improvement','Wins / ties / losses','Raw p','Holm p (10)'],
      [[r['n'], r['variables'], r['N'], pct(r['mean_relative']), pct(r['median_relative']),
        f"{{r['wins']}} / {{r['ties']}} / {{r['losses']}}", pv(r['p']), pv(r['p_holm10'])]
       for r in groups if r['objective']=='{objective}'])''')]
    nb.cells += [md("""## Interpretation and scope

The following statements are generated from the stored paired results, without
choosing favorable sizes or deleting ties. A significant improvement supports
the value of the one-time EELS enhancement **within this persistent-TT protocol**.
It does not establish superiority over SA, which used a different protocol and
budgets, nor a speed advantage over Best Cost.
"""), code('''for r in summary:
    significant = float(r['p_holm2']) < .05
    direction = "improves" if float(r['mean_relative']) > 0 else "does not improve"
    display(Markdown(f"**{r['objective'].title()}:** EELS {direction} the mean relative objective: {pct(r['mean_relative'])}; median {pct(r['median_relative'])}. "
                     f"There are {r['wins']} wins, {r['ties']} ties, and {r['losses']} losses out of {r['N']} tasks. "
                     f"The one-sided advantage {'passes' if significant else 'does not pass'} the two-test Holm correction (p = {pv(r['p_holm2'])})."))
table(['Objective','n','Mean structural support ratio: EELS / baseline','Mean distinct evaluations: baseline / EELS'],
      [[o,n,f"{sum(float(r['eels_support'])/float(r['baseline_support']) for r in pairs if r['objective']==o and int(r['n'])==n)/50:.2f}",
        ' / '.join(f"{sum(float(r[k]) for r in pairs if r['objective']==o and int(r['n'])==n)/50:,.0f}" for k in ['baseline_unique_evaluations','eels_unique_evaluations'])]
       for o in ['linear','quadratic'] for n in range(12,17)])'''),md("""Here the evaluation diagnostic is the number of **distinct cached points**,
not total invocations as in the SA report. It must not be compared directly to
the SA objective-call counts. Support refers to the initial structural indicator,
not the norm of a trained probability distribution. Saved concurrent elapsed
times are not isolated single-core speed measurements.

## Reproduction

```text
julia --project=. --threads=1,0 scripts/report_EELS_persistent.jl
python scripts/build_EELS_persistent_notebook.py --execute
```

These commands audit saved outputs and regenerate the report without running
optimization. The audit reproduces the initial training samples from seeds and
checks their fingerprints, all input hashes, final feasibility/costs, recorded
single initialization, and bounded batch-only training sizes. Figures are
generated with Julia; tables render as HTML. The manuscript is unchanged.
""")]
    nbf.write(nb, DEST)
    if execute:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        scratch = ROOT / "tmp" / "eels_report_kernel"; scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            spec = Path(tmp)/"eels-report"; spec.mkdir()
            (spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='EELS report',language='python')))
            km=KernelManager(kernel_name='eels-report',kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
            try:
                NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(ROOT/'examples')}}).execute()
                nbf.write(nb, DEST)
            finally:
                if km.has_kernel: km.shutdown_kernel(now=True)
    print(DEST)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    main(parser.parse_args().execute)
