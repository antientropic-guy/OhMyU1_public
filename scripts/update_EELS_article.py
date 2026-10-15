"""Refresh manuscript EELS sections and figure assets from audited protocol-3 CSVs.

Writes a staged manuscript by default through explicit --source/--output paths.
No optimizer is run, and no older numerical results are reused.
"""
import argparse
import csv
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/eels_persistent_analysis"


def read(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(x):
    return f"{float(x):.1f}"


def pvalue(x):
    x = float(x)
    exponent = math.floor(math.log10(x))
    return rf"${x / 10**exponent:.2f}\times10^{{{exponent}}}$"


def main(source, output):
    summary = {r["objective"]: r for r in read("summary.csv")}
    text = source.read_text(encoding="utf-8")
    main_text = r"""\subsection{EELS impact on optimization results}
\label{subsec:eels_optimization}
To test whether enhanced expressibility translates into lower objective values,
we compare EELS with $\gamma=1$ against Method (ii), using Best Cost selection
in both arms. For each of the linear and quadratic objective families, we use
50 assignment instances at each $n\in\{12,13,14,15,16\}$: 250 paired comparisons
with $N=n^2\in\{144,169,196,225,256\}$ binary variables. No EELS-specific
hyperparameter selection is performed.

For each pair, 400 feasible assignments are drawn uniformly from random
permutations. Both arms receive the same deduplicated, cost-sorted initial
matrix; its equality is verified by stored hashes and regeneration from seeds.
Both initialize their TT once, with EELS applied only at initialization in
the enhanced arm. Each of 20 outer iterations first draws $10{,}000$ samples
using the legacy backend, then trains the existing TT on the best at most
400 distinct assignments from that new batch alone. Training uses one complete
right--left sweep, learning rate $0.05$, and virtual charge degeneracy one.
Boltzmann weights use the standard deviation of the current elite costs as
temperature, with uniform weights when this deviation is zero. Learned
amplitudes and structural charge sets persist across iterations. Initial and
historical points are not merged into subsequent training batches; the best
evaluated solution is retained separately. No external data, restarts, or
subsequent TT reinitializations are used. Sampling seeds are paired at each
iteration, although different models need not produce identical samples.
This fixed-data setting differs from ADD, which continually receives external
feasible candidates and rebuilds its TT.

For best recorded costs $C_{\mathrm{ii}}$ and $C_{\mathrm{EELS}}$, define
\begin{equation}
 R=100\,\frac{C_{\mathrm{ii}}-C_{\mathrm{EELS}}}{|C_{\mathrm{ii}}|}.
 \label{eq:eels_relative_improvement}
\end{equation}
All baseline costs are nonzero; positive $R$ favors EELS.
\Cref{fig:eels_optimization} shows the distributions by size, including
group medians and arithmetic means. As in the ADD analysis, a one-sided
paired Wilcoxon signed-rank test is applied to unscaled final costs, pooled
over the five sizes within each objective. \Cref{tab:eels_optimization}
reports raw $p$-values and Holm adjustment across the two objective-level
comparisons. The test concerns paired cost differences, not specifically
the mean percentage improvement.

The mean improvement is @LINEAR_MEAN@\% for linear and @QUADRATIC_MEAN@\%
for quadratic objectives, with @LINEAR_WINS@ and @QUADRATIC_WINS@ wins out
of 250 pairs, respectively. Both advantages remain significant after the
Holm correction. However, there are @LINEAR_TIES@ and @QUADRATIC_TIES@
exact ties, and the pooled medians are only @LINEAR_MEDIAN@\% and
@QUADRATIC_MEDIAN@\% after rounding. Thus the improvement is heterogeneous,
not a uniform gain on every instance. These results
support the one-time expressibility enhancement under equal sampling and
iteration settings; they do not establish a wall-time advantage or robustness
across repeated optimization seeds on the same task.

\begin{table*}[t]
\centering
\caption{Persistent-TT EELS versus Method (ii), with Best Cost selection in
both arms: 250 paired instances per objective. W/T/L denotes EELS wins,
exact ties, and losses. Means and medians refer to the paired relative
improvement $R$ (\%). Holm correction covers the two one-sided tests;
percentages are rounded to one decimal place.}
\label{tab:eels_optimization}
\begin{tabular}{lrrrrr}
\toprule
Objective & W/T/L & Mean (\%) & Median (\%) & Raw $p$ & Holm $p$ \\
\midrule
@MAIN_ROWS@
\bottomrule
\end{tabular}
\end{table*}

\begin{figure*}[t]
\centering
\begin{subfigure}{0.48\textwidth}
\includegraphics[width=\linewidth]{pictures/test_EELS_Linear.pdf}
\caption{Linear objective.}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
\includegraphics[width=\linewidth]{pictures/test_EELS_Quadratic.pdf}
\caption{Quadratic objective.}
\end{subfigure}
\caption{EELS ($\gamma=1$) versus Method (ii), both with Best Cost selection
and persistent learned amplitudes. Each box contains 50 paired instances.
Purple curves mark group medians and orange curves group arithmetic means.
Box edges are the 25th and 75th percentiles; whiskers extend to the most
extreme observations within 1.5 interquartile ranges of the box. Outliers
are shown individually. Both panels use identical axes. Positive values
favor EELS.}
\label{fig:eels_optimization}
\end{figure*}

"""
    for o in ("linear", "quadratic"):
        r = summary[o]
        for suffix, field in (("MEAN", "mean_relative"), ("MEDIAN", "median_relative"), ("WINS", "wins"), ("TIES", "ties")):
            v = number(r[field]) if suffix in ("MEAN", "MEDIAN") else r[field]
            main_text = main_text.replace(f"@{o.upper()}_{suffix}@", v)
    table_rows = []
    for o in ("linear", "quadratic"):
        r = summary[o]
        table_rows.append(f"{o.title()} & {r['wins']}/{r['ties']}/{r['losses']} & {number(r['mean_relative'])} & {number(r['median_relative'])} & {pvalue(r['p'])} & {pvalue(r['p_holm2'])}" + r" \\")
    main_text = main_text.replace("@MAIN_ROWS@", "\n".join(table_rows))
    start = text.index(r"\subsection{EELS impact on optimization results}")
    end = text.index(r"\subsection {ADD impact on optimization results}", start)
    text = text[:start] + main_text + text[end:]
    # Remove the retired boxplot-diagnostic appendix when updating older drafts.
    heading = r"\section{Quadratic EELS: distributions within problem-size groups}"
    if heading in text:
        start = text.index(heading)
        end = text.index(r"\section{Complete ADD strategy and weight comparison}", start)
        text = text[:start] + text[end:]
    # The PROTES benchmark intentionally retains its separately measured old
    # protocol; remove the now-false claim that it uses the same EELS updates.
    text = text.replace("These are\nthe article's EELS updates with the fixed 20-iteration cap replaced by a\nwall-time limit; realized iteration counts are reported below.",
        "This timing benchmark uses the earlier reset/rebuild protocol, unlike\nthe persistent-TT comparison in \\cref{subsec:eels_optimization}. The number\nof iterations is set by the wall-time limit and reported below.")
    text = text.replace("and rebuilds its MPS from the retained data", "and rebuilds its TT from the retained data")
    text = text.replace("The normalized trained MPS supplies", "The normalized trained TT supplies")
    text = text.replace(
        "but the current optimization driver does not explicitly guard against\nzero cost variance or zero denominators.",
        "and the persistent-TT EELS driver explicitly uses uniform weights when the cost variance is zero.\nThe per-sample gradient still requires nonzero amplitude denominators.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    picture_dir = output.parent / "pictures"; picture_dir.mkdir(exist_ok=True)
    for source_name, destination_name in {
        "EELS_linear.pdf": "test_EELS_Linear.pdf",
        "EELS_quadratic.pdf": "test_EELS_Quadratic.pdf",
    }.items():
        shutil.copy2(DATA / source_name, picture_dir / destination_name)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
