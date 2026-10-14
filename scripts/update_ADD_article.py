"""Stage the refreshed ADD article section and full tables from audited CSVs."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/add_tuning_analysis"

MAIN = r"""\subsection {ADD impact on optimization results}
\label{subsec:diversify_data_res}
We use graph distance (\cref{eq:graph_distance}) to balance cost-based
selection with exploration of the charge graph. Assignment admits a simple
uniform feasible sampler based on random permutations, so this experiment
examines which available data to learn from, rather than how to obtain
feasible data. In the ADD variant of \cref{alg:tt_generative},
we reconstruct the TT from the selected data at every outer iteration.
We compare the following selection rules on a candidate pool $S$, with
selection quota $\mu$ and charge history $\mathcal H$:
\begin{enumerate}
\item \textit{Best Cost}: select the $\mu$ candidates with lowest cost.
\item \textit{Best Cost + Graph Distance Filter (GDF)}: first select
$(1-w)\mu$ candidates by cost, then select up to $w\mu$ lowest-cost
remaining candidates satisfying $\rho(\mathcal S,\mathcal H)>0$.
If no such candidates remain, use Best Cost as a fallback.
\item \textit{Mixed Selection}: select $w\mu$ candidates with largest
graph distance and fill the remaining quota by cost from the other candidates.
\item \textit{Weighted Rank}: select the $\mu$ candidates with smallest
$r_C+w r_\rho$, where rank one denotes lowest cost or largest graph
distance, respectively.
\end{enumerate}
The history stores charge sets at each TT link. Computing one graph distance
requires $O(|b|(N-1))$ operations on average with hashed charge sets;
history storage is an additional cost of the diversity-aware rules.

For each of the linear and quadratic objectives we use 50 instances at
each $n\in\{12,13,14,15,16\}$, with $N=n^2$ binary variables.
Every configuration receives the same 1,000 initial feasible assignments
for each instance, verified by stored hashes. Each run performs 20 outer
iterations, one training sweep per iteration, learning rate $0.05$ and
bond degeneracy one. After training a newly initialized TT, we draw
10,000 MPS samples, deduplicate them together with 50 retained elite
assignments, and supplement the pool with fresh uniform feasible samples
to reach 20,000 candidates. Each rule selects up to $\mu=1000$ candidates;
retaining the 50 elites can increase the next training set to at most
1,050 distinct assignments. No worst-sample retention, EELS enhancement
or barrier filter is used. All methods, including Best Cost, receive
external candidates. Thus the comparison fixes iteration and sampling
settings, not elapsed time.

We evaluate $w\in\{0.1,0.2,0.3,0.4\}$ for Mixed Selection and GDF,
and $w\in\{0.1,0.35,0.5,0.7\}$ for Weighted Rank.
For each strategy and objective, we select the weight with the largest
mean per-instance relative improvement
$R=100(C_{\mathrm{BestCost}}-C_{\mathrm{ADD}})/|C_{\mathrm{BestCost}}|$.
\Cref{fig:add_selected_strategies} shows all six selected configurations;
the complete grid and paired statistics are in
\cref{app:selection_strategy_screening}.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{pictures/ADD_selected_strategies.pdf}
\caption{ADD versus Best Cost on the parameter-screening instances.
Columns show linear (left) and quadratic (right) objectives; rows show
GDF ($w=0.4$), Mixed Selection ($w=0.4$), and Weighted Rank ($w=0.35$).
Each box contains 50 paired relative improvements at the indicated number
of binary variables. Positive values favor ADD. Box edges are the first
and third quartiles; whiskers extend to observations within 1.5 interquartile
ranges and remaining observations are shown as points. Green curves with
diamond markers and orange curves with circle markers show group medians
and arithmetic means, respectively.
All panels share axis limits. Each displayed weight maximizes mean relative
improvement within its strategy on these same instances.}
\label{fig:add_selected_strategies}
\end{figure*}

All three strategies improve on Best Cost at their selected weights.
The mean gains for Mixed Selection, Weighted Rank and GDF are,
respectively, $2.8\%$, $5.6\%$ and $6.9\%$ for linear objectives,
and $2.7\%$, $5.9\%$ and $6.6\%$ for quadratic objectives.
GDF with $w=0.4$ gives the largest mean improvement in both families.
It wins, ties and loses on 235/14/1 linear instances and 209/26/15
quadratic instances; its pooled median improvements are $5.9\%$ and
$6.2\%$. One-sided paired Wilcoxon tests on raw final costs give
$p=9.49\times10^{-41}$ and $p=2.05\times10^{-34}$, respectively;
Holm adjustment over all 24 baseline comparisons gives
$2.18\times10^{-39}$ and $3.49\times10^{-33}$.
The mean linear improvement of GDF increases from $2.5\%$ at $n=12$
to $11.2\%$ at $n=16$; its quadratic mean improves from $3.8\%$ to
$7.9\%$. This pattern is consistent with a benefit from charge-space
exploration as problem size grows, but does not isolate its mechanism.

These are within-benchmark comparisons with weights selected on the same
instances. They support GDF with $w=0.4$ as a practical choice for this
setting, not as a universally optimal configuration. In particular,
after Holm correction over 66 pairwise configuration comparisons within
the quadratic family, GDF $w=0.4$ is not statistically separated from
GDF $w=0.3$ ($p=0.0858$) or Weighted Rank $w=0.35$ ($p=0.2455$).
One solver seed per instance does not characterize seed-to-seed variability,
and these iteration-budget results do not establish a wall-time advantage
or performance on other constraint families.
\clearpage

"""

APPENDIX = r"""\section{Complete ADD strategy and weight comparison}
\label{app:selection_strategy_screening}
\Cref{tab:add_linear_all,tab:add_quadratic_all} report every configuration
under the protocol of \cref{subsec:diversify_data_res}, using the same
250 paired objective instances per table. Best Cost was computed once
per instance and objective (500 baseline runs); each of the 12 ADD
configurations was run on the corresponding 250 instances (6,000 ADD runs).
All 6,500 outcomes were checked for completeness, feasibility and consistency
of the stored final objective. Input hashes, initial training-data hashes,
initial costs and completed iteration counts were checked across configurations.

The linear coefficients follow $c_i\sim U[-50,50]$.
For the quadratic family a matrix $B$ is generated with independent
entries from $U[-7,7]$ and symmetrized as $Q=(B+B^T)/2$; hence
off-diagonal entries of $Q$ are averages of two uniform draws, rather
than uniform themselves. The cost is $x^TQx$, including both off-diagonal
directions and the diagonal. The source objective files are fixed across
strategies. For objective index $o=0$ (linear) or $o=1$ (quadratic),
instance index $i$ and size $n$, the initial/external source uses a
Mersenne Twister seeded by $20261012+100000o+100n+i$; the model sampling
stream is seeded by this value plus $10^7$. The source stream is independent
of MPS sampling. Subsequent source batch sizes can differ because the
number of unique MPS samples differs; identical starting data therefore
do not imply identical full adaptive trajectories.

We apply the paired Wilcoxon signed-rank test to raw differences
$C_{\mathrm{BestCost}}-C_{\mathrm{ADD}}$, with the right-tailed alternative
favoring ADD, as implemented by \texttt{SignedRankTest} in
\texttt{HypothesisTests.jl}. Zeros are handled by that implementation;
win/tie/loss counts use strict comparisons and exact equality of stored costs.
The tables give the positive-rank sum $W$, the unadjusted one-sided $p$,
Holm-adjusted $p$ over the 12 comparisons in that objective family, and
the more conservative adjustment over all 24 baseline comparisons.
Mean and median gains use the per-instance relative improvement $R$;
the signed-rank test is not a test of the mean percentage gain.
Percentages are displayed to one decimal place, while all calculations
use unrounded values. Rows are ordered by mean relative improvement,
matching the weight-selection criterion.

The signed-rank location-shift interpretation assumes independent problem
pairs and symmetric differences under the null; pooling sizes also allows
larger-magnitude differences to influence the ranks. The size-resolved
distributions in \cref{fig:add_selected_strategies} complement these pooled
statistics. Holm correction addresses multiplicity, but does not remove
selection-induced optimism in the displayed best-weight effect sizes.
Of the 24 baseline comparisons, 23 remain significant at $0.05$ after
the joint Holm adjustment; Mixed Selection with $w=0.1$ on quadratic
objectives is the exception ($p=0.0859$).

"""


def ptex(value):
    x = float(value)
    if x < 0.001:
        mantissa, exponent = f"{x:.2e}".split("e")
        return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"
    return f"${x:.4f}$"


def table(objective, rows):
    labels = {"gdf": "GDF", "mixed": "Mixed", "weighted_rank": "Weighted Rank"}
    text = [r"\begin{table*}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\caption{{{objective.title()}: every strategy and weight. All rows contain 250 pairs against Best Cost. W/T/L denotes wins/ties/losses for ADD; win rate, mean and median gains are percentages. Holm (12) and Holm (24) adjust the one-sided baseline comparisons within this objective and across both objectives, respectively.}}",
            rf"\label{{tab:add_{objective}_all}}",
            r"\begin{tabular}{lrcrrrrrrr}", r"\toprule",
            r"Strategy & $w$ & W/T/L & Win (\%) & Mean (\%) & Median (\%) & $W$ & $p$ & Holm (12) & Holm (24) \\", r"\midrule"]
    for r in sorted((r for r in rows if r["objective"] == objective), key=lambda r: -float(r["mean_relative"])):
        cells = [labels[r["strategy"]], r["weight"], f'{r["wins"]}/{r["ties"]}/{r["losses"]}',
                 f'{100*int(r["wins"])/int(r["N"]):.1f}', f'{float(r["mean_relative"]):.1f}',
                 f'{float(r["median_relative"]):.1f}', f'{float(r["W"]):g}',
                 ptex(r["p"]), ptex(r["holm12"]), ptex(r["holm24"])]
        text.append(" & ".join(cells) + r" \\")
    return "\n".join(text + [r"\bottomrule", r"\end{tabular}", r"\end{table*}", "", ""])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    start = text.index(r"\subsection {ADD impact on optimization results}")
    end = text.index(r"\subsection{Benchmarking library performance}", start)
    text = text[:start] + MAIN + text[end:]
    label = text.index(r"\label{app:selection_strategy_screening}")
    start = text.rfind(r"\section{", 0, label)
    end = text.index(r"\section{Detailed library performance benchmarking}", label)
    with (DATA / "all_configurations.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24
    tables = (table("linear", rows) + table("quadratic", rows)).replace(
        "\\end{table*}\n\n\\begin{table*}[t]", "\\vspace{1.5em}", 1)
    text = text[:start] + APPENDIX + tables + "\\clearpage\n\n" + text[end:]
    replacements = {
        "on held-out assignment\ninstances": "on assignment\ninstances",
        "50 test instances": "50 instances",
        "These are the held-out\nlinear and quadratic objective families used in the ADD evaluation below.":
            "Both linear and quadratic objective functions are considered.",
        "hyperparameter selection or validation split is needed for this comparison.":
            "hyperparameter selection is needed for this comparison.",
        "paired test instances": "paired instances",
        "new held-out SOTA evaluation": "new SOTA evaluation",
        "article's held-out objective\nfiles": "article's objective\nfiles",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    assert "held-out" not in text
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
