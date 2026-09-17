# Corrected A/B evaluation for assignment strategies

## Recommendation

Use the paired, one-sided Wilcoxon signed-rank test already used in the report
notebooks, but define the family *before looking at the p-values* and adjust all
of its p-values with Holm's step-down procedure. The executable implementation
is `examples/MultipleComparisonTests.jl`.

The conservative default treats all 24 comparisons as one family:

- 2 objective types: linear and quadratic;
- 3 strategies: `mixed`, `best nonzero mixed`, and `weighted rank`;
- 4 values of `w` for each strategy.

This is the right default for a paper-level claim such as "at least one proposed
configuration improves on `best_cost`". If the paper explicitly declares the
linear and quadratic experiments to be two distinct scientific questions, two
families of 12 can be used (`--by-objective`). Do not split families after seeing
the results. A correction only over the four `w` values is too narrow if the
paper also searches over strategies and objective types and then highlights the
best result.

Run:

```sh
julia --project=. examples/MultipleComparisonTests.jl
# or, for two predeclared objective-specific families:
julia --project=. examples/MultipleComparisonTests.jl --by-objective
```

## Mathematical definition

For objective instance \(i\), let \(A_i\) be the final cost from the control
`best_cost`, and let \(B_{ij}\) be the cost from candidate configuration \(j\).
The paired improvement is

\[
D_{ij}=A_i-B_{ij},
\]

so \(D_{ij}>0\) favors the candidate because this is a minimization problem.
For every configuration \(j\), the one-sided hypotheses are

\[
H_{0j}:\theta_j\leq 0, \qquad H_{1j}:\theta_j>0,
\]

where \(\theta_j\) is the location shift (the pseudomedian under the usual
signed-rank model) of the paired-difference distribution. Remove zero
differences, rank \(|D_{ij}|\) with average ranks for ties, and form

\[
W_j^+=\sum_i R_{ij}\,\mathbf 1[D_{ij}>0].
\]

The raw p-value is \(p_j=P_{H_{0j}}(W_j^+\geq W_{j,obs}^+)\). The signed-rank
test assumes independent instance-level pairs and a distribution of differences
that is symmetric about its location. Pairing is essential: candidate and
control must be evaluated on exactly the same random objective instance.

For \(m\) candidate configurations, order the raw p-values
\(p_{(1)}\leq\cdots\leq p_{(m)}\). Holm's procedure rejects in order while

\[
p_{(k)}\leq \frac{\alpha}{m-k+1}.
\]

Equivalently, the adjusted p-values reported by the script are

\[
\widetilde p_{(k)}=
\min\left(1,\max_{\ell\leq k}(m-\ell+1)p_{(\ell)}\right).
\]

Holm controls the probability of one or more false rejections (strong FWER) at
\(\alpha\), under arbitrary dependence among tests. That dependence matters
here because every candidate is compared with the same baseline and candidates
share the same instances. Holm is uniformly no worse than plain Bonferroni.

The report also gives quantities that should accompany p-values:

- wins/losses/ties for the paired differences;
- the Hodges--Lehmann pseudomedian (all Walsh averages), in cost units;
- matched-pairs rank-biserial correlation, in \([-1,1]\).

A small adjusted p-value is not evidence of a practically important gain, so
the effect estimates and their units should be reported in the paper.

## Hyperparameter selection versus hypothesis multiplicity

Holm correction solves the "many significance tests" problem, but not the
"winner's curse" from selecting the best \(w\) and estimating its performance on
the same 250 instances. There are two defensible analysis modes:

1. **Exploratory/current data.** Test all 24 configurations and report every
   Holm-adjusted p-value. It is acceptable to identify promising settings, but
   label the performance estimate as exploratory.
2. **Confirmatory/preferred.** Use the existing instances to choose one fixed
   configuration per strategy (or one overall configuration), then freeze it.
   Generate independent objectives, run `best_cost` and the frozen candidates
   on those same new objectives, and test only the predeclared comparisons.
   `scripts/generate_assignment_test_instances.jl` creates this untouched test
   set in separate directories. It does not run the expensive solvers.

If algorithmic randomness is material, use the same predeclared set of random
seeds for every configuration. Either average repeated runs within each
objective before testing (so the objective remains the replication unit), or
treat `(objective, seed)` as a block and use an analysis that respects the
nested structure. Do not treat correlated repeated runs as independent data.

The notebooks pool five problem sizes. Since raw cost differences can change
scale with `n`, also report results by `n` as a robustness analysis. Do not turn
those five breakdowns into five additional uncorrected confirmatory claims. A
sign test is a useful sensitivity check if symmetry of paired differences is
implausible; it tests only the probability of improvement and is usually less
powerful than signed-rank.

## Practice in computer-science publications

Practice is not uniform. Well-known ML guidance recommends Wilcoxon for two
paired algorithms and an omnibus Friedman test followed by multiplicity-aware
post-hoc comparisons when more algorithms are compared. Later JMLR work makes
adjusted p-values explicit for multiple classifier comparisons. In NLP, an ACL
survey found that significance testing was often absent or misused even in
papers that emphasized empirical results. Thus "uncorrected because there are
only four values of `w`" is seen in practice, but it is not strong statistical
practice.

For hyperparameters, the usual defensible workflow in CS/ML is selection on
training/validation data and a single final evaluation on an untouched test
set. Hyperparameter values are not automatically four scientific hypotheses if
they are only an internal tuning grid and only the frozen selected method is
tested once. They *are* part of the multiplicity family when each value is
tested on the benchmark and the paper chooses or highlights whichever is
significant. With only four values per strategy, Holm has little administrative
cost and is a clear choice. FDR methods such as Benjamini--Hochberg answer a
different question (expected proportion of false discoveries) and are less
appropriate when a paper makes a small number of confirmatory claims.

References:

- J. Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data
  Sets*, JMLR 7:1--30. https://www.jmlr.org/papers/v7/demsar06a.html
- S. García and F. Herrera (2008), *An Extension ... for all Pairwise
  Comparisons*, JMLR 9:2677--2694. https://www.jmlr.org/papers/v9/garcia08a.html
- R. Dror et al. (2018), *The Hitchhiker's Guide to Testing Statistical
  Significance in Natural Language Processing*, ACL 2018.
  https://aclanthology.org/P18-1128/

