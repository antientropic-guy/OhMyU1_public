## Selection and interpretation

Selection rule: choose the configuration with the largest mean per-instance relative improvement against the common baseline, using the Wilcoxon comparisons to assess evidence. This rule is stated for this analysis, not claimed to have been preregistered. A smaller p-value is not a larger effect.

| Objective | Best weight within strategy | Weight | Mean improvement | Median improvement | W / T / L |
|---|---|---:|---:|---:|---:|
| Linear | Mixed | 0.4 | 2.8% | 2.4% | 189 / 38 / 23 |
| Linear | Weighted Rank | 0.35 | 5.6% | 5.0% | 222 / 21 / 7 |
| Linear | GDF | 0.4 | 6.9% | 5.9% | 235 / 14 / 1 |
| Quadratic | Mixed | 0.4 | 2.7% | 1.8% | 165 / 35 / 50 |
| Quadratic | Weighted Rank | 0.35 | 5.9% | 5.4% | 189 / 28 / 33 |
| Quadratic | GDF | 0.4 | 6.6% | 6.2% | 209 / 26 / 15 |

Recommended configurations:

- **Linear: GDF, weight 0.4**. Mean improvement 6.9%; median 5.9%; wins 94.0%; baseline one-sided p=9.49e-41, Holm across all 24 tests=2.18e-39.
- **Quadratic: GDF, weight 0.4**. Mean improvement 6.6%; median 6.2%; wins 83.6%; baseline one-sided p=2.05e-34, Holm across all 24 tests=3.49e-33.

A single shared choice is therefore **GDF, weight 0.4** (`best nonzero mixed`). This is the best tested configuration by the stated effect-size criterion for both objectives.

Implementation setting: pass `"best nonzero mixed"` as the strategy argument to `solve` and set `mixed_proportion=0.4`.

### How clearly does the selected configuration beat alternatives?

Additional two-sided paired Wilcoxon tests compare all 66 configuration pairs within each objective, with Holm correction over those 66 tests. These are exploratory comparisons on the same tuning data. In this table, wins mean the selected configuration has lower cost than its competitor.

| Objective | Competitor | Weight | Selected W / T / L | Two-sided p | Holm (66) |
|---|---|---:|---:|---:|---:|
| Linear | Mixed | 0.4 | 199 / 42 / 9 | 2.68e-34 | 1.45e-32 |
| Linear | Weighted Rank | 0.35 | 143 / 66 / 41 | 2.94e-15 | 7.64e-14 |
| Linear | GDF | 0.3 | 90 / 102 / 58 | 0.0013 | 0.0066 |
| Quadratic | Mixed | 0.4 | 179 / 36 / 35 | 4.85e-24 | 2.81e-22 |
| Quadratic | Weighted Rank | 0.35 | 124 / 44 / 82 | 0.0273 | 0.2455 |
| Quadratic | GDF | 0.3 | 111 / 71 / 68 | 0.0057 | 0.0858 |

**Linear**: the selected configuration is separated from every other tested configuration by the Holm-adjusted pairwise tests.

**Quadratic**: the selected configuration is not statistically separated from GDF 0.3 (adjusted p=0.0858); Weighted Rank 0.35 (adjusted p=0.2455) after correction. Its largest observed mean supports a practical tuning choice, not a claim of uniquely established superiority.

### Selected configuration by problem size

| Objective | n | Variables | Mean improvement | Median improvement | W / T / L |
|---|---:|---:|---:|---:|---:|
| Linear | 12 | 144 | 2.5% | 2.4% | 39 / 11 / 0 |
| Linear | 13 | 169 | 4.9% | 4.9% | 49 / 1 / 0 |
| Linear | 14 | 196 | 6.4% | 5.8% | 47 / 2 / 1 |
| Linear | 15 | 225 | 9.3% | 9.0% | 50 / 0 / 0 |
| Linear | 16 | 256 | 11.2% | 11.0% | 50 / 0 / 0 |
| Quadratic | 12 | 144 | 3.8% | 1.9% | 33 / 12 / 5 |
| Quadratic | 13 | 169 | 6.5% | 6.0% | 44 / 4 / 2 |
| Quadratic | 14 | 196 | 7.1% | 7.1% | 46 / 1 / 3 |
| Quadratic | 15 | 225 | 8.0% | 8.5% | 43 / 4 / 3 |
| Quadratic | 16 | 256 | 7.9% | 6.9% | 43 / 5 / 2 |

23 of 24 configurations beat Best Cost at alpha=0.05 after Holm correction over the full baseline comparison family. See the full tables for the nonsignificant cases.

The recommendation is for subsequent held-out testing. The tuning data establish a strong improvement over Best Cost, but do not by themselves establish out-of-sample performance or a uniquely optimal continuous weight. In particular, inspect the adjusted pairwise p-values before claiming separation from nearby weights. There is one solver seed per objective instance, so this study does not separately estimate within-instance seed variability.

The Wilcoxon test matches the old notebooks exactly: paired raw cost differences, one-sided right tail for baseline minus candidate, exact zero differences treated by HypothesisTests. It is not a t-test of mean percentage improvement. Pooling sizes follows the old reports; larger-magnitude differences can influence rank sums, so the separate size summaries are also supplied. Its usual signed-rank interpretation assumes independent problem pairs and a symmetric difference distribution under a location-shift model.

The comparison uses equal iteration/sample settings, not equal wall-clock budgets. Concurrent timing diagnostics are not isolated speed benchmarks. Shared initial-data hashes verify the starting samples, but subsequent external-source batch sizes may differ by strategy; do not interpret the full adaptive trajectories as identical random draws.
