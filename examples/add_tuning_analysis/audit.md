## Input audit

Source: `data\assignment_add_tuning_v1\1f4c7d9256d7`. Recorded implementation: `1f4c7d9256d722e04683982cb5d6d25781a403f746f03e257e1134f56d91638e`.

- **Linear**: 250 Best Cost and 3,000 ADD outcomes; all expected keys present; objective/input hashes, initial-data hashes, seeds, initial costs, feasibility, final costs and 20 completed iterations verified.
- **Quadratic**: 250 Best Cost and 3,000 ADD outcomes; all expected keys present; objective/input hashes, initial-data hashes, seeds, initial costs, feasibility, final costs and 20 completed iterations verified.

All **6500** outcomes are complete. The runner explicitly schedules Best Cost as array task 0, once per training instance and objective (500 baseline runs total). There are 6,000 ADD runs. No test data, old result files or optimization reruns enter this report.

Recorded solver parameters: `(NUM_FEASIBLE_SAMPLES = 1000, NUM_MPS_SAMPLES = 10000, LINK_DEGENERACY = 1, NUM_SWEEP_ITER = 1, NUM_GLOBAL_ITER = 20, LEARNING_RATE = 0.05, UTILITY_FRACTION = 0.05, KEEP_NUM_WORST = 0.0)`. Both families use 1,000 initial feasible points, 10,000 MPS samples, 20 global iterations, one sweep, learning rate 0.05, elite fraction 0.05, and no worst retention. The runner disables EELS and the barrier. ADD and Best Cost both receive external feasible samples; this is the ADD comparison, not the closed-data EELS comparison.
