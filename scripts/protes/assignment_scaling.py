"""Analytic assignment ranks and reproducible, data-only PROTES size probe."""
from pathlib import Path
import argparse
import importlib.util
import itertools
import json
import math
import time
import hashlib
import platform

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("assignment_runner", ROOT / "PROTES/run_assignment.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
np, jax = baseline.np, baseline.jax
OUT = ROOT / "data/protes_comparison/scaling"


def choose(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def exact_rank(n, cut):
    """Minimal exact TT rank of permutation indicator, ROW-MAJOR binary order."""
    if cut in (0, n*n):
        return 1
    r, s = divmod(cut, n)
    return choose(n, r) - choose(s, r-(n-s)) + choose(n, r+1) - choose(n-s, r+1)


def max_rank(n):
    return max(exact_rank(n, k) for k in range(n*n+1))


def enumerate_check(n):
    """Independent sparse unfolding; delete only its zero rows and columns."""
    words = []
    for perm in itertools.permutations(range(n)):
        x = np.zeros((n, n), dtype=np.uint8)
        x[np.arange(n), perm] = 1
        words.append(tuple(x.ravel()))
    ranks = []
    for cut in range(n*n+1):
        left = {w[:cut] for w in words}
        right = {w[cut:] for w in words}
        li, ri = {v:i for i,v in enumerate(left)}, {v:i for i,v in enumerate(right)}
        matrix = np.zeros((len(left), len(right)))
        for w in words:
            matrix[li[w[:cut]], ri[w[cut:]]] = 1
        ranks.append(int(np.linalg.matrix_rank(matrix)))
    assert ranks == [exact_rank(n,k) for k in range(n*n+1)]
    return ranks


def probe(n, objective, repetition, budget, phase, parameters=None, initial_epochs=1):
    parameters = dict(baseline.CONFIG if parameters is None else parameters)
    batch_size = parameters["k"]
    assert 400 % batch_size == 0 and initial_epochs >= 1
    seed = 20261008 + n*1000 + repetition*10 + (objective == "quadratic")
    path = OUT / f"{phase}_{objective}_{n}_{repetition}.json"
    config = dict(n=n, objective=objective, repetition=repetition, seed=seed,
                  seconds=budget, phase=phase, protes=parameters, initial_count=400,
                  protocol=1, upstream=baseline.UPSTREAM)
    if initial_epochs != 1:
        config["initial_epochs"] = initial_epochs
    if path.exists():
        result = json.loads(path.read_text())
        assert result["config"] == config
        return result
    # NEW diagnostic instances, separate from train and held-out test files.
    # Same objective distribution as the project's test generator, but NumPy RNG.
    rng = np.random.default_rng(seed)
    initial = np.zeros((400, n*n), dtype=np.int32)
    for row in initial:
        row[np.arange(n)*n + rng.permutation(n)] = 1
    if objective == "linear":
        value = rng.uniform(-50, 50, n*n)
    else:
        raw = rng.uniform(-7, 7, (n*n, n*n))
        value = (raw + raw.T)/2
    costs = initial @ value if objective == "linear" else np.einsum("bi,ij,bj->b", initial, value, initial)
    initial_set = {x.tobytes() for x in initial}
    novel = set()
    seen = {x.tobytes():float(y) for x,y in zip(initial, costs)}
    incumbent = float(costs.min())
    order_rng = np.random.default_rng(seed)
    orders = [order_rng.permutation(400) for _ in range(initial_epochs)]
    batches_per_epoch = 400 // batch_size
    total_initial_batches = initial_epochs * batches_per_epoch
    state = dict(initial_batches=0, generated=0, feasible=0, novel_draws=0,
                 first_novel_seconds=None, initial_training_complete=False)
    history = []
    started = time.perf_counter()
    cpu_start = time.process_time()
    current_phase = ["initial"]

    def external(P, k, ignored_seed, info):
        b = state["initial_batches"]
        if b < total_initial_batches:
            state["initial_batches"] += 1
            current_phase[0] = "initial"
            epoch, within = divmod(b, batches_per_epoch)
            return jax.numpy.asarray(initial[orders[epoch][within*k:(within+1)*k]])
        current_phase[0] = "generated"
        state["initial_training_complete"] = True
        return None

    def evaluate(I):
        nonlocal incumbent
        x = np.asarray(I, dtype=np.int32)
        elapsed = time.perf_counter()-started
        if elapsed >= budget:
            return None
        valid = baseline.feasible(x, n)
        if current_phase[0] == "generated":
            state["generated"] += len(x)
            state["feasible"] += int(valid.sum())
        scores = np.full(len(x), 2.0)
        for i in np.flatnonzero(valid):
            if time.perf_counter()-started >= budget:
                return None
            row = x[i]
            key = row.tobytes()
            if current_phase[0] == "generated" and key not in initial_set:
                novel.add(key)
                state["novel_draws"] += 1
                if state["first_novel_seconds"] is None:
                    state["first_novel_seconds"] = time.perf_counter()-started
            if key not in seen:
                seen[key] = float(row @ value if objective == "linear" else row @ value @ row)
            incumbent = min(incumbent, seen[key])
            scores[i] = 2/np.pi * np.arctan(seen[key])
        if current_phase[0] == "generated" and (not history or elapsed-history[-1][0] >= .5):
            history.append([elapsed, state["generated"], state["feasible"], len(novel)])
        return scores if valid.any() else []

    info = {}
    baseline.protes(evaluate, n*n, 2, seed=seed, info=info, sample_ext=external,
                    with_info_p=True, **parameters)
    jax.effects_barrier()
    elapsed = time.perf_counter()-started
    assert all(np.isfinite(np.asarray(core)).all() for core in info["P"])
    result = dict(config=config, **state, novel_unique=len(novel),
                  initial_unique=len(initial_set), unseen_feasible=math.factorial(n)-len(initial_set),
                  c_min=incumbent, initial_c_min=float(costs.min()), elapsed_seconds=elapsed,
                  cpu_seconds=time.process_time()-cpu_start, cpu_affinity=baseline.PROCESS.cpu_affinity(),
                  initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                  objective_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                  host=platform.node(), numpy=np.__version__, jax=jax.__version__, history=history)
    baseline.atomic_json(path, result)
    print(f"{phase}: {objective} n={n} rep={repetition}, unique initial={len(initial_set)}, "
          f"new={len(novel)}, feasible={state['feasible']}/{state['generated']}, t={elapsed:.2f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="4:10")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--phase", default="pilot")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for n in baseline.integers(args.sizes):
        baseline.warmup(n*n)
        for objective in ("linear", "quadratic"):
            for rep in range(args.repetitions):
                probe(n, objective, rep, args.seconds, args.phase)
