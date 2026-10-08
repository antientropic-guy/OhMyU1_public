#!/usr/bin/env python3
"""Time-matched, single-CPU PROTES on the exact EELS binary datasets.

No permutation encoding, feasible mask, repair, or structural TT is supplied
to PROTES. Feasibility is known only to the external scoring function.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Must precede importing numpy/JAX and spawning Julia.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "JULIA_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import psutil
PROCESS = psutil.Process()
PROCESS.cpu_affinity([min(PROCESS.cpu_affinity())])
CPU = PROCESS.cpu_affinity()[0]

import numpy as np
import jax

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Load the unmodified upstream module without importing its optional plotting
# helpers (package __init__ imports matplotlib/seaborn even for headless runs).
_spec = importlib.util.spec_from_file_location("upstream_protes", HERE / "protes/protes.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
protes = _module.protes
UPSTREAM = "06d447845dd26ffccbc8e95c7545fa1d2617940e"
CONFIG = dict(k=100, k_top=10, k_gd=1, lr=0.05, r=5)
jax.config.update("jax_compilation_cache_dir", str(HERE / ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def integers(value):
    parts = list(map(int, value.split(":")))
    return list(range(parts[0], parts[-1] + 1))


def read_tsv(path):
    with path.open(newline="") as stream:
        return next(csv.DictReader(stream, delimiter="\t"))


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_input(folder):
    meta = read_tsv(folder / "metadata.tsv")
    n = int(meta["n"])
    raw = (folder / "initial.u8").read_bytes()
    # Julia vec(initial): each complete binary sample is contiguous.
    initial = np.frombuffer(raw, dtype=np.uint8).reshape(400, n*n).astype(np.int32)
    fingerprint = hashlib.sha256(f"({n*n}, 400):".encode() + raw).hexdigest()
    assert fingerprint == meta["initial_sha256"], "Julia/Python initial data differ"
    value = np.fromfile(folder / "objective.f64", dtype="<f8")
    if meta["objective"] == "quadratic":
        value = value.reshape((n*n, n*n), order="F")
    costs = np.loadtxt(folder / "initial_costs.csv", delimiter=",")
    actual = initial @ value if value.ndim == 1 else np.einsum("bi,ij,bj->b", initial, value, initial)
    np.testing.assert_allclose(actual, costs, rtol=0, atol=1e-9)
    assert np.all(feasible(initial, n))
    return meta, initial, value, costs


def feasible(samples, n):
    matrix = samples.reshape(-1, n, n)
    return ((matrix.sum(axis=1) == 1).all(axis=1)
            & (matrix.sum(axis=2) == 1).all(axis=1)
            & ((samples == 0) | (samples == 1)).all(axis=1))


def warmup(d):
    """Compile upstream sampling and training on synthetic data; no test tuning."""
    start = time.perf_counter()
    protes(lambda x: np.asarray(x).sum(axis=1), d, 2, m=200,
           info={}, seed=0, **CONFIG)
    jax.effects_barrier()
    return time.perf_counter() - start


def run_one(folder, output, seconds_override=None):
    meta, initial, value, initial_costs = load_input(folder)
    n, seed = int(meta["n"]), int(meta["seed"])
    reference = read_tsv(folder / "eels_timing.tsv")
    budget = float(reference["seconds"]) if seconds_override is None else seconds_override
    signature = dict(protocol=1, upstream=UPSTREAM, config=CONFIG, metadata=meta,
                     budget_seconds=budget, eels_seconds=float(reference["seconds"]))
    if output.exists():
        saved = json.loads(output.read_text())
        if saved["signature"] != signature:
            raise RuntimeError(f"Checkpoint mismatch: {output}")
        print(f"Already complete: {output.name}", flush=True)
        return saved

    # Fixed initial exposure: one shuffled pass over exactly the shared 400
    # points in 4 batches of 100; upstream performs its usual top-10 update.
    order = np.random.default_rng(seed).permutation(len(initial))
    start = time.perf_counter()
    cpu_start = time.process_time()
    cache = {row.tobytes(): float(y) for row, y in zip(initial, initial_costs)}
    best_index = int(initial_costs.argmin())
    incumbent = initial[best_index].copy()
    best = float(initial_costs[best_index])
    state = dict(initial_batches=0, generated=0, feasible_generated=0,
                 unique_evaluations=0, rejected_after_deadline=0,
                 mixed_batches_with_fewer_than_10_feasible=0, all_infeasible_batches=0)
    curve = []
    phase = ["initial"]

    def external(P, k, unused_seed, info):
        batch = state["initial_batches"]
        if batch < 4:
            state["initial_batches"] += 1
            phase[0] = "initial"
            return jax.numpy.asarray(initial[order[batch*k:(batch+1)*k]])
        phase[0] = "generated"
        return None

    def objective(I):
        nonlocal best, incumbent
        samples = np.asarray(I, dtype=np.int32)  # synchronizes sampling
        elapsed = time.perf_counter() - start
        if elapsed >= budget:
            state["rejected_after_deadline"] += len(samples)
            return None
        valid = feasible(samples, n)
        count = int(valid.sum())
        if phase[0] == "generated":
            state["generated"] += len(samples)
            state["feasible_generated"] += count
            state["mixed_batches_with_fewer_than_10_feasible"] += int(0 < count < CONFIG["k_top"])
            state["all_infeasible_batches"] += int(count == 0)
        # Only ranking is used by PROTES. This bounded monotone mapping places
        # EVERY feasible objective below the fixed invalid score 2. It avoids
        # a problem-specific penalty coefficient or access to A in the model.
        scores = np.full(len(samples), 2.0)
        for index in np.flatnonzero(valid):
            if time.perf_counter() - start >= budget:
                # Do not train on a partially evaluated batch.
                return None
            row = samples[index]
            key = row.tobytes()
            if key not in cache:
                cache[key] = float(row @ value if value.ndim == 1 else row @ value @ row)
                state["unique_evaluations"] += 1
            cost = cache[key]
            if not np.isfinite(cost):
                raise ValueError("Nonfinite objective")
            scores[index] = (2 / np.pi) * np.arctan(cost)
            if cost < best:
                best, incumbent = cost, row.copy()
        curve.append(dict(seconds=time.perf_counter()-start, best=best,
                          phase=phase[0], feasible=count, batch=len(samples),
                          evaluations=state["unique_evaluations"]))
        # No feasible training signal: upstream's documented empty result skips
        # this update. Otherwise retain its exact top-k logic, including invalid
        # points if fewer than k_top are feasible (recorded above).
        return scores if count else []

    info = {}
    protes(objective, n*n, 2, m=None, seed=seed, info=info,
           sample_ext=external, with_info_p=True, **CONFIG)
    jax.effects_barrier()
    elapsed = time.perf_counter() - start
    assert feasible(incumbent[None, :], n)[0]
    np.testing.assert_allclose(best, float(incumbent @ value if value.ndim == 1 else incumbent @ value @ incumbent), rtol=0, atol=1e-9)
    model_finite = all(np.isfinite(np.asarray(core)).all() for core in info["P"])
    if not model_finite:
        raise FloatingPointError("PROTES model contains NaN/Inf; refusing to save a successful run")
    result = dict(signature=signature, objective=meta["objective"], n=n,
                  instance=int(meta["instance"]), seed=seed, c_min=best,
                  initial_c_min=float(initial_costs.min()), incumbent=incumbent.tolist(),
                  elapsed_seconds=elapsed, cpu_seconds=time.process_time()-cpu_start,
                  deadline_overshoot_seconds=max(0, elapsed-budget),
                  eels_seconds=float(reference["seconds"]), eels_c_min=float(reference["c_min"]),
                  time_ratio=elapsed/float(reference["seconds"]),
                  initial_training_complete=state["initial_batches"] == 4 and len(curve) >= 4,
                  diagnostics=state, curve=curve, cpu_affinity=PROCESS.cpu_affinity(),
                  host=platform.node(), platform=platform.platform(),
                  python=sys.version, numpy=np.__version__, jax=jax.__version__,
                  device=str(jax.devices()), upstream_reported_seconds=info["t"])
    atomic_json(output, result)
    print(f"{meta['objective']} n={n} i={meta['instance']}: EELS {reference['seconds']}s, "
          f"PROTES {elapsed:.3f}s, best {best}, feasible "
          f"{state['feasible_generated']}/{state['generated']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/protes_comparison/server")
    parser.add_argument("--sizes", default="12:16")
    parser.add_argument("--instances", default="1:50")
    parser.add_argument("--objectives", default="linear,quadratic")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--seconds", type=float, help="Diagnostic override only; do not use for the main experiment")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM:
        raise RuntimeError(f"Unexpected PROTES revision {commit}")
    if subprocess.check_output(["git", "-C", str(HERE), "diff", "HEAD", "--", "protes/protes.py"], text=True):
        raise RuntimeError("Upstream protes.py has local modifications")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f"{args.objectives.replace(',', '-')}_{args.sizes.replace(':', '-')}_{args.instances.replace(':', '-')}"
    lock = args.output / (tag + ".running")
    lock.mkdir()  # separate group locks; stale locks require manual inspection
    try:
        manifest = dict(host=platform.node(), cpu=platform.processor(), platform=platform.platform(),
                        upstream=commit, config=CONFIG, jax=jax.__version__, numpy=np.__version__,
                        source_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in [HERE / "run_assignment.py", HERE / "prepare_assignment.jl",
                                                    ROOT / "scripts/run_EELS_optimization.jl", *sorted((ROOT / "src").glob("*.jl"))]})
        manifest_path = args.output / "experiment_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Experiment code/environment/host changed; use a new --output directory")
        atomic_json(manifest_path, manifest)
        print(f"Pinned to logical CPU {CPU}; CPU backend {jax.devices()}", flush=True)
        if not args.skip_prepare:
            subprocess.run(["julia", "--startup-file=no", f"--project={ROOT}", "--threads=1",
                            str(HERE / "prepare_assignment.jl"), "--output", str(args.output),
                            "--sizes", args.sizes, "--instances", args.instances,
                            "--objectives", args.objectives, "--timing", "all"], check=True)
        records = []
        warm = {}
        for n in integers(args.sizes):
            warm[str(n)] = warmup(n*n)
            for name in args.objectives.split(","):
                if name not in ("linear", "quadratic"):
                    parser.error("Unknown objective")
                for i in integers(args.instances):
                    stem = f"{name}_{n}_{i}"
                    records.append(run_one(args.output / "inputs" / stem,
                                           args.output / (stem + ".json"), args.seconds))
        atomic_json(args.output / (tag + "_warmup.json"), warm)
        fields = ["objective", "n", "instance", "seed", "initial_c_min", "c_min", "eels_c_min",
                  "eels_seconds", "elapsed_seconds", "cpu_seconds", "time_ratio", "deadline_overshoot_seconds"]
        with (args.output / (tag + "_summary.csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields + ["generated", "feasible_generated", "unique_evaluations"])
            writer.writeheader()
            for record in records:
                writer.writerow({**{key: record[key] for key in fields},
                                 **{key: record["diagnostics"][key] for key in ("generated", "feasible_generated", "unique_evaluations")}})
    finally:
        lock.rmdir()


if __name__ == "__main__":
    main()
