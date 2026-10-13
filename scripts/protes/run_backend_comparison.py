"""Run missing U1 backend measurements sequentially on one logical CPU.

Existing PROTES and ConstrainTNet measurements are retained. The archived full
table-sampler audit is reusable only if its inputs and library source match.
"""
import csv
import os
import hashlib
import json
from datetime import date
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'scripts/protes/run_julia_single_cpu.py'
OUT = ROOT / 'data/protes_comparison/scaling'

def run(script, *args):
    return subprocess.check_output(
        [sys.executable, str(RUNNER), str(ROOT / script), *args],
        cwd=ROOT, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})

if __name__ == '__main__':
    collected = []
    for backend in ['legacy', 'table']:
        output = run('scripts/benchmark_assignment.jl', '--backend=ohmy',
                     f'--sampling-backend={backend}')
        print(output, flush=True)
        for line in output.splitlines():
            if line.startswith('BENCH_RESULT|'):
                collected.append(dict(field.split('=', 1) for field in line.split('|')[1:]))
    with (ROOT / 'examples/assignment_benchmark.csv').open() as f:
        for row in csv.DictReader(f):
            if row['backend'] == 'ConstrainTNet':
                row['sampling_backend'] = 'upstream'
                collected.append(row)
    assert len(collected) == 21
    fields = list(collected[0])
    with (ROOT / 'examples/assignment_backend_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(collected)
    metadata = dict(date=date.today().isoformat(),sampling_backends=['legacy','table'],
        cpu_affinity=[0],julia_threads=1,blas_threads=1,constraintnet_results_reused=True,
        seed=20260923,training_samples=400,sampling_batch=400,repetitions=3,
        library_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.jl'))},
        note='Training and sampling medians are measured separately; training algorithm is unchanged between sampler variants.')
    (ROOT/'examples/assignment_backend_comparison_metadata.json').write_text(json.dumps(metadata,indent=2))
    for stage in ['short', 'long']:
        print(run('scripts/protes/benchmark_data_u1.jl', stage, 'table'), flush=True)
