"""Separate the deep memory audit from the final sampling-rate measurements."""
from pathlib import Path
import shutil,subprocess,sys
root=Path(__file__).resolve().parents[2]
runner=root/'scripts/protes/run_julia_single_cpu.py'
benchmark=root/'scripts/protes/benchmark_full_u1.jl'
out=root/'data/protes_comparison/scaling'
subprocess.run([sys.executable,str(runner),str(benchmark)],cwd=root,check=True)
audit=out/'full_u1_initial.csv'
shutil.copyfile(out/'full_u1.csv',audit)
subprocess.run([sys.executable,str(runner),str(benchmark),'4',str(audit)],cwd=root,check=True)
