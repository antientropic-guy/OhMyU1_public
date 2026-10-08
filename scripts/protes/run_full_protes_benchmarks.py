"""Separate processes release dense arrays/JIT caches between problem sizes."""
import subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
for n in range(4,11):
    for variant in ['fast','general']:
        subprocess.run([sys.executable,str(here/'benchmark_full_protes.py'),'--n',str(n),'--variant',variant],check=True)
