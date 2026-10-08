"""Export the exact NumPy diagnostic inputs for paired Julia experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/protes_comparison/scaling/paired_inputs'
OUT.mkdir(parents=True,exist_ok=True)
old=json.loads((ROOT/'examples/protes_scaling_results.json').read_text())['results']
lookup={(r['config']['n'],r['config']['objective'],r['config']['repetition']):r for r in old}
for n in range(4,17):
    for objective in ['linear','quadratic']:
        for rep in range(3):
            seed=20261008+n*1000+rep*10+(objective=='quadratic')
            rng=np.random.default_rng(seed)
            initial=np.zeros((400,n*n),dtype=np.int32)
            for row in initial: row[np.arange(n)*n+rng.permutation(n)]=1
            if objective=='linear': value=rng.uniform(-50,50,n*n)
            else:
                raw=rng.uniform(-7,7,(n*n,n*n));value=(raw+raw.T)/2
            ih=hashlib.sha256(initial.tobytes()).hexdigest()
            oh=hashlib.sha256(value.tobytes()).hexdigest()
            if (n,objective,rep) in lookup:
                ref=lookup[n,objective,rep]
                assert ih==ref['initial_sha256'] and oh==ref['objective_sha256']
            stem=OUT/f'{objective}_{n}_{rep}'
            initial.tofile(str(stem)+'.i32');value.tofile(str(stem)+'.f64')
            Path(str(stem)+'.sha256').write_text(ih+'\n'+oh+'\n')
print('Exported paired inputs; all historical hashes match.')
