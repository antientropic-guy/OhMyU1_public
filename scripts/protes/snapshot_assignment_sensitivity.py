"""Versionable inputs/results summary for the notebook sensitivity extension."""
from pathlib import Path
import hashlib
import json
import shutil
root=Path(__file__).resolve().parents[2]
out=root/"data/protes_comparison/scaling"
records=[json.loads(p.read_text()) for p in sorted(out.glob("sensitivity_*.json"))]
assert len([r for r in records if r["config"]["phase"].endswith("_short")])==56
assert len([r for r in records if r["config"]["phase"].endswith("_long")])==24
assert all(r["initial_training_complete"] for r in records)
sources=[root/"scripts/protes"/name for name in
         ["assignment_scaling.py","assignment_sensitivity.py","assignment_memory.jl","check_sparse_support.py"]]
data=dict(date="2026-10-08",hardware=json.loads((out/"hardware.json").read_text()),
          sparse_support=json.loads((out/"sparse_support.json").read_text()),
          source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
          results=[{k:v for k,v in r.items() if k!="history"} for r in records])
(root/"examples/protes_sensitivity_results.json").write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
shutil.copyfile(out/"memory.csv",root/"examples/protes_scaling_memory.csv")
print("Saved 80 additional trials, hardware, sparse-input checks and actual MPS memory")
