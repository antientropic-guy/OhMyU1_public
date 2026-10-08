"""Keep a compact, versionable evidence snapshot beside the analysis notebook."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

root = Path(__file__).resolve().parents[2]
out = root / "data/protes_comparison/scaling"
records = [json.loads(path.read_text()) for path in sorted(out.glob("*.json"))
           if path.name.startswith(("pilot_", "long_"))]
assert len([r for r in records if r["config"]["phase"] == "pilot"]) == 42
assert len([r for r in records if r["config"]["phase"] == "long"]) == 12
assert all(r["initial_training_complete"] for r in records)
sources = [root / "scripts/protes/assignment_scaling.py",
           root / "scripts/protes/assignment_rank_audit.jl",
           root / "PROTES/run_assignment.py", root / "PROTES/protes/protes.py",
           root / "src/mps_core.jl", root / "src/diversification.jl"]
snapshot = dict(date="2026-10-08", description="Exploratory diagnostic, not the held-out publication test set",
                protes_commit=subprocess.check_output(["git", "-C", str(root / "PROTES"), "rev-parse", "HEAD"], text=True).strip(),
                source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                results=[{k:v for k,v in r.items() if k != "history"} for r in records])
(root / "examples/protes_scaling_results.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8")
shutil.copyfile(out / "u1_rank_audit.csv", root / "examples/protes_scaling_u1_ranks.csv")
print("Saved 54 runs and measured Julia sector profiles")
