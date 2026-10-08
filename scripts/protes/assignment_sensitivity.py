"""Prespecified sensitivity configurations; same data, CPU and 5/50 s budgets."""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import assignment_scaling as experiment

CONFIGURATIONS = {
    "rank20": dict(r=20,k=100,k_top=10,k_gd=1,lr=.05,initial_epochs=1),
    "slow10": dict(r=5,k=100,k_top=10,k_gd=10,lr=.01,initial_epochs=1),
    "batch400": dict(r=10,k=400,k_top=40,k_gd=10,lr=.01,initial_epochs=1),
    "fit20": dict(r=5,k=100,k_top=10,k_gd=1,lr=.05,initial_epochs=20),
}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["short","long"],required=True)
    parser.add_argument("--configuration",choices=CONFIGURATIONS)
    args=parser.parse_args()
    experiment.OUT.mkdir(parents=True,exist_ok=True)
    selected=[args.configuration] if args.configuration else list(CONFIGURATIONS)
    # Broad first-seed screen, then three matched seeds at the previous failure
    # size, for EVERY configuration (not just the best one).
    sizes=range(4,11) if args.stage=="short" else [8]
    repetitions=range(1) if args.stage=="short" else range(3)
    budget=5 if args.stage=="short" else 50
    for name in selected:
        settings=CONFIGURATIONS[name].copy()
        epochs=settings.pop("initial_epochs")
        for n in sizes:
            start=time.perf_counter()
            experiment.baseline.protes(lambda x: experiment.np.asarray(x).sum(axis=1),
                n*n,2,m=2*settings["k"],seed=0,info={},**settings)
            experiment.jax.effects_barrier()
            print(f"Warmed {name}, n={n}: {time.perf_counter()-start:.2f} s",flush=True)
            for objective in ("linear","quadratic"):
                for rep in repetitions:
                    experiment.probe(n,objective,rep,budget,f"sensitivity_{name}_{args.stage}",
                                     parameters=settings,initial_epochs=epochs)

if __name__=="__main__": main()
