"""Check the unmodified public PROTES entry points with JAX sparse cores."""
import importlib.util
import json
from pathlib import Path
import psutil
process=psutil.Process()
process.cpu_affinity([max(process.cpu_affinity())])
import assignment_scaling as experiment
from jax.experimental import sparse

jax,np=experiment.jax,experiment.np
spec=importlib.util.spec_from_file_location("upstream_general",experiment.ROOT/"PROTES/protes/protes_general.py")
general=importlib.util.module_from_spec(spec);spec.loader.exec_module(general)
key=jax.random.PRNGKey(123)
fast=experiment.baseline._module._generate_initial(4,2,2,key)
ordinary=[fast[0],fast[1][0],fast[1][1],fast[2]]
out={}
for name,fn,dense in [("protes",experiment.baseline.protes,fast),("protes_general",general.protes_general,ordinary)]:
    for storage in ("dense","BCOO"):
        P=[sparse.BCOO.fromdense(x) for x in dense] if storage=="BCOO" else dense
        kwargs=dict(P=P,m=20,k=10,k_top=2,info={})
        try:
            if name=="protes": fn(lambda x:np.asarray(x).sum(axis=1),4,2,**kwargs)
            else: fn(lambda x:np.asarray(x).sum(axis=1),[2]*4,**kwargs)
            out[f"{name}_{storage}"]={"success":True}
        except Exception as exc:
            out[f"{name}_{storage}"]={"success":False,"exception":type(exc).__name__,"message":str(exc)[:1000]}
print(json.dumps(out,indent=2))
variable=[jax.numpy.ones(shape) for shape in [(1,2,2),(2,2,3),(3,2,2),(2,2,1)]]
general.protes_general(lambda x:np.asarray(x).sum(axis=1),[2]*4,
                      P=variable,m=20,k=10,k_top=2,info={})
out["protes_general_variable_dense_ranks"]={"success":True,"ranks":[1,2,3,2,1]}
path=experiment.OUT/"sparse_support.json"
path.write_text(json.dumps(out,indent=2))
assert out["protes_dense"]["success"] and out["protes_general_dense"]["success"]
assert not out["protes_BCOO"]["success"] and not out["protes_general_BCOO"]["success"]
