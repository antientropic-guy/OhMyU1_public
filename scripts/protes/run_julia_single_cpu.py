"""Pin Julia and all inherited threads to the same logical CPU as PROTES."""
import os,subprocess,sys
import psutil
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JULIA_NUM_THREADS']:
    os.environ[name]='1'
p=psutil.Process();p.cpu_affinity([min(p.cpu_affinity())])
raise SystemExit(subprocess.call(['julia','--project=.','--threads=1',*sys.argv[1:]]))
