"""Linux /proc monitor for one pinned Julia worker; no third-party dependencies."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    path = Path(sys.argv[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TIME_BUDGET_MONITOR=str(path.resolve()))
    process = subprocess.Popen(sys.argv[2:], env=env)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["unix_s", "pid", "rss_bytes", "anonymous_rss_bytes",
                         "process_hwm_bytes", "cpu_seconds"])
        try:
            while process.poll() is None:
                try:
                    status = {}
                    for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                        key, _, value = line.partition(":")
                        if key in ("VmRSS", "RssAnon", "VmHWM"):
                            status[key] = int(value.split()[0]) * 1024
                    fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                    cpu = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
                    writer.writerow([time.time(), process.pid, status.get("VmRSS", 0),
                                     status.get("RssAnon", 0), status.get("VmHWM", 0), cpu])
                    stream.flush()
                except (FileNotFoundError, ProcessLookupError):
                    pass
                time.sleep(1)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())
