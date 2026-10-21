#!/bin/bash
# The GPU-batch passes behind Table S27's DeepShift and LEGOLAS-GPU batch rows: LEGOLAS
# (eight preparation workers) and DeepShift's throughput arm, single model and ensemble,
# back to back. Waits until the machine is idle (1-minute load below 1.0 at three checks a
# minute apart, no GPU process), exactly as retime_idle_ledger.py does, then runs
#   legolas_batch_gpu_prep8   repeats 1, 2
#   deepshift_throughput_single   repeats 3, 4
#   deepshift_throughput_ens      repeats 2, 3
# Needs SI_SPEED_CENSUS and the LEGOLAS_* / REDUCE_BIN variables of run_all.sh.
set -eu
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
ok=0
while [ "$ok" -lt 3 ]; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if awk "BEGIN{exit !($load < 1.0)}" && [ "$gpu" = 0 ]; then ok=$((ok+1)); else ok=0; fi
  sleep 60
done
for r in 1 2; do
  "$PY" run_legolas_parallel_prep.py 8 "$r"
  "$PY" run_speed_throughput.py --members 1 --repeat $((r+2))
  "$PY" run_speed_throughput.py --members 5 --repeat $((r+1))
done
