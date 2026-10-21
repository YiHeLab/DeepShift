#!/bin/bash
# The per-invocation and cohort-batch passes behind Table S27, serially. ONE ARM AT A TIME.
#
# Gate: before every arm, wait until the machine is idle (no CPU-heavy job; >= 90 % CPU idle
# over three consecutive 3-second vmstat samples). A timing taken under contention is not
# reportable.
#
# Steady state, uniformly: before each arm its parameter/model tree is read into the page
# cache, then run_speed_census.py runs two off-cohort structures and discards them. Without
# this ProCS15 reads 25.7 GB from disk per invocation and measures 62 s per protein instead
# of its warm 16.8 s, a page-cache artifact and not a property of the method.
#
# Needs SI_SPEED_CENSUS, the tool variables of experiments/competitors (SPARTA_HOME,
# NAPSHIFT_*, LEGOLAS_*, NMRGNN_BIN, PROCS15_*, UCBSHIFT_*, SHIFTX2_*, REDUCE_BIN) and
# PYTHON (the interpreter that runs the census scripts; python3 by default).
#
# The two DeepShift per-invocation rows and the REDUCE-using batch rows are retimed on an
# idle machine by retime_idle_ledger.py; the GPU batch rows by retime_idle.sh.
set -u
HERE=$(dirname "$(readlink -f "$0")")
PY=${PYTHON:-python3}

prewarm() {
    case "$1" in
    procs15)      cat "$PROCS15_DATA"/* ;;
    ucbshift_*)   cat "$UCBSHIFT_HOME"/models/*.sav ;;
    legolas_*)    cat "$LEGOLAS_HOME"/*/*.pt 2>/dev/null ;;
    *) return 0 ;;
    esac >/dev/null 2>&1
}

wait_idle() {
    local idle
    while idle=$(vmstat 3 4 | tail -3 | awk '{s+=$15; n++} END {print int(s/n)}'); [ "$idle" -lt 90 ]; do
        echo "$(date -Is) WAIT: cpu_idle=${idle}%"
        sleep 60
    done
}

run() {
    wait_idle
    prewarm "$1"
    echo "$(date -Is) START $1"
    "$PY" "$HERE/run_speed_census.py" --arm "$1"
    echo "$(date -Is) END   $1 rc=$?"
}

for arm in ucbshift_x sparta shiftx2_noshifty legolas_cpu napshift nmrgnn_cpu procs15; do
    run "$arm"
done
for arm in shiftx2_batch napshift_batch procs15_batch; do
    wait_idle
    "$PY" "$HERE/run_speed_batch.py" --arm "$arm"
done
