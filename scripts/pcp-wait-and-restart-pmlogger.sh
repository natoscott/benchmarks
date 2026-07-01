#!/bin/bash
# Wait for all configured PCP metric sources to have live data, then
# restart pmlogger so its config picks up all available metrics.
#
# pmlogger evaluates pmlogconf once at startup. Any PMDA or openmetrics
# source not yet serving values at that point gets skipped. This script
# polls until all expected sources report data, then restarts pmlogger.
#
# Usage (inside a PCP container, after pmcd/pmlogger are running):
#   pcp-wait-and-restart-pmlogger.sh [metric1] [metric2] ...
#
# If no metrics are given, uses a built-in default set covering common
# benchmark sources (DCGM, vLLM, PostgreSQL).
#
# Can also be tested locally against any running pmcd:
#   ./pcp-wait-and-restart-pmlogger.sh kernel.all.cpu.user mem.util.used

. /etc/pcp.env

set -euo pipefail

POLL_INTERVAL="${PCP_PROBE_INTERVAL:-5}"
MAX_ATTEMPTS="${PCP_PROBE_ATTEMPTS:-90}"

_restart_pmlogger() {
    systemctl stop pmlogger
    rm -fr "/var/log/pcp/pmlogger/$(hostname)"
    systemctl start pmlogger
}

if [ $# -gt 0 ]; then
    PROBES=("$@")
else
    PROBES=(
        "openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL"
        "openmetrics.vllm.vllm:num_requests_running"
        "postgresql.stat.database.numbackends"
    )
fi

echo "pcp-wait-and-restart-pmlogger: waiting for ${#PROBES[@]} metric source(s)"
for m in "${PROBES[@]}"; do
    echo "  - ${m}"
done

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    ALL_LIVE=true
    MISSING=""
    for metric in "${PROBES[@]}"; do
        COUNT=$(pmprobe -v "${metric}" 2>/dev/null | awk '{print $2}')
        if [ "${COUNT:-0}" -lt 1 ]; then
            ALL_LIVE=false
            MISSING="${MISSING} ${metric}"
        fi
    done

    if [ "${ALL_LIVE}" = "true" ]; then
        echo "pcp-wait-and-restart-pmlogger: all sources live after $((attempt * POLL_INTERVAL))s"
        _restart_pmlogger
        exit 0
    fi

    if [ $((attempt % 6)) -eq 0 ]; then
        echo "  waiting... (${attempt}/${MAX_ATTEMPTS}, missing:${MISSING})"
    fi

    pmsleep "${POLL_INTERVAL}"
done

echo "pcp-wait-and-restart-pmlogger: timeout after $((MAX_ATTEMPTS * POLL_INTERVAL))s, restarting pmlogger anyway"
echo "  missing:${MISSING}"
_restart_pmlogger
exit 1
