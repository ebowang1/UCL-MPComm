#!/usr/bin/env bash
# Print receive throughput of an RDMA device from its InfiniBand port counter.
# port_rcv_data is defined in 32-bit words; convert delta words to bytes.
# For the concurrent experiment, mlx5_bond_1 RX corresponds to NIC->GPU RDMA GET
# when no other workload uses this bond.
#
# Usage:
#   ./monitor_rdma_rx_bw.sh [RDMA_DEVICE] [INTERVAL_SECONDS]
# Example:
#   ./monitor_rdma_rx_bw.sh mlx5_bond_1 1
set -euo pipefail

DEVICE="${1:-mlx5_bond_1}"
INTERVAL="${2:-1}"
COUNTER="/sys/class/infiniband/${DEVICE}/ports/1/counters/port_rcv_data"

if [[ ! -r "${COUNTER}" ]]; then
    echo "ERROR: cannot read ${COUNTER}" >&2
    exit 1
fi

if ! [[ "${INTERVAL}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "${INTERVAL}" == "0" ]]; then
    echo "ERROR: interval must be a positive number of seconds" >&2
    exit 1
fi

previous_words="$(<"${COUNTER}")"
previous_ns="$(date +%s%N)"

echo "[NIC-LIVE] device=${DEVICE} direction=RX interval=${INTERVAL}s"
echo "[NIC-LIVE] port_rcv_data uses 32-bit words; displayed throughput is aggregate bond RX"

while true; do
    sleep "${INTERVAL}"
    current_words="$(<"${COUNTER}")"
    current_ns="$(date +%s%N)"

    elapsed_ns=$((current_ns - previous_ns))
    delta_words=$((current_words - previous_words))

    # Guard against a counter reset or a device restart.
    if (( delta_words < 0 )); then
        echo "[NIC-LIVE] counter reset detected; restarting interval"
        previous_words="${current_words}"
        previous_ns="${current_ns}"
        continue
    fi

    awk -v words="${delta_words}" -v elapsed_ns="${elapsed_ns}" -v now="$(date '+%H:%M:%S')" '
        BEGIN {
            bytes = words * 4
            seconds = elapsed_ns / 1000000000.0
            gbs = bytes / seconds / 1000000000.0
            gbps = gbs * 8.0
            printf "[NIC-LIVE] time=%s interval=%.3fs rx=%.2f GB/s (%.2f Gb/s)\n", now, seconds, gbs, gbps
        }
    '

    previous_words="${current_words}"
    previous_ns="${current_ns}"
done
