#!/usr/bin/env bash
# Fixed-shape H2D benchmark: 2048 independent 272-byte blocks by default.
#
# Usage:
#   ./run-bench-h2d.sh [GPU_ID] [NUM_BLOCKS] [BLOCK_SIZE] [MODE] [OP]
#
# Examples:
#   ./run-bench-h2d.sh
#   ./run-bench-h2d.sh 0 2048 272 tma gather
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${1:-0}"
NUM_BLOCKS="${2:-2048}"
BLOCK_SIZE="${3:-272}"
MODE="${4:-tma}"
OP="${5:-both}"
CSV="${MPCOMM_BENCH_CSV:-h2d_${NUM_BLOCKS}x${BLOCK_SIZE}_gpu${GPU_ID}.csv}"

printf '[run-bench-h2d] gpu=%s blocks=%s block_size=%s mode=%s op=%s csv=%s\n' \
    "${GPU_ID}" "${NUM_BLOCKS}" "${BLOCK_SIZE}" "${MODE}" "${OP}" "${CSV}"

ARGS=(
    --path h2d
    --gpu "${GPU_ID}"
    --op "${OP}"
    --num-blocks "${NUM_BLOCKS}"
    --block-size "${BLOCK_SIZE}"
    --h2d-mode "${MODE}"
    --iterations "${MPCOMM_BENCH_ITERS:-50}"
    --report-every "${MPCOMM_REPORT_EVERY:-0}"
    --warmup "${MPCOMM_BENCH_WARMUP:-10}"
    --csv "${CSV}"
)

if [[ "${MPCOMM_CUDA_COPY_BASELINE:-1}" == "1" ]]; then
    ARGS+=(--cuda-copy-baseline)
fi

if [[ "${MPCOMM_HIDE_NATIVE_STDOUT:-1}" == "1" ]]; then
    MPCOMM_STATUS_STDERR=1 python3 "${SCRIPT_DIR}/bench_h2d_nic2gpu.py" "${ARGS[@]}" 1>/dev/null
else
    python3 "${SCRIPT_DIR}/bench_h2d_nic2gpu.py" "${ARGS[@]}"
fi
