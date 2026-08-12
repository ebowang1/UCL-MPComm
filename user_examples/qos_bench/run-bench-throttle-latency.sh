#!/usr/bin/env bash
# 运行时限速生效时延测试 (本地 GET 路径).
#
# 用法:
#   ./run-bench-throttle-latency.sh <TARGET> [GPU_ID] [SIZE] [DEVICE]
#
#   TARGET   <host_id>:<ip>:<port>, 例如 target143:29.209.105.143:12345
#   GPU_ID   本地 GPU 序号            (默认 0)
#   SIZE     单次 GET 大小            (默认 1G)
#   DEVICE   本地 RDMA 设备           (默认 mlx5_bond_1)
#
# 稳态与限速档位通过环境变量控制:
#   MPCOMM_QPS_PER_CONNECTION     稳态 QP 数            (建议 4)
#   MPCOMM_MAX_RDMA_TRANSFER_SIZE 稳态 chunk            (建议 65536)
#   MPCOMM_MAX_OUTSTANDING_PER_QP 稳态 outstanding      (建议 16)
#   MPCOMM_THROTTLE_OUTSTANDING限速时 outstanding(默认 1)
#   MPCOMM_THROTTLE_CHUNK         限速时 chunk, 0=不变(默认 0)
#   MPCOMM_THROTTLE_REPEAT        限速/解除循环次数     (默认 5)
#   MPCOMM_THROTTLE_MIN_SNAPSHOT_WQES
#                                  可选压力测试筛选：仅统计 t3 在途WQE不少于该值的样本
#                                  默认 0，保留所有实际触发时刻的样本
#
# 示例:
#   export MPCOMM_QPS_PER_CONNECTION=4
#   export MPCOMM_MAX_RDMA_TRANSFER_SIZE=65536
#   export MPCOMM_MAX_OUTSTANDING_PER_QP=16
#   ./run-bench-throttle-latency.sh target143:29.209.105.143:12345 0 1G mlx5_bond_1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <host_id>:<ip>:<port> [GPU_ID] [SIZE] [DEVICE]" >&2
    exit 1
fi

TARGET="$1"
GPU_ID="${2:-0}"
SIZE="${3:-1G}"
DEVICE="${4:-mlx5_bond_1}"

printf '[run-bench-throttle-latency] target=%s gpu=%s size=%s device=%s\n' \
    "${TARGET}" "${GPU_ID}" "${SIZE}" "${DEVICE}"

python3 "${SCRIPT_DIR}/bench_throttle_latency.py" \
    --target "${TARGET}" \
    --gpu "${GPU_ID}" \
    --size "${SIZE}" \
    --device "${DEVICE}" \
    --throttle-outstanding "${MPCOMM_THROTTLE_OUTSTANDING:-1}" \
    --throttle-chunk "${MPCOMM_THROTTLE_CHUNK:-0}" \
    --repeat "${MPCOMM_THROTTLE_REPEAT:-5}" \
    --min-snapshot-wqes "${MPCOMM_THROTTLE_MIN_SNAPSHOT_WQES:-0}"
