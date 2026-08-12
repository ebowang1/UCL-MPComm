#!/usr/bin/env bash
# 验证 H2D 触发的 NIC QoS (下沉到 MPComm C++).
#
# QoS 逻辑已内置于 MPComm: 调用 enable_h2d_qos后, 每次 tma_gather 会在其
# H2D kernel 运行期间自动限速后台 RDMA, kernel 完成后自动恢复. 本脚本只负责
# 打开开关并调用原生 tma_gather, 不含任何 QoS 逻辑.
#
# 用法:
#   ./run-bench-h2d-qos.sh <TARGET> [GPU_ID] [SIZE] [DEVICE]
#
#   TARGET   <host_id>:<ip>:<port>, 例如 target143:29.209.105.143:12345
#   GPU_ID   本地 GPU 序号            (默认 0)
#   SIZE     单次后台 GET 大小        (默认 1G)
#   DEVICE   本地 RDMA 设备           (默认 mlx5_bond_1)
#
# 环境变量:
#   MPCOMM_QPS_PER_CONNECTION     稳态 QP 数            (默认 4)
#   MPCOMM_MAX_RDMA_TRANSFER_SIZE 稳态 chunk            (默认 65536)
#   MPCOMM_MAX_OUTSTANDING_PER_QP 稳态 outstanding      (默认 16)
#   MPCOMM_H2D_QOS_RATE_BPS       H2D 活跃时匹配 RDMA 的总 post 速率
#                                  (默认 0，表示暂停新的匹配 post)
#   MPCOMM_H2D_QOS_BURST_BYTES    token bucket burst (默认 0，按 rate 自动推导)
#   MPCOMM_H2D_BLOCK_SIZE         H2D block 大小        (默认 1024)
#   MPCOMM_H2D_NUM_BLOCKS         H2D block 数(默认 10240 = 10MiB)
#   MPCOMM_H2D_POOL_MULTIPLIER    源 DRAM pool 倍数(默认 4，与run-bench-h2d一致)
#   MPCOMM_H2D_MODE               0=AUTO 1=SM 2=TMA     (默认 2)
#   MPCOMM_H2D_REPEAT             单次 H2D 试验次数      (默认 30)
#   MPCOMM_H2D_GAP                试验间隔秒(NIC 回稳)   (默认 0.05)
#
# 示例:
#   export MPCOMM_QPS_PER_CONNECTION=4
#   export MPCOMM_MAX_RDMA_TRANSFER_SIZE=65536
#   export MPCOMM_MAX_OUTSTANDING_PER_QP=16
#   ./run-bench-h2d-qos.sh target143:29.209.105.143:12345 0 1G mlx5_bond_1
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

# Keep baseline chunks reasonably small so post-rate pacing can react quickly
# and the amount of RDMA work already posted before an H2D stays bounded.
export MPCOMM_QPS_PER_CONNECTION="${MPCOMM_QPS_PER_CONNECTION:-4}"
export MPCOMM_MAX_RDMA_TRANSFER_SIZE="${MPCOMM_MAX_RDMA_TRANSFER_SIZE:-65536}"
export MPCOMM_MAX_OUTSTANDING_PER_QP="${MPCOMM_MAX_OUTSTANDING_PER_QP:-16}"

printf '[run-bench-h2d-qos] target=%s gpu=%s size=%s device=%s qps=%s chunk=%s outstanding=%s\n' \
    "${TARGET}" "${GPU_ID}" "${SIZE}" "${DEVICE}" \
    "${MPCOMM_QPS_PER_CONNECTION}" "${MPCOMM_MAX_RDMA_TRANSFER_SIZE}" \
    "${MPCOMM_MAX_OUTSTANDING_PER_QP}"

python3 "${SCRIPT_DIR}/bench_h2d_qos.py" \
    --target "${TARGET}" \
    --gpu "${GPU_ID}" \
    --size "${SIZE}" \
    --device "${DEVICE}" \
    --block-size "${MPCOMM_H2D_BLOCK_SIZE:-1024}" \
    --num-blocks "${MPCOMM_H2D_NUM_BLOCKS:-10240}" \
    --pool-multiplier "${MPCOMM_H2D_POOL_MULTIPLIER:-4}" \
    --mode "${MPCOMM_H2D_MODE:-2}" \
    --qos-rate-bps "${MPCOMM_H2D_QOS_RATE_BPS:-0}" \
    --qos-burst-bytes "${MPCOMM_H2D_QOS_BURST_BYTES:-0}" \
    --repeat "${MPCOMM_H2D_REPEAT:-30}" \
    --gap-seconds "${MPCOMM_H2D_GAP:-0.05}" \
    --h2d-only-warmup "${MPCOMM_H2D_ONLY_WARMUP:-100}" \
    --h2d-only-iters "${MPCOMM_H2D_ONLY_ITERS:-1000}"
