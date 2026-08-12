#!/usr/bin/env bash
# NIC->GPU 基准 —— 发起端(INITIATOR). 在【有 GPU 的本机】上运行.
# GPU HBM 作 local buffer, 经本地网卡与远端 target 做 GPUDirect RDMA.
#   get = 远端 NIC -> 本地 GPU (读带宽/时延)
#   put = 本地 GPU -> 远端      (写带宽/时延)
#
# 用法:
#   ./run-bench-nic2gpu-initiator.sh <TARGET> [GPU_ID] [OP] [SIZES] [DEVICE]
#
#   TARGET   目标: host_id:tcp_addr:tcp_port  (从 target 端输出复制, 必填)
#   GPU_ID   本机 CUDA 设备号                 (默认 0)
#   OP       get / put / both                 (默认 both)
#   SIZES    块大小扫描列表                   (默认 512K,1M,4M,16M,64M,256M,1G)
#   DEVICE   本机 RDMA 设备名                 (默认 空=自动检测)
#
# 前置条件:
#   1. 目标机已运行 run-bench-nic2gpu-target.sh
#   2. 本机已 modprobe nvidia-peermem  (lsmod | grep nvidia_peermem)
#   3. 本机 GPU 为可用 CUDA 设备
#
# 示例:
#   ./run-bench-nic2gpu-initiator.sh 10.0.0.2:10.0.0.2:12345 0 both
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="${1:-}"
GPU_ID="${2:-0}"
OP="${3:-both}"
SIZES="${4:-512K,1M,4M,16M,64M,256M,1G}"
DEVICE="${5:-}"
CSV="${MPCOMM_BENCH_CSV:-nic2gpu_bench_gpu${GPU_ID}.csv}"

if [[ -z "${TARGET}" ]]; then
    echo "用法: $0 <host_id:tcp_addr:tcp_port> [GPU_ID] [OP] [SIZES] [DEVICE]" >&2
    echo "  例: $0 10.0.0.2:10.0.0.2:12345 0 both" >&2
    exit 1
fi

# 检查 nvidia-peermem (GPUDirect RDMA 依赖)
if ! lsmod 2>/dev/null | grep -q nvidia_peermem; then
    echo "[warn] 未检测到 nvidia_peermem 内核模块, GPU 内存注册可能失败!" >&2
    echo "[warn] 请先: sudo modprobe nvidia-peermem" >&2
fi

echo "[run-bench-nic2gpu-initiator] target=${TARGET} GPU=${GPU_ID} op=${OP} sizes=${SIZES} csv=${CSV}"

python3 "${SCRIPT_DIR}/bench_h2d_nic2gpu.py" \
    --path nic2gpu \
    --gpu "${GPU_ID}" \
    --op "${OP}" \
    --sizes "${SIZES}" \
    --target "${TARGET}" \
    --device "${DEVICE}" \
    --host-id "${MPCOMM_HOST_ID:-initiator:0}" \
    --iters "${MPCOMM_BENCH_ITERS:-50}" \
    --warmup "${MPCOMM_BENCH_WARMUP:-10}" \
    --report-every "${MPCOMM_REPORT_EVERY:-0}" \
    --csv "${CSV}"
