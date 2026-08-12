#!/usr/bin/env bash
# NIC->GPU 基准 —— 目标端(TARGET). 在【被访问的远端服务器】上运行.
# 启动一个常驻 target, 发布一块 DRAM buffer, 等待 initiator 发起 RDMA.
#
# 用法:
#   ./run-bench-nic2gpu-target.sh [TCP_PORT] [DEVICE] [BUFFER_SIZE]
#
#   TCP_PORT     元数据交换端口   (默认 12345)
#   DEVICE       RDMA 设备名      (默认 空=自动检测, 例如 mlx5_bond_1,mlx5_bond_2)
#   BUFFER_SIZE  发布 buffer 大小 (默认 2G, 需>= initiator 测试的最大块)
#
# 说明:
#   - 本端buffer 是 DRAM (target 侧不必有 GPU); GPUDirect 发生在 initiator 端.
#   - 记下本机 IP, initiator 用<host_id>:<ip>:<port> 连接.
#
# 示例:
#   ./run-bench-nic2gpu-target.sh 12345 mlx5_bond_1,mlx5_bond_2 4G
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# mpcomm_target.py 位于上级 user_examples/ 目录
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TCP_PORT="${1:-12345}"
DEVICE="${2:-}"
BUFFER_SIZE="${3:-2G}"

# 取一个本机 IP 作为 host_id 前缀 (initiator 需要用到)
LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
LOCAL_IP="${LOCAL_IP:-127.0.0.1}"
HOST_ID="${MPCOMM_HOST_ID:-${LOCAL_IP}:${TCP_PORT}}"

echo "=============================================================="
echo " NIC->GPU 基准 TARGET 端"
echo " host_id: ${HOST_ID}"
echo " tcp_port    : ${TCP_PORT}"
echo " device      : ${DEVICE:-(auto)}"
echo " buffer_size : ${BUFFER_SIZE}"
echo "--------------------------------------------------------------"
echo " initiator 侧请使用如下 --target 参数连接:"
echo "     --target ${HOST_ID}"
echo "=============================================================="

python3 "${EXAMPLES_DIR}/mpcomm_target.py" \
    --host-id "${HOST_ID}" \
    --tcp-port "${TCP_PORT}" \
    --device "${DEVICE}" \
    --buffer-size "${BUFFER_SIZE}"
