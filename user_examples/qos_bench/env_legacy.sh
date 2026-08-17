# Legacy (pre-Stage-0) data-plane parameters — control group.
#
# Usage:  source env_legacy.sh   (then run any qos_bench script)
#
# These were the defaults before the chunk size was reduced for H2D QoS work.
# A single 1 GiB WQE occupies the PCIe uplink for ~21 ms at 50 GB/s, which is
# ~500x longer than a 30-50 us TMA H2D window. Since a posted WQE cannot be
# revoked, no admission control can take effect at this granularity — this
# profile exists purely as a baseline to quantify that.

export MPCOMM_MAX_RDMA_TRANSFER_SIZE=1073741824   # 1 GiB
export MPCOMM_QPS_PER_CONNECTION=1
export MPCOMM_MAX_OUTSTANDING_PER_QP=256

# Per-NIC in-flight ceiling = outstanding x QPs x chunk
#                           = 256 x 1 x 1 GiB = 256 GiB  (effectively unbounded)

echo "[env_legacy] chunk=1GiB qps=1 outstanding=256 (in-flight ceiling: unbounded)"
