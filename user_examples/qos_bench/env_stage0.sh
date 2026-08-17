# Stage 0 data-plane parameters + per-PCIe-domain instrumentation.
#
# Usage:  source env_stage0.sh   (then run any qos_bench script)
#
# Chunk size is the key knob: one chunk == one WQE, and a posted WQE cannot be
# revoked, so chunk_size / BW_pcie bounds how long a competing TMA H2D window
# must wait for the PCIe uplink.
#
#   chunk    residual @ 50 GB/s
#   -----    ------------------
#   1 GiB    ~21   ms      <- legacy, hopeless vs a 30-50 us window
#   128 KiB  ~2.6  us
#   64 KiB   ~1.3  us
#
# QPs are raised to keep the NIC pipeline deep enough after shrinking chunks.
# The in-flight ceiling must stay above the bandwidth-delay product, otherwise
# D2D throughput drops:
#
#   RTT      ~12.2 us   (measured: ib_write_lat one-way 6.07 us, x2)
#   BDP      = BW x RTT ~= 610 KB @ 50 GB/s
#   ceiling  = outstanding x QPs x chunk = 8 x 4 x 64 KiB = 2 MB  (~3.3x BDP)

export MPCOMM_MAX_RDMA_TRANSFER_SIZE=65536        # 64 KiB
export MPCOMM_QPS_PER_CONNECTION=4
export MPCOMM_MAX_OUTSTANDING_PER_QP=8

# Observation only: maintains posted/completed byte counters per PCIe domain.
# Does NOT gate posting. Read via mpcomm.get_pcie_domain_stats().
export MPCOMM_PCIE_STAT_ENABLE=1

echo "[env_stage0] chunk=64KiB qps=4 outstanding=8 (in-flight ceiling: 2 MB), pcie_stat=on"
