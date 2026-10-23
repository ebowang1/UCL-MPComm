// Copyright (C) 2026 Tencent. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TRMT_MPCOMM_INCLUDE_MPCOMM_H_
#define TRMT_MPCOMM_INCLUDE_MPCOMM_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

namespace mpcomm {

// =====================================================================
// Public Error Codes
// =====================================================================

enum MPCommError {
    MPCOMM_SUCCESS = 0,
    MPCOMM_ERR_DEVICE = -1,
    MPCOMM_ERR_CONTEXT = -2,
    MPCOMM_ERR_MEMORY = -3,
    MPCOMM_ERR_CONNECTION = -4,
    MPCOMM_ERR_TRANSFER = -5,
    MPCOMM_ERR_TIMEOUT = -6,
    MPCOMM_ERR_INVALID_ARG = -7,
    MPCOMM_ERR_INVALID_HANDLE = -8,
    MPCOMM_ERR_PENDING = -9,
};

// =====================================================================
// Public Types
// =====================================================================

// Async transfer handle type
using TransferHandle = uint64_t;
static constexpr TransferHandle INVALID_TRANSFER_HANDLE = 0;

/**
 * H2D (Host-to-Device) transfer mode for gather/scatter operations.
 *
 *   AUTO - Automatically select the best kernel based on hardware.
 *          On Hopper (sm_90+), defaults to TMA for optimal PCIe bandwidth
 *          with minimal SM occupancy.
 *   SM   - int4 vectorized Zero-Copy.  GPU SM threads directly read/write DRAM
 *          via PCIe using 128-bit LDG/STG instructions.  Optional optimization
 *          that trades SM compute resources for simplicity (no Smem staging).
 *          Requires block_size to be 16-byte aligned.
 *   TMA  - Hopper TMA engine (cp.async.bulk).  Data goes through Shared Memory
 *          staging with mbarrier synchronization.  Minimal SM occupancy; best
 *          for large contiguous transfers.  Requires sm_90+ (H100/H800/H20).
 */
enum H2DMode {
    H2D_MODE_AUTO = 0,   // Auto-select best kernel (default)
    H2D_MODE_SM   = 1,   // int4 vectorized Zero-Copy (SM-driven)
    H2D_MODE_TMA  = 2,   // Hopper TMA engine (cp.async.bulk via Smem)
};

// Maximum size for a single RDMA transfer (default: 1 GB)
// Can be configured via environment variable MPCOMM_MAX_RDMA_TRANSFER_SIZE
static constexpr size_t MPCOMM_DEFAULT_MAX_RDMA_TRANSFER_SIZE = 1ULL << 30;

// Number of QPs per NIC per connection (default: 1)
// Can be configured via environment variable MPCOMM_QPS_PER_CONNECTION
static constexpr size_t MPCOMM_DEFAULT_QPS_PER_CONNECTION = 1;

// Default GID index used for RoCE path setup (default: 3)
// Can be overridden via environment variable MPCOMM_GID_INDEX.
// Setting MPCOMM_GID_INDEX=-1 enables auto-selection of the first non-zero GID.
static constexpr int MPCOMM_DEFAULT_GID_INDEX = 3;

// NUMA topology information for a single NUMA node
struct NumaTopology {
    int numa_node;
    std::vector<std::string> local_nics;
    std::vector<std::string> remote_nics;
};

// NIC topology information
struct NicTopologyInfo {
    std::string nic_name;
    int numa_node;
};

// Remote buffer entry received from peer (single buffer)
struct RemoteBufferEntry {
    uint64_t addr;
    uint64_t length;
    int numa_node;
    std::vector<uint32_t> rkeys;
};

// Single published buffer entry with NUMA info
struct PublishedBufferEntry {
    uint64_t addr;
    uint64_t length;
    int numa_node;
    std::vector<uint32_t> rkeys;
};

// All published buffers
struct PublishedBufferInfo {
    std::vector<PublishedBufferEntry> buffers;
};

// Remote buffer info received from peer (all buffers)
struct RemoteBufferInfo {
    std::string host_id;
    std::vector<RemoteBufferEntry> buffers;
};

// Result of an async transfer operation
struct TransferResult {
    int error_code;
    size_t bytes_transferred;
    double elapsed_ms;
};

// =====================================================================
// MPComm Class (Pimpl)
// =====================================================================

/**
 * MPComm - Memory Pooling Communication using native ibverbs
 *
 * This class implements scatter/gather operations across multiple hosts
 * using multiple NICs, without depending on mooncake's TransferEngine.
 *
 * Environment Variables:
 *   MPCOMM_NIC_FILTER              - Comma-separated list of allowed NIC device names.
 *   MPCOMM_MAX_RDMA_TRANSFER_SIZE  - Max bytes per RDMA operation (default: 1 GB).
 *   MPCOMM_QPS_PER_CONNECTION      - Number of QPs per NIC per connection (default: 1).
 *   MPCOMM_POLL_BATCH_SIZE         - Max WCs per ibv_poll_cq call (default: 64).
 *   MPCOMM_MAX_IDLE_SPINS          - Worker idle spins before yield (default: 10000).
 *   MPCOMM_MAX_SEND_WR             - QP send queue depth (default: 512).
 *   MPCOMM_MAX_OUTSTANDING_PER_QP  - Max outstanding WRs per QP (default: 256).
 *   MPCOMM_LOG_LEVEL               - Log verbosity: "0"/"error", "1"/"warn", "2"/"info" (default),
 *                                    "3"/"debug" (includes transfer timing & NIC statistics).
 *   MPCOMM_TRANSFER_STATS_INTERVAL - Print stats every N transfers (default: 0 = every transfer,
 *                                    requires LOG_LEVEL=3/debug).
 */
class MPComm {
 public:
    MPComm();
    ~MPComm();

    // Non-copyable, non-movable
    MPComm(const MPComm &) = delete;
    MPComm &operator=(const MPComm &) = delete;
    MPComm(MPComm &&) noexcept;
    MPComm &operator=(MPComm &&) noexcept;

    /**
     * Initialize RDMA resources
     * @param local_host_id  Unique identifier for this host (e.g., "host1:12345")
     * @param device_names   Comma-separated list of device names (e.g., "mlx5_0,mlx5_1")
     *                       If empty, use all available devices
     * @param tcp_port       TCP port for metadata exchange (0 = auto select)
     * @return 0 on success, negative error code on failure
     */
    int init(const std::string &local_host_id,
             const std::string &device_names = "",
             int tcp_port = 0);

    /**
     * Initialize an explicit single-NIC pair, independent of NIC name suffixes.
     * Both peers must use this API; device_name must name exactly one NIC.
     * A nonempty MPCOMM_NIC_FILTER must equal device_name.
     * tcp_port = 0 binds an automatically selected TCP listening port.
     */
    int initSingleNicPair(const std::string &local_host_id,
                          const std::string &device_name,
                          int tcp_port = 0);

    /**
     * Shutdown and cleanup resources
     */
    void shutdown();

    /**
     * Register local memory for RDMA operations
     * @param addr    Memory address
     * @param length  Memory length in bytes
     * @return 0 on success, negative error code on failure
     */
    int registerMemory(void *addr, size_t length);

    /**
     * Unregister previously registered memory
     * @param addr  Memory address
     * @return 0 on success, negative error code on failure
     */
    int unregisterMemory(void *addr);

    /**
     * Connect to a remote host
     * @param remote_host_id  Remote host identifier
     * @param remote_tcp_addr Remote TCP address for handshake
     * @param remote_tcp_port Remote TCP port
     * @return 0 on success, negative error code on failure
     */
    int connect(const std::string &remote_host_id,
                const std::string &remote_tcp_addr,
                int remote_tcp_port);

    /**
     * Accept incoming connections (run in background)
     * @return 0 on success, negative error code on failure
     */
    int startAcceptThread();

    /**
     * Stop the accept thread
     */
    void stopAcceptThread();

    /**
     * Update remote memory info (rkey and address) for a connected host
     * @param remote_host_id  Remote host identifier
     * @param rkeys           Remote keys for each NIC (must match getNumNics())
     * @return 0 on success, negative error code on failure
     */
    int updateRemoteMemoryInfo(const std::string &remote_host_id,
                               const std::vector<uint32_t> &rkeys);

    /**
     * Publish a local buffer for remote access
     * @param addr       Buffer address (must be registered)
     * @param length     Buffer length
     * @param numa_node  NUMA node this buffer belongs to (-1 = auto-detect)
     * @return 0 on success, negative error code on failure
     */
    int publishBuffer(void *addr, size_t length, int numa_node = -1);

    /**
     * Unpublish a previously published buffer
     * @param addr  Buffer address
     * @return 0 on success, negative error code on failure
     */
    int unpublishBuffer(void *addr);

    /**
     * Unpublish all published buffers
     */
    void unpublishAllBuffers();

    /**
     * Query remote host's published buffer information via TCP
     * @param remote_host_id  Remote host identifier (must be connected)
     * @param remote_tcp_addr Remote TCP address
     * @param remote_tcp_port Remote TCP port
     * @param out_info        Output: remote buffer information
     * @return 0 on success, negative error code on failure
     */
    int queryRemoteBuffer(const std::string &remote_host_id,
                          const std::string &remote_tcp_addr,
                          int remote_tcp_port,
                          RemoteBufferInfo &out_info);

    /**
     * Query remote host's buffer by NUMA node
     * @param remote_host_id  Remote host identifier
     * @param remote_tcp_addr Remote TCP address
     * @param remote_tcp_port Remote TCP port
     * @param numa_node       NUMA node to query (-1 = first buffer)
     * @param out_entry       Output: matching buffer entry
     * @return 0 on success, negative error code on failure
     */
    int queryRemoteBufferByNuma(const std::string &remote_host_id,
                                const std::string &remote_tcp_addr,
                                int remote_tcp_port,
                                int numa_node,
                                RemoteBufferEntry &out_entry);

    /**
     * Get local published buffer info
     * @return Published buffer info, or nullptr if none published
     */
    const PublishedBufferInfo* getPublishedBufferInfo() const;

    /**
     * Get number of published buffers
     */
    size_t getPublishedBufferCount() const;

    /**
     * Get rkey for local memory region
     * @param nic_index  NIC index
     * @param addr       Memory address
     * @return rkey, or 0 if not found
     */
    uint32_t getRkey(size_t nic_index, void *addr) const;

    // ==================== Async Transfer API ====================

    /**
     * Start async scatter operation (returns immediately)
     *
     * Data layout:
     *   local_buffer[0..lengths[0]] -> host_list[0]:remote_addrs[0]
     *   local_buffer[lengths[0]..lengths[0]+lengths[1]] -> host_list[1]:remote_addrs[1]
     *   ...
     */
    TransferHandle scatterAsync(uintptr_t local_addr,
                                const std::vector<std::string> &host_list,
                                const std::vector<uintptr_t> &remote_addrs,
                                const std::vector<size_t> &lengths);

    /**
     * Start async gather operation (returns immediately)
     *
     * Data layout:
     *   host_list[0]:remote_addrs[0] -> local_buffer[0..lengths[0]]
     *   host_list[1]:remote_addrs[1] -> local_buffer[lengths[0]..lengths[0]+lengths[1]]
     *   ...
     */
    TransferHandle gatherAsync(uintptr_t local_addr,
                               const std::vector<std::string> &host_list,
                               const std::vector<uintptr_t> &remote_addrs,
                               const std::vector<size_t> &lengths);

    /**
     * Start async broadcast operation (returns immediately)
     *
     * Broadcasts the same local data to all destinations.
     */
    TransferHandle broadcastAsync(uintptr_t local_addr,
                                  size_t length,
                                  const std::vector<std::string> &host_list,
                                  const std::vector<uintptr_t> &remote_addrs);

    /**
     * Start async put operation (RDMA WRITE to single host)
     */
    TransferHandle putAsync(uintptr_t local_addr,
                            const std::string &remote_host_id,
                            uintptr_t remote_addr,
                            size_t length);

    /**
     * Start async get operation (RDMA READ from single host)
     */
    TransferHandle getAsync(uintptr_t local_addr,
                            const std::string &remote_host_id,
                            uintptr_t remote_addr,
                            size_t length);

    /**
     * Check if async transfer is complete (non-blocking)
     */
    bool isTransferComplete(TransferHandle handle);

    /**
     * Wait for async transfer to complete (blocking with optional timeout)
     * @param timeout_ms  -1 = wait forever
     */
    int waitTransfer(TransferHandle handle, int timeout_ms = -1);

    /**
     * Get result of completed async transfer
     */
    TransferResult getTransferResult(TransferHandle handle);

    /**
     * Release async transfer handle and associated resources
     */
    void releaseTransfer(TransferHandle handle);

    // ==================== End Async Transfer API ====================

    /** Get number of available NICs */
    size_t getNumNics() const;

    /** Get local host ID */
    const std::string &getLocalHostId() const;

    /** Get TCP port for metadata exchange */
    int getTcpPort() const;

    /** Get GID string for a specific NIC */
    std::string getGid(size_t nic_index) const;

    /** Get device name for a specific NIC */
    std::string getDeviceName(size_t nic_index) const;

    /** Get all active device names */
    std::vector<std::string> getActiveDevices() const;

    /** Get the NIC filter environment variable name */
    static const char* getNicFilterEnvVarName();

    /** Get the current max RDMA transfer size */
    size_t getMaxRdmaTransferSize() const;

    /** Get the number of QPs per connection */
    size_t getQpsPerConnection() const;

    /** Get the QPs per connection environment variable name */
    static const char* getQpsPerConnectionEnvVarName();

    /** Get NUMA topology information */
    const std::vector<NumaTopology>& getNumaTopology() const;

    /** Get NIC topology information */
    const std::vector<NicTopologyInfo>& getNicTopology() const;

    /** Get the NUMA node for a specific NIC */
    int getNicNumaNode(const std::string& nic_name) const;

    /** Get NUMA node for a given memory address */
    int getNumaNodeForAddr(void* addr) const;

    /** Get local NIC indices for a specific NUMA node */
    std::vector<size_t> getLocalNicIndicesForNuma(int numa_node) const;

    /** Get the NUMA node for a specific GPU device */
    int getGpuNumaNode(int gpu_device_id) const;

    // ==================== HBM-DRAM Mapping API ====================

    /**
     * Map a DRAM buffer to GPU address space via Zero-Copy (cudaHostRegister + Mapped)
     *
     * This registers a CPU DRAM buffer as pinned memory and obtains a GPU-accessible
     * device pointer. The GPU can then access DRAM data directly through PCIe BAR
     * without explicit cudaMemcpy.
     *
     * @param host_addr  CPU DRAM buffer address (should be page-aligned for best performance)
     * @param length     Buffer length in bytes
     * @return GPU-accessible device pointer (as uintptr_t), or 0 on failure
     */
    uintptr_t mapDRAMtoGPU(void *host_addr, size_t length);

    /**
     * Unmap a previously mapped DRAM buffer from GPU address space
     *
     * @param host_addr  The original CPU DRAM address passed to mapDRAMtoGPU
     * @return 0 on success, negative error code on failure
     */
    int unmapDRAMfromGPU(void *host_addr);

    // ==================== H2D Transfer API ====================

    /**
     * Gather: Load scattered data blocks from DRAM (via mapped pointer) to GPU HBM
     *
     * Supports two kernel backends selectable via `mode`:
     *   H2D_MODE_AUTO (default) - auto-select based on block_size and hardware
     *   H2D_MODE_SM   - int4 vectorized Zero-Copy (LDG.E.128, SM-driven)
     *   H2D_MODE_TMA  - Hopper TMA engine (cp.async.bulk, requires sm_90+)
     *
     * @param dram_dev_ptr   GPU-mapped DRAM device pointer (from mapDRAMtoGPU)
     * @param indices        Array of block indices to gather (on GPU)
     * @param gpu_dst        Destination GPU HBM buffer
     * @param num_blocks     Number of blocks to gather
     * @param block_size     Size of each block in bytes (must be aligned to 16 bytes)
     * @param max_sm_count   Maximum number of SMs to use (0 = auto)
     * @param mode           Transfer mode: H2D_MODE_AUTO, H2D_MODE_SM, or H2D_MODE_TMA
     * @return 0 on success, negative error code on failure
     */
    int tmaGather(uintptr_t dram_dev_ptr,
                  const int64_t *indices,
                  void *gpu_dst,
                  int num_blocks,
                  int block_size,
                  int max_sm_count = 0,
                  H2DMode mode = H2D_MODE_AUTO);

    /**
     * Scatter: Store data blocks from GPU HBM to DRAM (via mapped pointer)
     *
     * Supports two kernel backends selectable via `mode`:
     *   H2D_MODE_AUTO (default) - auto-select based on block_size and hardware
     *   H2D_MODE_SM   - int4 vectorized Zero-Copy (STG.E.128, SM-driven)
     *   H2D_MODE_TMA  - Hopper TMA engine (cp.async.bulk, requires sm_90+)
     *
     * @param gpu_src        Source GPU HBM buffer
     * @param indices        Array of block indices to scatter (on GPU)
     * @param dram_dev_ptr   GPU-mapped DRAM device pointer (from mapDRAMtoGPU)
     * @param num_blocks     Number of blocks to scatter
     * @param block_size     Size of each block in bytes (must be aligned to 16 bytes)
     * @param max_sm_count   Maximum number of SMs to use (0 = auto)
     * @param mode           Transfer mode: H2D_MODE_AUTO, H2D_MODE_SM, or H2D_MODE_TMA
     * @return 0 on success, negative error code on failure
     */
    int tmaScatter(void *gpu_src,
                   const int64_t *indices,
                   uintptr_t dram_dev_ptr,
                   int num_blocks,
                   int block_size,
                   int max_sm_count = 0,
                   H2DMode mode = H2D_MODE_AUTO);

    // ==================== End HBM-DRAM Mapping & TMA API ====================

 private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mpcomm

#endif  // TRMT_MPCOMM_INCLUDE_MPCOMM_H_
