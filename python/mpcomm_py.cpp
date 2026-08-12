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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <vector>

#include "mpcomm.h"

namespace py = pybind11;

namespace mpcomm {

/**
 * Python wrapper for MPComm class
 * Provides Python-friendly interface with uintptr_t for memory addresses
 */
class MPCommPy {
 public:
    MPCommPy() = default;
    ~MPCommPy() = default;

    /**
     * Initialize RDMA resources
     * @param local_host_id  Unique identifier for this host (e.g., "host1:12345")
     * @param device_names   Comma-separated list of device names (e.g., "mlx5_0,mlx5_1")
     * @param tcp_port       TCP port for metadata exchange (0 = auto select)
     * @return 0 on success, negative error code on failure
     */
    int init(const std::string &local_host_id,
             const std::string &device_names = "",
             int tcp_port = 0) {
        return comm_.init(local_host_id, device_names, tcp_port);
    }

    /**
     * Shutdown and cleanup resources
     */
    void shutdown() {
        comm_.shutdown();
    }

    /**
     * Register local memory for RDMA operations
     * @param addr    Memory address as uintptr_t
     * @param length  Memory length in bytes
     * @return 0 on success, negative error code on failure
     */
    int registerMemory(uintptr_t addr, size_t length) {
        return comm_.registerMemory(reinterpret_cast<void *>(addr), length);
    }

    /**
     * Unregister previously registered memory
     * @param addr  Memory address as uintptr_t
     * @return 0 on success, negative error code on failure
     */
    int unregisterMemory(uintptr_t addr) {
        return comm_.unregisterMemory(reinterpret_cast<void *>(addr));
    }

    /**
     * Connect to a remote host
     * @param remote_host_id  Remote host identifier
     * @param remote_tcp_addr Remote TCP address for handshake
     * @param remote_tcp_port Remote TCP port
     * @return 0 on success, negative error code on failure
     */
    int connect(const std::string &remote_host_id,
                const std::string &remote_tcp_addr,
                int remote_tcp_port) {
        return comm_.connect(remote_host_id, remote_tcp_addr, remote_tcp_port);
    }

    /**
     * Start accepting incoming connections in background
     * @return 0 on success, negative error code on failure
     */
    int startAcceptThread() {
        return comm_.startAcceptThread();
    }

    /**
     * Stop the accept thread
     */
    void stopAcceptThread() {
        comm_.stopAcceptThread();
    }

    /**
     * Update remote memory info (rkey) for a connected host
     * @param remote_host_id  Remote host identifier
     * @param rkeys           Remote keys for each NIC
     * @return 0 on success, negative error code on failure
     */
    int updateRemoteMemoryInfo(const std::string &remote_host_id,
                               const std::vector<uint32_t> &rkeys) {
        return comm_.updateRemoteMemoryInfo(remote_host_id, rkeys);
    }

    /**
     * Get rkey for local memory region
     * @param nic_index  NIC index
     * @param addr       Memory address as uintptr_t
     * @return rkey, or 0 if not found
     */
    uint32_t getRkey(size_t nic_index, uintptr_t addr) {
        return comm_.getRkey(nic_index, reinterpret_cast<void *>(addr));
    }

    /**
     * Get all rkeys for local memory region (one per NIC)
     * @param addr  Memory address as uintptr_t
     * @return vector of rkeys, one per NIC
     */
    std::vector<uint32_t> getAllRkeys(uintptr_t addr) {
        std::vector<uint32_t> rkeys;
        size_t num_nics = comm_.getNumNics();
        rkeys.reserve(num_nics);
        for (size_t i = 0; i < num_nics; ++i) {
            rkeys.push_back(comm_.getRkey(i, reinterpret_cast<void *>(addr)));
        }
        return rkeys;
    }

    // ==================== Async Transfer API ==

    /**
     * Start async scatter operation (returns immediately)
     * @param local_addr       Local buffer address
     * @param host_list        List of destination host IDs
     * @param remote_addrs     Remote buffer addresses on each host
     * @param lengths          Data lengths for each host
     * @return TransferHandle on success, 0 (INVALID_TRANSFER_HANDLE) on failure
     */
    uint64_t scatterAsync(uintptr_t local_addr,
                          const std::vector<std::string> &host_list,
                          const std::vector<uintptr_t> &remote_addrs,
                          const std::vector<size_t> &lengths) {
        return comm_.scatterAsync(local_addr, host_list, remote_addrs, lengths);
    }

    /**
     * Start async gather operation (returns immediately)
     * @param local_addr       Local buffer address
     * @param host_list        List of source host IDs
     * @param remote_addrs     Remote buffer addresses on each host
     * @param lengths          Data lengths for each host
     * @return TransferHandle on success, 0 (INVALID_TRANSFER_HANDLE) on failure
     */
    uint64_t gatherAsync(uintptr_t local_addr,
                         const std::vector<std::string> &host_list,
                         const std::vector<uintptr_t> &remote_addrs,
                         const std::vector<size_t> &lengths) {
        return comm_.gatherAsync(local_addr, host_list, remote_addrs, lengths);
    }

    /**
     * Start async broadcast operation (returns immediately)
     * Broadcasts the same local data to multiple remote hosts.
     * @param local_addr       Local buffer address
     * @param length           Data length to broadcast (same for all destinations)
     * @param host_list        List of destination host IDs
     * @param remote_addrs     Remote buffer addresses on each host
     * @return TransferHandle on success, 0 (INVALID_TRANSFER_HANDLE) on failure
     */
    uint64_t broadcastAsync(uintptr_t local_addr,
                            size_t length,
                            const std::vector<std::string> &host_list,
                            const std::vector<uintptr_t> &remote_addrs) {
        return comm_.broadcastAsync(local_addr, length, host_list, remote_addrs);
    }

    /**
     * Start async put operation (returns immediately)
     * Write local data to a single remote host using RDMA WRITE.
     * @param local_addr       Local buffer address
     * @param remote_host_id   Destination host ID
     * @param remote_addr      Remote buffer address
     * @param length           Data length to write
     * @return TransferHandle on success, 0 (INVALID_TRANSFER_HANDLE) on failure
     */
    uint64_t putAsync(uintptr_t local_addr,
                      const std::string &remote_host_id,
                      uintptr_t remote_addr,
                      size_t length) {
        return comm_.putAsync(local_addr, remote_host_id, remote_addr, length);
    }

    /**
     * Start async get operation (returns immediately)
     * Read data from a single remote host to local buffer using RDMA READ.
     * @param local_addr       Local buffer address
     * @param remote_host_id   Source host ID
     * @param remote_addr      Remote buffer address
     * @param length           Data length to read
     * @return TransferHandle on success, 0 (INVALID_TRANSFER_HANDLE) on failure
     */
    uint64_t getAsync(uintptr_t local_addr,
                      const std::string &remote_host_id,
                      uintptr_t remote_addr,
                      size_t length) {
        return comm_.getAsync(local_addr, remote_host_id, remote_addr, length);
    }

    /**
     * Unified async interface for scatter/gather/broadcast operations
     * @param comm_type        "scatter", "gather", or "broadcast"
     * @param host_list        List of remote host IDs
     * @param local_addr       Local buffer address
     * @param remote_addrs     Remote buffer addresses on each host
     * @param lengths          Data lengths for each host
     * @return TransferHandle on success, 0 (INVALID_TRANSFER_HANDLE) on failure
     */
    uint64_t mpReplicateAsync(const std::string &comm_type,
                              const std::vector<std::string> &host_list,
                              uintptr_t local_addr,
                              const std::vector<uintptr_t> &remote_addrs,
                              const std::vector<size_t> &lengths) {
        if (comm_type == "scatter") {
            return scatterAsync(local_addr, host_list, remote_addrs, lengths);
        } else if (comm_type == "gather") {
            return gatherAsync(local_addr, host_list, remote_addrs, lengths);
        } else if (comm_type == "broadcast") {
            // For broadcast, all lengths must be equal (use first length)
            if (lengths.empty()) return INVALID_TRANSFER_HANDLE;
            return broadcastAsync(local_addr, lengths[0], host_list, remote_addrs);
        }
        return INVALID_TRANSFER_HANDLE;
    }

    /**
     * Check if async transfer is complete (non-blocking)
     * @param handle  Transfer handle from async operations
     * @return true if transfer is complete, false if still in progress
     */
    bool isTransferComplete(uint64_t handle) {
        return comm_.isTransferComplete(handle);
    }

    /**
     * Wait for async transfer to complete (blocking with optional timeout)
     * @param handle      Transfer handle from async operations
     * @param timeout_ms  Timeout in milliseconds (-1 = wait forever)
     * @return 0 on success, MPCOMM_ERR_TIMEOUT on timeout, or other error code
     */
    int waitTransfer(uint64_t handle, int timeout_ms = -1) {
        // A blocking wait must not keep the CPython GIL. Otherwise a benchmark
        // control thread cannot issue set_nic_throttle() until the current
        // transfer ends, biasing every timing sample to a transfer boundary.
        py::gil_scoped_release release;
        return comm_.waitTransfer(handle, timeout_ms);
    }

    /**
     * Get result of completed async transfer
     * @param handle  Transfer handle from async operations
     * @return dict with error_code, bytes_transferred, elapsed_ms
     */
    py::dict getTransferResult(uint64_t handle) {
        TransferResult result = comm_.getTransferResult(handle);
        py::dict dict;
        dict["error_code"] = result.error_code;
        dict["bytes_transferred"] = result.bytes_transferred;
        dict["elapsed_ms"] = result.elapsed_ms;
        return dict;
    }

    /**
     * Release async transfer handle and associated resources
     * @param handle  Transfer handle from async operations
     */
    void releaseTransfer(uint64_t handle) {
        comm_.releaseTransfer(handle);
    }

    // ==================== End Async Transfer API ====================

    /**
     * Get number of available NICs
     */
    size_t getNumNics() const { return comm_.getNumNics(); }

    /**
     * Get local host ID
     */
    std::string getLocalHostId() const { return comm_.getLocalHostId(); }

    /**
     * Get TCP port for metadata exchange
     */
    int getTcpPort() const { return comm_.getTcpPort(); }

    /**
     * Get GID string for a specific NIC
     */
    std::string getGid(size_t nic_index) const { return comm_.getGid(nic_index); }

    /**
     * Get device name for a specific NIC
     */
    std::string getDeviceName(size_t nic_index) const { return comm_.getDeviceName(nic_index); }

    /**
     * Get all active device names
     */
    std::vector<std::string> getActiveDevices() const { return comm_.getActiveDevices(); }

    /**
     * Get the NIC filter environment variable name
     */
    static std::string getNicFilterEnvVarName() { return std::string(MPComm::getNicFilterEnvVarName()); }

    /**
     * Get the QPs per connection environment variable name
     */
    static std::string getQpsPerConnectionEnvVarName() { return std::string(MPComm::getQpsPerConnectionEnvVarName()); }

    /**
     * Get the number of QPs per connection
     */
    size_t getQpsPerConnection() const { return comm_.getQpsPerConnection(); }

    /**
     * Apply a runtime NIC throttle. 0 keeps the corresponding baseline value.
     */
    void setNicThrottle(size_t max_outstanding_per_qp, size_t chunk_size) {
        comm_.setNicThrottle(max_outstanding_per_qp, chunk_size);
    }

    /**
     * Remove the runtime NIC throttle.
     */
    void clearNicThrottle() { comm_.clearNicThrottle(); }

    /** Enable H2D-triggered GPU-specific RDMA post pacing. */
    void enableH2DQoS(uint64_t rate_bytes_per_sec, uint64_t burst_bytes) {
        comm_.enableH2DQoS(rate_bytes_per_sec, burst_bytes);
    }

    /** Update the rate/burst used by subsequently active H2D windows. */
    void setH2DQoSPolicy(uint64_t rate_bytes_per_sec, uint64_t burst_bytes) {
        comm_.setH2DQoSPolicy(rate_bytes_per_sec, burst_bytes);
    }

    bool isH2DQoSEnabled() const { return comm_.isH2DQoSEnabled(); }

    /** Disable H2D-triggered post pacing and restore affected RDMA flows. */
    void disableH2DQoS() { comm_.disableH2DQoS(); }

    /**
     * Cumulative NIC->GPU bytes whose RDMA CQE has completed since init.
     * Used for windowed background-NIC bandwidth measurement.
     */
    uint64_t getNicCompletedBytes() const {
        return comm_.getNicCompletedBytes();
    }

    /**
     * Read throttle latency instrumentation.
     * @return dict with timestamps, the t3 outstanding-WQE snapshot, and
     *         derived worker/drain/effective durations in microseconds.
     */
    py::dict getThrottleLatency() const {
        uint64_t set_ns = 0, worker_ns = 0, drained_ns = 0;
        size_t snapshot_outstanding_wqes = 0;
        bool drained = comm_.getThrottleLatency(&set_ns, &worker_ns, &drained_ns,
                                                &snapshot_outstanding_wqes);
        py::dict result;
        result["set_ns"] = set_ns;
        result["worker_seen_ns"] = worker_ns;
        result["drained_ns"] = drained_ns;
        result["snapshot_outstanding_wqes"] = snapshot_outstanding_wqes;
        result["drained"] = drained;
        result["worker_us"] = (worker_ns > set_ns)
                                  ? (worker_ns - set_ns) / 1000.0 : 0.0;
        result["drain_us"] = (drained && drained_ns > worker_ns && worker_ns > 0)
                                  ? (drained_ns - worker_ns) / 1000.0 : 0.0;
        result["effective_us"] = (drained && drained_ns > set_ns)
                                  ? (drained_ns - set_ns) / 1000.0 : 0.0;
        return result;
    }

    /**
     * Write bytes to a registered buffer (for testing)
     * @param addr    Memory address as uintptr_t
     * @param data    Data to write
     * @param length  Number of bytes to write
     * @return 0 on success, negative error code on failure
     */
    int writeBytesToBuffer(uintptr_t addr, const py::bytes &data, size_t length) {
        std::string str_data = data;
        if (str_data.size() < length) {
            return MPCOMM_ERR_INVALID_ARG;
        }
        std::memcpy(reinterpret_cast<void *>(addr), str_data.data(), length);
        return MPCOMM_SUCCESS;
    }

    /**
     * Read bytes from a registered buffer (for testing)
     * @param addr    Memory address as uintptr_t
     * @param length  Number of bytes to read
     * @return bytes object containing the data
     */
    py::bytes readBytesFromBuffer(uintptr_t addr, size_t length) {
        return py::bytes(reinterpret_cast<const char *>(addr), length);
    }

    /**
     * Publish a local buffer for remote access
     * Multiple buffers can be published - each call adds to the list
     * @param addr       Buffer address (must be registered)
     * @param length     Buffer length
     * @param numa_node  NUMA node this buffer belongs to (-1 = auto-detect)
     * @return 0 on success, negative error code on failure
     */
    int publishBuffer(uintptr_t addr, size_t length, int numa_node = -1) {
        return comm_.publishBuffer(reinterpret_cast<void *>(addr), length, numa_node);
    }

    /**
     * Unpublish a previously published buffer
     * @param addr  Buffer address
     * @return 0 on success, negative error code on failure
     */
    int unpublishBuffer(uintptr_t addr) {
        return comm_.unpublishBuffer(reinterpret_cast<void *>(addr));
    }

    /**
     * Unpublish all published buffers
     */
    void unpublishAllBuffers() {
        comm_.unpublishAllBuffers();
    }

    /**
     * Get number of published buffers
     * @return Number of buffers currently published
     */
    size_t getPublishedBufferCount() {
        return comm_.getPublishedBufferCount();
    }

    /**
     * Query remote host's published buffer information via TCP
     * Returns all published buffers with NUMA info
     * @param remote_host_id  Remote host identifier (must be connected)
     * @param remote_tcp_addr Remote TCP address
     * @param remote_tcp_port Remote TCP port
     * @return dict with buffer info: {'host_id': str,
     *         'buffers': [{'addr': int, 'length': int, 'numa_node': int, 'rkeys': [int...]}]}
     *         or empty dict on failure
     */
    py::dict queryRemoteBuffers(const std::string &remote_host_id,
                                const std::string &remote_tcp_addr,
                                int remote_tcp_port) {
        RemoteBufferInfo info;
        int ret = comm_.queryRemoteBuffer(remote_host_id, remote_tcp_addr,
                                          remote_tcp_port, info);
        py::dict result;
        if (ret == 0) {
            result["host_id"] = info.host_id;
            py::list buffers_list;
            for (const auto &buf : info.buffers) {
                py::dict buf_dict;
                buf_dict["addr"] = buf.addr;
                buf_dict["length"] = buf.length;
                buf_dict["numa_node"] = buf.numa_node;
                buf_dict["gpu_device_id"] = buf.gpu_device_id;
                buf_dict["pcie_affine_nic_indices"] = buf.pcie_affine_nic_indices;
                buf_dict["rkeys"] = buf.rkeys;
                buffers_list.append(buf_dict);
            }
            result["buffers"] = buffers_list;
        }
        return result;
    }

    /**
     * Query remote host's buffer by NUMA node (convenience method)
     * @param remote_host_id  Remote host identifier (must be connected)
     * @param remote_tcp_addr Remote TCP address
     * @param remote_tcp_port Remote TCP port
     * @param numa_node       NUMA node to query (-1 = first buffer)
     * @return dict with buffer info: {'addr': int, 'length': int, 'numa_node': int, 'rkeys': [int...]}
     *         or empty dict on failure
     */
    py::dict queryRemoteBufferByNuma(const std::string &remote_host_id,
                                     const std::string &remote_tcp_addr,
                                     int remote_tcp_port,
                                     int numa_node = -1) {
        RemoteBufferEntry entry;
        int ret = comm_.queryRemoteBufferByNuma(remote_host_id, remote_tcp_addr,
                                                remote_tcp_port, numa_node, entry);
        py::dict result;
        if (ret == 0) {
            result["addr"] = entry.addr;
            result["length"] = entry.length;
            result["numa_node"] = entry.numa_node;
            result["gpu_device_id"] = entry.gpu_device_id;
            result["pcie_affine_nic_indices"] = entry.pcie_affine_nic_indices;
            result["rkeys"] = entry.rkeys;
        }
        return result;
    }

    /**
     * Query remote host's published buffer (backward compatible version)
     * Returns first buffer info for backward compatibility
     * @param remote_host_id  Remote host identifier (must be connected)
     * @param remote_tcp_addr Remote TCP address
     * @param remote_tcp_port Remote TCP port
     * @return dict with buffer info: {'addr': int, 'length': int, 'rkeys': [int...]}
     *         or empty dict on failure
     */
    py::dict queryRemoteBuffer(const std::string &remote_host_id,
                               const std::string &remote_tcp_addr,
                               int remote_tcp_port) {
        return queryRemoteBufferByNuma(remote_host_id, remote_tcp_addr, remote_tcp_port, -1);
    }

    /**
     * Get local published buffer info (for debugging/display)
     * @return dict with all published buffers:
     *         {'buffers': [{'addr': int, 'length': int, 'numa_node': int, 'rkeys': [int...]}]}
     *         or empty dict if not published
     */
    py::dict getPublishedBufferInfo() {
        py::dict result;
        const PublishedBufferInfo* info = comm_.getPublishedBufferInfo();
        if (info && !info->buffers.empty()) {
            py::list buffers_list;
            for (const auto &buf : info->buffers) {
                py::dict buf_dict;
                buf_dict["addr"] = buf.addr;
                buf_dict["length"] = buf.length;
                buf_dict["numa_node"] = buf.numa_node;
                buf_dict["gpu_device_id"] = buf.gpu_device_id;
                buf_dict["pcie_affine_nic_indices"] = buf.pcie_affine_nic_indices;
                buf_dict["rkeys"] = buf.rkeys;
                buffers_list.append(buf_dict);
            }
            result["buffers"] = buffers_list;
        }
        return result;
    }

    // ==================== HBM-DRAM Mapping API ====================

    /**
     * Map a DRAM buffer to GPU address space via Zero-Copy
     * @param host_addr  CPU DRAM buffer address as uintptr_t
     * @param length     Buffer length in bytes
     * @return GPU device pointer as uintptr_t, or 0 on failure
     */
    uintptr_t mapDRAMtoGPU(uintptr_t host_addr, size_t length) {
        return comm_.mapDRAMtoGPU(reinterpret_cast<void *>(host_addr), length);
    }

    /**
     * Unmap a previously mapped DRAM buffer from GPU address space
     * @param host_addr  The original CPU DRAM address as uintptr_t
     * @return 0 on success, negative error code on failure
     */
    int unmapDRAMfromGPU(uintptr_t host_addr) {
        return comm_.unmapDRAMfromGPU(reinterpret_cast<void *>(host_addr));
    }

    // ==================== TMA Transfer API ====================

    /**
     * Gather: Load scattered blocks from DRAM to GPU HBM
     * @param dram_dev_ptr   GPU-mapped DRAM device pointer (from mapDRAMtoGPU)
     * @param indices_ptr    Pointer to block indices array (on GPU)
     * @param gpu_dst_ptr    GPU HBM destination address
     * @param num_blocks     Number of blocks to gather
     * @param block_size     Size of each block in bytes
     * @param max_sm_count   Maximum SMs to use (0 = auto)
     * @param mode           Transfer mode: 0=AUTO, 1=SM, 2=TMA
     * @return 0 on success, negative error code on failure
     */
    int tmaGather(uintptr_t dram_dev_ptr, uintptr_t indices_ptr,
                  uintptr_t gpu_dst_ptr, int num_blocks, int block_size,
                  int max_sm_count = 0, int mode = 0) {
        return comm_.tmaGather(dram_dev_ptr,
                               reinterpret_cast<const int64_t *>(indices_ptr),
                               reinterpret_cast<void *>(gpu_dst_ptr),
                               num_blocks, block_size, max_sm_count,
                               static_cast<H2DMode>(mode));
    }

    /**
     * Scatter: Store data blocks from GPU HBM to DRAM
     * @param gpu_src_ptr    GPU HBM source address
     * @param indices_ptr    Pointer to block indices array (on GPU)
     * @param dram_dev_ptr   GPU-mapped DRAM device pointer (from mapDRAMtoGPU)
     * @param num_blocks     Number of blocks to scatter
     * @param block_size     Size of each block in bytes
     * @param max_sm_count   Maximum SMs to use (0 = auto)
     * @param mode           Transfer mode: 0=AUTO, 1=SM, 2=TMA
     * @return 0 on success, negative error code on failure
     */
    int tmaScatter(uintptr_t gpu_src_ptr, uintptr_t indices_ptr,
                   uintptr_t dram_dev_ptr, int num_blocks, int block_size,
                   int max_sm_count = 0, int mode = 0) {
        return comm_.tmaScatter(reinterpret_cast<void *>(gpu_src_ptr),
                                reinterpret_cast<const int64_t *>(indices_ptr),
                                dram_dev_ptr, num_blocks, block_size,
                                max_sm_count, static_cast<H2DMode>(mode));
    }

 private:
    MPComm comm_;
};

PYBIND11_MODULE(mpcomm, m) {
m.doc() = "MPComm - Memory Pooling Communication using native ibverbs";

    // Error codes
    py::enum_<MPCommError>(m, "MPCommError")
        .value("SUCCESS", MPCOMM_SUCCESS)
        .value("ERR_DEVICE", MPCOMM_ERR_DEVICE)
        .value("ERR_CONTEXT", MPCOMM_ERR_CONTEXT)
        .value("ERR_MEMORY", MPCOMM_ERR_MEMORY)
        .value("ERR_CONNECTION", MPCOMM_ERR_CONNECTION)
        .value("ERR_TRANSFER", MPCOMM_ERR_TRANSFER)
        .value("ERR_TIMEOUT", MPCOMM_ERR_TIMEOUT)
        .value("ERR_INVALID_ARG", MPCOMM_ERR_INVALID_ARG)
        .value("ERR_INVALID_HANDLE", MPCOMM_ERR_INVALID_HANDLE)
        .value("ERR_PENDING", MPCOMM_ERR_PENDING)
        .export_values();

    // Constants
    m.attr("INVALID_TRANSFER_HANDLE") = INVALID_TRANSFER_HANDLE;

    // MPComm class
    py::class_<MPCommPy>(m, "MPComm")
        .def(py::init<>())
        .def("init", &MPCommPy::init,
             py::arg("local_host_id"),
             py::arg("device_names") = "",
             py::arg("tcp_port") = 0,
             "Initialize RDMA resources")
        .def("shutdown", &MPCommPy::shutdown,
             "Shutdown and cleanup resources")
        .def("register_memory", &MPCommPy::registerMemory,
             py::arg("addr"),
             py::arg("length"),
             "Register local memory for RDMA operations")
        .def("unregister_memory", &MPCommPy::unregisterMemory,
             py::arg("addr"),
             "Unregister previously registered memory")
        .def("connect", &MPCommPy::connect,
             py::arg("remote_host_id"),
             py::arg("remote_tcp_addr"),
             py::arg("remote_tcp_port"),
             "Connect to a remote host via TCP metadata exchange")
        .def("start_accept_thread", &MPCommPy::startAcceptThread,
             "Start accepting incoming connections in background")
        .def("stop_accept_thread", &MPCommPy::stopAcceptThread,
             "Stop the accept thread")
        .def("update_remote_memory_info", &MPCommPy::updateRemoteMemoryInfo,
             py::arg("remote_host_id"),
             py::arg("rkeys"),
             "Update remote memory info (rkeys) for a connected host")
        .def("get_rkey", &MPCommPy::getRkey,
             py::arg("nic_index"),
             py::arg("addr"),
             "Get rkey for local memory region on a specific NIC")
        .def("get_all_rkeys", &MPCommPy::getAllRkeys,
             py::arg("addr"),
             "Get all rkeys for local memory region (one per NIC)")
        // Async transfer API
        .def("scatter_async", &MPCommPy::scatterAsync,
             py::arg("local_addr"),
             py::arg("host_list"),
             py::arg("remote_addrs"),
             py::arg("lengths"),
             "Start async scatter operation, returns handle immediately")
        .def("gather_async", &MPCommPy::gatherAsync,
             py::arg("local_addr"),
             py::arg("host_list"),
             py::arg("remote_addrs"),
             py::arg("lengths"),
             "Start async gather operation, returns handle immediately")
        .def("broadcast_async", &MPCommPy::broadcastAsync,
             py::arg("local_addr"),
             py::arg("length"),
             py::arg("host_list"),
             py::arg("remote_addrs"),
             "Broadcast same local data to multiple remote hosts, returns handle immediately")
        .def("put_async", &MPCommPy::putAsync,
             py::arg("local_addr"),
             py::arg("remote_host_id"),
             py::arg("remote_addr"),
             py::arg("length"),
             "Write local data to a single remote host (RDMA WRITE), returns handle")
        .def("get_async", &MPCommPy::getAsync,
             py::arg("local_addr"),
             py::arg("remote_host_id"),
             py::arg("remote_addr"),
             py::arg("length"),
             "Read data from a single remote host to local buffer (RDMA READ), returns handle")
        .def("mp_replicate_async", &MPCommPy::mpReplicateAsync,
             py::arg("comm_type"),
             py::arg("host_list"),
             py::arg("local_addr"),
             py::arg("remote_addrs"),
             py::arg("lengths"),
             "Unified async interface for scatter/gather/broadcast operations")
        .def("is_transfer_complete", &MPCommPy::isTransferComplete,
             py::arg("handle"),
             "Check if async transfer is complete (non-blocking)")
        .def("wait_transfer", &MPCommPy::waitTransfer,
             py::arg("handle"),
             py::arg("timeout_ms") = -1,
             "Wait for async transfer to complete with optional timeout")
        .def("get_transfer_result", &MPCommPy::getTransferResult,
             py::arg("handle"),
             "Get result of completed async transfer")
        .def("release_transfer", &MPCommPy::releaseTransfer,
             py::arg("handle"),
             "Release async transfer handle and resources")
        // End async transfer API
        .def("get_num_nics", &MPCommPy::getNumNics,
             "Get number of available NICs")
        .def("get_local_host_id", &MPCommPy::getLocalHostId,
             "Get local host ID")
        .def("get_tcp_port", &MPCommPy::getTcpPort,
             "Get TCP port for metadata exchange")
        .def("get_gid", &MPCommPy::getGid,
             py::arg("nic_index"),
             "Get GID string for a specific NIC")
        .def("get_device_name", &MPCommPy::getDeviceName,
             py::arg("nic_index"),
             "Get device name for a specific NIC")
        .def("get_active_devices", &MPCommPy::getActiveDevices,
             "Get all active device names")
        .def_static("get_nic_filter_env_var_name", &MPCommPy::getNicFilterEnvVarName,
             "Get the NIC filter environment variable name (MPCOMM_NIC_FILTER)")
        .def_static("get_qps_per_connection_env_var_name", &MPCommPy::getQpsPerConnectionEnvVarName,
             "Get the QPs per connection environment variable name (MPCOMM_QPS_PER_CONNECTION)")
        .def("get_qps_per_connection", &MPCommPy::getQpsPerConnection,
             "Get the number of QPs per NIC connection")
        .def("set_nic_throttle", &MPCommPy::setNicThrottle,
             py::arg("max_outstanding_per_qp"),
             py::arg("chunk_size") = 0,
             "Apply a runtime NIC throttle (0 keeps the baseline value)")
        .def("clear_nic_throttle", &MPCommPy::clearNicThrottle,
             "Remove the runtime NIC throttle")
        .def("enable_h2d_qos", &MPCommPy::enableH2DQoS,
             py::arg("rate_bytes_per_sec") = 0,
             py::arg("burst_bytes") = 0,
             "Enable H2D-triggered GPU-specific RDMA post pacing; rate 0 pauses matching posts")
        .def("set_h2d_qos_policy", &MPCommPy::setH2DQoSPolicy,
             py::arg("rate_bytes_per_sec"),
             py::arg("burst_bytes") = 0,
             "Update H2D QoS token-bucket rate and burst")
        .def("is_h2d_qos_enabled", &MPCommPy::isH2DQoSEnabled,
             "Return whether H2D post-rate QoS is enabled")
        .def("disable_h2d_qos", &MPCommPy::disableH2DQoS,
             "Disable H2D-triggered NIC QoS and restore baseline limits")
        .def("get_nic_completed_bytes", &MPCommPy::getNicCompletedBytes,
             "Cumulative NIC->GPU bytes whose RDMA CQE has completed since "
             "init (for windowed background-NIC bandwidth measurement)")
        .def("get_throttle_latency", &MPCommPy::getThrottleLatency,
             "Read throttle latency instrumentation (ns timestamps + derived us)")
        .def("write_bytes_to_buffer", &MPCommPy::writeBytesToBuffer,
             py::arg("addr"),
             py::arg("data"),
             py::arg("length"),
             "Write bytes to a registered buffer")
        .def("read_bytes_from_buffer", &MPCommPy::readBytesFromBuffer,
             py::arg("addr"),
             py::arg("length"),
             "Read bytes from a registered buffer")
        .def("publish_buffer", &MPCommPy::publishBuffer,
             py::arg("addr"),
             py::arg("length"),
             py::arg("numa_node") = -1,
             "Publish a local buffer for remote access (supports multiple buffers with NUMA info)")
        .def("unpublish_buffer", &MPCommPy::unpublishBuffer,
             py::arg("addr"),
             "Unpublish a previously published buffer")
        .def("unpublish_all_buffers", &MPCommPy::unpublishAllBuffers,
             "Unpublish all published buffers")
        .def("get_published_buffer_count", &MPCommPy::getPublishedBufferCount,
             "Get number of published buffers")
        .def("query_remote_buffers", &MPCommPy::queryRemoteBuffers,
             py::arg("remote_host_id"),
             py::arg("remote_tcp_addr"),
             py::arg("remote_tcp_port"),
             "Query all remote host's published buffers with NUMA info")
        .def("query_remote_buffer_by_numa", &MPCommPy::queryRemoteBufferByNuma,
             py::arg("remote_host_id"),
             py::arg("remote_tcp_addr"),
             py::arg("remote_tcp_port"),
             py::arg("numa_node") = -1,
             "Query remote host's buffer by NUMA node")
        .def("query_remote_buffer", &MPCommPy::queryRemoteBuffer,
             py::arg("remote_host_id"),
             py::arg("remote_tcp_addr"),
             py::arg("remote_tcp_port"),
             "Query remote host's first published buffer (backward compatible)")
        .def("get_published_buffer_info", &MPCommPy::getPublishedBufferInfo,
             "Get local published buffer info")
        // HBM-DRAM Mapping API
        .def("map_dram_to_gpu", &MPCommPy::mapDRAMtoGPU,
             py::arg("host_addr"),
             py::arg("length"),
             "Map DRAM buffer to GPU address space via Zero-Copy. Returns GPU device pointer.")
        .def("unmap_dram_from_gpu", &MPCommPy::unmapDRAMfromGPU,
             py::arg("host_addr"),
             "Unmap a previously mapped DRAM buffer from GPU address space")
        // H2D Transfer API (supports SM and TMA modes)
        .def("tma_gather", &MPCommPy::tmaGather,
             py::arg("dram_dev_ptr"),
             py::arg("indices_ptr"),
             py::arg("gpu_dst_ptr"),
             py::arg("num_blocks"),
             py::arg("block_size"),
             py::arg("max_sm_count") = 0,
             py::arg("mode") = 0,
             "Gather: Load scattered blocks from DRAM to GPU HBM.\n"
             "mode: 0=AUTO (default), 1=SM (int4 Zero-Copy), 2=TMA (cp.async.bulk)")
        .def("tma_scatter", &MPCommPy::tmaScatter,
             py::arg("gpu_src_ptr"),
             py::arg("indices_ptr"),
             py::arg("dram_dev_ptr"),
             py::arg("num_blocks"),
             py::arg("block_size"),
             py::arg("max_sm_count") = 0,
             py::arg("mode") = 0,
             "Scatter: Store blocks from GPU HBM to DRAM.\n"
             "mode: 0=AUTO (default), 1=SM (int4 Zero-Copy), 2=TMA (cp.async.bulk)");

    // Expose H2DMode constants
    m.attr("H2D_MODE_AUTO") = static_cast<int>(H2D_MODE_AUTO);
    m.attr("H2D_MODE_SM")   = static_cast<int>(H2D_MODE_SM);
    m.attr("H2D_MODE_TMA")  = static_cast<int>(H2D_MODE_TMA);
}

}  // namespace mpcomm
