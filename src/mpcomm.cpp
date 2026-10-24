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

#include "mpcomm.h"
#include "mpcomm_log.h"
#include "mpcomm_pxn.h"

#include <arpa/inet.h>
#include <emmintrin.h>  // For _mm_pause()
#include <fcntl.h>
#include <linux/limits.h>  // For PATH_MAX
#include <netdb.h>
#include <netinet/tcp.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <infiniband/verbs.h>
#include <string.h>

#ifdef USE_MLNX
#include <infiniband/mlx5dv.h>
#elif defined(USE_BNXT)
#include <infiniband/bnxt_re_dv.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace mpcomm {

// =====================================================================
// Internal Types (moved from mpcomm.h for Pimpl encapsulation)
// =====================================================================

// wr_id encoding for async transfers
//
// 64-bit layout:
//   [63:56]  NIC index    (8 bits, max 256 NICs)
//   [55:48]  QP index     (8 bits, max 256 QPs per connection)
//   [47:32]  Handle       (16 bits, recycled via HandlePool, max 65535 concurrent)
//   [31:0]   Chunk index  (32 bits, max ~4 billion chunks per transfer)
//
// Handle values are recycled: when a transfer is released, its handle is returned
// to the pool for reuse. This ensures handles stay within the 16-bit encoding
// space indefinitely, regardless of how many total transfers are performed.
// Fixed-size limits for per-NIC flow control arrays in TransferContext.
// Using inline arrays instead of std::vector eliminates heap allocation
// in the submit hot path (~0.8 us per call saved).
static constexpr size_t kMaxNics = 16;
static constexpr size_t kMaxQPsPerNic = 16;

// =====================================================================
// NUMA-aware allocation utilities
// =====================================================================

// Custom deleter that calls destructor + numa_free (for objects allocated
// via numa_alloc_onnode).  Falls back to plain delete when the object was
// allocated with operator new (indicated by alloc_bytes == 0).
template<typename T>
struct NumaDeleter {
    size_t alloc_bytes = 0;  // 0 means normal delete
    void operator()(T* ptr) const noexcept {
        if (!ptr) return;
        if (alloc_bytes > 0) {
            ptr->~T();
            numa_free(ptr, alloc_bytes);
        } else {
            delete ptr;
        }
    }
};

template<typename T>
using NumaUniquePtr = std::unique_ptr<T, NumaDeleter<T>>;

// Allocate an object of type T on a specific NUMA node using
// numa_alloc_onnode + placement new.  Returns a NumaUniquePtr with the
// matching deleter.  If numa_node < 0, falls back to normal allocation.
template<typename T>
NumaUniquePtr<T> make_numa_unique(int numa_node) {
    if (numa_node < 0 || numa_available() < 0) {
        // Fallback: normal heap allocation (current NUMA policy)
        return NumaUniquePtr<T>(new T(), NumaDeleter<T>{0});
    }
    void* mem = numa_alloc_onnode(sizeof(T), numa_node);
    if (!mem) {
        // Fallback on allocation failure
        return NumaUniquePtr<T>(new T(), NumaDeleter<T>{0});
    }
    T* obj = new (mem) T();
    return NumaUniquePtr<T>(obj, NumaDeleter<T>{sizeof(T)});
}

// wr_id layout: [63:60] NIC | [59:56] QP | [55:48] NIC-limit generation tag |
//               [47:32] transfer handle | [31:0] chunk index
static_assert(kMaxNics <= 16 && kMaxQPsPerNic <= 16, "wr_id NIC/QP fields are 4 bits wide");

struct WrIdEncoding {
    static constexpr uint64_t NIC_SHIFT = 60;
    static constexpr uint64_t QP_SHIFT = 56;
    static constexpr uint64_t TAG_SHIFT = 48;
    static constexpr uint64_t HANDLE_SHIFT = 32;
    static constexpr uint64_t CHUNK_MASK = 0xFFFFFFFFULL;
    static constexpr uint64_t HANDLE_MASK = 0xFFFFULL;
    static constexpr uint64_t TAG_MASK = 0xFFULL;
    static constexpr uint64_t QP_MASK = 0xFULL;
    static constexpr uint64_t NIC_MASK = 0xFULL;

    static inline uint64_t encode(size_t nic_index, size_t qp_index,
                                  TransferHandle handle, size_t chunk_index,
                                  uint32_t tag = 0) {
        return (static_cast<uint64_t>(nic_index & NIC_MASK) << NIC_SHIFT) |
               (static_cast<uint64_t>(qp_index & QP_MASK) << QP_SHIFT) |
               (static_cast<uint64_t>(tag & TAG_MASK) << TAG_SHIFT) |
               (static_cast<uint64_t>(handle & HANDLE_MASK) << HANDLE_SHIFT) |
               (chunk_index & CHUNK_MASK);
    }

    static inline size_t decodeNic(uint64_t wr_id) {
        return static_cast<size_t>((wr_id >> NIC_SHIFT) & NIC_MASK);
    }

    static inline size_t decodeQp(uint64_t wr_id) {
        return static_cast<size_t>((wr_id >> QP_SHIFT) & QP_MASK);
    }

    static inline uint32_t decodeTag(uint64_t wr_id) {
        return static_cast<uint32_t>((wr_id >> TAG_SHIFT) & TAG_MASK);
    }

    static inline TransferHandle decodeHandle(uint64_t wr_id) {
        return static_cast<TransferHandle>((wr_id >> HANDLE_SHIFT) & HANDLE_MASK);
    }

    static inline size_t decodeChunk(uint64_t wr_id) {
        return static_cast<size_t>(wr_id & CHUNK_MASK);
    }
};

// Remote endpoint information exchanged via TCP
struct RemoteEndpointInfo {
    char gid[64];
    uint16_t lid;
    uint32_t qp_num;
    uint32_t rkey;
    uint64_t addr;
    uint64_t length;
};

// Memory region info
struct MemoryRegionInfo {
    void *addr;
    size_t length;
    struct ibv_mr *mr;
    uint32_t lkey;
    uint32_t rkey;
    int numa_node;
    bool is_gpu;
    int gpu_device_id;
    std::vector<size_t> pcie_affine_nic_indices;
};

// Per-NIC context
struct NicContext {
    std::string device_name;
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    uint8_t port;
    int gid_index;
    uint16_t lid;
    union ibv_gid gid;

    std::vector<MemoryRegionInfo> memory_regions;
    std::mutex mr_mutex;

    std::unordered_map<std::string, std::vector<struct ibv_qp *>> qp_map;
    std::mutex qp_mutex;
};

// Connection info for a remote host
struct ConnectionInfo {
    std::string host_id;
    int tcp_port;
    std::vector<RemoteEndpointInfo> nic_endpoints;

    std::vector<int> remote_nic_numa_nodes;
    int remote_numa_count;
    std::vector<std::string> remote_nic_names;
    std::unordered_map<size_t, std::vector<size_t>> local_to_remote_nic_map;

    std::unordered_map<uint64_t, RemoteBufferEntry> remote_buffers;

    ConnectionInfo() : tcp_port(0), remote_numa_count(0) {}

    uint32_t getRkeyForAddr(uint64_t remote_addr, size_t nic_idx) const {
        for (const auto& [buf_addr, buf_entry] : remote_buffers) {
            if (remote_addr >= buf_addr && remote_addr < buf_addr + buf_entry.length) {
                if (nic_idx < buf_entry.rkeys.size()) {
                    return buf_entry.rkeys[nic_idx];
                }
                break;
            }
        }
        if (nic_idx < nic_endpoints.size()) {
            return nic_endpoints[nic_idx].rkey;
        }
        return 0;
    }
};

// Chunk task for transfer operations
struct ChunkTask {
    size_t host_idx;
    uintptr_t local_addr;
    uintptr_t remote_addr;
    size_t length;
};

// Transfer direction
enum class TransferDirection {
    SCATTER,
    GATHER,
    BROADCAST
};

// Context for tracking async transfer operations
struct TransferContext {
    TransferHandle handle;
    std::atomic<size_t> total_chunks;
    std::atomic<size_t> total_completed;
    std::atomic<int> error_code;
    std::atomic<bool> finished;
    std::atomic<bool> submitted;

    uintptr_t local_addr;
    std::vector<std::string> host_list;
    std::vector<uintptr_t> remote_addrs;
    std::vector<size_t> lengths;
    TransferDirection direction;

    size_t max_chunk_size;
    std::vector<size_t> host_chunk_starts;
    std::vector<size_t> host_local_offsets;

    inline ChunkTask getChunk(size_t chunk_idx) const {
        size_t host_idx = 0;
        size_t lo = 0, hi = host_chunk_starts.size();
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (host_chunk_starts[mid] <= chunk_idx) {
                host_idx = mid;
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        size_t chunk_within_host = chunk_idx - host_chunk_starts[host_idx];
        size_t offset = chunk_within_host * max_chunk_size;
        size_t remaining = lengths[host_idx] - offset;
        size_t len = (remaining < max_chunk_size) ? remaining : max_chunk_size;
        return ChunkTask{
            host_idx,
            static_cast<uintptr_t>(local_addr + host_local_offsets[host_idx] + offset),
            static_cast<uintptr_t>(remote_addrs[host_idx] + offset),
            len
        };
    }

    std::atomic<size_t> next_chunk_idx;

    // Fixed-size per-NIC flow control arrays (no heap allocation).
    // Indexed by NIC index; only [0..num_nics_used) are valid.
    size_t num_nics_used;  // set during init, == nic_contexts_.size()
    size_t num_qps_used;   // set during init, == qps_per_connection_
    size_t per_nic_posted[kMaxNics];
    size_t per_nic_completed[kMaxNics];
    size_t per_nic_bytes[kMaxNics];
    size_t per_nic_qp_posted[kMaxNics][kMaxQPsPerNic];
    size_t per_nic_qp_completed[kMaxNics][kMaxQPsPerNic];

    std::vector<size_t> candidate_nic_indices;

    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point prep_chunks_calc_time;
    std::chrono::steady_clock::time_point prep_chunks_meta_time;
    std::chrono::steady_clock::time_point prep_numa_query_time;
    std::chrono::steady_clock::time_point prep_flowctrl_time;
    std::chrono::steady_clock::time_point queued_time;
    std::chrono::steady_clock::time_point worker_start_time;
    std::chrono::steady_clock::time_point cache_done_time;
    std::chrono::steady_clock::time_point first_post_time;
    std::chrono::steady_clock::time_point all_posted_time;
    std::chrono::steady_clock::time_point end_time;

    int numa_id;  // NUMA node that owns this transfer
    std::string timing_breakdown_str;  // Buffered timing breakdown (written by worker, printed by releaseTransfer)

    // PXN (NVLink Proxy) fields
    bool pxn_enabled;           // Whether this transfer uses PXN proxy forwarding
    int pxn_source_gpu;         // Source GPU device_id (valid when pxn_enabled)
    void* pxn_source_ctx;       // Actual CUDA context (CUcontext) that owns the source buffer
    bool pxn_is_proxy_nic[kMaxNics];  // true if NIC requires NVLink proxy (precomputed)

    // ---- Dynamic PXN fields (min_outstanding load balancing) ----
    // Each proxy pipeline corresponds to one proxy GPU + NIC pair.
    // Chunks are dynamically assigned by the worker loop (not pre-partitioned).
    // Uses a per-chunk task queue: copy done one → post one (no batch waiting).
    struct PxnCopyingChunk {
        size_t chunk_idx;                   // Original chunk index in TransferContext
        uint64_t request_id;                // Copy thread request ID
        uint64_t proxy_addr;             // Address in proxy buffer (CUdeviceptr-compatible)
        size_t length;                      // Chunk length (for proxy buffer free)
        std::atomic<bool>* copy_done_flag;  // Set by CUDA callback when copy completes
        bool result_received;               // Whether PxnCopyResult has been received
    };
    // Tracks a proxy chunk whose RDMA has been posted but not yet completed.
    // Needed to defer proxy buffer free() until RDMA DMA is done.
    struct PxnInflightRdma {
        size_t length;              // Chunk length (for proxy buffer free)
    };
    struct PxnProxyPipeline {
        int proxy_gpu_id;           // Proxy GPU device ID
        size_t nic_index;           // NIC attached to this proxy GPU

        // Pending queue: chunks assigned but not yet submitted to copy thread
        // (waiting for proxy buffer space)
        std::vector<size_t> pending_chunk_indices;

        // Copying queue: chunks submitted to copy thread, waiting for completion.
        // Each chunk is tracked individually — copy done one, post one.
        std::vector<PxnCopyingChunk> copying_chunks;

        // RDMA inflight queue: chunks whose RDMA has been posted but not completed.
        // Proxy buffer space is freed in FIFO order when RDMA completions arrive.
        std::vector<PxnInflightRdma> rdma_inflight;
        size_t rdma_completed_snapshot;  // Last seen per_nic_completed for this NIC
        size_t assigned_outstanding;     // Chunks assigned but not yet RDMA-completed

        // Cumulative stats
        size_t total_bytes_assigned;  // Total bytes assigned to this pipeline
        size_t total_bytes_copied;    // Total bytes copied via NVLink
        size_t total_chunks_posted;   // Total RDMA chunks posted from this pipeline
        size_t total_copies_submitted;  // Total NVLink copy requests submitted
    };
    std::vector<PxnProxyPipeline> pxn_proxy_pipelines;
    std::vector<size_t> pxn_direct_nic_indices;   // Direct NIC indices for this transfer
    std::vector<size_t> pxn_proxy_nic_indices;     // Proxy NIC indices (one per pipeline)

    // PXN diagnostic counters
    size_t pxn_diag_direct_chunks;       // Total chunks sent via direct NICs
    size_t pxn_diag_proxy_chunks;        // Total chunks sent via proxy NICs
    size_t pxn_diag_proxy_rounds;        // Total NVLink copy rounds across all proxies
    size_t pxn_diag_proxy_copy_bytes;    // Total bytes copied via NVLink

    // PXN per-round timing aggregates
    size_t pxn_diag_proxy_chunks_timed;  // Number of proxy chunks with timing data
    double pxn_diag_submit_to_result_us;  // Sum of (t_result_recv - t_submit) across all proxy chunks
    double pxn_diag_result_to_event_us;  // Sum of (t_event_done - t_result_recv) across all proxy chunks
    double pxn_diag_event_to_rdma_us;    // Sum of (t_rdma_posted - t_event_done) across all proxy chunks
    double pxn_diag_submit_to_rdma_us;   // Sum of (t_rdma_posted - t_submit) across all proxy chunks
    double pxn_diag_max_submit_to_result_us;
    double pxn_diag_max_result_to_event_us;
    double pxn_diag_max_event_to_rdma_us;
    double pxn_diag_max_submit_to_rdma_us;
    // First/last proxy RDMA post timestamps (relative to all_posted_time)
    std::chrono::steady_clock::time_point pxn_diag_first_rdma_post;
    std::chrono::steady_clock::time_point pxn_diag_last_rdma_post;

    // PXN posting loop diagnostics (for identifying direct NIC underutilization)
    size_t pxn_diag_loop_iterations;         // Total outer while(true) loop iterations
    size_t pxn_diag_direct_break_nic_full;   // Times direct posting broke due to NIC full
    size_t pxn_diag_direct_break_qp_full;    // Times direct posting broke due to QP full
    size_t pxn_diag_direct_break_no_chunks;  // Times direct posting exited (all chunks done)
    size_t pxn_diag_proxy_not_ready;         // Times proxy posting was called but copy not ready
    size_t pxn_diag_max_direct_per_iter;     // Max direct chunks posted in a single iteration
    size_t pxn_diag_max_proxy_per_iter;      // Max proxy chunks posted in a single iteration
    size_t pxn_diag_total_cq_completions;    // Total CQ completions polled
    size_t pxn_diag_max_outstanding_direct;  // Peak outstanding on direct NIC
    size_t pxn_diag_max_outstanding_proxy;   // Peak outstanding on proxy NIC
    // Time breakdown (nanoseconds) for PXN loop phases
    uint64_t pxn_diag_ns_advance_copies;     // Time in advance_pxn_copies_fn
    uint64_t pxn_diag_ns_direct_posting;     // Time in direct chunk posting loop
    uint64_t pxn_diag_ns_proxy_posting;      // Time in post_pxn_proxy_chunks_fn (interleaved + tail)
    uint64_t pxn_diag_ns_cq_polling;         // Time in pollAllNicsForWorker

    // Per-NIC first post / last completion timestamps (for per-NIC BW measurement)
    std::chrono::steady_clock::time_point per_nic_first_post[kMaxNics];
    std::chrono::steady_clock::time_point per_nic_last_completion[kMaxNics];

    // Bench mode: wait for all NVLink copies to finish before posting any RDMA
    bool pxn_bench_sequential;       // Enabled via MPCOMM_PXN_BENCH_SEQUENTIAL=1
    bool pxn_bench_copies_done;      // Set to true when all copies are complete
    double pxn_proxy_nvlink_alpha;    // Weight for NVLink outstanding in proxy load calculation (default 1.0)

    TransferContext()
        : handle(INVALID_TRANSFER_HANDLE)
        , total_chunks(0)
        , total_completed(0)
        , error_code(0)
        , finished(false)
        , submitted(false)
        , local_addr(0)
        , direction(TransferDirection::SCATTER)
        , max_chunk_size(0)
        , next_chunk_idx(0)
        , num_nics_used(0)
        , num_qps_used(0)
        , numa_id(0)
        , pxn_enabled(false)
        , pxn_source_gpu(-1)
        , pxn_source_ctx(nullptr)
        , pxn_diag_direct_chunks(0)
        , pxn_diag_proxy_chunks(0)
        , pxn_diag_proxy_rounds(0)
        , pxn_diag_proxy_copy_bytes(0)
        , pxn_diag_proxy_chunks_timed(0)
        , pxn_diag_submit_to_result_us(0.0)
        , pxn_diag_result_to_event_us(0.0)
        , pxn_diag_event_to_rdma_us(0.0)
        , pxn_diag_submit_to_rdma_us(0.0)
        , pxn_diag_max_submit_to_result_us(0.0)
        , pxn_diag_max_result_to_event_us(0.0)
        , pxn_diag_max_event_to_rdma_us(0.0)
        , pxn_diag_max_submit_to_rdma_us(0.0)
        , pxn_diag_loop_iterations(0)
        , pxn_diag_direct_break_nic_full(0)
        , pxn_diag_direct_break_qp_full(0)
        , pxn_diag_direct_break_no_chunks(0)
        , pxn_diag_proxy_not_ready(0)
        , pxn_diag_max_direct_per_iter(0)
        , pxn_diag_max_proxy_per_iter(0)
        , pxn_diag_total_cq_completions(0)
        , pxn_diag_max_outstanding_direct(0)
        , pxn_diag_max_outstanding_proxy(0)
        , pxn_diag_ns_advance_copies(0)
        , pxn_diag_ns_direct_posting(0)
        , pxn_diag_ns_proxy_posting(0)
        , pxn_diag_ns_cq_polling(0)
        , pxn_bench_sequential(false)
        , pxn_bench_copies_done(false)
        , pxn_proxy_nvlink_alpha(1.0) {
        memset(pxn_is_proxy_nic, 0, sizeof(pxn_is_proxy_nic));
        memset(per_nic_posted, 0, sizeof(per_nic_posted));
        memset(per_nic_completed, 0, sizeof(per_nic_completed));
        memset(per_nic_bytes, 0, sizeof(per_nic_bytes));
        memset(per_nic_qp_posted, 0, sizeof(per_nic_qp_posted));
        memset(per_nic_qp_completed, 0, sizeof(per_nic_qp_completed));
        // Initialize per-NIC timestamps to epoch
        for (size_t i = 0; i < kMaxNics; ++i) {
            per_nic_first_post[i] = std::chrono::steady_clock::time_point{};
            per_nic_last_completion[i] = std::chrono::steady_clock::time_point{};
        }
    }

    TransferContext(const TransferContext&) = delete;
    TransferContext& operator=(const TransferContext&) = delete;
};

// =====================================================================
// MPComm::Impl - Internal Implementation Class
// =====================================================================

static constexpr size_t kMaxNumaNodes = 8;
static constexpr size_t kWorkersPerNuma = 1;
static constexpr size_t kLockFreeQueueSize = 4096;

// Lock-free MPSC queue per worker
struct LockFreeQueue {
    std::vector<std::atomic<TransferHandle>> buffer;
    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};

    LockFreeQueue() : buffer(kLockFreeQueueSize) {
        for (auto& slot : buffer) {
            slot.store(INVALID_TRANSFER_HANDLE, std::memory_order_relaxed);
        }
    }

    bool tryPush(TransferHandle handle) {
        size_t current_head, next_head;
        do {
            current_head = head.load(std::memory_order_relaxed);
            next_head = (current_head + 1) % kLockFreeQueueSize;
            if (next_head == tail.load(std::memory_order_acquire)) {
                return false;
            }
        } while (!head.compare_exchange_weak(current_head, next_head,
                                              std::memory_order_release,
                                              std::memory_order_relaxed));
        buffer[current_head].store(handle, std::memory_order_release);
        return true;
    }

    TransferHandle tryPop() {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        if (current_tail == head.load(std::memory_order_acquire)) {
            return INVALID_TRANSFER_HANDLE;
        }
        TransferHandle handle;
        do {
            handle = buffer[current_tail].load(std::memory_order_acquire);
        } while (handle == INVALID_TRANSFER_HANDLE);
        buffer[current_tail].store(INVALID_TRANSFER_HANDLE, std::memory_order_relaxed);
        tail.store((current_tail + 1) % kLockFreeQueueSize, std::memory_order_release);
        return handle;
    }

    bool empty() const {
        return tail.load(std::memory_order_acquire) == head.load(std::memory_order_acquire);
    }
};

// ==================== Experimental per-NIC outstanding limit ====================

static inline int64_t steadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline void atomicMaxI64(std::atomic<int64_t>& target, int64_t value) {
    int64_t current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

// (generation << 32) | limit, so workers read both atomically.
static inline uint64_t packNicLimit(uint32_t generation, uint32_t limit) {
    return (static_cast<uint64_t>(generation) << 32) | limit;
}

// Trace buffer of one NIC. Never freed while workers run: retired buffers are
// kept until shutdown so a worker holding a stale pointer stays safe.
struct NicLimitTraceBuffer {
    struct Event {
        uint32_t generation = 0;
        uint32_t limit = 0;
        int64_t set_ns = 0;
        std::atomic<int64_t> applied_ns{0};
        std::atomic<int64_t> drained_ns{0};
    };

    size_t nic = 0;
    uint32_t first_generation = 0;
    int64_t origin_ns = 0;
    uint32_t bin_ns = 1;
    size_t max_bins = 0;
    size_t max_events = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> bins;
    std::unique_ptr<Event[]> events;
    std::atomic<size_t> event_count{0};
    std::atomic<uint64_t> dropped_events{0};
    std::atomic<uint64_t> dropped_completions{0};

    void recordBytes(int64_t now_ns, uint64_t bytes) {
        const int64_t offset = now_ns - origin_ns;
        if (offset < 0) return;
        const uint64_t bin = static_cast<uint64_t>(offset) / bin_ns;
        if (bin < max_bins) bins[bin].fetch_add(bytes, std::memory_order_relaxed);
        else dropped_completions.fetch_add(1, std::memory_order_relaxed);
    }

    // Raise applied_ns or drained_ns of generations in (after, up_to] to now_ns.
    void mark(uint32_t after, uint32_t up_to, int64_t now_ns, bool applied) {
        const int64_t count = static_cast<int64_t>(event_count.load(std::memory_order_acquire));
        const int64_t lo = std::max<int64_t>(static_cast<int32_t>(after - first_generation) + 1, 0);
        const int64_t hi = std::min<int64_t>(static_cast<int32_t>(up_to - first_generation), count - 1);
        for (int64_t i = lo; i <= hi; ++i)
            atomicMaxI64(applied ? events[i].applied_ns : events[i].drained_ns, now_ns);
    }
};

class MPComm::Impl {
 public:
    Impl();
    ~Impl();

    // Public API implementation
    int init(const std::string &local_host_id,
             const std::string &device_names, int tcp_port);
    int initSingleNicPair(const std::string &local_host_id,
                          const std::string &device_name, int tcp_port);
    void shutdown();
    int registerMemory(void *addr, size_t length);
    int unregisterMemory(void *addr);
    int connect(const std::string &remote_host_id,
                const std::string &remote_tcp_addr, int remote_tcp_port);
    int startAcceptThread();
    void stopAcceptThread();
    int updateRemoteMemoryInfo(const std::string &remote_host_id,
                               const std::vector<uint32_t> &rkeys);
    int publishBuffer(void *addr, size_t length, int numa_node);
    int unpublishBuffer(void *addr);
    void unpublishAllBuffers();
    int queryRemoteBuffer(const std::string &remote_host_id,
                          const std::string &remote_tcp_addr,
                          int remote_tcp_port, RemoteBufferInfo &out_info);
    int queryRemoteBufferByNuma(const std::string &remote_host_id,
                                const std::string &remote_tcp_addr,
                                int remote_tcp_port, int numa_node,
                                RemoteBufferEntry &out_entry);
    const PublishedBufferInfo* getPublishedBufferInfo() const;
    size_t getPublishedBufferCount() const;
    uint32_t getRkey(size_t nic_index, void *addr) const;

    TransferHandle scatterAsync(uintptr_t local_addr,
                                const std::vector<std::string> &host_list,
                                const std::vector<uintptr_t> &remote_addrs,
                                const std::vector<size_t> &lengths);
    TransferHandle gatherAsync(uintptr_t local_addr,
                               const std::vector<std::string> &host_list,
                               const std::vector<uintptr_t> &remote_addrs,
                               const std::vector<size_t> &lengths);
    TransferHandle broadcastAsync(uintptr_t local_addr, size_t length,
                                  const std::vector<std::string> &host_list,
                                  const std::vector<uintptr_t> &remote_addrs);
    TransferHandle putAsync(uintptr_t local_addr,
                            const std::string &remote_host_id,
                            uintptr_t remote_addr, size_t length);
    TransferHandle getAsync(uintptr_t local_addr,
                            const std::string &remote_host_id,
                            uintptr_t remote_addr, size_t length);
    bool isTransferComplete(TransferHandle handle);
    int waitTransfer(TransferHandle handle, int timeout_ms);
    TransferResult getTransferResult(TransferHandle handle);
    void releaseTransfer(TransferHandle handle);

    // Experimental per-NIC outstanding limit
    int setNicOutstandingLimit(size_t nic_index, uint32_t limit);
    int getNicLimitState(size_t nic_index, NicLimitState &out) const;
    int waitNicLimitApplied(size_t nic_index, int timeout_ms);
    int startNicLimitTrace(size_t nic_index, uint32_t bin_ns, size_t max_bins, size_t max_events);
    int stopNicLimitTrace(NicLimitTrace &out);

    size_t getNumNics() const { return nic_contexts_.size(); }
    const std::string &getLocalHostId() const { return local_host_id_; }
    int getTcpPort() const { return tcp_port_; }
    std::string getGid(size_t nic_index) const;
    std::string getDeviceName(size_t nic_index) const;
    std::vector<std::string> getActiveDevices() const;
    size_t getMaxRdmaTransferSize() const;
    size_t getQpsPerConnection() const;
    const std::vector<NumaTopology>& getNumaTopology() const;
    const std::vector<NicTopologyInfo>& getNicTopology() const;
    int getNicNumaNode(const std::string& nic_name) const;
    int getNumaNodeForAddr(void* addr) const;
    std::vector<size_t> getLocalNicIndicesForNuma(int numa_node) const;
    int getGpuNumaNode(int gpu_device_id) const;

    // HBM-DRAM Mapping
    uintptr_t mapDRAMtoGPU(void *host_addr, size_t length);
    int unmapDRAMfromGPU(void *host_addr);

    // TMA Transfer
    int tmaGather(uintptr_t dram_dev_ptr, const int64_t *indices,
                  void *gpu_dst, int num_blocks, int block_size,
                  int max_sm_count, int mode);
    int tmaScatter(void *gpu_src, const int64_t *indices,
                   uintptr_t dram_dev_ptr, int num_blocks, int block_size,
                   int max_sm_count, int mode);

 private:
    // Topology discovery
    int getNumaNodeCount();
    int readNicNumaNode(const std::string& nic_name);
    std::vector<std::string> getCandidateNics();
    void discoverTopology();
    void printTopologyInfo();
    int detectGpuDevice(void* addr) const;
    std::vector<size_t> getGpuPcieAffinityNics(int gpu_device_id) const;
    std::vector<size_t> getPcieAffinityNicsForAddr(void* addr) const;

    // Internal helpers
    int openDevices(const std::string &device_names);
    int setupNicContext(const std::string &device_name, NicContext &ctx);
    void cleanupNicContext(NicContext &ctx);
    int createQP(NicContext &ctx, struct ibv_qp **qp);
    int modifyQPToInit(NicContext &ctx, struct ibv_qp *qp);
    int modifyQPToRTR(NicContext &ctx, struct ibv_qp *qp,
                      const RemoteEndpointInfo &remote);
    int modifyQPToRTS(struct ibv_qp *qp);
    void destroyQP(struct ibv_qp *qp);
    uint32_t getLkey(size_t nic_index, void *addr) const;
    struct ibv_qp *getOrCreateQP(size_t local_nic_index,
                                 const std::string &remote_host_id,
                                 size_t remote_nic_index,
                                 size_t qp_index = 0);
    void acceptLoop();
    void handleBufferQuery(int client_fd);
    TransferHandle transferAsyncStart(uintptr_t local_addr,
                                      const std::vector<std::string> &host_list,
                                      const std::vector<uintptr_t> &remote_addrs,
                                      const std::vector<size_t> &lengths,
                                      TransferDirection direction);
    int pollAllNicsForAsync(TransferContext& ctx);
    void workerThreadLoop(size_t worker_id, int numa_id, std::vector<int> cpu_ids);
    // workerThreadLoop helpers (extracted to keep the loop body focused):
    //   bindWorkerToNuma    - NUMA-only CPU affinity for the worker thread.
    //   probeQuarantinedNics- worker 0 periodic NIC quarantine recovery probe.
    void bindWorkerToNuma(size_t worker_id, int numa_id,
                          const std::vector<int>& cpu_ids);
    void probeQuarantinedNics(size_t num_nics);

    // Fault-injection debugging state (driven by MPCOMM_FAULT_INJECT_* env vars).
    // Only exercised on worker 0; simulates a port flap for recovery testing.
    struct FaultInjectionState {
        // Record of a faulted QP, kept so it can be recovered later.
        struct FaultedQPInfo {
            size_t nic_idx;
            std::string qp_map_key;  // "host_id:local_nic:remote_nic"
            size_t qp_index;
            struct ibv_qp* qp;
            RemoteEndpointInfo remote_info;
        };
        double delay = 0.0;          // MPCOMM_FAULT_INJECT_AFTER_SEC
        size_t nic_idx = 0;          // MPCOMM_FAULT_INJECT_NIC
        std::string mode = "single_qp";  // MPCOMM_FAULT_INJECT_MODE
        double recover_delay = 0.0;  // MPCOMM_FAULT_INJECT_RECOVER_AFTER_SEC
        bool triggered = false;
        bool recovered = false;
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point fault_time;
        std::vector<FaultedQPInfo> faulted_qps;

        void initFromEnv() {
            const char* e = std::getenv("MPCOMM_FAULT_INJECT_AFTER_SEC");
            delay = e ? std::atof(e) : 0.0;
            const char* ne = std::getenv("MPCOMM_FAULT_INJECT_NIC");
            nic_idx = ne ? static_cast<size_t>(std::atoi(ne)) : 0;
            const char* me = std::getenv("MPCOMM_FAULT_INJECT_MODE");
            mode = me ? me : "single_qp";
            const char* re = std::getenv("MPCOMM_FAULT_INJECT_RECOVER_AFTER_SEC");
            recover_delay = re ? std::atof(re) : 0.0;
            start_time = std::chrono::steady_clock::now();
            fault_time = start_time;
        }
    };
    void maybeInjectFault(FaultInjectionState& fi, size_t num_nics);
    void maybeRecoverFault(FaultInjectionState& fi);
    // Frees PXN proxy-buffer space and recycles copy-done flags held by a
    // context's copying_chunks / rdma_inflight when the transfer is finalized.
    // No-op when built without CUDA.
    void cleanupPxnContextState(TransferContext& ctx);
    int getNumaForAddr(uintptr_t addr) const;
    size_t selectWorkerForNuma(int numa_id);

    // Pipelined worker helpers: process multiple handles concurrently
    // to keep NIC pipelines full without handle-boundary stalls.
    struct NicConnInfo {
        size_t remote_nic;
        uint32_t rkey;
        uint32_t lkey;
        static constexpr size_t kMaxQPsPerConn = 16;
        struct ibv_qp* qps[kMaxQPsPerConn];
        size_t num_qps;
    };
    using NicConnCache = std::unordered_map<uint64_t, NicConnInfo>;
    void initContextCache(TransferContext& ctx, NicConnCache& cache);
    // ActiveContextInfo is passed to pollAllNicsForWorker for lock-free completion routing
    struct ActiveContextInfo {
        TransferHandle handle;
        TransferContext* ctx;
    };

    // Experimental per-NIC outstanding limit, worker-local view (owned by one worker).
    // out_by_tag counts this worker's outstanding WQEs per generation tag; live_gens
    // keeps adopted generations in order while older ones still have WQEs in flight.
    struct WorkerLimitView {
        uint64_t seen_version = ~0ULL;
        size_t limit[kMaxNics];
        uint32_t generation[kMaxNics];
        uint32_t out_by_tag[kMaxNics][256];
        std::deque<uint32_t> live_gens[kMaxNics];

        WorkerLimitView() {
            for (size_t nic = 0; nic < kMaxNics; ++nic) {
                limit[nic] = SIZE_MAX;
                generation[nic] = 0;
                std::fill(std::begin(out_by_tag[nic]), std::end(out_by_tag[nic]), 0u);
                live_gens[nic].push_back(0);
            }
        }

        void posted(size_t nic) { ++out_by_tag[nic][generation[nic] & 0xFF]; }
        uint32_t tag(size_t nic) const { return generation[nic] & 0xFF; }
    };

    // Published by each worker, read by getNicLimitState/waitNicLimitApplied.
    struct alignas(64) WorkerLimitShared {
        std::atomic<uint32_t> applied_generation[kMaxNics];
        std::atomic<uint64_t> outstanding[kMaxNics];
        std::atomic<uint64_t> posted[kMaxNics];

        WorkerLimitShared() {
            for (size_t nic = 0; nic < kMaxNics; ++nic) {
                applied_generation[nic].store(0, std::memory_order_relaxed);
                outstanding[nic].store(0, std::memory_order_relaxed);
                posted[nic].store(0, std::memory_order_relaxed);
            }
        }
    };

    void adoptNicLimits(size_t worker_id, WorkerLimitView& view, size_t num_nics,
                        const size_t (&posted)[kMaxNics], const size_t (&completed)[kMaxNics]);
    void advanceNicLimitDrain(WorkerLimitView& view, size_t nic, NicLimitTraceBuffer* trace);
    void resetNicLimits();

    int pollAllNicsForWorker(int numa_id,
                             const std::vector<size_t>& poll_nic_indices,
                             size_t (&worker_nic_completed)[kMaxNics],
                             size_t (&worker_nic_qp_completed)[kMaxNics][kMaxQPsPerNic],
                             const std::vector<ActiveContextInfo>& active_ctx_info,
                             WorkerLimitView& limits);
    void finalizeTransferStats(TransferContext& ctx);

    // Member variables
    std::string local_host_id_;
    int tcp_port_;
    int listen_fd_;
    size_t max_rdma_transfer_size_;
    size_t qps_per_connection_;
    int gid_index_;

    std::vector<NumaUniquePtr<NicContext>> nic_contexts_;

    std::unordered_map<std::string, ConnectionInfo> connections_;
    mutable std::mutex connections_mutex_;

    std::unique_ptr<PublishedBufferInfo> published_buffer_;
    std::mutex published_buffer_mutex_;

    std::atomic<bool> accept_running_;
    std::unique_ptr<std::thread> accept_thread_;

    // Per-NUMA transfer state to avoid cross-NUMA lock contention
    struct alignas(64) NumaTransferState {
        std::unordered_map<TransferHandle, NumaUniquePtr<TransferContext>> active_transfers;
        mutable std::mutex mutex;
    };
    std::array<NumaTransferState, kMaxNumaNodes> per_numa_transfers_;

    // Lightweight handle-to-NUMA routing map (only used by user-facing API, not worker hot path)
    std::unordered_map<TransferHandle, int> handle_numa_map_;
    mutable std::mutex handle_numa_mutex_;

    // Handle pool for recycling transfer handles within 16-bit wr_id encoding space.
    // Handles are allocated from the pool and returned on releaseTransfer().
    // This prevents handle overflow that would cause completion routing errors.
    struct HandlePool {
        static constexpr TransferHandle kMaxHandle = 0xFFFF;  // 16-bit max
        static constexpr TransferHandle kMinHandle = 1;       // 0 is INVALID_TRANSFER_HANDLE

        std::vector<TransferHandle> free_list;
        TransferHandle next_new_handle = kMinHandle;  // For initial allocation before recycling kicks in
        mutable std::mutex mutex;

        // Allocate a handle. Returns INVALID_TRANSFER_HANDLE if pool is exhausted.
        TransferHandle allocate() {
            std::lock_guard<std::mutex> lock(mutex);
            if (!free_list.empty()) {
                TransferHandle h = free_list.back();
                free_list.pop_back();
                return h;
            }
            if (next_new_handle <= kMaxHandle) {
                return next_new_handle++;
            }
            return INVALID_TRANSFER_HANDLE;  // All 65535 handles are in use
        }

        // Return a handle to the pool for reuse.
        void release(TransferHandle handle) {
            if (handle == INVALID_TRANSFER_HANDLE) return;
            std::lock_guard<std::mutex> lock(mutex);
            free_list.push_back(handle);
        }
    };
    HandlePool handle_pool_;

    size_t num_numa_nodes_;
    size_t total_workers_;
    std::vector<std::unique_ptr<std::thread>> worker_threads_;
    std::atomic<bool> worker_running_{false};

    std::vector<std::unique_ptr<LockFreeQueue>> worker_queues_;
    std::vector<std::unique_ptr<std::atomic<size_t>>> numa_worker_rr_;

    size_t poll_batch_size_;
    size_t max_idle_spins_;
    int max_send_wr_;
    size_t max_outstanding_per_qp_;

    // Experimental per-NIC outstanding limit state
    std::array<std::atomic<uint64_t>, kMaxNics> nic_limit_packed_;  // packNicLimit(gen, limit)
    std::atomic<uint64_t> nic_limit_version_{0};                    // bumped on every change
    mutable std::mutex nic_limit_mutex_;
    std::vector<std::unique_ptr<WorkerLimitShared>> worker_limit_shared_;
    std::atomic<NicLimitTraceBuffer*> nic_limit_trace_{nullptr};
    std::vector<std::unique_ptr<NicLimitTraceBuffer>> nic_limit_traces_;  // freed at shutdown

    bool initialized_;
    bool single_nic_pair_;
    size_t transfer_stats_interval_;              // Print stats every N transfers (0 = every transfer)
    std::atomic<size_t> transfer_stats_counter_;  // Global transfer completion counter

    std::vector<NumaTopology> numa_topology_;
    std::vector<NicTopologyInfo> nic_topology_;

    // PXN (NVLink Proxy) subsystem
    PxnManager pxn_manager_;

    // ---- NIC Quarantine: fault detection + auto-recovery ----
    // When a NIC has a WC error, it is placed in quarantine for
    // MPCOMM_NIC_QUARANTINE_SEC seconds (default 5s).  After the
    // quarantine period a probe (ibv_query_port) checks if the port
    // is back to ACTIVE; if so the NIC is re-admitted to the pool.
    // Only worker_id==0 drives the probe loop; other workers only
    // read in_quarantine (atomic) for per-transfer NIC filtering.
    struct NicQuarantineState {
        std::atomic<bool> in_quarantine{false};
        std::atomic<size_t> consecutive_errors{0};
        std::chrono::steady_clock::time_point quarantine_since;
        // Deleted copy/move to be explicit; access via unique_ptr in vector
        NicQuarantineState() = default;
        NicQuarantineState(const NicQuarantineState&) = delete;
        NicQuarantineState& operator=(const NicQuarantineState&) = delete;
    };
    std::vector<std::unique_ptr<NicQuarantineState>> nic_quarantine_;  // size = num_nics

#ifdef MPCOMM_ENABLE_CUDA_KERNELS
    // DRAM-GPU mapping tracking (used by H2D/D2H TMA kernels)
    struct DRAMGPUMapping {
        void *host_addr;
        size_t length;
        uintptr_t device_ptr;
    };
    std::unordered_map<uintptr_t, DRAMGPUMapping> dram_gpu_mappings_;  // key: host_addr as uintptr_t
    std::mutex dram_gpu_mappings_mutex_;
#endif
};

// ============================================================================
// Environment Variables and Helper Functions
// ============================================================================

// Environment variable name for NIC filtering
static const char* kNicFilterEnvVar = "MPCOMM_NIC_FILTER";
// Environment variable name for max RDMA transfer size
static const char* kMaxRdmaTransferSizeEnvVar = "MPCOMM_MAX_RDMA_TRANSFER_SIZE";
// Environment variable name for QPs per connection
static const char* kQpsPerConnectionEnvVar = "MPCOMM_QPS_PER_CONNECTION";
// Environment variable names for flow control tuning
static const char* kPollBatchSizeEnvVar = "MPCOMM_POLL_BATCH_SIZE";
static const char* kMaxIdleSpinsEnvVar = "MPCOMM_MAX_IDLE_SPINS";
static const char* kMaxSendWREnvVar = "MPCOMM_MAX_SEND_WR";
static const char* kMaxOutstandingPerQPEnvVar = "MPCOMM_MAX_OUTSTANDING_PER_QP";
static const char* kTransferStatsIntervalEnvVar = "MPCOMM_TRANSFER_STATS_INTERVAL";
static const char* kGidIndexEnvVar = "MPCOMM_GID_INDEX";

// Constants for QP setup
static const uint8_t kMaxHopLimit = 16;
static const uint8_t kTimeout = 14;
static const uint8_t kRetryCnt = 7;
static const int kMaxCQE = 1024;
static const int kMaxRecvWR = 128;
static const int kMaxSGE = 1;

// QP UDP source port list for multi-path ECMP load balancing
// These ports are used in round-robin fashion for each QP
#if defined(USE_BNXT) || defined(USE_MLNX)
static const uint16_t qp_port_list[] = {60000, 60051, 57663, 57804};
static std::atomic<int> qp_port_index{0};
#endif

// TCP message types for metadata exchange
// Use high values (magic numbers) to avoid conflict with NIC counts
// NIC count is typically 1-16, so values > 0x10000000 are safe
static const uint32_t kMsgTypeBufferQuery = 0x4D504251;  // "MPBQ" in hex
static const uint32_t kMsgTypeSingleNicConnect = 0x4D50534E;

static bool setSingleNicControlTimeouts(int fd) {
    struct timeval timeout = {};
    timeout.tv_sec = 10;
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
           setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string gidToString(const union ibv_gid &gid) {
    char buf[64];
    snprintf(buf, sizeof(buf),
             "%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:"
             "%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x",
             (unsigned)gid.raw[0], (unsigned)gid.raw[1],
             (unsigned)gid.raw[2], (unsigned)gid.raw[3],
             (unsigned)gid.raw[4], (unsigned)gid.raw[5],
             (unsigned)gid.raw[6], (unsigned)gid.raw[7],
             (unsigned)gid.raw[8], (unsigned)gid.raw[9],
             (unsigned)gid.raw[10], (unsigned)gid.raw[11],
             (unsigned)gid.raw[12], (unsigned)gid.raw[13],
             (unsigned)gid.raw[14], (unsigned)gid.raw[15]);
    return std::string(buf);
}

void stringToGid(const std::string &str, union ibv_gid &gid) {
    unsigned int values[16];
    sscanf(str.c_str(),
           "%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:"
           "%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x",
           &values[0], &values[1], &values[2], &values[3],
           &values[4], &values[5], &values[6], &values[7],
           &values[8], &values[9], &values[10], &values[11],
           &values[12], &values[13], &values[14], &values[15]);
    for (int i = 0; i < 16; ++i) {
        gid.raw[i] = static_cast<uint8_t>(values[i]);
    }
}

// Helper to check if GID is null
static bool isNullGid(const union ibv_gid *gid) {
    for (int i = 0; i < 16; ++i) {
        if (gid->raw[i] != 0) return false;
    }
    return true;
}

// ============================================================================
// MPComm Implementation
// ============================================================================

MPComm::Impl::Impl()
    : tcp_port_(0),
      listen_fd_(-1),
      max_rdma_transfer_size_(MPCOMM_DEFAULT_MAX_RDMA_TRANSFER_SIZE),
      qps_per_connection_(MPCOMM_DEFAULT_QPS_PER_CONNECTION),
      gid_index_(MPCOMM_DEFAULT_GID_INDEX),
      accept_running_(false),
      num_numa_nodes_(0),
      total_workers_(0),
      poll_batch_size_(64),
      max_idle_spins_(10000),
      max_send_wr_(512),
      max_outstanding_per_qp_(256),
      initialized_(false),
      single_nic_pair_(false),
      transfer_stats_interval_(0),
      transfer_stats_counter_(0) {
    resetNicLimits();

    // Read max RDMA transfer size from environment variable
    const char* env_val = std::getenv(kMaxRdmaTransferSizeEnvVar);
    if (env_val && env_val[0] != '\0') {
        char* endptr = nullptr;
        uint64_t val = strtoull(env_val, &endptr, 10);
        if (endptr != env_val && *endptr == '\0' && val > 0) {
            max_rdma_transfer_size_ = static_cast<size_t>(val);
            MPCOMM_LOG_INFO("MPComm: Using max RDMA transfer size from %s: %zu bytes\n",
                   kMaxRdmaTransferSizeEnvVar, max_rdma_transfer_size_);
        } else {
            MPCOMM_LOG_WARN("MPComm: Invalid %s value '%s', using default %zu\n",
                    kMaxRdmaTransferSizeEnvVar, env_val,
                    max_rdma_transfer_size_);
        }
    }

    // Read QPs per connection from environment variable
    env_val = std::getenv(kQpsPerConnectionEnvVar);
    if (env_val && env_val[0] != '\0') {
        char* endptr = nullptr;
        uint64_t val = strtoul(env_val, &endptr, 10);
        if (endptr != env_val && *endptr == '\0' && val > 0 && val <= 64) {
            qps_per_connection_ = static_cast<size_t>(val);
            MPCOMM_LOG_INFO("MPComm: Using QPs per connection from %s: %zu\n",
                   kQpsPerConnectionEnvVar, qps_per_connection_);
        } else {
            MPCOMM_LOG_WARN("MPComm: Invalid %s value '%s' (must be 1-64), using default %zu\n",
                    kQpsPerConnectionEnvVar, env_val,
                    qps_per_connection_);
        }
    }

    // Read GID index from environment variable
    // -1 means auto-select the first non-zero GID on each NIC.
    env_val = std::getenv(kGidIndexEnvVar);
    if (env_val && env_val[0] != '\0') {
        char* endptr = nullptr;
        int64_t val = strtol(env_val, &endptr, 10);
        if (endptr != env_val && *endptr == '\0' && val >= -1 && val <= 255) {
            gid_index_ = static_cast<int>(val);
            MPCOMM_LOG_INFO("MPComm: Using GID index from %s: %d\n",
                   kGidIndexEnvVar, gid_index_);
        } else {
            MPCOMM_LOG_WARN("MPComm: Invalid %s value '%s' (must be -1 or 0-255), using default %d\n",
                    kGidIndexEnvVar, env_val, gid_index_);
        }
    }

    // Read poll batch size from environment variable
    env_val = std::getenv(kPollBatchSizeEnvVar);
    if (env_val && env_val[0] != '\0') {
        char* endptr = nullptr;
        uint64_t val = strtoul(env_val, &endptr, 10);
        if (endptr != env_val && *endptr == '\0' && val > 0 && val <= 256) {
            poll_batch_size_ = static_cast<size_t>(val);
            MPCOMM_LOG_INFO("MPComm: Using poll batch size from %s: %zu\n",
                   kPollBatchSizeEnvVar, poll_batch_size_);
        } else {
            MPCOMM_LOG_WARN("MPComm: Invalid %s value '%s' (must be 1-256), using default %zu\n",
                    kPollBatchSizeEnvVar, env_val, poll_batch_size_);
        }
    }

    // Read max send WR from environment variable
    env_val = std::getenv(kMaxSendWREnvVar);
    if (env_val && env_val[0] != '\0') {
        char* endptr = nullptr;
        uint64_t val = strtoul(env_val, &endptr, 10);
        if (endptr != env_val && *endptr == '\0' && val > 0 && val <= 8192) {
            max_send_wr_ = static_cast<int>(val);
            MPCOMM_LOG_INFO("MPComm: Using max send WR from %s: %d\n",
                   kMaxSendWREnvVar, max_send_wr_);
        } else {
            MPCOMM_LOG_WARN("MPComm: Invalid %s value '%s' (must be 1-8192), using default %d\n",
                    kMaxSendWREnvVar, env_val, max_send_wr_);
        }
    }

    // Read max outstanding per QP from environment variable
    env_val = std::getenv(kMaxOutstandingPerQPEnvVar);
    if (env_val && env_val[0] != '\0') {
        char* endptr = nullptr;
        uint64_t val = strtoul(env_val, &endptr, 10);
        if (endptr != env_val && *endptr == '\0' && val > 0 && val <= 8192) {
            max_outstanding_per_qp_ = static_cast<size_t>(val);
            MPCOMM_LOG_INFO("MPComm: Using max outstanding per QP from %s: %zu\n",
                   kMaxOutstandingPerQPEnvVar, max_outstanding_per_qp_);
        } else {
            MPCOMM_LOG_WARN("MPComm: Invalid %s value '%s' (must be 1-8192), using default %zu\n",
                    kMaxOutstandingPerQPEnvVar, env_val, max_outstanding_per_qp_);
        }
    }

    // Read max idle spins from environment variable
    env_val = std::getenv(kMaxIdleSpinsEnvVar);
    if (env_val && env_val[0] != '\0') {
        char* endptr = nullptr;
        uint64_t val = strtoul(env_val, &endptr, 10);
        if (endptr != env_val && *endptr == '\0' && val > 0 && val <= 10000000) {
            max_idle_spins_ = static_cast<size_t>(val);
            MPCOMM_LOG_INFO("MPComm: Using max idle spins from %s: %zu\n",
                   kMaxIdleSpinsEnvVar, max_idle_spins_);
        } else {
            MPCOMM_LOG_WARN("MPComm: Invalid %s value '%s' (must be 1-10000000), using default %zu\n",
                    kMaxIdleSpinsEnvVar, env_val, max_idle_spins_);
        }
    }

    // Read transfer stats sampling interval from environment variable
    // When set to N (N > 0), only print stats for every Nth transfer.
    // Default 0 means print stats for every transfer (when DEBUG log level is enabled).
    env_val = std::getenv(kTransferStatsIntervalEnvVar);
    if (env_val && env_val[0] != '\0') {
        char* endptr = nullptr;
        uint64_t val = strtoull(env_val, &endptr, 10);
        if (endptr != env_val && *endptr == '\0') {
            transfer_stats_interval_ = static_cast<size_t>(val);
            MPCOMM_LOG_INFO("MPComm: Using transfer stats interval from %s: %zu\n",
                   kTransferStatsIntervalEnvVar, transfer_stats_interval_);
        } else {
            MPCOMM_LOG_WARN("MPComm: Invalid %s value '%s', using default 0 (every transfer)\n",
                    kTransferStatsIntervalEnvVar, env_val);
        }
    }
}

MPComm::Impl::~Impl() {
    shutdown();
}

/*
init()
  │
  ├─ 1. 防重复 + openDevices() → 打开 RDMA NIC
  │
  ├─ 2. socket/bind/listen → TCP 控制面监听
  │
  ├─ 3. discoverTopology() → 探测 NUMA 拓扑 + 读取 CPU 列表
  │
  ├─ 4. 创建 worker_queues_ + worker_threads_ → RDMA 异步引擎启动
  │
  ├─ 5. 日志
  │
  └─ 6. 写版本文件（可选）
*/


int MPComm::Impl::initSingleNicPair(const std::string &local_host_id,
                                   const std::string &device_name,
                                   int tcp_port) {
    if (initialized_) {
        MPCOMM_LOG_ERROR("MPComm: Already initialized\n");
        return MPCOMM_ERR_CONTEXT;
    }
    if (device_name.empty() || device_name.find(',') != std::string::npos ||
        device_name.find('\0') != std::string::npos || tcp_port < 0 || tcp_port > 65535) {
        MPCOMM_LOG_ERROR("MPComm: Single-NIC pair requires one explicit device and a valid TCP port\n");
        return MPCOMM_ERR_INVALID_ARG;
    }
    const char* env_filter = std::getenv(kNicFilterEnvVar);
    if (env_filter && env_filter[0] != '\0' && device_name != env_filter) {
        MPCOMM_LOG_ERROR("MPComm: %s='%s' conflicts with single-NIC device '%s'\n",
                         kNicFilterEnvVar, env_filter, device_name.c_str());
        return MPCOMM_ERR_INVALID_ARG;
    }
    single_nic_pair_ = true;
    int ret = init(local_host_id, device_name, tcp_port);
    if (ret != MPCOMM_SUCCESS) {
        shutdown();
        return ret;
    }
    if (getNumNics() != 1 || getDeviceName(0) != device_name) {
        MPCOMM_LOG_ERROR("MPComm: Single-NIC initialization did not select exactly '%s'\n",
                         device_name.c_str());
        shutdown();
        return MPCOMM_ERR_DEVICE;
    }
    return MPCOMM_SUCCESS;
}

int MPComm::Impl::init(const std::string &local_host_id,
                 const std::string &device_names,
                 int tcp_port) {
    if (initialized_) {
        MPCOMM_LOG_ERROR("MPComm: Already initialized\n");
        return MPCOMM_ERR_CONTEXT;
    }

    local_host_id_ = local_host_id;
    tcp_port_ = tcp_port;

    // Open RDMA devices
    int ret = openDevices(device_names);
    if (ret != 0) {
        MPCOMM_LOG_ERROR("MPComm: Failed to open devices\n");
        return ret;
    }

    if (nic_contexts_.empty()) {
        MPCOMM_LOG_ERROR("MPComm: No RDMA devices found\n");
        return MPCOMM_ERR_DEVICE;
    }

    // Validate fixed-size array limits
    if (nic_contexts_.size() > kMaxNics) {
        MPCOMM_LOG_ERROR("MPComm: Too many NICs (%zu > kMaxNics=%zu). "
                "Increase kMaxNics and rebuild.\n",
                nic_contexts_.size(), kMaxNics);
        shutdown();
        return MPCOMM_ERR_DEVICE;
    }
    if (qps_per_connection_ > kMaxQPsPerNic) {
        MPCOMM_LOG_ERROR("MPComm: Too many QPs per connection (%zu > kMaxQPsPerNic=%zu). "
                "Increase kMaxQPsPerNic and rebuild.\n",
                qps_per_connection_, kMaxQPsPerNic);
        shutdown();
        return MPCOMM_ERR_INVALID_ARG;
    }

    // Setup TCP listener if port specified
    if (tcp_port > 0 || single_nic_pair_) {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to create socket");
            shutdown();
            return MPCOMM_ERR_CONNECTION;
        }

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(tcp_port);

        if (bind(listen_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to bind");
            close(listen_fd_);
            listen_fd_ = -1;
            shutdown();
            return MPCOMM_ERR_CONNECTION;
        }

        if (listen(listen_fd_, 16) < 0) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to listen");
            close(listen_fd_);
            listen_fd_ = -1;
            shutdown();
            return MPCOMM_ERR_CONNECTION;
        }

        // Get actual port if tcp_port was 0
        struct sockaddr_in bound_addr;
        socklen_t len = sizeof(bound_addr);
        getsockname(listen_fd_, (struct sockaddr *)&bound_addr, &len);
        tcp_port_ = ntohs(bound_addr.sin_port);
    }

    initialized_ = true;

    // Discover NUMA topology and print results
    discoverTopology();
    printTopologyInfo();

    // Start per-NUMA worker threads for fully async transfer processing
    // Multiple workers per NUMA node, each bound to different CPUs
    int numa_count = getNumaNodeCount();
    num_numa_nodes_ = static_cast<size_t>(std::max(1, numa_count));
    if (num_numa_nodes_ > kMaxNumaNodes) {
        num_numa_nodes_ = kMaxNumaNodes;
    }
    total_workers_ = num_numa_nodes_ * kWorkersPerNuma;

    // Initialize lock-free queues for each worker
    worker_queues_.resize(total_workers_);
    for (size_t i = 0; i < total_workers_; ++i) {
        worker_queues_[i] = std::make_unique<LockFreeQueue>();
    }

    // Initialize per-NUMA round-robin counters
    numa_worker_rr_.resize(num_numa_nodes_);
    for (size_t i = 0; i < num_numa_nodes_; ++i) {
        numa_worker_rr_[i] = std::make_unique<std::atomic<size_t>>(0);
    }

    // Per-worker published state of the experimental NIC outstanding limit
    worker_limit_shared_.clear();
    for (size_t i = 0; i < total_workers_; ++i) {
        worker_limit_shared_.push_back(std::make_unique<WorkerLimitShared>());
    }

    // Get CPU list for each NUMA node
    std::vector<std::vector<int>> numa_cpus(num_numa_nodes_);
    for (size_t numa = 0; numa < num_numa_nodes_; ++numa) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%zu/cpulist", numa);
        FILE* f = fopen(path, "r");
        if (f) {
            char buf[256];
            if (fgets(buf, sizeof(buf), f)) {
                // Parse CPU list (format: "0-7,16-23" or "0,1,2,3")
                char* p = buf;
                while (*p) {
                    int start, end;
                    if (sscanf(p, "%d-%d", &start, &end) == 2) {
                        for (int cpu = start; cpu <= end; ++cpu) {
                            numa_cpus[numa].push_back(cpu);
                        }
                    } else if (sscanf(p, "%d", &start) == 1) {
                        numa_cpus[numa].push_back(start);
                    }
                    // Move to next number
                    while (*p && *p != ',' && *p != '-') ++p;
                    if (*p == '-') {
                        while (*p && *p != ',') ++p;
                    }
                    if (*p == ',') ++p;
                }
            }
            fclose(f);
        }
        if (numa_cpus[numa].empty()) {
            // Fallback: no specific CPU binding
            numa_cpus[numa].push_back(-1);
        }
    }

    // Start worker threads (kWorkersPerNuma workers per NUMA node)
    // NUMA-only affinity: each worker is bound to the entire CPU set of its NUMA node,
    // and the kernel scheduler picks an idle core. This avoids collisions when multiple
    // mpcomm processes run on the same host.
    worker_running_.store(true);
    worker_threads_.resize(total_workers_);
    for (size_t numa = 0; numa < num_numa_nodes_; ++numa) {
        for (size_t w = 0; w < kWorkersPerNuma; ++w) {
            size_t worker_id = numa * kWorkersPerNuma + w;
            // Pass the full CPU list of this NUMA node; workerThreadLoop will set
            // affinity to all of them (intersected with sched_getaffinity).
            std::vector<int> cpu_ids = numa_cpus[numa];
            worker_threads_[worker_id] = std::make_unique<std::thread>(
                &MPComm::Impl::workerThreadLoop, this, worker_id,
                static_cast<int>(numa), std::move(cpu_ids));
        }
    }
    MPCOMM_LOG_INFO("MPComm: Started %zu async worker threads (%zu per NUMA, %zu NUMA nodes)\n",
           total_workers_, kWorkersPerNuma, num_numa_nodes_);

    MPCOMM_LOG_INFO("MPComm: Initialized with %zu NICs, TCP port %d\n",
           nic_contexts_.size(), tcp_port_);

    // Initialize PXN (NVLink Proxy) subsystem if enabled
#ifdef USE_CUDA
    {
        size_t nic_count = nic_contexts_.size();
        auto get_gpu_pcie_nics = [this](int gpu_device_id) -> std::vector<size_t> {
            return getGpuPcieAffinityNics(gpu_device_id);
        };
        if (pxn_manager_.init(nic_count, get_gpu_pcie_nics)) {
            // Register proxy buffers with all NICs for RDMA access
            auto reg_fn = [this](void* addr, size_t length) -> int {
                return registerMemory(addr, length);
            };
            int ret = pxn_manager_.registerProxyBuffers(reg_fn);
            if (ret != 0) {
                MPCOMM_LOG_ERROR("MPComm: Failed to register PXN proxy buffers\n");
                pxn_manager_.shutdown();
            } else {
                // Cache proxy buffer lkeys for each NIC
                int gpu_count = 0;
                cuDeviceGetCount(&gpu_count);
                for (int dev_id = 0; dev_id < gpu_count; ++dev_id) {
                    auto* pb = pxn_manager_.getProxyBuffer(dev_id);
                    if (!pb || pb->buffer == 0) continue;
                    for (size_t nic = 0; nic < nic_count; ++nic) {
                        uint32_t lkey = getLkey(nic, reinterpret_cast<void*>(pb->buffer));
                        if (lkey != 0) {
                            pxn_manager_.setProxyLkey(dev_id, nic, lkey);
                        }
                    }
                }
                MPCOMM_LOG_INFO("MPComm: PXN proxy buffers registered and lkeys cached\n");
            }
        }
    }
#endif  // USE_CUDA

    // Generate MPComm version config file
    {
        namespace fs = std::filesystem;
        std::string MPCOMM_VERSION = MPCOMM_VERSION_STRING;
        std::string platform = "cuda";
        std::string target_dir = "/dockerdata/.trmt/";
        fs::path full_path = fs::path(target_dir) / "mpcomm.config.json";
        std::fstream file;

        try {
            fs::create_directories(target_dir);
        } catch (const fs::filesystem_error& e) {
            MPCOMM_LOG_WARN("MPComm: No version file generated since the target directory %s is not accessible.\n",
                            target_dir.c_str());
            return MPCOMM_SUCCESS;
        }

        if (fs::exists(full_path)) {
            try {
                std::ifstream tmp(full_path);
                if (!tmp.is_open()) throw std::exception();

                std::string content((std::istreambuf_iterator<char>(tmp)),
                                    std::istreambuf_iterator<char>());
                std::string key = "\"MPCOMM_VERSION\"";
                size_t keyPos, colonPos, valueStart, valueEnd;
                if ((keyPos = content.find(key)) == std::string::npos) throw std::exception();
                if ((colonPos = content.find(':', keyPos + key.length())) == std::string::npos) throw std::exception();
                if ((valueStart = content.find_first_not_of(" \t\n\r", colonPos + 1)) == std::string::npos)
                    throw std::exception();
                if ((valueEnd = content.find('"', valueStart + 1)) == std::string::npos) throw std::exception();
                std::string version_prev = content.substr(valueStart + 1, valueEnd - valueStart - 1);
                if (version_prev == MPCOMM_VERSION) {
                    return MPCOMM_SUCCESS;
                }
            } catch (const std::exception& e) {
                MPCOMM_LOG_WARN("MPComm: Version file %s exists but cannot be updated. Will be rebuilt.\n",
                                full_path.c_str());
            }
        }

        file.open(full_path, std::ios::out | std::ios::trunc);

        auto now = std::chrono::system_clock::now();
        std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_time_t);
        auto now_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch() % std::chrono::seconds(1));
        std::stringstream ss;
        ss << std::put_time(&now_tm, "%Y-%m-%dT%H:%M:%S")
           << "." << std::setfill('0') << std::setw(6) << now_ms.count();

        // Generate JSON
        std::string jsonStr = "{\n";
        jsonStr += "    \"TIMESTAMP\": \"" + ss.str() + "\",\n";
        jsonStr += "    \"MPCOMM_VERSION\": \"" + MPCOMM_VERSION + "\",\n";
        jsonStr += "    \"PLATFORM\": \"" + platform + "\"\n";
        jsonStr += "}\n";

        file << jsonStr;
        file.close();
    }

    return MPCOMM_SUCCESS;
}
/*
shutdown():
  停 Accept 线程 → 原子标记 worker 停止 → join 所有 worker
  → 清理队列/线程 → 关 TCP socket → 销毁 NIC RDMA 资源
  → 清空连接 → 标记未初始化
*/

void MPComm::Impl::shutdown() {
    stopAcceptThread();

    // Stop all worker threads first (they use busy-poll, just set flag and wait)
    if (worker_running_.load()) {
        worker_running_.store(false);
        // Join all worker threads (they will exit when they see worker_running_ = false)
        for (size_t i = 0; i < worker_threads_.size(); ++i) {
            if (worker_threads_[i] && worker_threads_[i]->joinable()) {
                worker_threads_[i]->join();
            }
        }
        worker_threads_.clear();
        worker_queues_.clear();
        numa_worker_rr_.clear();
        MPCOMM_LOG_INFO("MPComm: All %zu worker threads stopped\n", total_workers_);
    }

    // Workers are joined: limit/trace state can be released safely.
    {
        std::lock_guard<std::mutex> lock(nic_limit_mutex_);
        nic_limit_trace_.store(nullptr, std::memory_order_release);
        nic_limit_traces_.clear();
        worker_limit_shared_.clear();
        resetNicLimits();
    }

    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
    }

    // Shutdown PXN subsystem (must happen before NIC cleanup since proxy
    // buffers are registered as memory regions on the NICs)
#ifdef USE_CUDA
    if (pxn_manager_.isEnabled()) {
        auto unreg_fn = [this](void* addr) -> int {
            return unregisterMemory(addr);
        };
        pxn_manager_.unregisterProxyBuffers(unreg_fn);
        pxn_manager_.shutdown();
    }
#endif

    // Cleanup NIC contexts
    for (auto &ctx_ptr : nic_contexts_) {
        cleanupNicContext(*ctx_ptr);
    }
    nic_contexts_.clear();

    connections_.clear();
    initialized_ = false;
    single_nic_pair_ = false;
}
/*
openDevices():
  枚举 RDMA 设备 → 构建设备白名单(参数 < 环境变量优先级)
  → 逐设备: 过滤 → NUMA本地内存分配 NicContext
  → setupNicContext 初始化 RDMA 资源 → 收集到 nic_contexts_
*/
int MPComm::Impl::openDevices(const std::string &device_names) {
    int num_devices = 0;
    struct ibv_device **devices = ibv_get_device_list(&num_devices);
    if (!devices || num_devices <= 0) {
        MPCOMM_LOG_ERROR("MPComm: No RDMA devices found\n");
        return MPCOMM_ERR_DEVICE;
    }

    // Parse device filter from parameter
    std::vector<std::string> filter;
    if (!device_names.empty()) {
        std::istringstream iss(device_names);
        std::string token;
        while (std::getline(iss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
                filter.push_back(token.substr(start, end - start + 1));
            }
        }
    }

    // Check environment variable for additional/override filter
    const char* env_filter = std::getenv(kNicFilterEnvVar);
    if (!single_nic_pair_ && env_filter != nullptr && strlen(env_filter) > 0) {
        // If env var is set, it takes precedence (overrides device_names if provided)
        // Clear existing filter and use env var
        filter.clear();
        std::istringstream iss(env_filter);
        std::string token;
        while (std::getline(iss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
                filter.push_back(token.substr(start, end - start + 1));
            }
        }
        MPCOMM_LOG_INFO("MPComm: Using NIC filter from environment variable %s: %s\n",
               kNicFilterEnvVar, env_filter);
    }

    // Log the active filter
    if (!filter.empty()) {
        std::string devs_str;
        for (size_t i = 0; i < filter.size(); ++i) {
            devs_str += filter[i];
            if (i < filter.size() - 1) devs_str += ", ";
        }
        MPCOMM_LOG_INFO("MPComm: NIC filter active, allowed devices: %s\n", devs_str.c_str());
    }

    for (int i = 0; i < num_devices; ++i) {
        const char *name = ibv_get_device_name(devices[i]);

        // Apply filter if specified
        if (!filter.empty()) {
            bool found = false;
            for (const auto &f : filter) {
                if (f == name) {
                    found = true;
                    break;
                }
            }
            if (!found) continue;
        }

        // Allocate NicContext on the NUMA node where this NIC physically resides.
        // This ensures the worker thread on that NUMA node accesses local memory
        // when polling CQ, posting WRs, and reading NIC metadata.
        int nic_numa = readNicNumaNode(name);
        auto ctx = make_numa_unique<NicContext>(nic_numa);
        ctx->device_name = name;
        ctx->context = nullptr;
        ctx->pd = nullptr;
        ctx->cq = nullptr;
        ctx->port = 1;
        ctx->gid_index = gid_index_;

        int ret = setupNicContext(name, *ctx);
        if (ret == 0) {
            MPCOMM_LOG_INFO("MPComm: Opened device %s, GID=%s (NUMA %d)\n",
                   name, gidToString(ctx->gid).c_str(), nic_numa);
            nic_contexts_.push_back(std::move(ctx));
        }
    }

    ibv_free_device_list(devices);

    // Initialize quarantine state array (one entry per NIC)
    nic_quarantine_.resize(nic_contexts_.size());
    for (auto& ptr : nic_quarantine_) {
        ptr = std::make_unique<NicQuarantineState>();
    }

    return MPCOMM_SUCCESS;
}

int MPComm::Impl::setupNicContext(const std::string &device_name, NicContext &ctx) {
    int num_devices = 0;
    struct ibv_device **devices = ibv_get_device_list(&num_devices);
    if (!devices) return MPCOMM_ERR_DEVICE;

    struct ibv_device *target_device = nullptr;
    for (int i = 0; i < num_devices; ++i) {
        if (device_name == ibv_get_device_name(devices[i])) {
            target_device = devices[i];
            break;
        }
    }

    if (!target_device) {
        ibv_free_device_list(devices);
        return MPCOMM_ERR_DEVICE;
    }

    // Open device
    ctx.context = ibv_open_device(target_device);
    ibv_free_device_list(devices);

    if (!ctx.context) {
        MPCOMM_LOG_ERROR("MPComm: Failed to open device %s\n",
                device_name.c_str());
        return MPCOMM_ERR_CONTEXT;
    }

    // Query port
    struct ibv_port_attr port_attr;
    if (ibv_query_port(ctx.context, ctx.port, &port_attr) != 0) {
        MPCOMM_LOG_ERROR("MPComm: Failed to query port on %s\n",
                device_name.c_str());
        ibv_close_device(ctx.context);
        ctx.context = nullptr;
        return MPCOMM_ERR_CONTEXT;
    }

    if (port_attr.state != IBV_PORT_ACTIVE) {
        MPCOMM_LOG_WARN("MPComm: Port not active on %s\n",
                device_name.c_str());
        ibv_close_device(ctx.context);
        ctx.context = nullptr;
        return MPCOMM_ERR_CONTEXT;
    }

    ctx.lid = port_attr.lid;

    // Find best GID index (prefer RoCEv2 with network device)
    if (ctx.gid_index < 0) {
        for (int i = 0; i < port_attr.gid_tbl_len; ++i) {
            union ibv_gid gid;
            if (ibv_query_gid(ctx.context, ctx.port, i, &gid) != 0) continue;
            if (isNullGid(&gid)) continue;

            ctx.gid_index = i;
            ctx.gid = gid;
            break;
        }
    } else {
        if (ibv_query_gid(ctx.context, ctx.port, ctx.gid_index, &ctx.gid) != 0) {
            MPCOMM_LOG_ERROR("MPComm: Failed to query GID on %s\n",
                    device_name.c_str());
            ibv_close_device(ctx.context);
            ctx.context = nullptr;
            return MPCOMM_ERR_CONTEXT;
        }
        // Check if the queried GID is all-zeros (no routable IPv4-mapped address).
        // This happens on NICs that only have link-local (fe80::) GIDs but no
        // IPv4 configuration. Such NICs cannot establish RoCEv2 connections and
        // would cause modifyQPToRTR to fail with EINVAL.
        if (isNullGid(&ctx.gid)) {
            MPCOMM_LOG_WARN("MPComm: GID index %d on %s is all-zeros (no routable address), skipping device\n",
                    ctx.gid_index, device_name.c_str());
            ibv_close_device(ctx.context);
            ctx.context = nullptr;
            return MPCOMM_ERR_CONTEXT;
        }
    }

    if (ctx.gid_index < 0) {
        MPCOMM_LOG_ERROR("MPComm: No valid GID found on %s\n",
                device_name.c_str());
        ibv_close_device(ctx.context);
        ctx.context = nullptr;
        return MPCOMM_ERR_CONTEXT;
    }

    // Allocate PD
    ctx.pd = ibv_alloc_pd(ctx.context);
    if (!ctx.pd) {
        MPCOMM_LOG_ERROR("MPComm: Failed to allocate PD on %s\n",
                device_name.c_str());
        ibv_close_device(ctx.context);
        ctx.context = nullptr;
        return MPCOMM_ERR_CONTEXT;
    }

    // Create CQ
    ctx.cq = ibv_create_cq(ctx.context, kMaxCQE, nullptr, nullptr, 0);
    if (!ctx.cq) {
        MPCOMM_LOG_ERROR("MPComm: Failed to create CQ on %s\n",
                device_name.c_str());
        ibv_dealloc_pd(ctx.pd);
        ctx.pd = nullptr;
        ibv_close_device(ctx.context);
        ctx.context = nullptr;
        return MPCOMM_ERR_CONTEXT;
    }

    return MPCOMM_SUCCESS;
}

void MPComm::Impl::cleanupNicContext(NicContext &ctx) {
    // Destroy QPs
    {
        std::lock_guard<std::mutex> lock(ctx.qp_mutex);
        for (auto &kv : ctx.qp_map) {
            for (auto *qp : kv.second) {
                if (qp) {
                    ibv_destroy_qp(qp);
                }
            }
        }
        ctx.qp_map.clear();
    }

    // Deregister memory regions
    {
        std::lock_guard<std::mutex> lock(ctx.mr_mutex);
        for (auto &mr_info : ctx.memory_regions) {
            if (mr_info.mr) {
                ibv_dereg_mr(mr_info.mr);
            }
        }
        ctx.memory_regions.clear();
    }

    if (ctx.cq) {
        ibv_destroy_cq(ctx.cq);
        ctx.cq = nullptr;
    }

    if (ctx.pd) {
        ibv_dealloc_pd(ctx.pd);
        ctx.pd = nullptr;
    }

    if (ctx.context) {
        ibv_close_device(ctx.context);
        ctx.context = nullptr;
    }
}

int MPComm::Impl::registerMemory(void *addr, size_t length) {
    if (!initialized_) return MPCOMM_ERR_CONTEXT;
    if (!addr || length == 0) return MPCOMM_ERR_INVALID_ARG;

    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_WRITE;

    // Detect if this is GPU memory and get GPU device ID
    int gpu_device_id = detectGpuDevice(addr);
    bool is_gpu = (gpu_device_id >= 0);

    // Detect NUMA node for this memory region
    int numa_node = -1;
    if (is_gpu) {
        // For GPU memory, move_pages() won't work.
        // Look up the GPU's NUMA node from topology discovery instead.
        numa_node = getGpuNumaNode(gpu_device_id);
        MPCOMM_LOG_INFO("MPComm: Detected GPU memory (device %d, NUMA %d), using nvidia-peermem\n",
               gpu_device_id, numa_node);
        // nvidia-peermem works through the standard ibv_reg_mr path:
        // The kernel's ib_umem_get() calls get_user_pages() which is intercepted
        // by nvidia-peermem (ib_peer_memory_client) to pin GPU pages via the
        // NVIDIA driver. Do NOT add IBV_ACCESS_ON_DEMAND here - ODP uses a
        // different kernel path that does not invoke peer_memory callbacks,
        // resulting in EFAULT for GPU addresses.
    } else {
        // CPU memory: use move_pages() as before
        int status = -1;
        void* pages[1] = { addr };
        if (move_pages(0, 1, pages, nullptr, &status, 0) == 0 && status >= 0) {
            numa_node = status;
        }
    }

    // Discover PCIe-affine NICs for GPU memory (once, before registering on each NIC)
    std::vector<size_t> pcie_affine_nics;
    if (is_gpu) {
        pcie_affine_nics = getGpuPcieAffinityNics(gpu_device_id);
    }

    // Register on all NICs
    for (size_t i = 0; i < nic_contexts_.size(); ++i) {
        auto &ctx = *nic_contexts_[i];

        struct ibv_mr *mr = ibv_reg_mr(ctx.pd, addr, length, access_flags);
        if (!mr) {
            MPCOMM_LOG_ERROR("MPComm: Failed to register %s memory on %s (errno=%d: %s)\n",
                    is_gpu ? "GPU" : "CPU", ctx.device_name.c_str(),
                    errno, strerror(errno));
            if (is_gpu) {
                MPCOMM_LOG_WARN("MPComm: Hint: Ensure nvidia-peermem kernel module is loaded "
                        "(lsmod | grep nvidia_peermem).\n");
                MPCOMM_LOG_WARN("MPComm: Hint: Also verify the GPU memory is valid and "
                        "CUDA context is active on device %d.\n", gpu_device_id);
            }
            // Unregister from previous NICs
            for (size_t j = 0; j < i; ++j) {
                auto &prev_ctx = *nic_contexts_[j];
                std::lock_guard<std::mutex> lock(prev_ctx.mr_mutex);
                for (auto it = prev_ctx.memory_regions.begin();
                     it != prev_ctx.memory_regions.end(); ++it) {
                    if (it->addr == addr) {
                        ibv_dereg_mr(it->mr);
                        prev_ctx.memory_regions.erase(it);
                        break;
                    }
                }
            }
            return MPCOMM_ERR_MEMORY;
        }

        MemoryRegionInfo info;
        info.addr = addr;
        info.length = length;
        info.mr = mr;
        info.lkey = mr->lkey;
        info.rkey = mr->rkey;
        info.numa_node = numa_node;
        info.is_gpu = is_gpu;
        info.gpu_device_id = gpu_device_id;
        info.pcie_affine_nic_indices = pcie_affine_nics;

        std::lock_guard<std::mutex> lock(ctx.mr_mutex);
        ctx.memory_regions.push_back(info);
    }

    MPCOMM_LOG_INFO("MPComm: Registered %s memory %p, length %zu, NUMA node %d%s\n",
           is_gpu ? "GPU" : "CPU", addr, length, numa_node,
           is_gpu ? " (nvidia-peermem)" : "");
    return MPCOMM_SUCCESS;
}

int MPComm::Impl::unregisterMemory(void *addr) {
    if (!initialized_) return MPCOMM_ERR_CONTEXT;
    if (!addr) return MPCOMM_ERR_INVALID_ARG;

    for (auto &ctx_ptr : nic_contexts_) {
        auto &ctx = *ctx_ptr;
        std::lock_guard<std::mutex> lock(ctx.mr_mutex);
        for (auto it = ctx.memory_regions.begin();
             it != ctx.memory_regions.end(); ++it) {
            if (it->addr == addr) {
                ibv_dereg_mr(it->mr);
                ctx.memory_regions.erase(it);
                break;
            }
        }
    }

    return MPCOMM_SUCCESS;
}

uint32_t MPComm::Impl::getLkey(size_t nic_index, void *addr) const {
    if (nic_index >= nic_contexts_.size()) return 0;

    const auto &ctx = *nic_contexts_[nic_index];
    // Note: We can't lock here in const method, assume read-only after init

    for (const auto &mr_info : ctx.memory_regions) {
        char *mr_start = static_cast<char *>(mr_info.addr);
        char *mr_end = mr_start + mr_info.length;
        char *target = static_cast<char *>(addr);

        if (target >= mr_start && target < mr_end) {
            return mr_info.lkey;
        }
    }

    return 0;
}

uint32_t MPComm::Impl::getRkey(size_t nic_index, void *addr) const {
    if (nic_index >= nic_contexts_.size()) return 0;

    const auto &ctx = *nic_contexts_[nic_index];

    for (const auto &mr_info : ctx.memory_regions) {
        char *mr_start = static_cast<char *>(mr_info.addr);
        char *mr_end = mr_start + mr_info.length;
        char *target = static_cast<char *>(addr);

        if (target >= mr_start && target < mr_end) {
            return mr_info.rkey;
        }
    }

    return 0;
}

int MPComm::Impl::updateRemoteMemoryInfo(const std::string &remote_host_id,
                                   const std::vector<uint32_t> &rkeys) {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    auto it = connections_.find(remote_host_id);
    if (it == connections_.end()) {
        MPCOMM_LOG_ERROR("MPComm: Host %s not connected\n", remote_host_id.c_str());
        return MPCOMM_ERR_CONNECTION;
    }

    size_t count = std::min(rkeys.size(), it->second.nic_endpoints.size());
    for (size_t i = 0; i < count; ++i) {
        it->second.nic_endpoints[i].rkey = rkeys[i];
    }

    MPCOMM_LOG_INFO("MPComm: Updated rkeys for %s\n", remote_host_id.c_str());
    return MPCOMM_SUCCESS;
}

std::string MPComm::Impl::getGid(size_t nic_index) const {
    if (nic_index >= nic_contexts_.size()) return "";
    return gidToString(nic_contexts_[nic_index]->gid);
}

std::string MPComm::Impl::getDeviceName(size_t nic_index) const {
    if (nic_index >= nic_contexts_.size()) return "";
    return nic_contexts_[nic_index]->device_name;
}

std::vector<std::string> MPComm::Impl::getActiveDevices() const {
    std::vector<std::string> devices;
    devices.reserve(nic_contexts_.size());
    for (const auto &ctx : nic_contexts_) {
        devices.push_back(ctx->device_name);
    }
    return devices;
}

size_t MPComm::Impl::getMaxRdmaTransferSize() const {
    return max_rdma_transfer_size_;
}

size_t MPComm::Impl::getQpsPerConnection() const {
    return qps_per_connection_;
}

// ============================================================================
// NUMA Topology Discovery
// ============================================================================

// Extract trailing numeric suffix from NIC name (e.g., "mlx5_bond_0" -> 0, "mlx5_2" -> 2)
// Returns -1 if no numeric suffix found
static int extractNicSuffix(const std::string& nic_name) {
    if (nic_name.empty()) return -1;

    // Find the last sequence of digits
    size_t end = nic_name.length();
    size_t start = end;

    // Scan backwards to find digits
    while (start > 0 && std::isdigit(nic_name[start - 1])) {
        --start;
    }

    if (start == end) {
        return -1;  // No digits found
    }

    // Extract the number
    std::string num_str = nic_name.substr(start, end - start);
    return std::stoi(num_str);
}

// Get the number of NUMA nodes in the system
int MPComm::Impl::getNumaNodeCount() {
    int count = 0;
    for (int i = 0; i < 256; i++) {
        std::string path = "/sys/devices/system/node/node" + std::to_string(i);
        if (access(path.c_str(), F_OK) == 0) {
            count = i + 1;
        } else {
            break;
        }
    }
    return std::max(1, count);
}

// Read the NUMA node of a NIC from sysfs
int MPComm::Impl::readNicNumaNode(const std::string& nic_name) {
    std::string path = "/sys/class/infiniband/" + nic_name + "/device/numa_node";
    std::ifstream file(path);
    if (file.is_open()) {
        int numa_node;
        file >> numa_node;
        return numa_node;  // May return -1 if unknown
    }
    return -1;  // Failed to read
}

// Get candidate NICs (either from MPCOMM_NIC_FILTER or all available)
std::vector<std::string> MPComm::Impl::getCandidateNics() {
    std::vector<std::string> candidates;

    // Check MPCOMM_NIC_FILTER environment variable
    const char* filter = std::getenv(kNicFilterEnvVar);
    if (filter && strlen(filter) > 0) {
        // Parse comma-separated list
        std::istringstream iss(filter);
        std::string token;
        while (std::getline(iss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
                candidates.push_back(token.substr(start, end - start + 1));
            }
        }
    } else {
        // Get all available RDMA NICs
        int num_devices = 0;
        struct ibv_device **devices = ibv_get_device_list(&num_devices);
        if (devices) {
            for (int i = 0; i < num_devices; i++) {
                candidates.push_back(ibv_get_device_name(devices[i]));
            }
            ibv_free_device_list(devices);
        }
    }
    return candidates;
}

// Discover NUMA topology and build NIC-to-NUMA mappings
void MPComm::Impl::discoverTopology() {
    int numa_count = getNumaNodeCount();
    std::vector<std::string> candidates = getCandidateNics();

    // Initialize NUMA topology for each node
    numa_topology_.resize(numa_count);
    for (int i = 0; i < numa_count; i++) {
        numa_topology_[i].numa_node = i;
    }

    // Build NIC topology info and assign NICs to NUMA nodes
    nic_topology_.clear();
    nic_topology_.reserve(candidates.size());

    for (const auto& nic : candidates) {
        int numa_node = readNicNumaNode(nic);

        // Store NIC topology info
        NicTopologyInfo info;
        info.nic_name = nic;
        info.numa_node = numa_node;
        nic_topology_.push_back(info);

        // Assign NIC to appropriate NUMA node
        if (numa_node >= 0 && numa_node < numa_count) {
            // NIC is local to this NUMA node
            numa_topology_[numa_node].local_nics.push_back(nic);
        } else {
            // NUMA node unknown (-1), add to all NUMA nodes as remote
            for (int i = 0; i < numa_count; i++) {
                numa_topology_[i].remote_nics.push_back(nic);
            }
        }
    }

    // For NUMA nodes without local NICs, populate remote_nics as fallback
    for (int i = 0; i < numa_count; i++) {
        if (numa_topology_[i].local_nics.empty()) {
            // No local NICs, add all NICs from other NUMA nodes as remote
            for (const auto& info : nic_topology_) {
                if (info.numa_node != i && info.numa_node >= 0) {
                    numa_topology_[i].remote_nics.push_back(info.nic_name);
                }
            }
        }
    }
}

// Print topology discovery results
void MPComm::Impl::printTopologyInfo() {
    if (mpcomm_get_log_level() < MPCOMM_LOG_LEVEL_INFO) return;

    std::string msg;
    msg += "\n================== MPCOMM Topology Discovery ==================\n";
    msg += "System: " + std::to_string(numa_topology_.size()) + " NUMA nodes, "
         + std::to_string(nic_topology_.size()) + " candidate NICs\n";

    // NIC Filter setting
    const char* filter = std::getenv(kNicFilterEnvVar);
    if (filter && strlen(filter) > 0) {
        msg += std::string("NIC Filter: ") + filter + "\n";
    } else {
        msg += "NIC Filter: (none, using all available NICs)\n";
    }
    msg += "\n";

    // NUMA topology
    for (const auto& topo : numa_topology_) {
        msg += "NUMA Node " + std::to_string(topo.numa_node) + ":\n";

        if (!topo.local_nics.empty()) {
            msg += "  Local NICs (optimal): ";
            for (size_t i = 0; i < topo.local_nics.size(); i++) {
                msg += topo.local_nics[i];
                if (i < topo.local_nics.size() - 1) msg += ", ";
            }
            msg += " (" + std::to_string(topo.local_nics.size()) + " NICs)\n";
        } else {
            msg += "  Local NICs: (none)\n";
        }

        if (!topo.remote_nics.empty() && topo.local_nics.empty()) {
            msg += "  Fallback NICs (cross-NUMA): ";
            for (size_t i = 0; i < topo.remote_nics.size(); i++) {
                msg += topo.remote_nics[i];
                if (i < topo.remote_nics.size() - 1) msg += ", ";
            }
            msg += "\n";
        }
        msg += "\n";
    }

    // NIC-to-NUMA mapping
    msg += "NIC -> NUMA Mapping:\n";
    for (const auto& info : nic_topology_) {
        if (info.numa_node >= 0) {
            msg += "  " + info.nic_name + " -> NUMA " + std::to_string(info.numa_node) + "\n";
        } else {
            msg += "  " + info.nic_name + " -> NUMA unknown\n";
        }
    }
    msg += "================================================================\n";

    MPCOMM_LOG_INFO("%s", msg.c_str());
}

// Get NUMA topology information
const std::vector<NumaTopology>& MPComm::Impl::getNumaTopology() const {
    return numa_topology_;
}

// Get NIC topology information
const std::vector<NicTopologyInfo>& MPComm::Impl::getNicTopology() const {
    return nic_topology_;
}

// Get the NUMA node for a specific NIC
int MPComm::Impl::getNicNumaNode(const std::string& nic_name) const {
    for (const auto& info : nic_topology_) {
        if (info.nic_name == nic_name) {
            return info.numa_node;
        }
    }
    return -1;
}

// Get NUMA node for a given memory address (check registered memory regions)
int MPComm::Impl::getNumaNodeForAddr(void* addr) const {
    if (!addr || nic_contexts_.empty()) return -1;

    uintptr_t target = reinterpret_cast<uintptr_t>(addr);

    // Check the first NIC's memory regions (all NICs share the same regions)
    const auto& ctx = *nic_contexts_[0];
    for (const auto& mr : ctx.memory_regions) {
        uintptr_t start = reinterpret_cast<uintptr_t>(mr.addr);
        uintptr_t end = start + mr.length;
        if (target >= start && target < end) {
            return mr.numa_node;
        }
    }
    return -1;  // Not found in registered regions
}

// Get local NIC indices for a specific NUMA node
std::vector<size_t> MPComm::Impl::getLocalNicIndicesForNuma(int numa_node) const {
    std::vector<size_t> indices;

    // If numa_node is invalid, return empty (will fallback to all NICs)
    if (numa_node < 0) {
        return indices;
    }

    // Find NIC indices that belong to this NUMA node
    for (size_t i = 0; i < nic_contexts_.size(); ++i) {
        const auto& device_name = nic_contexts_[i]->device_name;
        for (const auto& topo : nic_topology_) {
            if (topo.nic_name == device_name && topo.numa_node == numa_node) {
                indices.push_back(i);
                break;
            }
        }
    }

    return indices;
}

// Detect if an address is GPU memory using CUDA driver API
// Returns the CUDA device ordinal, or -1 if CPU memory / detection fails
int MPComm::Impl::detectGpuDevice(void* addr) const {
#ifdef USE_CUDA
    // Use cuPointerGetAttribute to query the memory type
    unsigned int mem_type = 0;
    CUresult res = cuPointerGetAttribute(
        &mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        reinterpret_cast<CUdeviceptr>(addr));
    if (res != CUDA_SUCCESS || mem_type != CU_MEMORYTYPE_DEVICE) {
        return -1;  // Not device memory
    }

    // Get the CUDA device ordinal for this pointer
    // CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL requires CUDA 11.x+
    int device_ordinal = -1;
    res = cuPointerGetAttribute(
        &device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        reinterpret_cast<CUdeviceptr>(addr));
    if (res == CUDA_SUCCESS && device_ordinal >= 0) {
        return device_ordinal;
    }

    // Fallback: use cuMemGetAddressRange + iterate devices
    // This is less efficient but works on older CUDA versions
    CUdeviceptr base;
    size_t alloc_size;
    res = cuMemGetAddressRange(&base, &alloc_size,
                                reinterpret_cast<CUdeviceptr>(addr));
    if (res != CUDA_SUCCESS) {
        return -1;
    }

    // Try to determine device by setting context
    int device_count = 0;
    cuDeviceGetCount(&device_count);
    for (int dev = 0; dev < device_count; ++dev) {
        CUdevice cu_dev;
        if (cuDeviceGet(&cu_dev, dev) != CUDA_SUCCESS) continue;
        CUcontext ctx;
        if (cuDevicePrimaryCtxRetain(&ctx, cu_dev) != CUDA_SUCCESS) continue;

        CUcontext old_ctx;
        cuCtxPushCurrent(ctx);

        // Try to get attribute with this device's context
        unsigned int check_type = 0;
        CUresult check = cuPointerGetAttribute(
            &check_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            reinterpret_cast<CUdeviceptr>(addr));

        cuCtxPopCurrent(&old_ctx);
        cuDevicePrimaryCtxRelease(cu_dev);

        if (check == CUDA_SUCCESS && check_type == CU_MEMORYTYPE_DEVICE) {
            return dev;
        }
    }

    // If we confirmed it's device memory but can't determine which device,
    // return 0 as best guess (single-GPU case)
    return 0;
#else
    (void)addr;
    return -1;  // No CUDA support compiled in
#endif
}

// Get the NUMA node for a specific GPU device (delegates to TopologyManager via cached topology)
int MPComm::Impl::getGpuNumaNode(int gpu_device_id) const {
    if (gpu_device_id < 0) return -1;

    // Search GPU topology info discovered during init
    // The topology info is stored in nic_topology_ style - we need to check
    // the topology manager's GPU topology. Since we don't hold a reference
    // to TopologyManager after init, we search the registered memory regions
    // for GPU entries with matching device ID, or use sysfs directly.
#ifdef USE_CUDA
    // Use CUDA driver API to get the PCI bus ID, then read NUMA from sysfs
    CUdevice cu_dev;
    if (cuDeviceGet(&cu_dev, gpu_device_id) != CUDA_SUCCESS) {
        return -1;
    }

    char pci_bus_id[64] = {0};
    if (cuDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cu_dev) != CUDA_SUCCESS) {
        return -1;
    }

    // Normalize to lower case for sysfs
    std::string bdf(pci_bus_id);
    std::transform(bdf.begin(), bdf.end(), bdf.begin(), ::tolower);

    std::string numa_path = "/sys/bus/pci/devices/" + bdf + "/numa_node";
    std::ifstream numa_file(numa_path);
    if (numa_file.is_open()) {
        int numa_node = -1;
        numa_file >> numa_node;
        return numa_node;
    }
#else
    (void)gpu_device_id;
#endif
    return -1;
}

// Discover PCIe-affine NICs for a GPU device by comparing sysfs PCIe paths.
// Returns NIC indices sorted by PCIe affinity (closest first, sharing the
// longest common PCIe path prefix with the GPU).
std::vector<size_t> MPComm::Impl::getGpuPcieAffinityNics(int gpu_device_id) const {
    std::vector<size_t> result;
    if (gpu_device_id < 0) return result;

#ifdef USE_CUDA
    // Get GPU's PCIe BDF from CUDA driver API
    CUdevice cu_dev;
    if (cuDeviceGet(&cu_dev, gpu_device_id) != CUDA_SUCCESS) {
        return result;
    }

    char pci_bus_id_buf[64] = {0};
    if (cuDeviceGetPCIBusId(pci_bus_id_buf, sizeof(pci_bus_id_buf), cu_dev) != CUDA_SUCCESS) {
        return result;
    }

    // Normalize GPU BDF to lower case
    std::string gpu_bdf(pci_bus_id_buf);
    std::transform(gpu_bdf.begin(), gpu_bdf.end(), gpu_bdf.begin(), ::tolower);

    // Resolve GPU's full sysfs PCIe path (e.g., /sys/devices/pci0000:00/0000:00:01.0/.../0000:3b:00.0)
    std::string gpu_sysfs = "/sys/bus/pci/devices/" + gpu_bdf;
    char gpu_resolved[PATH_MAX];
    if (realpath(gpu_sysfs.c_str(), gpu_resolved) == nullptr) {
        MPCOMM_LOG_ERROR("MPComm: Cannot resolve GPU %d sysfs path: %s\n",
                gpu_device_id, gpu_sysfs.c_str());
        return result;
    }
    std::string gpu_path(gpu_resolved);

    // For each NIC, resolve its sysfs PCIe path and compute common prefix with GPU
    struct NicAffinity {
        size_t nic_index;
        size_t common_prefix_len;  // Length of common PCIe path prefix with GPU
    };
    std::vector<NicAffinity> affinities;

    for (size_t i = 0; i < nic_contexts_.size(); ++i) {
        const auto& nic_name = nic_contexts_[i]->device_name;
        // Resolve NIC's PCIe device path via sysfs
        std::string nic_sysfs = "/sys/class/infiniband/" + nic_name + "/device";
        char nic_resolved[PATH_MAX];
        if (realpath(nic_sysfs.c_str(), nic_resolved) == nullptr) {
            continue;
        }
        std::string nic_path(nic_resolved);

        // Compute longest common prefix length between GPU and NIC PCIe paths
        size_t common_len = 0;
        size_t min_len = std::min(gpu_path.size(), nic_path.size());
        for (size_t j = 0; j < min_len; ++j) {
            if (gpu_path[j] == nic_path[j]) {
                common_len = j + 1;
            } else {
                break;
            }
        }

        affinities.push_back({i, common_len});
    }

    if (affinities.empty()) return result;

    // Sort by common prefix length descending (closest PCIe affinity first)
    std::sort(affinities.begin(), affinities.end(),
              [](const NicAffinity& a, const NicAffinity& b) {
                  return a.common_prefix_len > b.common_prefix_len;
              });

    // Find the maximum common prefix length
    size_t max_common = affinities[0].common_prefix_len;

    // Select NICs that share the maximum common PCIe path with GPU
    // (these are under the same PCIe switch)
    for (const auto& aff : affinities) {
        if (aff.common_prefix_len == max_common) {
            result.push_back(aff.nic_index);
        }
    }

    // Print discovery result
    MPCOMM_LOG_INFO("MPComm: GPU %d PCIe affinity discovery: %zu/%zu NICs share closest PCIe switch\n",
           gpu_device_id, result.size(), nic_contexts_.size());
    MPCOMM_LOG_INFO("MPComm:   GPU path: %s\n", gpu_path.c_str());
    for (const auto& aff : affinities) {
        const auto& nic_name = nic_contexts_[aff.nic_index]->device_name;
        bool is_affine = (aff.common_prefix_len == max_common);
        MPCOMM_LOG_INFO("MPComm:   %s: common_prefix=%zu%s\n",
               nic_name.c_str(), aff.common_prefix_len,
               is_affine ? " [AFFINE]" : "");
    }
#else
    (void)gpu_device_id;
#endif  // USE_CUDA

    return result;
}

// Get PCIe-affine NIC indices for a registered memory address
std::vector<size_t> MPComm::Impl::getPcieAffinityNicsForAddr(void* addr) const {
    if (!addr || nic_contexts_.empty()) return {};

    uintptr_t target = reinterpret_cast<uintptr_t>(addr);

    const auto& ctx = *nic_contexts_[0];
    for (const auto& mr : ctx.memory_regions) {
        uintptr_t start = reinterpret_cast<uintptr_t>(mr.addr);
        uintptr_t end = start + mr.length;
        if (target >= start && target < end) {
            return mr.pcie_affine_nic_indices;
        }
    }
    return {};
}

// ============================================================================
// Buffer Publishing and Query (Multi-buffer with NUMA support)
// ============================================================================

int MPComm::Impl::publishBuffer(void *addr, size_t length, int numa_node) {
    if (!initialized_) return MPCOMM_ERR_CONTEXT;
    if (!addr || length == 0) return MPCOMM_ERR_INVALID_ARG;

    std::lock_guard<std::mutex> lock(published_buffer_mutex_);

    // Create published buffer info if not exists
    if (!published_buffer_) {
        published_buffer_ = std::make_unique<PublishedBufferInfo>();
    }

    // Check if buffer already published (update if so)
    uint64_t buf_addr = reinterpret_cast<uint64_t>(addr);
    for (auto &entry : published_buffer_->buffers) {
        if (entry.addr == buf_addr) {
            // Update existing entry
            entry.length = length;
            entry.numa_node = (numa_node >= 0) ? numa_node : getNumaNodeForAddr(addr);
            MPCOMM_LOG_INFO("MPComm: Updated published buffer addr=%p, length=%zu, numa=%d\n",
                   addr, length, entry.numa_node);
            return MPCOMM_SUCCESS;
        }
    }

    // Create new buffer entry
    PublishedBufferEntry entry;
    entry.addr = buf_addr;
    entry.length = length;
    entry.numa_node = (numa_node >= 0) ? numa_node : getNumaNodeForAddr(addr);

    // Get rkeys for all NICs
    entry.rkeys.reserve(nic_contexts_.size());
    for (size_t i = 0; i < nic_contexts_.size(); ++i) {
        uint32_t rkey = getRkey(i, addr);
        if (rkey == 0) {
            MPCOMM_LOG_ERROR("MPComm: Buffer not registered on NIC %zu\n", i);
            return MPCOMM_ERR_MEMORY;
        }
        entry.rkeys.push_back(rkey);
    }

    published_buffer_->buffers.push_back(std::move(entry));

    const auto &added = published_buffer_->buffers.back();
    {
        std::string rkeys_str;
        for (size_t i = 0; i < added.rkeys.size(); ++i) {
            if (i > 0) rkeys_str += ",";
            rkeys_str += std::to_string(added.rkeys[i]);
        }
        MPCOMM_LOG_INFO("MPComm: Published buffer addr=%p, length=%zu, numa=%d, rkeys=[%s] (total %zu buffers)\n",
               addr, length, added.numa_node, rkeys_str.c_str(), published_buffer_->buffers.size());
    }

    return MPCOMM_SUCCESS;
}

int MPComm::Impl::unpublishBuffer(void *addr) {
    if (!initialized_) return MPCOMM_ERR_CONTEXT;

    std::lock_guard<std::mutex> lock(published_buffer_mutex_);

    if (!published_buffer_) {
        return MPCOMM_ERR_INVALID_ARG;
    }

    uint64_t buf_addr = reinterpret_cast<uint64_t>(addr);
    auto &buffers = published_buffer_->buffers;

    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
        if (it->addr == buf_addr) {
            buffers.erase(it);
            MPCOMM_LOG_INFO("MPComm: Unpublished buffer addr=%p (remaining %zu buffers)\n",
                   addr, buffers.size());
            return MPCOMM_SUCCESS;
        }
    }

    return MPCOMM_ERR_INVALID_ARG;
}

void MPComm::Impl::unpublishAllBuffers() {
    std::lock_guard<std::mutex> lock(published_buffer_mutex_);

    if (published_buffer_) {
        size_t count = published_buffer_->buffers.size();
        published_buffer_->buffers.clear();
        MPCOMM_LOG_INFO("MPComm: Unpublished all %zu buffers\n", count);
    }
}

size_t MPComm::Impl::getPublishedBufferCount() const {
    // Note: Not fully thread-safe, but adequate for informational purposes
    if (!published_buffer_) return 0;
    return published_buffer_->buffers.size();
}

int MPComm::Impl::queryRemoteBuffer(const std::string &remote_host_id,
                              const std::string &remote_tcp_addr,
                              int remote_tcp_port,
                              RemoteBufferInfo &out_info) {
    if (!initialized_) return MPCOMM_ERR_CONTEXT;

    MPCOMM_LOG_INFO("MPComm: Querying buffers from %s at %s:%d\n",
           remote_host_id.c_str(), remote_tcp_addr.c_str(), remote_tcp_port);

    // Create TCP connection
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to create socket");
        return MPCOMM_ERR_CONNECTION;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(remote_tcp_port);

    if (inet_pton(AF_INET, remote_tcp_addr.c_str(), &addr.sin_addr) <= 0) {
        struct hostent *he = gethostbyname(remote_tcp_addr.c_str());
        if (!he) {
            MPCOMM_LOG_ERROR("MPComm: Failed to resolve %s\n",
                    remote_tcp_addr.c_str());
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }
        memcpy(&addr.sin_addr, he->h_addr, he->h_length);
    }

    if (single_nic_pair_ && !setSingleNicControlTimeouts(sock_fd)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to set single-NIC buffer query timeouts");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    if (::connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to connect for buffer query");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    int flag = 1;
    setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // Send message type: buffer query
    uint32_t msg_type = kMsgTypeBufferQuery;
    if (send(sock_fd, &msg_type, sizeof(msg_type), single_nic_pair_ ? MSG_NOSIGNAL : 0) !=
            sizeof(msg_type)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to send message type");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    // Receive response: number of buffers (0 = no published buffers)
    uint32_t num_buffers;
    if (recv(sock_fd, &num_buffers, sizeof(num_buffers), MSG_WAITALL) != sizeof(num_buffers)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to receive buffer count");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    if (num_buffers == 0) {
        MPCOMM_LOG_ERROR("MPComm: Remote host has no published buffers\n");
        close(sock_fd);
        return MPCOMM_ERR_INVALID_ARG;
    }

    MPCOMM_LOG_INFO("MPComm: Remote has %u published buffer(s)\n", num_buffers);

    // Receive number of NICs (same for all buffers)
    uint32_t num_nics;
    if (recv(sock_fd, &num_nics, sizeof(num_nics), MSG_WAITALL) != sizeof(num_nics)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to receive num_nics");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    // Receive GIDs for NIC matching (once, same for all buffers)
    std::vector<std::string> remote_gids(num_nics);
    for (uint32_t i = 0; i < num_nics; ++i) {
        char gid_buf[64];
        if (recv(sock_fd, gid_buf, sizeof(gid_buf), MSG_WAITALL) != sizeof(gid_buf)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to receive GID");
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }
        remote_gids[i] = std::string(gid_buf);
    }

    // Receive each buffer's info
    out_info.host_id = remote_host_id;
    out_info.buffers.clear();
    out_info.buffers.reserve(num_buffers);

    for (uint32_t buf_idx = 0; buf_idx < num_buffers; ++buf_idx) {
        RemoteBufferEntry entry;

        // Receive buffer address
        if (recv(sock_fd, &entry.addr, sizeof(entry.addr), MSG_WAITALL) != sizeof(entry.addr)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to receive buffer address");
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }

        // Receive buffer length
        if (recv(sock_fd, &entry.length, sizeof(entry.length), MSG_WAITALL) != sizeof(entry.length)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to receive buffer length");
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }

        // Receive NUMA node
        int32_t numa_node;
        if (recv(sock_fd, &numa_node, sizeof(numa_node), MSG_WAITALL) != sizeof(numa_node)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to receive NUMA node");
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }
        entry.numa_node = numa_node;

        // Receive rkeys for each NIC
        entry.rkeys.resize(num_nics);
        for (uint32_t nic_idx = 0; nic_idx < num_nics; ++nic_idx) {
            uint32_t rkey;
            if (recv(sock_fd, &rkey, sizeof(rkey), MSG_WAITALL) != sizeof(rkey)) {
                MPCOMM_PLOG_ERROR("MPComm: Failed to receive rkey");
                close(sock_fd);
                return MPCOMM_ERR_CONNECTION;
            }
            entry.rkeys[nic_idx] = rkey;
        }

        {
            std::string rkeys_str;
            for (size_t i = 0; i < entry.rkeys.size(); ++i) {
                if (i > 0) rkeys_str += ",";
                rkeys_str += std::to_string(entry.rkeys[i]);
            }
            MPCOMM_LOG_INFO("MPComm: Buffer %u: addr=0x%lx, length=%lu, numa=%d, rkeys=[%s]\n",
                   buf_idx, entry.addr, entry.length, entry.numa_node, rkeys_str.c_str());
        }

        out_info.buffers.push_back(std::move(entry));
    }

    close(sock_fd);

    // Match rkeys to connection endpoints by GID (use first buffer's rkeys for connection)
    // Also store all buffers in remote_buffers for multi-NUMA parallel access
    if (!out_info.buffers.empty()) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto conn_it = connections_.find(remote_host_id);
        if (conn_it != connections_.end()) {
            MPCOMM_LOG_INFO("MPComm: Matching rkeys for %zu endpoints\n",
                   conn_it->second.nic_endpoints.size());

            // Store all remote buffers for multi-NUMA parallel access
            conn_it->second.remote_buffers.clear();
            for (const auto &buf : out_info.buffers) {
                // Map GID to local NIC index to get correct rkey order
                RemoteBufferEntry reordered_entry;
                reordered_entry.addr = buf.addr;
                reordered_entry.length = buf.length;
                reordered_entry.numa_node = buf.numa_node;
                reordered_entry.rkeys.resize(conn_it->second.nic_endpoints.size());

                // Reorder rkeys to match local NIC endpoint indices
                for (size_t ep_idx = 0; ep_idx < conn_it->second.nic_endpoints.size(); ++ep_idx) {
                    const std::string &ep_gid = conn_it->second.nic_endpoints[ep_idx].gid;
                    for (uint32_t remote_idx = 0; remote_idx < num_nics; ++remote_idx) {
                        if (remote_gids[remote_idx] == ep_gid) {
                            reordered_entry.rkeys[ep_idx] = buf.rkeys[remote_idx];
                            break;
                        }
                    }
                }

                conn_it->second.remote_buffers[buf.addr] = reordered_entry;
                MPCOMM_LOG_INFO("MPComm: Stored remote buffer: addr=0x%lx, numa=%d, rkeys=[",
                       buf.addr, buf.numa_node);
                for (size_t i = 0; i < reordered_entry.rkeys.size(); ++i) {
                    MPCOMM_LOG_INFO("%u%s", reordered_entry.rkeys[i],
                           i < reordered_entry.rkeys.size() - 1 ? "," : "");
                }
                MPCOMM_LOG_INFO("]\n");
            }

            // Also update nic_endpoints with first buffer's rkeys for backward compatibility
            const auto &first_buf = out_info.buffers[0];
            for (size_t ep_idx = 0; ep_idx < conn_it->second.nic_endpoints.size(); ++ep_idx) {
                const std::string &ep_gid = conn_it->second.nic_endpoints[ep_idx].gid;
                bool matched = false;
                for (uint32_t remote_idx = 0; remote_idx < num_nics; ++remote_idx) {
                    if (remote_gids[remote_idx] == ep_gid) {
                        conn_it->second.nic_endpoints[ep_idx].rkey = first_buf.rkeys[remote_idx];
                        MPCOMM_LOG_INFO("MPComm: Matched endpoint %zu (GID=%s) -> rkey=%u\n",
                               ep_idx, ep_gid.c_str(), first_buf.rkeys[remote_idx]);
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    MPCOMM_LOG_INFO("MPComm: WARNING: No matching GID found for endpoint %zu\n", ep_idx);
                }
            }
        }
    }

    MPCOMM_LOG_INFO("MPComm: Received %zu buffer(s) from %s\n",
           out_info.buffers.size(), remote_host_id.c_str());

    return MPCOMM_SUCCESS;
}

int MPComm::Impl::queryRemoteBufferByNuma(const std::string &remote_host_id,
                                    const std::string &remote_tcp_addr,
                                    int remote_tcp_port,
                                    int numa_node,
                                    RemoteBufferEntry &out_entry) {
    RemoteBufferInfo all_buffers;
    int ret = queryRemoteBuffer(remote_host_id, remote_tcp_addr, remote_tcp_port, all_buffers);
    if (ret != MPCOMM_SUCCESS) {
        return ret;
    }

    if (all_buffers.buffers.empty()) {
        return MPCOMM_ERR_INVALID_ARG;
    }

    // If numa_node < 0, return first buffer
    if (numa_node < 0) {
        out_entry = all_buffers.buffers[0];
        return MPCOMM_SUCCESS;
    }

    // Find buffer matching NUMA node
    for (const auto &buf : all_buffers.buffers) {
        if (buf.numa_node == numa_node) {
            out_entry = buf;
            return MPCOMM_SUCCESS;
        }
    }

    // Not found, return first buffer as fallback
    MPCOMM_LOG_WARN("MPComm: No buffer found for NUMA node %d, using first buffer\n", numa_node);
    out_entry = all_buffers.buffers[0];
    return MPCOMM_SUCCESS;
}

const PublishedBufferInfo* MPComm::Impl::getPublishedBufferInfo() const {
    // Note: This is not thread-safe for simplicity
    return published_buffer_.get();
}

void MPComm::Impl::handleBufferQuery(int client_fd) {
    const auto send = [this](int fd, const void* data, size_t size, int flags) {
        return ::send(fd, data, size, flags | (single_nic_pair_ ? MSG_NOSIGNAL : 0));
    };
    std::lock_guard<std::mutex> lock(published_buffer_mutex_);

    // Check if any buffers are published
    uint32_t num_buffers = 0;
    if (published_buffer_) {
        num_buffers = static_cast<uint32_t>(published_buffer_->buffers.size());
    }

    // Send number of buffers (0 = no published buffers)
    if (send(client_fd, &num_buffers, sizeof(num_buffers), 0) != sizeof(num_buffers)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to send buffer count");
        return;
    }

    if (num_buffers == 0) {
        return;
    }

    // Send number of NICs (same for all buffers)
    uint32_t num_nics = static_cast<uint32_t>(nic_contexts_.size());
    if (send(client_fd, &num_nics, sizeof(num_nics), 0) != sizeof(num_nics)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to send num_nics");
        return;
    }

    // Send GIDs for all NICs (once, for NIC matching)
    for (uint32_t i = 0; i < num_nics; ++i) {
        std::string gid_str = gidToString(nic_contexts_[i]->gid);
        char gid_buf[64];
        memset(gid_buf, 0, sizeof(gid_buf));
        strncpy(gid_buf, gid_str.c_str(), sizeof(gid_buf) - 1);
        if (send(client_fd, gid_buf, sizeof(gid_buf), 0) != sizeof(gid_buf)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send GID");
            return;
        }
    }

    // Send each buffer's info
    for (uint32_t buf_idx = 0; buf_idx < num_buffers; ++buf_idx) {
        const auto &entry = published_buffer_->buffers[buf_idx];

        // Send buffer address
        if (send(client_fd, &entry.addr, sizeof(entry.addr), 0) != sizeof(entry.addr)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send buffer address");
            return;
        }

        // Send buffer length
        if (send(client_fd, &entry.length, sizeof(entry.length), 0) != sizeof(entry.length)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send buffer length");
            return;
        }

        // Send NUMA node
        int32_t numa_node = entry.numa_node;
        if (send(client_fd, &numa_node, sizeof(numa_node), 0) != sizeof(numa_node)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send NUMA node");
            return;
        }

        // Send rkeys for each NIC
        for (uint32_t nic_idx = 0; nic_idx < num_nics; ++nic_idx) {
            uint32_t rkey = entry.rkeys[nic_idx];
            if (send(client_fd, &rkey, sizeof(rkey), 0) != sizeof(rkey)) {
                MPCOMM_PLOG_ERROR("MPComm: Failed to send rkey");
                return;
            }
        }

        MPCOMM_LOG_INFO("MPComm: Sent buffer %u: addr=0x%lx, length=%lu, numa=%d\n",
               buf_idx, entry.addr, entry.length, entry.numa_node);
    }

    MPCOMM_LOG_INFO("MPComm: Sent %u buffer(s) to client\n", num_buffers);
}

// ============================================================================
// QP Management
// ============================================================================

int MPComm::Impl::createQP(NicContext &ctx, struct ibv_qp **qp) {
    struct ibv_qp_init_attr init_attr;
    memset(&init_attr, 0, sizeof(init_attr));

    init_attr.send_cq = ctx.cq;
    init_attr.recv_cq = ctx.cq;
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.sq_sig_all = 0;
    init_attr.cap.max_send_wr = max_send_wr_;
    init_attr.cap.max_recv_wr = kMaxRecvWR;
    init_attr.cap.max_send_sge = kMaxSGE;
    init_attr.cap.max_recv_sge = kMaxSGE;

    *qp = ibv_create_qp(ctx.pd, &init_attr);
    if (!*qp) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to create QP");
        return MPCOMM_ERR_CONTEXT;
    }

    return MPCOMM_SUCCESS;
}

int MPComm::Impl::modifyQPToInit(NicContext &ctx, struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = ctx.port;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                IBV_QP_ACCESS_FLAGS;

    if (ibv_modify_qp(qp, &attr, flags) != 0) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to modify QP to INIT");
        return MPCOMM_ERR_CONNECTION;
    }

    return MPCOMM_SUCCESS;
}

int MPComm::Impl::modifyQPToRTR(NicContext &ctx, struct ibv_qp *qp,
                          const RemoteEndpointInfo &remote) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 12;

    // Parse remote GID
    stringToGid(remote.gid, attr.ah_attr.grh.dgid);
    attr.ah_attr.grh.sgid_index = ctx.gid_index;
    attr.ah_attr.grh.hop_limit = kMaxHopLimit;
    attr.ah_attr.dlid = remote.lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = ctx.port;

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    if (ibv_modify_qp(qp, &attr, flags) != 0) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to modify QP to RTR");
        MPCOMM_LOG_ERROR("MPComm:   Local NIC: %s, port=%u, gid_index=%d, gid=%s\n",
                ctx.device_name.c_str(), ctx.port, ctx.gid_index,
                gidToString(ctx.gid).c_str());
        MPCOMM_LOG_ERROR("MPComm:   Remote: gid=%s, lid=%u, qp_num=%u\n",
                remote.gid, remote.lid, remote.qp_num);
        return MPCOMM_ERR_CONNECTION;
    }

    return MPCOMM_SUCCESS;
}

int MPComm::Impl::modifyQPToRTS(struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = kTimeout;
    attr.retry_cnt = kRetryCnt;
    attr.rnr_retry = 7;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 16;

    int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

    if (ibv_modify_qp(qp, &attr, flags) != 0) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to modify QP to RTS");
        return MPCOMM_ERR_CONNECTION;
    }

    // Set QP UDP source port for multi-path ECMP load balancing
    // This helps distribute traffic across multiple network paths
#ifdef USE_BNXT
    {
        int port_idx = qp_port_index.fetch_add(1) % (sizeof(qp_port_list) / sizeof(qp_port_list[0]));
        bnxt_re_dv_modify_qp_udp_sport(qp, qp_port_list[port_idx]);
    }
#elif defined(USE_MLNX)
    {
        int port_idx = qp_port_index.fetch_add(1) % (sizeof(qp_port_list) / sizeof(qp_port_list[0]));
        mlx5dv_modify_qp_udp_sport(qp, qp_port_list[port_idx]);
    }
#endif

    return MPCOMM_SUCCESS;
}

void MPComm::Impl::destroyQP(struct ibv_qp *qp) {
    if (qp) {
        ibv_destroy_qp(qp);
    }
}

// ============================================================================
// TCP Handshake for Metadata Exchange
// ============================================================================

int MPComm::Impl::connect(const std::string &remote_host_id,
                    const std::string &remote_tcp_addr,
                    int remote_tcp_port) {
    if (!initialized_) return MPCOMM_ERR_CONTEXT;
    const auto send = [this](int fd, const void* data, size_t size, int flags) {
        return ::send(fd, data, size, flags | (single_nic_pair_ ? MSG_NOSIGNAL : 0));
    };

    MPCOMM_LOG_INFO("MPComm: Connecting to %s at %s:%d\n",
           remote_host_id.c_str(), remote_tcp_addr.c_str(), remote_tcp_port);

    // Create TCP connection
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to create socket");
        return MPCOMM_ERR_CONNECTION;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(remote_tcp_port);

    if (inet_pton(AF_INET, remote_tcp_addr.c_str(), &addr.sin_addr) <= 0) {
        // Try hostname resolution
        struct hostent *he = gethostbyname(remote_tcp_addr.c_str());
        if (!he) {
            MPCOMM_LOG_ERROR("MPComm: Failed to resolve %s\n",
                    remote_tcp_addr.c_str());
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }
        memcpy(&addr.sin_addr, he->h_addr, he->h_length);
    }

    if (single_nic_pair_ && !setSingleNicControlTimeouts(sock_fd)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to set single-NIC control timeouts");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    if (::connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to connect");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    // Disable Nagle's algorithm
    int flag = 1;
    setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // Exchange metadata for each NIC
    ConnectionInfo conn_info;
    conn_info.host_id = remote_host_id;
    conn_info.tcp_port = remote_tcp_port;

    // Send number of local NICs
    uint32_t num_nics = static_cast<uint32_t>(nic_contexts_.size());
    if (single_nic_pair_) {
        if (num_nics != 1 ||
            send(sock_fd, &kMsgTypeSingleNicConnect, sizeof(kMsgTypeSingleNicConnect), 0) !=
                sizeof(kMsgTypeSingleNicConnect)) {
            MPCOMM_LOG_ERROR("MPComm: Failed to start single-NIC handshake\n");
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }
    }
    if (send(sock_fd, &num_nics, sizeof(num_nics), 0) != sizeof(num_nics)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to send num_nics");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    // Receive number of remote NICs
    uint32_t remote_num_nics;
    if (recv(sock_fd, &remote_num_nics, sizeof(remote_num_nics), MSG_WAITALL) !=
        sizeof(remote_num_nics)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_num_nics");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }
    if (single_nic_pair_ && remote_num_nics != 1) {
        MPCOMM_LOG_ERROR("MPComm: Single-NIC peer advertised %u NICs, expected 1\n",
                         remote_num_nics);
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    MPCOMM_LOG_INFO("MPComm: Local NICs=%u, Remote NICs=%u, QPs per connection=%zu\n",
           num_nics, remote_num_nics, qps_per_connection_);

    // Exchange NUMA topology information for NUMA-aware NIC selection
    // Send local NUMA count and per-NIC NUMA node info
    int32_t local_numa_count = static_cast<int32_t>(numa_topology_.size());
    if (send(sock_fd, &local_numa_count, sizeof(local_numa_count), 0) != sizeof(local_numa_count)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to send local_numa_count");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    // Send per-NIC NUMA node info
    std::vector<int32_t> local_nic_numa(num_nics);
    for (size_t i = 0; i < num_nics; ++i) {
        local_nic_numa[i] = getNicNumaNode(nic_contexts_[i]->device_name);
    }
    if (send(sock_fd, local_nic_numa.data(), num_nics * sizeof(int32_t), 0) !=
        static_cast<ssize_t>(num_nics * sizeof(int32_t))) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to send local_nic_numa");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    // Receive remote NUMA topology
    int32_t remote_numa_count;
    if (recv(sock_fd, &remote_numa_count, sizeof(remote_numa_count), MSG_WAITALL) !=
        sizeof(remote_numa_count)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_numa_count");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    std::vector<int32_t> remote_nic_numa(remote_num_nics);
    if (recv(sock_fd, remote_nic_numa.data(), remote_num_nics * sizeof(int32_t), MSG_WAITALL) !=
        static_cast<ssize_t>(remote_num_nics * sizeof(int32_t))) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_nic_numa");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    // Store remote NUMA topology in connection info
    conn_info.remote_numa_count = remote_numa_count;
    conn_info.remote_nic_numa_nodes.resize(remote_num_nics);
    for (size_t i = 0; i < remote_num_nics; ++i) {
        conn_info.remote_nic_numa_nodes[i] = remote_nic_numa[i];
    }

    MPCOMM_LOG_INFO("MPComm: Remote NUMA count=%d, Remote NIC NUMA mapping: ", remote_numa_count);
    for (size_t i = 0; i < remote_num_nics; ++i) {
        MPCOMM_LOG_INFO("NIC%zu->NUMA%d%s", i, remote_nic_numa[i],
               (i < remote_num_nics - 1) ? ", " : "\n");
    }

    // Exchange NIC names for name-based matching
    // Send local NIC names (format: length-prefixed strings)
    for (size_t i = 0; i < num_nics; ++i) {
        const std::string& name = nic_contexts_[i]->device_name;
        uint32_t name_len = static_cast<uint32_t>(name.size());
        if (send(sock_fd, &name_len, sizeof(name_len), 0) != sizeof(name_len)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send nic_name length");
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }
        if (name_len > 0 && send(sock_fd, name.c_str(), name_len, 0) != static_cast<ssize_t>(name_len)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send nic_name");
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }
    }

    // Receive remote NIC names
    std::vector<std::string> remote_nic_names(remote_num_nics);
    for (size_t i = 0; i < remote_num_nics; ++i) {
        uint32_t name_len;
        if (recv(sock_fd, &name_len, sizeof(name_len), MSG_WAITALL) != sizeof(name_len)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to receive nic_name length");
            close(sock_fd);
            return MPCOMM_ERR_CONNECTION;
        }
        if (name_len > 0) {
            std::vector<char> buf(name_len + 1, 0);
            if (recv(sock_fd, buf.data(), name_len, MSG_WAITALL) != static_cast<ssize_t>(name_len)) {
                MPCOMM_PLOG_ERROR("MPComm: Failed to receive nic_name");
                close(sock_fd);
                return MPCOMM_ERR_CONNECTION;
            }
            remote_nic_names[i] = std::string(buf.data());
        }
    }
    conn_info.remote_nic_names = remote_nic_names;

    // Build suffix-to-index mapping for remote NICs
    std::map<int, std::vector<size_t>> remote_suffix_to_nics;  // suffix -> list of remote NIC indices
    for (size_t i = 0; i < remote_num_nics; ++i) {
        int suffix = extractNicSuffix(remote_nic_names[i]);
        if (suffix >= 0) {
            remote_suffix_to_nics[suffix].push_back(i);
        }
    }

    MPCOMM_LOG_INFO("MPComm: Remote NIC names: ");
    for (size_t i = 0; i < remote_num_nics; ++i) {
        int suffix = extractNicSuffix(remote_nic_names[i]);
        MPCOMM_LOG_INFO("%s(suffix=%d)%s", remote_nic_names[i].c_str(), suffix,
               (i < remote_num_nics - 1) ? ", " : "\n");
    }

    // Check how many distinct NUMA nodes the remote NICs actually span
    // This is different from remote_numa_count (system NUMA count)
    std::set<int> remote_nic_numa_set;
    for (size_t i = 0; i < remote_num_nics; ++i) {
        if (remote_nic_numa[i] >= 0) {
            remote_nic_numa_set.insert(remote_nic_numa[i]);
        }
    }
    size_t actual_remote_numa_count = remote_nic_numa_set.size();

    // Build per-NUMA NIC lists for remote side
    std::map<int, std::vector<size_t>> remote_numa_to_nics;
    for (size_t i = 0; i < remote_num_nics; ++i) {
        int numa = remote_nic_numa[i];
        if (numa >= 0) {
            remote_numa_to_nics[numa].push_back(i);
        }
    }

    MPCOMM_LOG_INFO("MPComm: Remote NICs span %zu distinct NUMA node(s)\n", actual_remote_numa_count);

    // Store all remote endpoints (one per remote NIC)
    conn_info.nic_endpoints.resize(remote_num_nics);

    // First, send number of QPs per connection
    uint32_t qps_per_conn = static_cast<uint32_t>(qps_per_connection_);
    if (send(sock_fd, &qps_per_conn, sizeof(qps_per_conn), 0) != sizeof(qps_per_conn)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to send qps_per_conn");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    // Receive remote's QPs per connection (use min of both)
    uint32_t remote_qps_per_conn;
    if (recv(sock_fd, &remote_qps_per_conn, sizeof(remote_qps_per_conn), MSG_WAITALL) !=
        sizeof(remote_qps_per_conn)) {
        MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_qps_per_conn");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }

    size_t actual_qps = std::min(static_cast<size_t>(qps_per_conn),
                                  static_cast<size_t>(remote_qps_per_conn));
    if (single_nic_pair_ && actual_qps == 0) {
        MPCOMM_LOG_ERROR("MPComm: Single-NIC peer advertised zero QPs\n");
        close(sock_fd);
        return MPCOMM_ERR_CONNECTION;
    }
    MPCOMM_LOG_INFO("MPComm: Using %zu QPs per NIC connection\n", actual_qps);

    // Name-based NUMA-aware connection strategy:
    // - Match local and remote NICs by their name suffix (e.g., mlx5_bond_0 matches with mlx5_0)
    // - If remote NICs span multiple NUMA nodes: connect to matching NIC AND
    //   corresponding NIC in other NUMA nodes with same suffix offset
    //   Example: local aa0 connects to remote bb0 (same suffix) and bb4 (same position in NUMA1)
    // - If all remote NICs are on the same NUMA node: simple suffix-based matching

    size_t total_connections = 0;
    size_t skipped_connections = 0;

    // Suffix-based matching strategy with cross-NUMA support:
    // Local NIC with suffix N connects to:
    //   1. Remote NIC with same suffix N (same relative position, primary)
    //   2. Remote NIC with suffix N%4 or N-4 (cross-NUMA, secondary)
    // This ensures that local NUMA1 NICs (suffix 4-7) can reach remote NUMA0 NICs (suffix 0-3)
    // Example: local mlx5_bond_5 -> remote mlx5_bond_5 (same NUMA) + mlx5_bond_1 (cross NUMA)
    //          local mlx5_bond_1 -> remote mlx5_bond_1 (same NUMA) + mlx5_bond_5 (cross NUMA)

    MPCOMM_LOG_INFO("MPComm: Using suffix-based matching (N -> N and N%%4 for cross-NUMA)\n");

    for (size_t local_nic = 0; local_nic < num_nics; ++local_nic) {
        auto &ctx = *nic_contexts_[local_nic];
        const std::string& local_name = ctx.device_name;
        int local_suffix = extractNicSuffix(local_name);

        // Calculate which remote NICs this local NIC should connect to
        std::vector<size_t> target_remote_nics;

        int cross_numa_suffix = (local_suffix >= 4) ? (local_suffix - 4) : (local_suffix + 4);
        if (single_nic_pair_) {
            target_remote_nics.push_back(0);
        } else {
            // Primary: find remote NIC with same suffix N
            auto it = remote_suffix_to_nics.find(local_suffix);
            if (it != remote_suffix_to_nics.end() && !it->second.empty()) {
                target_remote_nics.push_back(it->second[0]);
            }

            // Secondary: find remote NIC with cross-NUMA suffix
            auto it2 = remote_suffix_to_nics.find(cross_numa_suffix);
            if (it2 != remote_suffix_to_nics.end() && !it2->second.empty()) {
                target_remote_nics.push_back(it2->second[0]);
            }
        }

        if (target_remote_nics.empty()) {
            // No matching suffix found, skip this local NIC
            MPCOMM_LOG_INFO("MPComm: Local NIC%zu (%s, suffix=%d) has no matching remote NIC "
                            "(checked %d and %d), skipping\n",
                   local_nic, local_name.c_str(), local_suffix, local_suffix, cross_numa_suffix);
            continue;
        }

        if (target_remote_nics.empty()) {
            continue;  // No remote NICs to connect to
        }

        MPCOMM_LOG_INFO("MPComm: Local NIC%zu (%s, suffix=%d) connecting to remote NICs: ",
               local_nic, local_name.c_str(), local_suffix);
        for (size_t idx = 0; idx < target_remote_nics.size(); ++idx) {
            size_t r = target_remote_nics[idx];
            MPCOMM_LOG_INFO("%zu(%s)%s", r, remote_nic_names[r].c_str(),
                   (idx < target_remote_nics.size() - 1) ? ", " : "\n");
        }

        // Store the local-to-remote NIC mapping for later use in transfers
        conn_info.local_to_remote_nic_map[local_nic] = target_remote_nics;

        // Create connections to each target remote NIC
        for (size_t remote_nic : target_remote_nics) {
            bool connection_failed = false;
            std::string key = remote_host_id + ":" + std::to_string(local_nic) +
                              ":" + std::to_string(remote_nic);
            std::vector<struct ibv_qp *> qp_list;
            qp_list.reserve(actual_qps);

            // Create multiple QPs for this NIC connection
            for (size_t qp_idx = 0; qp_idx < actual_qps; ++qp_idx) {
                // Create QP for this connection
                struct ibv_qp *qp = nullptr;
                int ret = createQP(ctx, &qp);
                if (ret != 0) {
                    // Cleanup already created QPs
                    for (auto *created_qp : qp_list) {
                        destroyQP(created_qp);
                    }
                    close(sock_fd);
                    return ret;
                }

                ret = modifyQPToInit(ctx, qp);
                if (ret != 0) {
                    destroyQP(qp);
                    for (auto *created_qp : qp_list) {
                        destroyQP(created_qp);
                    }
                    close(sock_fd);
                    return ret;
                }

                // Prepare local info with target remote NIC index
                RemoteEndpointInfo local_info;
                memset(&local_info, 0, sizeof(local_info));
                strncpy(local_info.gid, gidToString(ctx.gid).c_str(),
                        sizeof(local_info.gid) - 1);
                local_info.lid = ctx.lid;
                local_info.qp_num = qp->qp_num;
                // Encode target remote NIC in addr field for passive side to know
                local_info.addr = remote_nic;
                // Encode source local NIC in length field for passive side to track
                local_info.length = local_nic;

                // Send local info
                if (send(sock_fd, &local_info, sizeof(local_info), 0) !=
                    sizeof(local_info)) {
                    MPCOMM_PLOG_ERROR("MPComm: Failed to send local_info");
                    destroyQP(qp);
                    for (auto *created_qp : qp_list) {
                        destroyQP(created_qp);
                    }
                    close(sock_fd);
                    return MPCOMM_ERR_CONNECTION;
                }

                // Receive remote info
                RemoteEndpointInfo remote_info;
                if (recv(sock_fd, &remote_info, sizeof(remote_info), MSG_WAITALL) !=
                    sizeof(remote_info)) {
                    MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_info");
                    destroyQP(qp);
                    for (auto *created_qp : qp_list) {
                        destroyQP(created_qp);
                    }
                    close(sock_fd);
                    return MPCOMM_ERR_CONNECTION;
                }

                MPCOMM_LOG_INFO("MPComm: NIC %s->%s QP[%zu]: Local QPN=%u, Remote QPN=%u\n",
                       nic_contexts_[local_nic]->device_name.c_str(),
                       remote_nic_names[remote_nic].c_str(),
                       qp_idx, local_info.qp_num, remote_info.qp_num);

                // Complete QP setup
                ret = modifyQPToRTR(ctx, qp, remote_info);
                if (ret != 0) {
                    // Send failure status to passive side so it can skip too
                    uint8_t status = 1;  // 1 = failure
                    send(sock_fd, &status, sizeof(status), 0);
                    MPCOMM_LOG_WARN("MPComm: Skipping NIC %s->%s (modifyQPToRTR failed), continuing with other NICs\n",
                           nic_contexts_[local_nic]->device_name.c_str(),
                           remote_nic_names[remote_nic].c_str());
                    destroyQP(qp);
                    for (auto *created_qp : qp_list) {
                        destroyQP(created_qp);
                    }
                    qp_list.clear();
                    connection_failed = true;
                    break;
                }

                ret = modifyQPToRTS(qp);
                if (ret != 0) {
                    // Send failure status to passive side so it can skip too
                    uint8_t status = 1;  // 1 = failure
                    send(sock_fd, &status, sizeof(status), 0);
                    MPCOMM_LOG_WARN("MPComm: Skipping NIC %s->%s (modifyQPToRTS failed), continuing with other NICs\n",
                           nic_contexts_[local_nic]->device_name.c_str(),
                           remote_nic_names[remote_nic].c_str());
                    destroyQP(qp);
                    for (auto *created_qp : qp_list) {
                        destroyQP(created_qp);
                    }
                    qp_list.clear();
                    connection_failed = true;
                    break;
                }

                // Send success status to passive side
                {
                    uint8_t status = 0;  // 0 = success
                    if (send(sock_fd, &status, sizeof(status), 0) != sizeof(status)) {
                        MPCOMM_PLOG_ERROR("MPComm: Failed to send QP status");
                        destroyQP(qp);
                        for (auto *created_qp : qp_list) {
                            destroyQP(created_qp);
                        }
                        close(sock_fd);
                        return MPCOMM_ERR_CONNECTION;
                    }
                }

                if (single_nic_pair_) {
                    uint8_t passive_status = 1;
                    if (recv(sock_fd, &passive_status, sizeof(passive_status), MSG_WAITALL) !=
                            sizeof(passive_status) || passive_status != 0) {
                        MPCOMM_LOG_ERROR("MPComm: Single-NIC passive QP[%zu] setup failed\n", qp_idx);
                        destroyQP(qp);
                        for (auto *created_qp : qp_list) {
                            destroyQP(created_qp);
                        }
                        close(sock_fd);
                        return MPCOMM_ERR_CONNECTION;
                    }
                }

                qp_list.push_back(qp);

                // Store remote endpoint info for this remote NIC
                if (qp_idx == 0) {
                    conn_info.nic_endpoints[remote_nic] = remote_info;
                }
            }

            if (connection_failed) {
                if (single_nic_pair_) {
                    MPCOMM_LOG_ERROR("MPComm: Single-NIC QP setup failed; connection rejected\n");
                    close(sock_fd);
                    return MPCOMM_ERR_CONNECTION;
                }
                // Remove this remote NIC from the mapping since connection failed
                auto map_it = conn_info.local_to_remote_nic_map.find(local_nic);
                if (map_it != conn_info.local_to_remote_nic_map.end()) {
                    auto &targets = map_it->second;
                    targets.erase(std::remove(targets.begin(), targets.end(), remote_nic), targets.end());
                    if (targets.empty()) {
                        conn_info.local_to_remote_nic_map.erase(map_it);
                    }
                }
                skipped_connections++;
                continue;  // Skip to next remote NIC
            }

            // Store QP list with new key format: "host:local_nic:remote_nic"
            {
                std::lock_guard<std::mutex> lock(ctx.qp_mutex);
                ctx.qp_map[key] = std::move(qp_list);
            }
            total_connections++;
        }
    }

    close(sock_fd);

    if (total_connections == 0) {
        MPCOMM_LOG_ERROR("MPComm: No NIC connections established to %s (all %zu skipped)\n",
               remote_host_id.c_str(), skipped_connections);
        return MPCOMM_ERR_CONNECTION;
    }

    // Store connection info
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        connections_[remote_host_id] = conn_info;
    }

    if (skipped_connections > 0) {
        MPCOMM_LOG_WARN("MPComm: Connected to %s with %zu NIC connections "
                        "(%zu skipped due to unreachable NICs), %zu QPs each\n",
               remote_host_id.c_str(), total_connections, skipped_connections, actual_qps);
    } else {
        MPCOMM_LOG_INFO("MPComm: Connected to %s with %zu NIC connections (NUMA-aware), %zu QPs each\n",
               remote_host_id.c_str(), total_connections, actual_qps);
    }

    return MPCOMM_SUCCESS;
}

int MPComm::Impl::startAcceptThread() {
    if (listen_fd_ < 0) {
        MPCOMM_LOG_ERROR("MPComm: TCP listener not initialized\n");
        return MPCOMM_ERR_CONNECTION;
    }

    accept_running_ = true;
    accept_thread_ = std::make_unique<std::thread>(&MPComm::Impl::acceptLoop, this);
    return MPCOMM_SUCCESS;
}

void MPComm::Impl::stopAcceptThread() {
    accept_running_ = false;

    // Interrupt accept() by connecting to ourselves
    if (listen_fd_ >= 0 && accept_thread_) {
        int dummy_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (dummy_sock >= 0) {
            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            addr.sin_port = htons(tcp_port_);
            ::connect(dummy_sock, (struct sockaddr *)&addr, sizeof(addr));
            close(dummy_sock);
        }
    }

    if (accept_thread_ && accept_thread_->joinable()) {
        accept_thread_->join();
    }
    accept_thread_.reset();
}

void MPComm::Impl::acceptLoop() {
    const auto send = [this](int fd, const void* data, size_t size, int flags) {
        return ::send(fd, data, size, flags | (single_nic_pair_ ? MSG_NOSIGNAL : 0));
    };
    while (accept_running_) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);

        int client_fd = accept(listen_fd_, (struct sockaddr *)&client_addr,
                               &addr_len);
        if (client_fd < 0) {
            if (accept_running_) {
                MPCOMM_PLOG_ERROR("MPComm: Accept failed");
            }
            continue;
        }

        if (!accept_running_) {
            close(client_fd);
            break;
        }

        // Handle client in same thread (simple implementation)
        MPCOMM_LOG_INFO("MPComm: Accepted connection from %s:%d\n",
               inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

        // Disable Nagle's algorithm
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        if (single_nic_pair_ && !setSingleNicControlTimeouts(client_fd)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to set single-NIC control timeouts");
            close(client_fd);
            continue;
        }

        // Receive message type (first 4 bytes)
        uint32_t msg_type;
        if (recv(client_fd, &msg_type, sizeof(msg_type), MSG_WAITALL) !=
            sizeof(msg_type)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to receive message type");
            close(client_fd);
            continue;
        }

        if (msg_type == kMsgTypeBufferQuery) {
            // Handle buffer query request
            handleBufferQuery(client_fd);
            close(client_fd);
            continue;
        }

        const bool single_request = msg_type == kMsgTypeSingleNicConnect;
        if (single_request != single_nic_pair_) {
            MPCOMM_LOG_ERROR("MPComm: Single-NIC and legacy connection modes do not match\n");
            close(client_fd);
            continue;
        }

        // Legacy requests encode num_nics directly in the first word.
        uint32_t remote_num_nics = msg_type;
        uint32_t num_nics = static_cast<uint32_t>(nic_contexts_.size());
        if (single_request) {
            if (recv(client_fd, &remote_num_nics, sizeof(remote_num_nics), MSG_WAITALL) !=
                    sizeof(remote_num_nics) || remote_num_nics != 1 || num_nics != 1) {
                MPCOMM_LOG_ERROR("MPComm: Single-NIC handshake requires exactly one NIC on both peers\n");
                close(client_fd);
                continue;
            }
        }

        // Send number of local NICs
        if (send(client_fd, &num_nics, sizeof(num_nics), 0) !=
            sizeof(num_nics)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send num_nics");
            close(client_fd);
            continue;
        }

        // Generate temporary remote host ID
        char remote_id_buf[64];
        snprintf(remote_id_buf, sizeof(remote_id_buf), "%s:%d",
                 inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
        std::string remote_host_id = remote_id_buf;

        ConnectionInfo conn_info;
        conn_info.host_id = remote_host_id;
        conn_info.tcp_port = ntohs(client_addr.sin_port);

        // Receive remote NUMA topology (part of new protocol)
        int32_t remote_numa_count;
        if (recv(client_fd, &remote_numa_count, sizeof(remote_numa_count), MSG_WAITALL) !=
            sizeof(remote_numa_count)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_numa_count");
            close(client_fd);
            continue;
        }

        std::vector<int32_t> remote_nic_numa(remote_num_nics);
        if (recv(client_fd, remote_nic_numa.data(), remote_num_nics * sizeof(int32_t), MSG_WAITALL) !=
            static_cast<ssize_t>(remote_num_nics * sizeof(int32_t))) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_nic_numa");
            close(client_fd);
            continue;
        }

        // Send local NUMA topology
        int32_t local_numa_count = static_cast<int32_t>(numa_topology_.size());
        if (send(client_fd, &local_numa_count, sizeof(local_numa_count), 0) !=
            sizeof(local_numa_count)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send local_numa_count");
            close(client_fd);
            continue;
        }

        std::vector<int32_t> local_nic_numa(num_nics);
        for (size_t i = 0; i < num_nics; ++i) {
            local_nic_numa[i] = getNicNumaNode(nic_contexts_[i]->device_name);
        }
        if (send(client_fd, local_nic_numa.data(), num_nics * sizeof(int32_t), 0) !=
            static_cast<ssize_t>(num_nics * sizeof(int32_t))) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send local_nic_numa");
            close(client_fd);
            continue;
        }

        // Store remote NUMA topology
        conn_info.remote_numa_count = remote_numa_count;
        conn_info.remote_nic_numa_nodes.resize(remote_num_nics);
        for (size_t i = 0; i < remote_num_nics; ++i) {
            conn_info.remote_nic_numa_nodes[i] = remote_nic_numa[i];
        }

        MPCOMM_LOG_INFO("MPComm: Passive side: Remote NUMA count=%d, Remote NIC NUMA mapping: ", remote_numa_count);
        for (size_t i = 0; i < remote_num_nics; ++i) {
            MPCOMM_LOG_INFO("NIC%zu->NUMA%d%s", i, remote_nic_numa[i],
                   (i < remote_num_nics - 1) ? ", " : "\n");
        }

        // Receive remote NIC names (passive side receives first, then sends)
        bool nic_names_ok = true;
        std::vector<std::string> remote_nic_names(remote_num_nics);
        for (size_t i = 0; i < remote_num_nics; ++i) {
            uint32_t name_len;
            if (recv(client_fd, &name_len, sizeof(name_len), MSG_WAITALL) != sizeof(name_len)) {
                MPCOMM_PLOG_ERROR("MPComm: Failed to receive nic_name length");
                if (single_request) {
                    nic_names_ok = false;
                    break;
                }
                close(client_fd);
                continue;
            }
            if (name_len > 0) {
                std::vector<char> buf(name_len + 1, 0);
                if (recv(client_fd, buf.data(), name_len, MSG_WAITALL) != static_cast<ssize_t>(name_len)) {
                    MPCOMM_PLOG_ERROR("MPComm: Failed to receive nic_name");
                    if (single_request) {
                        nic_names_ok = false;
                        break;
                    }
                    close(client_fd);
                    continue;
                }
                remote_nic_names[i] = std::string(buf.data());
            }
        }
        if (!nic_names_ok) {
            close(client_fd);
            continue;
        }
        conn_info.remote_nic_names = remote_nic_names;

        // Send local NIC names
        for (size_t i = 0; i < num_nics; ++i) {
            const std::string& name = nic_contexts_[i]->device_name;
            uint32_t name_len = static_cast<uint32_t>(name.size());
            if (send(client_fd, &name_len, sizeof(name_len), 0) != sizeof(name_len)) {
                MPCOMM_PLOG_ERROR("MPComm: Failed to send nic_name length");
                if (single_request) {
                    nic_names_ok = false;
                    break;
                }
                close(client_fd);
                continue;
            }
            if (name_len > 0 && send(client_fd, name.c_str(), name_len, 0) != static_cast<ssize_t>(name_len)) {
                MPCOMM_PLOG_ERROR("MPComm: Failed to send nic_name");
                if (single_request) {
                    nic_names_ok = false;
                    break;
                }
                close(client_fd);
                continue;
            }
        }
        if (!nic_names_ok) {
            close(client_fd);
            continue;
        }

        // Build suffix-to-index mapping for remote NICs
        std::map<int, std::vector<size_t>> remote_suffix_to_nics;
        for (size_t i = 0; i < remote_num_nics; ++i) {
            int suffix = extractNicSuffix(remote_nic_names[i]);
            if (suffix >= 0) {
                remote_suffix_to_nics[suffix].push_back(i);
            }
        }

        MPCOMM_LOG_INFO("MPComm: Passive side: Remote NIC names: ");
        for (size_t i = 0; i < remote_num_nics; ++i) {
            int suffix = extractNicSuffix(remote_nic_names[i]);
            MPCOMM_LOG_INFO("%s(suffix=%d)%s", remote_nic_names[i].c_str(), suffix,
                   (i < remote_num_nics - 1) ? ", " : "\n");
        }

        // Store all remote endpoints (one per remote NIC)
        conn_info.nic_endpoints.resize(remote_num_nics);

        // Receive remote's QPs per connection
        uint32_t remote_qps_per_conn;
        if (recv(client_fd, &remote_qps_per_conn, sizeof(remote_qps_per_conn),
                 MSG_WAITALL) != sizeof(remote_qps_per_conn)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_qps_per_conn");
            close(client_fd);
            continue;
        }

        // Send our QPs per connection
        uint32_t qps_per_conn = static_cast<uint32_t>(qps_per_connection_);
        if (send(client_fd, &qps_per_conn, sizeof(qps_per_conn), 0) !=
            sizeof(qps_per_conn)) {
            MPCOMM_PLOG_ERROR("MPComm: Failed to send qps_per_conn");
            close(client_fd);
            continue;
        }

        size_t actual_qps = std::min(static_cast<size_t>(qps_per_conn),
                                      static_cast<size_t>(remote_qps_per_conn));
        if (single_request && actual_qps == 0) {
            MPCOMM_LOG_ERROR("MPComm: Single-NIC peer advertised zero QPs\n");
            close(client_fd);
            continue;
        }
        MPCOMM_LOG_INFO("MPComm: Passive side: Using %zu QPs per NIC connection\n", actual_qps);

        // Check how many distinct NUMA nodes the remote NICs actually span
        // Must match the logic in connect() on active side
        std::set<int> remote_nic_numa_set;
        for (size_t i = 0; i < remote_num_nics; ++i) {
            if (remote_nic_numa[i] >= 0) {
                remote_nic_numa_set.insert(remote_nic_numa[i]);
            }
        }

        // Build per-NUMA NIC lists for remote side
        std::map<int, std::vector<size_t>> remote_numa_to_nics;
        for (size_t i = 0; i < remote_num_nics; ++i) {
            int numa = remote_nic_numa[i];
            if (numa >= 0) {
                remote_numa_to_nics[numa].push_back(i);
            }
        }

        // Build suffix to local NIC mapping for quick lookup
        std::map<int, size_t> local_suffix_to_nic;
        for (size_t i = 0; i < num_nics; ++i) {
            int suffix = extractNicSuffix(nic_contexts_[i]->device_name);
            if (suffix >= 0) {
                local_suffix_to_nic[suffix] = i;
            }
        }

        MPCOMM_LOG_INFO("MPComm: Passive side: Using suffix-based matching (N -> N and cross-NUMA)\n");

        // Calculate expected number of connections (must match active side exactly)
        // Active side: for each local NIC with suffix N, connects to remote NICs with suffix N and cross-NUMA
        // On passive side, we need to calculate how many remote NICs will connect to us
        size_t expected_connections = single_request ? 1 : 0;

        // For each remote NIC (which is the active side's local NIC)
        for (size_t remote_nic = 0; !single_request && remote_nic < remote_num_nics; ++remote_nic) {
            int remote_suffix = extractNicSuffix(remote_nic_names[remote_nic]);

            // Active side with suffix N will try to connect to local NICs with suffix N and cross-NUMA
            // Count connections where we have matching suffix N (same as remote)
            if (local_suffix_to_nic.count(remote_suffix) > 0) {
                expected_connections++;  // Primary: N -> N
            }

            // Count connections where we have cross-NUMA suffix
            // Active side N connects to our N-4 (if N>=4) or N+4 (if N<4)
            int cross_numa_suffix = (remote_suffix >= 4) ? (remote_suffix - 4) : (remote_suffix + 4);
            if (local_suffix_to_nic.count(cross_numa_suffix) > 0) {
                expected_connections++;  // Secondary: cross-NUMA
            }
        }

        MPCOMM_LOG_INFO("MPComm: Passive side: Expecting %zu connections\n", expected_connections);

        bool success = true;
        size_t total_connections = 0;

        // Receive connections from active side - the active side determines which
        // local NIC connects to which remote NIC (encoded in remote_info)
        // remote_info.addr = target local NIC on passive side
        // remote_info.length = source remote NIC (active side's local NIC)
        while (total_connections < expected_connections && success) {
            // Receive first QP info to determine connection mapping
            RemoteEndpointInfo remote_info;
            if (recv(client_fd, &remote_info, sizeof(remote_info), MSG_WAITALL) !=
                sizeof(remote_info)) {
                MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_info");
                success = false;
                break;
            }

            // Decode: remote_info.addr contains the target local NIC index on passive side
            size_t local_nic = static_cast<size_t>(remote_info.addr);
            if (local_nic >= num_nics) {
                MPCOMM_LOG_WARN("MPComm: Invalid local NIC index %zu from active side\n", local_nic);
                success = false;
                break;
            }

            // Decode: remote_info.length contains the source remote NIC (active side's local NIC)
            size_t remote_nic = static_cast<size_t>(remote_info.length);
            if (remote_nic >= remote_num_nics) {
                MPCOMM_LOG_WARN("MPComm: Invalid remote NIC index %zu from active side\n", remote_nic);
                success = false;
                break;
            }

            auto &ctx = *nic_contexts_[local_nic];
            std::string key = remote_host_id + ":" + std::to_string(local_nic) +
                              ":" + std::to_string(remote_nic);
            std::vector<struct ibv_qp *> qp_list;
            qp_list.reserve(actual_qps);
            bool connection_skipped = false;

            // Process first QP (already received remote_info)
            for (size_t qp_idx = 0; qp_idx < actual_qps && success; ++qp_idx) {
                // For subsequent QPs, receive remote_info
                if (qp_idx > 0) {
                    if (recv(client_fd, &remote_info, sizeof(remote_info), MSG_WAITALL) !=
                        sizeof(remote_info)) {
                        MPCOMM_PLOG_ERROR("MPComm: Failed to receive remote_info");
                        success = false;
                        break;
                    }
                }

                // Create QP
                struct ibv_qp *qp = nullptr;
                if (createQP(ctx, &qp) != 0) {
                    success = false;
                    break;
                }

                if (modifyQPToInit(ctx, qp) != 0) {
                    destroyQP(qp);
                    success = false;
                    break;
                }

                // Prepare local info
                RemoteEndpointInfo local_info;
                memset(&local_info, 0, sizeof(local_info));
                strncpy(local_info.gid, gidToString(ctx.gid).c_str(),
                        sizeof(local_info.gid) - 1);
                local_info.lid = ctx.lid;
                local_info.qp_num = qp->qp_num;

                // Send local info
                if (send(client_fd, &local_info, sizeof(local_info), 0) !=
                    sizeof(local_info)) {
                    MPCOMM_PLOG_ERROR("MPComm: Failed to send local_info");
                    destroyQP(qp);
                    success = false;
                    break;
                }

                // Receive status from active side (whether modifyQPToRTR/RTS succeeded)
                uint8_t active_status = 0;
                if (recv(client_fd, &active_status, sizeof(active_status), MSG_WAITALL) !=
                    sizeof(active_status)) {
                    MPCOMM_PLOG_ERROR("MPComm: Failed to receive QP status from active side");
                    destroyQP(qp);
                    success = false;
                    break;
                }

                if (active_status != 0) {
                    // Active side failed to establish this connection, skip it
                    MPCOMM_LOG_WARN("MPComm: Passive side: Active side reported failure for NIC %s<-%s, skipping\n",
                           nic_contexts_[local_nic]->device_name.c_str(),
                           remote_nic_names[remote_nic].c_str());
                    destroyQP(qp);
                    for (auto *created_qp : qp_list) {
                        destroyQP(created_qp);
                    }
                    qp_list.clear();
                    if (single_request) {
                        uint8_t passive_status = 1;
                        send(client_fd, &passive_status, sizeof(passive_status), 0);
                        success = false;
                    } else {
                        connection_skipped = true;
                    }
                    break;
                }

                // Complete QP setup
                if (modifyQPToRTR(ctx, qp, remote_info) != 0 ||
                    modifyQPToRTS(qp) != 0) {
                    if (single_request) {
                        uint8_t passive_status = 1;
                        send(client_fd, &passive_status, sizeof(passive_status), 0);
                    }
                    destroyQP(qp);
                    success = false;
                    break;
                }
                if (single_request) {
                    uint8_t passive_status = 0;
                    if (send(client_fd, &passive_status, sizeof(passive_status), 0) !=
                            sizeof(passive_status)) {
                        MPCOMM_PLOG_ERROR("MPComm: Failed to send passive QP status");
                        destroyQP(qp);
                        success = false;
                        break;
                    }
                }

                qp_list.push_back(qp);

                // Store remote endpoint info
                if (qp_idx == 0) {
                    // Clear the encoded local_nic from addr field
                    remote_info.addr = 0;
                    conn_info.nic_endpoints[remote_nic] = remote_info;
                }

                MPCOMM_LOG_INFO("MPComm: Passive NIC %s<-%s QP[%zu]: Local QPN=%u, Remote QPN=%u\n",
                       nic_contexts_[local_nic]->device_name.c_str(),
                       remote_nic_names[remote_nic].c_str(),
                       qp_idx, local_info.qp_num, remote_info.qp_num);
            }

            if (connection_skipped) {
                // Active side reported failure, skip this connection but continue
                total_connections++;  // Still counts toward expected_connections
                continue;
            } else if (success) {
                // Store QP list with new key format
                std::lock_guard<std::mutex> lock(ctx.qp_mutex);
                ctx.qp_map[key] = std::move(qp_list);
                total_connections++;

                // Store the local-to-remote NIC mapping
                // On passive side: local_nic is our NIC, remote_nic is the active side's NIC
                conn_info.local_to_remote_nic_map[local_nic].push_back(remote_nic);
            } else {
                // Cleanup on failure
                for (auto *qp : qp_list) {
                    destroyQP(qp);
                }
            }
        }

        close(client_fd);

        if (success) {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            connections_[remote_host_id] = conn_info;
            MPCOMM_LOG_INFO("MPComm: Passive connection established with %s (%zu NIC connections, NUMA-aware)\n",
                   remote_host_id.c_str(), total_connections);
        }
    }
}

struct ibv_qp *MPComm::Impl::getOrCreateQP(size_t local_nic_index,
                                     const std::string &remote_host_id,
                                     size_t remote_nic_index,
                                     size_t qp_index) {
    if (local_nic_index >= nic_contexts_.size()) {
        return nullptr;
    }

    auto &ctx = *nic_contexts_[local_nic_index];
    // New key format: "host_id:local_nic:remote_nic" for NUMA-aware connections
    std::string key = remote_host_id + ":" + std::to_string(local_nic_index) +
                      ":" + std::to_string(remote_nic_index);

    std::lock_guard<std::mutex> lock(ctx.qp_mutex);
    auto it = ctx.qp_map.find(key);
    if (it != ctx.qp_map.end()) {
        const auto &qp_list = it->second;
        if (qp_index < qp_list.size()) {
            return qp_list[qp_index];
        }
        // If qp_index is out of range, return first QP (fallback)
        if (!qp_list.empty()) {
            return qp_list[0];
        }
    }

    return nullptr;  // QP should be created during connect()
}

// ==================== Async Transfer Implementation ====================

TransferHandle MPComm::Impl::scatterAsync(uintptr_t local_addr,
                                    const std::vector<std::string> &host_list,
                                    const std::vector<uintptr_t> &remote_addrs,
                                    const std::vector<size_t> &lengths) {
    return transferAsyncStart(local_addr, host_list, remote_addrs, lengths,
                              TransferDirection::SCATTER);
}

TransferHandle MPComm::Impl::broadcastAsync(uintptr_t local_addr,
                                      size_t length,
                                      const std::vector<std::string> &host_list,
                                      const std::vector<uintptr_t> &remote_addrs) {
    // Broadcast: same length for all hosts, expand to lengths vector
    size_t host_count = host_list.size();
    if (host_count == 0 || remote_addrs.size() != host_count) {
        MPCOMM_LOG_ERROR("MPComm: BroadcastAsync failed - invalid arguments\n");
        return INVALID_TRANSFER_HANDLE;
    }
    if (length == 0) {
        MPCOMM_LOG_ERROR("MPComm: BroadcastAsync failed - zero length\n");
        return INVALID_TRANSFER_HANDLE;
    }
    std::vector<size_t> lengths(host_count, length);
    return transferAsyncStart(local_addr, host_list, remote_addrs, lengths,
                              TransferDirection::BROADCAST);
}

TransferHandle MPComm::Impl::gatherAsync(uintptr_t local_addr,
                                   const std::vector<std::string> &host_list,
                                   const std::vector<uintptr_t> &remote_addrs,
                                   const std::vector<size_t> &lengths) {
    return transferAsyncStart(local_addr, host_list, remote_addrs, lengths,
                              TransferDirection::GATHER);
}

TransferHandle MPComm::Impl::putAsync(uintptr_t local_addr,
                                const std::string &remote_host_id,
                                uintptr_t remote_addr,
                                size_t length) {
    if (length == 0) {
        MPCOMM_LOG_ERROR("MPComm: PutAsync failed - zero length\n");
        return INVALID_TRANSFER_HANDLE;
    }
    // Put = RDMA WRITE to a single host, reuse scatter path with 1 host
    return transferAsyncStart(local_addr,
                              {remote_host_id},
                              {remote_addr},
                              {length},
                              TransferDirection::SCATTER);
}

TransferHandle MPComm::Impl::getAsync(uintptr_t local_addr,
                                const std::string &remote_host_id,
                                uintptr_t remote_addr,
                                size_t length) {
    if (length == 0) {
        MPCOMM_LOG_ERROR("MPComm: GetAsync failed - zero length\n");
        return INVALID_TRANSFER_HANDLE;
    }
    // Get = RDMA READ from a single host, reuse gather path with 1 host
    return transferAsyncStart(local_addr,
                              {remote_host_id},
                              {remote_addr},
                              {length},
                              TransferDirection::GATHER);
}

TransferHandle MPComm::Impl::transferAsyncStart(uintptr_t local_addr,
                                          const std::vector<std::string> &host_list,
                                          const std::vector<uintptr_t> &remote_addrs,
                                          const std::vector<size_t> &lengths,
                                          TransferDirection direction) {
    const char* op_name = (direction == TransferDirection::SCATTER) ? "ScatterAsync" :
                          (direction == TransferDirection::GATHER) ? "GatherAsync" : "BroadcastAsync";

    // Validation
    if (!initialized_) {
        MPCOMM_LOG_ERROR("MPComm: %s failed - not initialized\n", op_name);
        return INVALID_TRANSFER_HANDLE;
    }

    size_t host_count = host_list.size();
    if (host_count == 0 || remote_addrs.size() != host_count ||
        lengths.size() != host_count) {
        MPCOMM_LOG_ERROR("MPComm: %s failed - invalid arguments\n", op_name);
        return INVALID_TRANSFER_HANDLE;
    }

    size_t num_nics = nic_contexts_.size();
    if (num_nics == 0) {
        MPCOMM_LOG_ERROR("MPComm: %s failed - no NICs available\n", op_name);
        return INVALID_TRANSFER_HANDLE;
    }

    // Determine NUMA node for this transfer early so we can allocate
    // TransferContext on the correct NUMA node (avoids cross-NUMA accesses
    // from the worker thread that processes this transfer).
    int transfer_numa_id = getNumaForAddr(local_addr);

    // Create transfer context on the target NUMA node
    auto ctx = make_numa_unique<TransferContext>(transfer_numa_id);
    ctx->handle = handle_pool_.allocate();
    if (ctx->handle == INVALID_TRANSFER_HANDLE) {
        MPCOMM_LOG_ERROR("MPComm: %s failed - handle pool exhausted "
                "(all %u handles in use, call releaseTransfer to free handles)\n",
                op_name, (unsigned)HandlePool::kMaxHandle);
        return INVALID_TRANSFER_HANDLE;
    }
    ctx->local_addr = local_addr;
    ctx->host_list = host_list;
    ctx->remote_addrs = remote_addrs;
    ctx->lengths = lengths;
    ctx->direction = direction;
    ctx->start_time = std::chrono::steady_clock::now();

    // Prepare all chunks (same logic as transferImplDynamic)
    const size_t max_chunk_size = max_rdma_transfer_size_;
    size_t total_chunks = 0;

    // Calculate total chunks and local offsets
    std::vector<size_t> host_chunk_starts(host_count);
    std::vector<size_t> host_local_offsets(host_count);

    size_t running_local_offset = 0;
    for (size_t i = 0; i < host_count; ++i) {
        host_chunk_starts[i] = total_chunks;
        host_local_offsets[i] = running_local_offset;
        total_chunks += (lengths[i] + max_chunk_size - 1) / max_chunk_size;
        // Broadcast: all hosts read from the same local offset (0)
        // Scatter/Gather: local offset advances consecutively
        if (direction != TransferDirection::BROADCAST) {
            running_local_offset += lengths[i];
        }
    }

    ctx->prep_chunks_calc_time = std::chrono::steady_clock::now();

    // Handle empty transfer
    if (total_chunks == 0) {
        ctx->total_chunks.store(0);
        ctx->total_completed.store(0);
        ctx->finished.store(true);
        ctx->error_code.store(MPCOMM_SUCCESS);
        auto now = std::chrono::steady_clock::now();
        ctx->prep_chunks_meta_time = now;
        ctx->prep_numa_query_time = now;
        ctx->prep_flowctrl_time = now;
        ctx->queued_time = now;
        ctx->worker_start_time = now;
        ctx->cache_done_time = now;
        ctx->first_post_time = now;
        ctx->all_posted_time = now;
        ctx->end_time = now;

        TransferHandle handle = ctx->handle;
        ctx->numa_id = transfer_numa_id;
        {
            std::lock_guard<std::mutex> lock(per_numa_transfers_[transfer_numa_id].mutex);
            per_numa_transfers_[transfer_numa_id].active_transfers[handle] = std::move(ctx);
        }
        {
            std::lock_guard<std::mutex> lock(handle_numa_mutex_);
            handle_numa_map_[handle] = transfer_numa_id;
        }
        return handle;
    }

    // Store chunk metadata for lazy computation (no pre-allocated chunk array)
    ctx->max_chunk_size = max_chunk_size;
    ctx->host_chunk_starts = std::move(host_chunk_starts);
    ctx->host_local_offsets = std::move(host_local_offsets);

    ctx->total_chunks.store(total_chunks);
    ctx->next_chunk_idx.store(0);

    ctx->prep_chunks_meta_time = std::chrono::steady_clock::now();

    // NUMA-aware NIC selection (with PCIe-affine upgrade for GPU memory)
    int memory_numa_node = getNumaNodeForAddr(reinterpret_cast<void*>(local_addr));

    // For GPU memory, prefer PCIe-affine NICs (sharing closest PCIe switch)
    // This provides much finer granularity than NUMA-level selection,
    // especially when all NICs are on the same NUMA node.
    auto pcie_affine = getPcieAffinityNicsForAddr(reinterpret_cast<void*>(local_addr));

    // PXN: When enabled and source is GPU memory in SCATTER/PUT direction,
    // use static data partitioning: front X% via direct NICs, back (100-X)%
    // via NVLink proxy GPUs (one proxy GPU per proxy NIC).
    bool use_pxn = false;
#ifdef USE_CUDA
    int gpu_device_id = -1;
    if (pxn_manager_.isEnabled() && !pcie_affine.empty() &&
        direction == TransferDirection::SCATTER) {
        gpu_device_id = detectGpuDevice(reinterpret_cast<void*>(local_addr));
        if (gpu_device_id >= 0) {
            auto reachable = pxn_manager_.getAllReachableNics(gpu_device_id);
            if (reachable.size() > pcie_affine.size()) {
                // Filter reachable NICs by remote DRAM NUMA affinity.
                // When the remote buffer sits on a specific NUMA node, only
                // the remote NICs local to that NUMA can write efficiently.
                // We therefore restrict local NICs to those that map to
                // remote-NUMA-affine NICs, avoiding cross-NUMA DRAM writes
                // on the remote side.
                std::vector<size_t> numa_filtered;
                {
                    std::lock_guard<std::mutex> conn_lock(connections_mutex_);
                    // Collect the set of remote NIC indices that are affine
                    // to the remote buffer's NUMA node.
                    std::set<size_t> affine_remote_nics;
                    for (const auto& host_id : host_list) {
                        auto conn_it = connections_.find(host_id);
                        if (conn_it == connections_.end()) continue;
                        const auto& conn = conn_it->second;

                        // Determine remote buffer NUMA node from remote_buffers
                        int remote_buf_numa = -1;
                        for (size_t hi = 0; hi < host_list.size(); ++hi) {
                            if (host_list[hi] != host_id) continue;
                            uintptr_t raddr = remote_addrs[hi];
                            for (const auto& [buf_addr, buf_entry] : conn.remote_buffers) {
                                if (raddr >= buf_addr && raddr < buf_addr + buf_entry.length) {
                                    remote_buf_numa = buf_entry.numa_node;
                                    break;
                                }
                            }
                            if (remote_buf_numa >= 0) break;
                        }

                        if (remote_buf_numa < 0) {
                            // Cannot determine remote NUMA — skip filtering for this host
                            affine_remote_nics.clear();
                            break;
                        }

                        // Find remote NICs that belong to this NUMA node
                        for (size_t ri = 0; ri < conn.remote_nic_numa_nodes.size(); ++ri) {
                            if (conn.remote_nic_numa_nodes[ri] == remote_buf_numa) {
                                affine_remote_nics.insert(ri);
                            }
                        }
                    }

                    if (!affine_remote_nics.empty()) {
                        // Keep only local NICs whose primary (same-rail) remote
                        // NIC is NUMA-affine.  local_to_remote_nic_map[nic][0]
                        // is the same-rail NIC used by initContextCache; the
                        // secondary entry ([1]) is a cross-NUMA fallback and
                        // must NOT be considered here, otherwise every local NIC
                        // would pass the filter through its cross-NUMA mapping.
                        for (const auto& host_id : host_list) {
                            auto conn_it = connections_.find(host_id);
                            if (conn_it == connections_.end()) continue;
                            const auto& nic_map = conn_it->second.local_to_remote_nic_map;
                            for (size_t nic : reachable) {
                                auto map_it = nic_map.find(nic);
                                if (map_it != nic_map.end() && !map_it->second.empty()) {
                                    // Only check the primary (index 0) remote NIC
                                    if (affine_remote_nics.count(map_it->second[0])) {
                                        numa_filtered.push_back(nic);
                                    }
                                }
                            }
                            break;  // Use first host's mapping (all hosts share topology)
                        }
                    }
                }

                if (!numa_filtered.empty()) {
                    // De-duplicate (a local NIC may appear once already)
                    std::sort(numa_filtered.begin(), numa_filtered.end());
                    numa_filtered.erase(std::unique(numa_filtered.begin(), numa_filtered.end()),
                                        numa_filtered.end());
                    reachable = std::move(numa_filtered);
                }

                // Also filter pcie_affine to only NICs still in reachable
                {
                    std::vector<size_t> filtered_pcie_affine;
                    for (size_t nic : pcie_affine) {
                        if (std::find(reachable.begin(), reachable.end(), nic) != reachable.end()) {
                            filtered_pcie_affine.push_back(nic);
                        }
                    }
                    if (!filtered_pcie_affine.empty()) {
                        pcie_affine = std::move(filtered_pcie_affine);
                    }
                }

                MPCOMM_LOG_DEBUG("MPComm PXN: After remote NUMA affinity filter: "
                                 "%zu reachable NICs, %zu direct NICs (GPU %d)\n",
                                 reachable.size(), pcie_affine.size(), gpu_device_id);

                // Determine max proxy NICs from env (0 or unset = unlimited)
                int max_proxy_nics = 0;
                const char* env_val = std::getenv("MPCOMM_PXN_PROXY_NICS");
                if (env_val && env_val[0] != '\0') {
                    max_proxy_nics = std::atoi(env_val);
                }

                // Collect proxy NICs (non-direct NICs reachable via NVLink)
                std::vector<size_t> proxy_nics;
                for (size_t nic : reachable) {
                    bool is_direct = false;
                    for (size_t d : pcie_affine) {
                        if (nic == d) { is_direct = true; break; }
                    }
                    if (!is_direct) {
                        proxy_nics.push_back(nic);
                        if (max_proxy_nics > 0 &&
                            static_cast<int>(proxy_nics.size()) >= max_proxy_nics)
                            break;
                    }
                }

                if (!proxy_nics.empty()) {
                    ctx->pxn_enabled = true;
                    ctx->pxn_source_gpu = gpu_device_id;

                    // Query the actual CUDA context that owns the source buffer.
                    // This may differ from the primary context retained by PxnManager
                    // (e.g. if the user allocated memory via a different context).
                    // Using the wrong src_ctx in cuMemcpyPeerAsync causes the driver
                    // to fall back to a slow CPU staging path (~30 GB/s vs ~384 GB/s).
#ifdef USE_CUDA
                    CUcontext src_buf_ctx = nullptr;
                    CUresult ctx_res = cuPointerGetAttribute(
                        &src_buf_ctx, CU_POINTER_ATTRIBUTE_CONTEXT,
                        static_cast<CUdeviceptr>(local_addr));
                    if (ctx_res == CUDA_SUCCESS && src_buf_ctx != nullptr) {
                        ctx->pxn_source_ctx = src_buf_ctx;
                        CUcontext primary_ctx = pxn_manager_.getProxyBuffer(gpu_device_id)
                                                    ? pxn_manager_.getProxyBuffer(gpu_device_id)->cuda_ctx
                                                    : nullptr;
                        if (src_buf_ctx != primary_ctx) {
                            MPCOMM_LOG_DEBUG("MPComm PXN: Source buffer context (%p) differs "
                                           "from primary context (%p) for GPU %d, "
                                           "enabling P2P access from user context\n",
                                           reinterpret_cast<void*>(src_buf_ctx), reinterpret_cast<void*>(primary_ctx),
                                           gpu_device_id);
                            // The user allocated GPU memory in a non-primary context.
                            // cuCtxEnablePeerAccess is per-context, so we must also
                            // enable P2P from the user's context to each NVLink peer's
                            // primary context.  Without this, cuMemcpyPeerAsync falls
                            // back to a slow CPU staging path (~30 GB/s vs ~384 GB/s).
                            const auto* gpu_info = pxn_manager_.getGpuInfo(gpu_device_id);
                            if (gpu_info) {
                                CUcontext old_ctx;
                                cuCtxPushCurrent(src_buf_ctx);
                                for (int peer_dev : gpu_info->nvlink_peers) {
                                    auto* peer_pb = pxn_manager_.getProxyBuffer(peer_dev);
                                    if (!peer_pb) continue;
                                    CUresult pa_res = cuCtxEnablePeerAccess(peer_pb->cuda_ctx, 0);
                                    if (pa_res != CUDA_SUCCESS &&
                                        pa_res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
                                        MPCOMM_LOG_WARN("MPComm PXN: cuCtxEnablePeerAccess "
                                                       "user_ctx(%p) -> GPU %d primary_ctx(%p) "
                                                       "failed: %d\n",
                                                       reinterpret_cast<void*>(src_buf_ctx),
                                                       peer_dev,
                                                       reinterpret_cast<void*>(peer_pb->cuda_ctx),
                                                       pa_res);
                                    } else {
                                        MPCOMM_LOG_DEBUG("MPComm PXN: Enabled P2P access from "
                                                        "user_ctx(%p) -> GPU %d primary_ctx(%p)\n",
                                                        reinterpret_cast<void*>(src_buf_ctx),
                                                        peer_dev,
                                                        reinterpret_cast<void*>(peer_pb->cuda_ctx));
                                    }
                                }
                                cuCtxPopCurrent(&old_ctx);
                            }
                        }
                    } else {
                        // Fallback to primary context if query fails
                        ctx->pxn_source_ctx = pxn_manager_.getProxyBuffer(gpu_device_id)
                                                  ? pxn_manager_.getProxyBuffer(gpu_device_id)->cuda_ctx
                                                  : nullptr;
                        MPCOMM_LOG_WARN("MPComm PXN: cuPointerGetAttribute(CONTEXT) failed: %d, "
                                       "falling back to primary context\n", ctx_res);
                    }
#endif  // USE_CUDA
                    ctx->pxn_direct_nic_indices = pcie_affine;
                    ctx->pxn_proxy_nic_indices = proxy_nics;

                    // Bench mode: wait for all copies before posting RDMA
                    const char* bench_env = std::getenv(kPxnBenchSequentialEnvVar);
                    ctx->pxn_bench_sequential = (bench_env && bench_env[0] == '1');
                    ctx->pxn_bench_copies_done = false;

                    // NVLink alpha: weight for NVLink outstanding in proxy load calc
                    {
                        double alpha = 1.0;
                        const char* alpha_env = std::getenv(kPxnProxyNvlinkAlphaEnvVar);
                        if (alpha_env) {
                            alpha = std::strtod(alpha_env, nullptr);
                            if (alpha < 0.0) alpha = 0.0;
                        }
                        ctx->pxn_proxy_nvlink_alpha = alpha;
                    }

                    // All NICs are candidates (for CQ polling)
                    std::vector<size_t> all_nics = pcie_affine;
                    for (size_t nic : proxy_nics) all_nics.push_back(nic);
                    ctx->candidate_nic_indices = std::move(all_nics);

                    // Precompute proxy NIC bitmap
                    for (size_t nic : ctx->candidate_nic_indices) {
                        bool is_direct = false;
                        for (size_t d : pcie_affine) {
                            if (nic == d) { is_direct = true; break; }
                        }
                        ctx->pxn_is_proxy_nic[nic] = !is_direct;
                    }

                    // Dynamic pipeline: create one pipeline per proxy NIC.
                    // No chunk pre-assignment — chunks are dynamically assigned
                    // by the worker loop using min_outstanding strategy.
                    for (size_t p = 0; p < proxy_nics.size(); ++p) {
                        TransferContext::PxnProxyPipeline pp;
                        pp.proxy_gpu_id = pxn_manager_.getProxyGpuForNic(
                            gpu_device_id, proxy_nics[p]);
                        pp.nic_index = proxy_nics[p];
                        pp.total_bytes_assigned = 0;
                        pp.total_bytes_copied = 0;
                        pp.total_chunks_posted = 0;
                        pp.total_copies_submitted = 0;
                        pp.rdma_completed_snapshot = 0;
                        pp.assigned_outstanding = 0;

                        // Reset proxy buffer allocator for clean state.
                        // This is safe because the previous transfer's cleanup
                        // should have drained all inflight RDMA and freed all space.
                        auto* pb = pxn_manager_.getProxyBuffer(pp.proxy_gpu_id);
                        if (pb) {
                            pb->resetAllocator();
                        }

                        ctx->pxn_proxy_pipelines.push_back(std::move(pp));
                    }

                    use_pxn = true;
                    MPCOMM_LOG_DEBUG("MPComm PXN: Dynamic pipeline: "
                                     "%zu direct NICs, %zu proxy NICs "
                                     "(GPU %d, handle=%lu, "
                                     "nvlink_alpha=%.2f)\n",
                                     pcie_affine.size(),
                                     proxy_nics.size(),
                                     gpu_device_id, ctx->handle,
                                     ctx->pxn_proxy_nvlink_alpha);
                }
            }
        }
    }
#endif  // USE_CUDA

    if (!use_pxn) {
        if (!pcie_affine.empty()) {
            ctx->candidate_nic_indices = std::move(pcie_affine);
        } else {
            // CPU memory or PCIe affinity not available: fall back to NUMA-level selection
            ctx->candidate_nic_indices = getLocalNicIndicesForNuma(memory_numa_node);
        }
    }

    if (ctx->candidate_nic_indices.empty()) {
        ctx->candidate_nic_indices.reserve(num_nics);
        for (size_t i = 0; i < num_nics; ++i) {
            ctx->candidate_nic_indices.push_back(i);
        }
    }

    // Filter out quarantined NICs from the candidate list.
    // If all NICs are quarantined, keep the full list (degraded but functional).
    if (!nic_quarantine_.empty()) {
        std::vector<size_t> healthy;
        healthy.reserve(ctx->candidate_nic_indices.size());
        for (size_t nic : ctx->candidate_nic_indices) {
            if (nic < nic_quarantine_.size() && !nic_quarantine_[nic]->in_quarantine.load(std::memory_order_relaxed)) {
                healthy.push_back(nic);
            }
        }
        if (!healthy.empty()) {
            ctx->candidate_nic_indices = std::move(healthy);
        }
        // else: all quarantined — keep original list to avoid total blackout
    }

    ctx->prep_numa_query_time = std::chrono::steady_clock::now();

    // Record NIC/QP counts for flow control arrays (arrays already zero-initialized in ctor)
    ctx->num_nics_used = num_nics;
    ctx->num_qps_used = qps_per_connection_;

    ctx->prep_flowctrl_time = std::chrono::steady_clock::now();


    // MPCOMM_LOG_INFO("MPComm: %s queued with %zu chunks across %zu candidate NICs (handle=%lu, numa=%d)\n",
    //        op_name, total_chunks, ctx->candidate_nic_indices.size(), ctx->handle, memory_numa_node);

    // Fully async mode: submit task to a worker thread via lock-free queue
    // Worker thread will handle all post and poll operations
    TransferHandle handle = ctx->handle;

    // Use the NUMA node determined earlier (before allocation)
    int numa_id = transfer_numa_id;
    ctx->numa_id = numa_id;

    // Record queued time before moving ctx
    ctx->queued_time = std::chrono::steady_clock::now();

    // Store context in per-NUMA map (avoids cross-NUMA lock contention)
    {
        std::lock_guard<std::mutex> lock(per_numa_transfers_[numa_id].mutex);
        per_numa_transfers_[numa_id].active_transfers[handle] = std::move(ctx);
    }
    {
        std::lock_guard<std::mutex> lock(handle_numa_mutex_);
        handle_numa_map_[handle] = numa_id;
    }

    // Select a worker within the NUMA node (round-robin)
    size_t worker_id = selectWorkerForNuma(numa_id);

    // Submit to worker's lock-free queue (busy-wait if queue is full)
    while (!worker_queues_[worker_id]->tryPush(handle)) {
        _mm_pause();  // CPU spin hint
    }

    return handle;
}



int MPComm::Impl::pollAllNicsForAsync(TransferContext& ctx) {
    // Poll only candidate NICs for better performance in single-transfer scenarios
    // When parallel transfers share the same NICs, completions will still be properly routed
    // because we poll CQs (shared per NIC) and decode the transfer handle from wr_id
    const int poll_batch_size = static_cast<int>(poll_batch_size_);
    struct ibv_wc wc_array[64];  // Fixed max size; actual poll count controlled by poll_batch_size
    size_t num_nics = nic_contexts_.size();
    size_t num_candidate_nics = ctx.candidate_nic_indices.size();

    // Poll only candidate NICs for better performance
    for (size_t idx = 0; idx < num_candidate_nics; ++idx) {
        size_t nic = ctx.candidate_nic_indices[idx];
        auto &nic_ctx = *nic_contexts_[nic];
        int n = ibv_poll_cq(nic_ctx.cq, poll_batch_size, wc_array);
        if (n < 0) {
            MPCOMM_LOG_ERROR("MPComm: ibv_poll_cq failed on NIC %zu in async transfer\n", nic);
            return MPCOMM_ERR_TRANSFER;
        }
        for (int i = 0; i < n; ++i) {
            if (wc_array[i].status != IBV_WC_SUCCESS) {
                MPCOMM_LOG_ERROR("MPComm: WC error on NIC %zu in async transfer: status=%d, wr_id=0x%lx\n",
                        nic, wc_array[i].status, wc_array[i].wr_id);
                // Mark error on the transfer that owns this completion
                TransferHandle wc_handle = WrIdEncoding::decodeHandle(wc_array[i].wr_id);
                if (wc_handle == ctx.handle) {
                    return MPCOMM_ERR_TRANSFER;
                }
                // Error belongs to another transfer on the same NUMA, mark it there
                {
                    auto& nts = per_numa_transfers_[ctx.numa_id];
                    std::lock_guard<std::mutex> lock(nts.mutex);
                    auto it = nts.active_transfers.find(wc_handle);
                    if (it != nts.active_transfers.end()) {
                        it->second->error_code.store(MPCOMM_ERR_TRANSFER);
                        it->second->finished.store(true);
                    }
                }
                continue;
            }

            // Decode nic_index, qp_index, and transfer_handle from wr_id
            size_t completed_nic = WrIdEncoding::decodeNic(wc_array[i].wr_id);
            size_t completed_qp = WrIdEncoding::decodeQp(wc_array[i].wr_id);
            TransferHandle wc_handle = WrIdEncoding::decodeHandle(wc_array[i].wr_id);

            // Route completion to the correct TransferContext
            if (wc_handle == ctx.handle) {
                // Completion belongs to current context
                if (completed_nic < num_nics) {
                    ctx.per_nic_completed[completed_nic]++;
                    ctx.per_nic_last_completion[completed_nic] = std::chrono::steady_clock::now();
                    if (completed_qp < qps_per_connection_) {
                        ctx.per_nic_qp_completed[completed_nic][completed_qp]++;
                    }
                }
                ctx.total_completed.fetch_add(1);
            } else {
                // Completion belongs to another transfer on the same NUMA - route it there
                auto& nts = per_numa_transfers_[ctx.numa_id];
                std::lock_guard<std::mutex> lock(nts.mutex);
                auto it = nts.active_transfers.find(wc_handle);
                if (it != nts.active_transfers.end()) {
                    TransferContext* other_ctx = it->second.get();
                    if (completed_nic < num_nics) {
                        other_ctx->per_nic_completed[completed_nic]++;
                        other_ctx->per_nic_last_completion[completed_nic] = std::chrono::steady_clock::now();
                        if (completed_qp < qps_per_connection_) {
                            other_ctx->per_nic_qp_completed[completed_nic][completed_qp]++;
                        }
                    }
                    other_ctx->total_completed.fetch_add(1);
                }
                // If transfer not found, completion is orphaned (transfer already released)
                // This is OK - just ignore it
            }
        }
    }
    return MPCOMM_SUCCESS;
}

bool MPComm::Impl::isTransferComplete(TransferHandle handle) {
    // Fully async mode: just check the finished flag (no polling here)
    // Worker thread handles all post and poll operations
    int numa_id = -1;
    {
        std::lock_guard<std::mutex> lock(handle_numa_mutex_);
        auto it = handle_numa_map_.find(handle);
        if (it == handle_numa_map_.end()) return true;  // Invalid handle is considered "complete"
        numa_id = it->second;
    }
    auto& nts = per_numa_transfers_[numa_id];
    std::lock_guard<std::mutex> lock(nts.mutex);
    auto it = nts.active_transfers.find(handle);
    if (it == nts.active_transfers.end()) {
        return true;
    }
    return it->second->finished.load();
}

int MPComm::Impl::waitTransfer(TransferHandle handle, int timeout_ms) {
    // Resolve handle to NUMA node once (avoids repeated map lookup)
    int numa_id = -1;
    TransferContext* ctx_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(handle_numa_mutex_);
        auto it = handle_numa_map_.find(handle);
        if (it == handle_numa_map_.end()) return MPCOMM_ERR_INVALID_HANDLE;
        numa_id = it->second;
    }
    {
        auto& nts = per_numa_transfers_[numa_id];
        std::lock_guard<std::mutex> lock(nts.mutex);
        auto it = nts.active_transfers.find(handle);
        if (it == nts.active_transfers.end()) return MPCOMM_ERR_INVALID_HANDLE;
        ctx_ptr = it->second.get();
    }

    // Busy-wait on the finished flag (no mutex, no sleep)
    auto start_time = std::chrono::steady_clock::now();
    while (!ctx_ptr->finished.load(std::memory_order_acquire)) {
        _mm_pause();  // CPU spin hint

        // Check timeout
        if (timeout_ms >= 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            if (elapsed_ms >= timeout_ms) {
                return MPCOMM_ERR_TIMEOUT;
            }
        }
    }
    return ctx_ptr->error_code.load(std::memory_order_acquire);
}

TransferResult MPComm::Impl::getTransferResult(TransferHandle handle) {
    TransferResult result = {MPCOMM_ERR_INVALID_HANDLE, 0, 0.0};

    int numa_id = -1;
    {
        std::lock_guard<std::mutex> lock(handle_numa_mutex_);
        auto nit = handle_numa_map_.find(handle);
        if (nit == handle_numa_map_.end()) return result;
        numa_id = nit->second;
    }
    auto& nts = per_numa_transfers_[numa_id];
    std::lock_guard<std::mutex> lock(nts.mutex);
    auto it = nts.active_transfers.find(handle);
    if (it == nts.active_transfers.end()) {
        return result;
    }

    TransferContext& ctx = *it->second;
    result.error_code = ctx.error_code.load();

    // Calculate bytes transferred
    size_t total_bytes = 0;
    for (size_t i = 0; i < ctx.num_nics_used; ++i) {
        total_bytes += ctx.per_nic_bytes[i];
    }
    result.bytes_transferred = total_bytes;

    // Calculate elapsed time
    auto end = ctx.finished.load() ? ctx.end_time : std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - ctx.start_time).count();

    return result;
}

void MPComm::Impl::releaseTransfer(TransferHandle handle) {
    int numa_id = -1;
    {
        std::lock_guard<std::mutex> lock(handle_numa_mutex_);
        auto nit = handle_numa_map_.find(handle);
        if (nit == handle_numa_map_.end()) {
            // Handle not found in NUMA map - still return it to pool to avoid leak
            handle_pool_.release(handle);
            return;
        }
        numa_id = nit->second;
        handle_numa_map_.erase(nit);
    }
    auto& nts = per_numa_transfers_[numa_id];
    std::lock_guard<std::mutex> lock(nts.mutex);
    auto it = nts.active_transfers.find(handle);
    if (it != nts.active_transfers.end()) {
        // Print statistics if transfer was completed
        TransferContext& ctx = *it->second;
        if (ctx.finished.load() && ctx.error_code.load() == MPCOMM_SUCCESS) {
            const char* op_name = (ctx.direction == TransferDirection::SCATTER) ? "ScatterAsync" :
                                  (ctx.direction == TransferDirection::GATHER) ? "GatherAsync" : "BroadcastAsync";
            double transfer_ms = std::chrono::duration<double, std::milli>(
                ctx.end_time - ctx.start_time).count();

            size_t total_bytes = 0;
            for (size_t i = 0; i < ctx.num_nics_used; ++i) {
                total_bytes += ctx.per_nic_bytes[i];
            }
            double total_bandwidth_gbps = (total_bytes * 8.0) / (transfer_ms * 1e6);

            if (!ctx.timing_breakdown_str.empty()) {
                // Print timing breakdown first (buffered from worker thread)
                MPCOMM_LOG_DEBUG("%s", ctx.timing_breakdown_str.c_str());
                MPCOMM_LOG_DEBUG("\n========== %s Statistics (handle=%lu) ==========\n", op_name, handle);
                MPCOMM_LOG_DEBUG("%-20s %15s %12s %12s %12s %8s\n",
                       "NIC", "Bytes", "Chunks", "Share(%)", "BW(Gbps)",
                       ctx.pxn_enabled ? "Type" : "");
                MPCOMM_LOG_DEBUG("------------------------------------------------------------------------\n");
                size_t num_nics = nic_contexts_.size();
                for (size_t nic = 0; nic < num_nics; ++nic) {
                    double share_pct = (total_bytes > 0) ?
                                       (100.0 * ctx.per_nic_bytes[nic] / total_bytes) : 0.0;
                    double nic_bandwidth_gbps = (ctx.per_nic_bytes[nic] * 8.0) / (transfer_ms * 1e6);
                    const char* nic_type = "";
                    if (ctx.pxn_enabled && ctx.per_nic_bytes[nic] > 0) {
                        nic_type = ctx.pxn_is_proxy_nic[nic] ? "proxy" : "direct";
                    }
                    MPCOMM_LOG_DEBUG("%-20s %15zu %12zu %11.1f%% %12.2f %8s\n",
                           nic_contexts_[nic]->device_name.c_str(),
                           ctx.per_nic_bytes[nic], ctx.per_nic_posted[nic], share_pct, nic_bandwidth_gbps,
                           nic_type);
                }
                MPCOMM_LOG_DEBUG("------------------------------------------------------------------------\n");
                MPCOMM_LOG_DEBUG("%-20s %15zu %12zu %12s %12.2f\n",
                       "Total", total_bytes, ctx.total_chunks.load(), "-", total_bandwidth_gbps);
                MPCOMM_LOG_DEBUG("Time: %.2f ms\n", transfer_ms);

                // Per-NIC independent timing: first_post -> last_completion
                // Shows each NIC's own bandwidth (useful for bench sequential mode)
                if (ctx.pxn_enabled) {
                    MPCOMM_LOG_DEBUG("\n  Per-NIC independent timing (first_post -> last_completion):\n");
                    for (size_t nic = 0; nic < num_nics; ++nic) {
                        if (ctx.per_nic_bytes[nic] == 0) continue;
                        auto fp = ctx.per_nic_first_post[nic];
                        auto lc = ctx.per_nic_last_completion[nic];
                        if (fp.time_since_epoch().count() == 0 ||
                            lc.time_since_epoch().count() == 0) continue;
                        double nic_time_us = std::chrono::duration<double, std::micro>(lc - fp).count();
                        double nic_own_bw = (ctx.per_nic_bytes[nic] * 8.0) / (nic_time_us * 1000.0);
                        const char* nic_type = ctx.pxn_is_proxy_nic[nic] ? "proxy" : "direct";
                        MPCOMM_LOG_DEBUG("    %-20s %.1f us  %.2f Gbps  (%s)\n",
                               nic_contexts_[nic]->device_name.c_str(),
                               nic_time_us, nic_own_bw, nic_type);
                    }
                    if (ctx.pxn_bench_sequential) {
                        MPCOMM_LOG_DEBUG("  [BENCH SEQUENTIAL MODE: copies completed before posting]\n");
                    }
                }

                MPCOMM_LOG_DEBUG("==============================================================\n\n");
            }
        }

        nts.active_transfers.erase(it);
    }

    // Return handle to the pool for reuse
    handle_pool_.release(handle);
}

// ============================================================================
// Worker Thread Implementation for Fully Async Transfer
// ============================================================================

/**
┌─────────────────────────────────────────┐
│          while (worker_running_)        │
│                                         │
│  Phase 1: 取任务 ──→ 加入活跃集合          │
│  Phase 2: 发 RDMA WR（流控 + 负载均衡）    │
│  Phase 3: 轮询 CQ，收割完成事件            │
│  Phase 4: 清理已完成的上下文               │
│                                         │
└─────────────────────────────────────────┘
 */

void MPComm::Impl::bindWorkerToNuma(size_t worker_id, int numa_id,
                                    const std::vector<int>& cpu_ids) {
    // NUMA-only affinity: bind this thread to ALL CPUs of the given NUMA node,
    // letting the kernel scheduler pick an idle core. This avoids multiple
    // mpcomm processes on the same host colliding on the same single core.
#ifdef __linux__
    if (!cpu_ids.empty() && cpu_ids[0] >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        // Intersect with the process's allowed CPU set so we respect external
        // taskset / numactl --physcpubind constraints.
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        bool have_allowed = (sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
        int set_count = 0;
        std::string cpu_list_str;
        for (int c : cpu_ids) {
            if (c < 0) continue;
            if (have_allowed && !CPU_ISSET(c, &allowed)) continue;
            CPU_SET(c, &cpuset);
            ++set_count;
            if (!cpu_list_str.empty()) cpu_list_str += ",";
            cpu_list_str += std::to_string(c);
        }
        if (set_count > 0) {
            if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0) {
                MPCOMM_LOG_INFO("MPComm: Worker %zu (NUMA %d) bound to CPUs [%s] (%d cores)\n",
                       worker_id, numa_id, cpu_list_str.c_str(), set_count);
            } else {
                MPCOMM_LOG_WARN("MPComm: Worker %zu (NUMA %d) pthread_setaffinity_np failed, "
                       "falling back to kernel scheduling\n", worker_id, numa_id);
            }
        } else {
            MPCOMM_LOG_WARN("MPComm: Worker %zu (NUMA %d) has no allowed CPU after intersecting "
                   "with process affinity, leaving thread unbound\n", worker_id, numa_id);
        }
    }
#else
    (void)worker_id;
    (void)numa_id;
    (void)cpu_ids;  // Suppress unused parameter warning
#endif
}

// Fault injection: after fi.delay seconds, force QP(s) into ERROR state.
// Simulates a NIC/port failure for recovery testing. Worker 0 only.
void MPComm::Impl::maybeInjectFault(FaultInjectionState& fi, size_t num_nics) {
    if (fi.delay <= 0.0 || fi.triggered) return;

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - fi.start_time).count();
    if (elapsed < fi.delay) return;

    fi.triggered = true;
    fi.fault_time = now;
    size_t fi_fault_count = 0;

    // Determine which NICs to fault
    size_t nic_start = 0, nic_end = num_nics;
    if (fi.mode != "all_nic") {
        nic_start = fi.nic_idx < num_nics ? fi.nic_idx : 0;
        nic_end = nic_start + 1;
    }

    for (size_t nic = nic_start; nic < nic_end; ++nic) {
        auto& nic_ctx = *nic_contexts_[nic];
        std::lock_guard<std::mutex> lock(nic_ctx.qp_mutex);
        for (auto& [key, qps] : nic_ctx.qp_map) {
            for (size_t qi = 0; qi < qps.size(); ++qi) {
                struct ibv_qp_attr attr = {};
                attr.qp_state = IBV_QPS_ERR;
                int rc = ibv_modify_qp(qps[qi], &attr, IBV_QP_STATE);
                fi_fault_count++;

                // Record for potential recovery
                if (fi.recover_delay > 0.0 && rc == 0) {
                    // Parse key "host_id:local_nic:remote_nic" to get remote_nic
                    size_t last_colon = key.rfind(':');
                    size_t second_colon = key.rfind(':', last_colon - 1);
                    size_t remote_nic = std::stoul(key.substr(last_colon + 1));
                    std::string host_id = key.substr(0, second_colon);

                    RemoteEndpointInfo rinfo = {};
                    {
                        std::lock_guard<std::mutex> conn_lock(connections_mutex_);
                        auto conn_it = connections_.find(host_id);
                        if (conn_it != connections_.end() &&
                            remote_nic < conn_it->second.nic_endpoints.size()) {
                            rinfo = conn_it->second.nic_endpoints[remote_nic];
                            // Each QP has its own remote QPN; nic_endpoints only stores
                            // the first QP's info. For QP[0] this is correct.
                            // For QP[>0] we need the actual remote QPN from the QP itself.
                            // Query it before RESET clears it.
                            struct ibv_qp_attr qattr = {};
                            struct ibv_qp_init_attr qiattr = {};
                            if (ibv_query_qp(qps[qi], &qattr, IBV_QP_DEST_QPN, &qiattr) == 0) {
                                rinfo.qp_num = qattr.dest_qp_num;
                            }
                        }
                    }
                    fi.faulted_qps.push_back({nic, key, qi, qps[qi], rinfo});
                }

                if (rc != 0) {
                    MPCOMM_LOG_ERROR("MPComm: [FAULT INJECT] Failed to fault QP[%zu] on NIC %zu (rc=%d)\n",
                           qi, nic, rc);
                }
                if (fi.mode == "single_qp") break;
            }
            if (fi.mode == "single_qp") break;
        }
    }

    MPCOMM_LOG_WARN("MPComm: [FAULT INJECT] mode=%s, faulted %zu QP(s) after %.1fs "
           "(NIC range [%zu, %zu))%s\n",
           fi.mode.c_str(), fi_fault_count, elapsed, nic_start, nic_end,
           fi.recover_delay > 0.0 ? ", recovery scheduled" : "");
}

// Fault recovery: after fi.recover_delay seconds past the fault, restore QPs
// through RESET->INIT->RTR->RTS. Worker 0 only.
void MPComm::Impl::maybeRecoverFault(FaultInjectionState& fi) {
    if (!fi.triggered || fi.recovered || fi.recover_delay <= 0.0) return;

    auto now = std::chrono::steady_clock::now();
    double since_fault = std::chrono::duration<double>(now - fi.fault_time).count();
    if (since_fault < fi.recover_delay) return;

    fi.recovered = true;
    size_t recovered = 0, failed = 0;

    for (auto& fqp : fi.faulted_qps) {
        auto& nic_ctx = *nic_contexts_[fqp.nic_idx];

        // RESET the QP first
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RESET;
        if (ibv_modify_qp(fqp.qp, &attr, IBV_QP_STATE) != 0) {
            MPCOMM_LOG_ERROR("MPComm: [FAULT RECOVER] Failed to RESET QP on NIC %zu\n", fqp.nic_idx);
            failed++;
            continue;
        }

        // INIT
        if (modifyQPToInit(nic_ctx, fqp.qp) != 0) {
            MPCOMM_LOG_ERROR("MPComm: [FAULT RECOVER] Failed INIT on NIC %zu\n", fqp.nic_idx);
            failed++;
            continue;
        }

        // RTR (needs remote endpoint info)
        if (modifyQPToRTR(nic_ctx, fqp.qp, fqp.remote_info) != 0) {
            MPCOMM_LOG_ERROR("MPComm: [FAULT RECOVER] Failed RTR on NIC %zu\n", fqp.nic_idx);
            failed++;
            continue;
        }

        // RTS
        if (modifyQPToRTS(fqp.qp) != 0) {
            MPCOMM_LOG_ERROR("MPComm: [FAULT RECOVER] Failed RTS on NIC %zu\n", fqp.nic_idx);
            failed++;
            continue;
        }

        recovered++;
    }

    MPCOMM_LOG_WARN("MPComm: [FAULT RECOVER] Recovered %zu/%zu QP(s) after %.1fs "
           "(failed=%zu)\n",
           recovered, fi.faulted_qps.size(), since_fault, failed);
    fi.faulted_qps.clear();
}

void MPComm::Impl::cleanupPxnContextState(TransferContext& ctx) {
#ifdef USE_CUDA
    // Clean up PXN proxy copying state and drain inflight RDMA
    if (!ctx.pxn_enabled) return;
    for (auto& pp : ctx.pxn_proxy_pipelines) {
        // Free proxy buffer space held by copying_chunks
        // (these chunks were tryAlloc'd but not yet RDMA-posted,
        //  so their space was never moved to rdma_inflight for freeing)
        if (!pp.copying_chunks.empty()) {
            auto* pb_cc = pxn_manager_.getProxyBuffer(pp.proxy_gpu_id);
            if (pb_cc) {
                for (auto& cc : pp.copying_chunks) {
                    pb_cc->free(cc.length);
                }
            }
        }
        for (auto& cc : pp.copying_chunks) {
            if (cc.copy_done_flag) {
                pxn_manager_.recycleFlagToThread(cc.copy_done_flag);
            }
        }
        pp.copying_chunks.clear();
        // Drain remaining inflight RDMA and free proxy buffer
        if (!pp.rdma_inflight.empty()) {
            auto* pb = pxn_manager_.getProxyBuffer(pp.proxy_gpu_id);
            if (pb) {
                for (auto& inf : pp.rdma_inflight) {
                    pb->free(inf.length);
                }
            }
            pp.rdma_inflight.clear();
        }
    }
#else
    (void)ctx;
#endif
}

void MPComm::Impl::workerThreadLoop(size_t worker_id, int numa_id, std::vector<int> cpu_ids) {
    bindWorkerToNuma(worker_id, numa_id, cpu_ids);

    MPCOMM_LOG_INFO("MPComm: Worker %zu (NUMA %d) started (tid=%" PRIu64 ")\n",
           worker_id, numa_id, static_cast<uint64_t>(pthread_self()));

    const size_t kMaxIdleSpins = max_idle_spins_;
    const size_t max_outstanding_per_nic = max_outstanding_per_qp_ * qps_per_connection_;
    const size_t kMaxOutstandingPerQP = max_outstanding_per_qp_;
    size_t num_nics = nic_contexts_.size();

    // Worker-level outstanding tracking (shared across all active contexts on this worker).
    // These track the ACTUAL NIC-level outstanding WRs, which is what matters for flow control
    // since all contexts share the same NIC CQs.
    // Using fixed-size arrays to avoid heap allocation.
    size_t worker_nic_posted[kMaxNics] = {};
    size_t worker_nic_completed[kMaxNics] = {};
    size_t worker_nic_qp_posted[kMaxNics][kMaxQPsPerNic] = {};
    size_t worker_nic_qp_completed[kMaxNics][kMaxQPsPerNic] = {};

    // Experimental per-NIC outstanding limit: worker-local snapshot of the limits.
    auto limit_view = std::make_unique<WorkerLimitView>();
    WorkerLimitView& lv = *limit_view;

    // Active context set: contexts currently being processed by this worker.
    // Contexts are added when popped from the queue, removed when all chunks are completed.
    struct ActiveContext {
        TransferContext* ctx;
        NicConnCache cache;       // Per-context connection cache (lkey, rkey, QP pointers)
        bool first_post_recorded;
    };
    std::vector<ActiveContext> active_contexts;

    // Union of all candidate NIC indices across all active contexts (for CQ polling)
    std::vector<size_t> poll_nic_indices;
    std::vector<ActiveContextInfo> active_ctx_info;  // Reusable buffer for poll routing

    // PXN concurrency cap: handles deferred because too many PXN contexts are active.
    // Re-tried at the start of each Phase 1 before draining the lock-free queue.
    std::vector<TransferHandle> pending_pxn_handles;

#ifdef USE_CUDA
    // PXN dynamic pipeline: split into two lambdas for unified Phase 2 processing.
    //   0) dispatch_pxn_copy_results_fn: Polls copy results from the copy thread
    //      and dispatches them to the correct transfer's pipeline. Must be called
    //      once before iterating active_contexts (destructive read).
    //   1) advance_pxn_copies_fn: Submits new NVLink copies from pending queue.
    //      Per-chunk granularity — no batch waiting.
    //   2) post_pxn_proxy_chunks_fn: Posts RDMA for chunks whose copy is done.
    //      Uses the same outstanding-based flow control as direct NICs.
    uint64_t pxn_next_request_id = 1;  // Monotonic request ID counter
    std::vector<PxnCopyResult> pxn_copy_results;  // Reusable buffer for poll results
    size_t pxn_unmatched_results_total = 0;  // Cumulative count of unmatched copy results

    // Poll all PXN copy results and dispatch to the correct transfer's pipeline.
    // MUST be called ONCE before iterating active_contexts, because
    // pollAllCopyResults is a destructive read — results are dequeued from the
    // copy thread's result queue. If we polled inside per-transfer
    // advance_pxn_copies_fn, results belonging to other transfers would be
    // silently dropped, causing those transfers to hang forever.
    auto dispatch_pxn_copy_results_fn = [&]() {
        pxn_copy_results.clear();
        pxn_manager_.pollAllCopyResults(pxn_copy_results);
        if (pxn_copy_results.empty()) return;

        size_t unmatched_this_poll = 0;
        for (const auto& cr : pxn_copy_results) {
            bool dispatched = false;
            for (auto& ac : active_contexts) {
                TransferContext& ctx = *ac.ctx;
                if (!ctx.pxn_enabled) continue;
                for (auto& pp : ctx.pxn_proxy_pipelines) {
                    for (auto& cc : pp.copying_chunks) {
                        if (cc.request_id == cr.request_id && !cc.result_received) {
                            cc.copy_done_flag = cr.copy_done_flag;
                            cc.result_received = true;
                            if (!cr.success) {
                                ctx.error_code.store(MPCOMM_ERR_TRANSFER);
                                ctx.end_time = std::chrono::steady_clock::now();
                                ctx.finished.store(true);
                            }
                            dispatched = true;
                            break;
                        }
                    }
                    if (dispatched) break;
                }
                if (dispatched) break;
            }
            if (!dispatched) {
                unmatched_this_poll++;
                // Recycle the orphaned flag to prevent leak
                if (cr.copy_done_flag) {
                    pxn_manager_.recycleFlagToThread(cr.copy_done_flag);
                }
            }
        }
        if (unmatched_this_poll > 0) {
            pxn_unmatched_results_total += unmatched_this_poll;
            MPCOMM_LOG_WARN("MPComm PXN: %zu copy results unmatched this poll "
                            "(total unmatched: %zu, active_contexts: %zu)\n",
                            unmatched_this_poll, pxn_unmatched_results_total,
                            active_contexts.size());
        }
    };

    // Advance NVLink copy pipeline: submit new copies from pending queue.
    // Copy result polling is handled separately by dispatch_pxn_copy_results_fn.
    auto advance_pxn_copies_fn = [&](TransferContext& ctx) {
        if (!ctx.pxn_enabled || ctx.pxn_proxy_pipelines.empty()) return;

        // Submit new copies from pending queue.
        // For each pipeline, try to move pending chunks → copying_chunks
        // as long as proxy buffer has space and copy queue is not full.
        for (auto& pp : ctx.pxn_proxy_pipelines) {
            if (pp.pending_chunk_indices.empty()) continue;

            auto* pb = pxn_manager_.getProxyBuffer(pp.proxy_gpu_id);
            if (!pb) continue;

            // Submit as many pending chunks as possible
            while (!pp.pending_chunk_indices.empty()) {
                size_t ci = pp.pending_chunk_indices.front();
                auto ck = ctx.getChunk(ci);

                // Try to allocate space in proxy buffer
                CUdeviceptr proxy_addr = pb->tryAlloc(ck.length);
                if (proxy_addr == 0) {
                    break;  // No space — wait for RDMA completions to free
                }

                // Build and submit copy request
                PxnCopyRequest copy_req;
                copy_req.src_addr = static_cast<CUdeviceptr>(ck.local_addr);
                copy_req.src_ctx = static_cast<CUcontext>(ctx.pxn_source_ctx);
                copy_req.dst_addr = proxy_addr;
                copy_req.dst_ctx = pb->cuda_ctx;
                copy_req.length = ck.length;
                copy_req.request_id = pxn_next_request_id++;

                if (!pxn_manager_.submitCopyRequest(pp.proxy_gpu_id, copy_req)) {
                    // Copy queue full — free the already-allocated proxy buffer
                    // space to prevent a permanent leak. Without this, alloc_offset
                    // advances but free_offset never catches up, eventually
                    // exhausting the proxy buffer and causing a deadlock.
                    pb->free(ck.length);
                    break;  // Copy queue full — try again next iteration
                }

                // Move from pending to copying
                TransferContext::PxnCopyingChunk cc;
                cc.chunk_idx = ci;
                cc.request_id = copy_req.request_id;
                cc.proxy_addr = proxy_addr;
                cc.length = ck.length;
                cc.copy_done_flag = nullptr;
                cc.result_received = false;
                pp.copying_chunks.push_back(cc);
                pp.pending_chunk_indices.erase(pp.pending_chunk_indices.begin());
                pp.total_copies_submitted++;
                ctx.pxn_diag_proxy_copy_bytes += ck.length;
            }
        }
    };

    // Post RDMA for proxy chunks whose NVLink copy is done.
    // Per-chunk granularity: scan copying_chunks, post any with flag=true.
    // Uses the same outstanding-based flow control as direct NICs.
    // Proxy buffer free is deferred until RDMA completion (NIC DMA must finish
    // reading from proxy buffer before the space can be reused).
    // Returns number of RDMA posts made.
    auto post_pxn_proxy_chunks_fn = [&](TransferContext& ctx,
                                        NicConnCache& cache,
                                        bool& first_post_recorded) -> size_t {
        if (!ctx.pxn_enabled || ctx.pxn_proxy_pipelines.empty()) return 0;

        size_t rdma_posted_count = 0;

        for (auto& pp : ctx.pxn_proxy_pipelines) {
            auto* pb = pxn_manager_.getProxyBuffer(pp.proxy_gpu_id);
            if (!pb) continue;

            size_t nic = pp.nic_index;

            // Step 0: Drain completed RDMA inflight chunks and free proxy buffer.
            // Compare current per_nic_completed with our snapshot to determine
            // how many RDMA completions have arrived since last check.
            {
                size_t cur_completed = ctx.per_nic_completed[nic];
                size_t new_completions = cur_completed - pp.rdma_completed_snapshot;
                size_t to_free = std::min(new_completions, pp.rdma_inflight.size());
                for (size_t f = 0; f < to_free; ++f) {
                    pb->free(pp.rdma_inflight[f].length);
                }
                if (to_free > 0) {
                    pp.rdma_inflight.erase(pp.rdma_inflight.begin(),
                                           pp.rdma_inflight.begin() + to_free);
                }
                pp.rdma_completed_snapshot = cur_completed;
            }

            if (pp.copying_chunks.empty()) continue;

            // Step 1: Scan copying_chunks front-to-back. Post RDMA for completed ones.
            // We process in order to maintain proxy buffer free() ordering
            // (linear allocator requires FIFO free).
            while (!pp.copying_chunks.empty()) {
                auto& cc = pp.copying_chunks.front();

                // Check if this chunk's copy is done
                if (!cc.result_received || !cc.copy_done_flag) break;
                if (!cc.copy_done_flag->load(std::memory_order_acquire)) break;

                // Copy is done — try to post RDMA

                // Flow control: check NIC outstanding (same as direct NICs),
                // including the experimental dynamic NIC limit.
                size_t outstanding = worker_nic_posted[nic] - worker_nic_completed[nic];
                if (outstanding >= std::min(max_outstanding_per_nic, lv.limit[nic])) break;

                // Select QP with lowest outstanding
                size_t qp_index = 0;
                size_t min_qp_out = SIZE_MAX;
                bool found_qp = false;
                for (size_t qp = 0; qp < qps_per_connection_; ++qp) {
                    size_t qp_out = worker_nic_qp_posted[nic][qp] -
                                    worker_nic_qp_completed[nic][qp];
                    if (qp_out < kMaxOutstandingPerQP && qp_out < min_qp_out) {
                        min_qp_out = qp_out;
                        qp_index = qp;
                        found_qp = true;
                    }
                }
                if (!found_qp) break;

                size_t ci = cc.chunk_idx;
                auto ck = ctx.getChunk(ci);

                // Resolve connection info
                size_t host_idx = ck.host_idx;
                uint64_t cache_key = (static_cast<uint64_t>(host_idx) << 16) | nic;
                auto cache_it = cache.find(cache_key);

                uint32_t lkey, rkey;
                struct ibv_qp* qp_ptr;

                if (cache_it != cache.end()) {
                    const auto& cached = cache_it->second;
                    lkey = pxn_manager_.getProxyLkey(pp.proxy_gpu_id, nic);
                    rkey = cached.rkey;
                    qp_ptr = (qp_index < cached.num_qps) ?
                             cached.qps[qp_index] : cached.qps[0];
                } else {
                    break;  // Should not happen with proper cache init
                }

                if (lkey == 0 || rkey == 0 || !qp_ptr) {
                    MPCOMM_LOG_ERROR("MPComm PXN: Invalid conn info for proxy "
                                     "chunk (nic=%zu, lkey=%u, rkey=%u)\n",
                                     nic, lkey, rkey);
                    ctx.error_code.store(MPCOMM_ERR_TRANSFER);
                    ctx.end_time = std::chrono::steady_clock::now();
                    ctx.finished.store(true);
                    break;
                }

                struct ibv_sge p_sge;
                memset(&p_sge, 0, sizeof(p_sge));
                p_sge.addr = static_cast<uint64_t>(cc.proxy_addr);
                p_sge.length = static_cast<uint32_t>(ck.length);
                p_sge.lkey = lkey;

                struct ibv_send_wr p_wr;
                memset(&p_wr, 0, sizeof(p_wr));
                p_wr.wr_id = WrIdEncoding::encode(nic, qp_index,
                                                   ctx.handle, ci, lv.tag(nic));
                p_wr.opcode = IBV_WR_RDMA_WRITE;
                p_wr.sg_list = &p_sge;
                p_wr.num_sge = 1;
                p_wr.send_flags = IBV_SEND_SIGNALED;
                p_wr.wr.rdma.remote_addr = ck.remote_addr;
                p_wr.wr.rdma.rkey = rkey;

                struct ibv_send_wr* p_bad_wr = nullptr;
                int ret = ibv_post_send(qp_ptr, &p_wr, &p_bad_wr);
                if (ret != 0) {
                    MPCOMM_LOG_ERROR("MPComm PXN: ibv_post_send failed "
                                     "on NIC %zu: %d\n", nic, ret);
                    ctx.error_code.store(MPCOMM_ERR_TRANSFER);
                    ctx.end_time = std::chrono::steady_clock::now();
                    ctx.finished.store(true);
                    break;
                }

                worker_nic_posted[nic]++;
                worker_nic_qp_posted[nic][qp_index]++;
                lv.posted(nic);
                ctx.per_nic_posted[nic]++;
                ctx.per_nic_qp_posted[nic][qp_index]++;
                ctx.per_nic_bytes[nic] += ck.length;
                pp.total_chunks_posted++;
                pp.total_bytes_copied += ck.length;
                rdma_posted_count++;
                ctx.pxn_diag_proxy_chunks++;

                // Record per-NIC first post time (for per-NIC BW measurement)
                if (ctx.per_nic_first_post[nic].time_since_epoch().count() == 0) {
                    ctx.per_nic_first_post[nic] = std::chrono::steady_clock::now();
                }

                if (!first_post_recorded) {
                    ctx.first_post_time = std::chrono::steady_clock::now();
                    first_post_recorded = true;
                }

                if (ctx.pxn_diag_first_rdma_post.time_since_epoch().count() == 0)
                    ctx.pxn_diag_first_rdma_post = std::chrono::steady_clock::now();
                ctx.pxn_diag_last_rdma_post = std::chrono::steady_clock::now();

                // Recycle the copy-done flag (no longer needed).
                pxn_manager_.recycleFlagToThread(cc.copy_done_flag);

                // Move to RDMA inflight queue — proxy buffer free is deferred
                // until RDMA completion (NIC DMA must finish reading first).
                TransferContext::PxnInflightRdma inflight;
                inflight.length = cc.length;
                pp.rdma_inflight.push_back(inflight);

                // Remove from front of copying_chunks
                pp.copying_chunks.erase(pp.copying_chunks.begin());
            }
        }
        return rdma_posted_count;
    };
#endif

    size_t idle_spins = 0;

    // ---- Fault injection support ----
    // MPCOMM_FAULT_INJECT_AFTER_SEC: delay before injecting fault
    // MPCOMM_FAULT_INJECT_NIC: which NIC index to fault (default: 0)
    // MPCOMM_FAULT_INJECT_MODE: "single_qp" (default), "all_qp", "all_nic"
    // MPCOMM_FAULT_INJECT_RECOVER_AFTER_SEC: seconds after fault to recover QPs (0=no recovery)
    //   Simulates a port flap: ERROR for N seconds, then RESET->INIT->RTR->RTS
    FaultInjectionState fi;
    fi.initFromEnv();

    while (worker_running_.load(std::memory_order_relaxed)) {
        // Fault injection / recovery (worker 0 only, debug builds via env vars)
        if (worker_id == 0) {
            maybeInjectFault(fi, num_nics);
            maybeRecoverFault(fi);
        }

        // Experimental per-NIC outstanding limit: adopt changes and publish counters.
        adoptNicLimits(worker_id, lv, num_nics, worker_nic_posted, worker_nic_completed);

        // ---- Phase 1: Drain new handles from the queue into active set ----
        // PXN concurrency cap: when too many PXN contexts are active, defer
        // new PXN handles into a local pending list instead of activating them.
        // This keeps the active_contexts list small so Phase 2 iteration + CQ
        // polling stays fast, which is critical for high NIC utilization.
        // Non-PXN transfers are always activated immediately.
        {
            // Count active (unfinished) PXN contexts
            size_t active_pxn_count = 0;
            for (const auto& ac : active_contexts) {
                if (ac.ctx->pxn_enabled && !ac.ctx->finished.load()) {
                    active_pxn_count++;
                }
            }

            // Max concurrent PXN contexts per worker.  With 100 batch requests,
            // allowing ~10 active PXN contexts keeps Phase 2 iteration cost low
            // while still overlapping NVLink copies across multiple transfers.
            static constexpr size_t kMaxActivePxnContexts = 5;
            bool need_rebuild_nics = false;

            // Helper lambda: try to activate a handle.  Returns true if
            // activated, false if deferred (PXN cap reached).
            auto try_activate_handle = [&](TransferHandle h) -> bool {
                TransferContext* ctx_ptr = nullptr;
                {
                    auto& nts = per_numa_transfers_[numa_id];
                    std::lock_guard<std::mutex> lock(nts.mutex);
                    auto it = nts.active_transfers.find(h);
                    if (it == nts.active_transfers.end()) {
                        return true;  // Gone — treat as consumed
                    }
                    ctx_ptr = it->second.get();
                }

                // PXN concurrency cap check
                if (ctx_ptr->pxn_enabled && active_pxn_count >= kMaxActivePxnContexts) {
                    return false;  // Defer
                }

                // Initialize this context
                ctx_ptr->submitted.store(true);
                ctx_ptr->worker_start_time = std::chrono::steady_clock::now();

                ActiveContext ac;
                ac.ctx = ctx_ptr;
                ac.first_post_recorded = false;
                initContextCache(*ctx_ptr, ac.cache);
                ctx_ptr->cache_done_time = std::chrono::steady_clock::now();

                active_contexts.push_back(std::move(ac));
                need_rebuild_nics = true;

                if (ctx_ptr->pxn_enabled) {
                    active_pxn_count++;
                }
                return true;
            };

            // First, retry previously deferred PXN handles
            if (!pending_pxn_handles.empty()) {
                size_t write_idx = 0;
                for (size_t i = 0; i < pending_pxn_handles.size(); ++i) {
                    if (!try_activate_handle(pending_pxn_handles[i])) {
                        // Still can't activate — keep in pending list
                        pending_pxn_handles[write_idx++] = pending_pxn_handles[i];
                    }
                }
                pending_pxn_handles.resize(write_idx);
            }

            // Then drain new handles from the lock-free queue
            TransferHandle handle;
            while ((handle = worker_queues_[worker_id]->tryPop()) != INVALID_TRANSFER_HANDLE) {
                idle_spins = 0;
                if (!try_activate_handle(handle)) {
                    // PXN cap reached — defer this handle
                    pending_pxn_handles.push_back(handle);
                }
            }

            // Rebuild poll NIC indices only when contexts changed
            if (need_rebuild_nics) {
                poll_nic_indices.clear();
                std::vector<bool> nic_seen(num_nics, false);
                for (const auto& actx : active_contexts) {
                    for (size_t nic : actx.ctx->candidate_nic_indices) {
                        if (!nic_seen[nic]) {
                            nic_seen[nic] = true;
                            poll_nic_indices.push_back(nic);
                        }
                    }
                }
            }
        }

        // If no active contexts and no pending PXN handles, just idle-spin
        if (active_contexts.empty() && pending_pxn_handles.empty()) {
            ++idle_spins;
            if (idle_spins >= kMaxIdleSpins) {
                _mm_pause();
                idle_spins = 0;
            }
            continue;
        }
        idle_spins = 0;

        // ---- Phase 2: Post chunks from all active contexts (unified flow control) ----
        // For PXN-enabled transfers with static partition:
        //   - Direct chunks (front X%) are posted to direct NICs only
        //   - Proxy chunks (back 100-X%) are posted from proxy buffer after NVLink copy
        //   - Both direct and proxy posting happen in this unified loop
        // For non-PXN transfers: same as before (all NICs are candidates)
        // PXN diagnostics: per-iteration counters
        size_t diag_iter_direct_posted = 0;
        size_t diag_iter_proxy_posted = 0;
        size_t diag_iter_direct_break_nic_full = 0;
        size_t diag_iter_direct_break_qp_full = 0;

        // Accumulate global proxy assigned_outstanding per NIC across all active transfers.
        // This is needed for correct load balancing when multiple PXN transfers share NICs:
        // assigned_outstanding is per-transfer, but rdma_out (worker_nic_posted - completed)
        // is global, so we must use a global assigned_outstanding to correctly decompose
        // the pipeline into NVLink vs RDMA stages.
        size_t global_proxy_assigned[kMaxNics] = {};
        size_t num_active_pxn_transfers = 0;
        for (const auto& ac : active_contexts) {
            const TransferContext& actx = *ac.ctx;
            if (actx.pxn_enabled && !actx.finished.load()) {
                num_active_pxn_transfers++;
                for (const auto& pp : actx.pxn_proxy_pipelines) {
                    global_proxy_assigned[pp.nic_index] += pp.assigned_outstanding;
                }
            }
        }
        if (num_active_pxn_transfers == 0) num_active_pxn_transfers = 1;  // Avoid zero

        // Poll PXN copy results ONCE and dispatch to all active transfers.
        // This must happen before the per-context loop to avoid losing results.
#ifdef USE_CUDA
        dispatch_pxn_copy_results_fn();
#endif

        for (auto& ac : active_contexts) {
            TransferContext& ctx = *ac.ctx;
            size_t total_chunks = ctx.total_chunks.load();

#ifdef USE_CUDA
            // Advance PXN NVLink copy state machine: submit new copies from
            // pending queue. Copy result polling is handled by
            // dispatch_pxn_copy_results_fn above.
            if (ctx.pxn_enabled && !ctx.finished.load()) {
                auto t_adv0 = std::chrono::steady_clock::now();
                advance_pxn_copies_fn(ctx);
                auto t_adv1 = std::chrono::steady_clock::now();
                ctx.pxn_diag_ns_advance_copies += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t_adv1 - t_adv0).count());
            }

            // Bench sequential mode: wait for ALL NVLink copies to complete
            // before posting any RDMA chunks (direct or proxy). This isolates
            // RDMA posting from NVLink copy to measure per-NIC peak bandwidth.
            if (ctx.pxn_enabled && ctx.pxn_bench_sequential && !ctx.pxn_bench_copies_done) {
                bool all_copies_done = true;
                for (const auto& pp : ctx.pxn_proxy_pipelines) {
                    if (!pp.copying_chunks.empty() || !pp.pending_chunk_indices.empty()) {
                        all_copies_done = false;
                        break;
                    }
                }
                if (!all_copies_done) {
                    continue;  // Skip posting, only advance copies + poll CQ
                }
                // All copies done — record timestamp and proceed to posting
                ctx.pxn_bench_copies_done = true;
                MPCOMM_LOG_DEBUG("MPComm PXN BENCH: All NVLink copies done, "
                                 "starting RDMA posting (handle=%lu)\n", ctx.handle);
            }
#endif  // USE_CUDA

            // Post as many chunks as possible from this context.
            // For PXN: unified min_outstanding NIC selection across direct + proxy NICs.
            // If a proxy NIC is selected, the chunk is assigned to its pipeline for
            // NVLink copy + deferred RDMA posting. Interleave proxy RDMA posting
            // periodically to keep proxy NICs' RDMA queues fed.
            size_t post_batch_count = 0;
            auto t_direct0 = std::chrono::steady_clock::now();
            while (ctx.next_chunk_idx.load() < total_chunks) {
                size_t chunk_idx = ctx.next_chunk_idx.load();

#ifdef USE_CUDA
                static constexpr size_t kPostBatchBeforeProxyRdma = 16;
                // PXN: interleave proxy RDMA posting after every batch of posts.
                if (ctx.pxn_enabled && post_batch_count >= kPostBatchBeforeProxyRdma &&
                    !ctx.finished.load()) {
                    auto t_px0 = std::chrono::steady_clock::now();
                    diag_iter_proxy_posted += post_pxn_proxy_chunks_fn(ctx, ac.cache, ac.first_post_recorded);
                    auto t_px1 = std::chrono::steady_clock::now();
                    ctx.pxn_diag_ns_proxy_posting += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t_px1 - t_px0).count());
                    post_batch_count = 0;
                    // Also advance copies to keep pipeline flowing
                    dispatch_pxn_copy_results_fn();
                    advance_pxn_copies_fn(ctx);
                }
#endif  // USE_CUDA

                auto chunk = ctx.getChunk(chunk_idx);

                // Unified NIC selection: min effective outstanding across all candidates.
                //
                // For direct NICs, effective outstanding = RDMA outstanding
                // (worker_nic_posted - worker_nic_completed), max = max_outstanding_per_nic.
                //
                // For proxy NICs, effective outstanding =
                //   nvlink_outstanding * alpha + rdma_outstanding
                // Both values are GLOBAL (summed across all active transfers on this NIC):
                //   rdma_outstanding = worker_nic_posted[nic] - worker_nic_completed[nic]
                //   nvlink_outstanding = global_proxy_assigned[nic] - rdma_outstanding
                // Alpha (MPCOMM_PXN_NVLINK_ALPHA, default 1.0) controls how much
                // weight NVLink-stage chunks carry in load balancing decisions.
                //
                // The max for proxy NICs scales with active PXN transfers:
                //   eff_max = max_outstanding_per_nic + proxy_pipeline_depth * N
                // where N = number of active PXN transfers. Each transfer contributes
                // its own NVLink copy pipeline capacity, while RDMA depth is shared
                // and independently limited to max_outstanding_per_nic by
                // post_pxn_proxy_chunks_fn.
                //
                const size_t proxy_pipeline_depth = max_outstanding_per_nic * 2;
                size_t best_nic = num_nics;  // Invalid initially
                size_t num_candidate_nics = ctx.candidate_nic_indices.size();

                {
                    // Min effective outstanding strategy
                    size_t min_eff_outstanding = SIZE_MAX;

                    for (size_t idx = 0; idx < num_candidate_nics; ++idx) {
                        size_t nic = ctx.candidate_nic_indices[idx];

                        // Skip quarantined NICs in real-time (even for in-flight transfers)
                        if (nic < nic_quarantine_.size() && nic_quarantine_[nic] &&
                            nic_quarantine_[nic]->in_quarantine.load(std::memory_order_relaxed)) {
                            continue;
                        }

                        // Experimental dynamic NIC limit 0: no new WQE on this NIC
                        // (direct or proxy); completions keep being polled.
                        if (lv.limit[nic] == 0) {
                            continue;
                        }

                        size_t eff_outstanding;
                        size_t eff_max;
                        if (ctx.pxn_enabled && ctx.pxn_is_proxy_nic[nic]) {
                            // Use global assigned_outstanding and global rdma_out so both
                            // are in the same dimension (summed across all active transfers).
                            size_t rdma_out = worker_nic_posted[nic] - worker_nic_completed[nic];
                            size_t total_assigned = global_proxy_assigned[nic];
                            size_t nvlink_out = (total_assigned > rdma_out)
                                                ? (total_assigned - rdma_out) : 0;
                            eff_outstanding = static_cast<size_t>(
                                nvlink_out * ctx.pxn_proxy_nvlink_alpha + rdma_out);
                            // NVLink pipeline capacity scales with number of active PXN transfers.
                            eff_max = max_outstanding_per_nic +
                                      proxy_pipeline_depth * num_active_pxn_transfers;
                        } else {
                            eff_outstanding = worker_nic_posted[nic] - worker_nic_completed[nic];
                            eff_max = std::min(max_outstanding_per_nic, lv.limit[nic]);
                        }

                        if (eff_outstanding < eff_max && eff_outstanding < min_eff_outstanding) {
                            min_eff_outstanding = eff_outstanding;
                            best_nic = nic;
                        }
                    }
                }

                if (best_nic == num_nics) {
                    diag_iter_direct_break_nic_full++;
                    break;  // All NICs are full — break out and poll
                }

#ifdef USE_CUDA
                // If selected NIC is a proxy NIC, assign chunk to its pipeline
                if (ctx.pxn_enabled && ctx.pxn_is_proxy_nic[best_nic]) {
                    for (auto& pp : ctx.pxn_proxy_pipelines) {
                        if (pp.nic_index == best_nic) {
                            pp.pending_chunk_indices.push_back(chunk_idx);
                            pp.total_bytes_assigned += chunk.length;
                            pp.assigned_outstanding++;
                            global_proxy_assigned[best_nic]++;  // Keep global snapshot in sync
                            break;
                        }
                    }
                    ctx.next_chunk_idx.fetch_add(1);
                    post_batch_count++;
                    continue;  // No direct RDMA post — chunk goes through NVLink copy pipeline
                }
#endif  // USE_CUDA

                if (ctx.pxn_enabled) {
                    ctx.pxn_diag_direct_chunks++;
                }

                // Select QP with lowest outstanding within this NIC (worker-level)
                size_t qp_index = 0;
                size_t min_qp_outstanding = SIZE_MAX;
                bool found_available_qp = false;
                for (size_t qp = 0; qp < qps_per_connection_; ++qp) {
                    size_t qp_outstanding = worker_nic_qp_posted[best_nic][qp] -
                                            worker_nic_qp_completed[best_nic][qp];
                    if (qp_outstanding < kMaxOutstandingPerQP && qp_outstanding < min_qp_outstanding) {
                        min_qp_outstanding = qp_outstanding;
                        qp_index = qp;
                        found_available_qp = true;
                    }
                }
                if (!found_available_qp) {
                    diag_iter_direct_break_qp_full++;
                    break;  // All QPs full on best NIC — poll for completions
                }

                // Resolve connection info from cache
                size_t host_idx_for_qp = chunk.host_idx;
                uint64_t cache_key = (static_cast<uint64_t>(host_idx_for_qp) << 16) | best_nic;
                auto cache_it = ac.cache.find(cache_key);

                uint32_t lkey, rkey;
                size_t remote_nic;
                struct ibv_qp *qp;

                if (cache_it != ac.cache.end()) {
                    const auto& cached = cache_it->second;
                    lkey = cached.lkey;
                    rkey = cached.rkey;
                    remote_nic = cached.remote_nic;
                    qp = (qp_index < cached.num_qps) ? cached.qps[qp_index] : cached.qps[0];
                } else {
                    // Fallback: resolve on the fly (should rarely happen)
                    lkey = getLkey(best_nic, reinterpret_cast<void *>(chunk.local_addr));
                    const std::string& chunk_host_id = ctx.host_list[host_idx_for_qp];
                    remote_nic = best_nic;
                    rkey = 0;
                    {
                        std::lock_guard<std::mutex> lock(connections_mutex_);
                        auto it = connections_.find(chunk_host_id);
                        if (it != connections_.end()) {
                            const auto& nic_map = it->second.local_to_remote_nic_map;
                            auto map_it = nic_map.find(best_nic);
                            if (map_it != nic_map.end() && !map_it->second.empty()) {
                                remote_nic = map_it->second[0];
                            }
                            rkey = it->second.getRkeyForAddr(chunk.remote_addr, remote_nic);
                        }
                    }
                    qp = getOrCreateQP(best_nic, ctx.host_list[host_idx_for_qp], remote_nic, qp_index);
                }

                // Validate lkey, rkey, qp
                if (lkey == 0 || rkey == 0 || !qp) {
                    if (lkey == 0)
                        MPCOMM_LOG_ERROR("MPComm: Worker: No lkey for address %p on NIC %zu\n",
                                reinterpret_cast<void *>(chunk.local_addr), best_nic);
                    else if (rkey == 0)
                        MPCOMM_LOG_ERROR("MPComm: Worker: No rkey for remote NIC %zu (remote_addr=0x%lx)\n",
                                remote_nic, chunk.remote_addr);
                    else
                        MPCOMM_LOG_ERROR("MPComm: Worker: No QP for local NIC %zu -> remote NIC %zu\n",
                                best_nic, remote_nic);
                    int err = (lkey == 0) ? MPCOMM_ERR_MEMORY : MPCOMM_ERR_CONNECTION;
                    ctx.error_code.store(err);
                    ctx.end_time = std::chrono::steady_clock::now();
                    ctx.finished.store(true);  // Must be last write to ctx (release fence)
                    break;  // Skip this context, will be removed in Phase 4
                }

                // Prepare and post RDMA WR
                struct ibv_sge sge;
                memset(&sge, 0, sizeof(sge));
                sge.length = static_cast<uint32_t>(chunk.length);
                sge.lkey = lkey;

                sge.addr = chunk.local_addr;

                struct ibv_send_wr wr;
                memset(&wr, 0, sizeof(wr));
                wr.wr_id = WrIdEncoding::encode(best_nic, qp_index, ctx.handle, chunk_idx,
                                                lv.tag(best_nic));
                wr.opcode = (ctx.direction == TransferDirection::GATHER) ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
                wr.sg_list = &sge;
                wr.num_sge = 1;
                wr.send_flags = IBV_SEND_SIGNALED;
                wr.wr.rdma.remote_addr = chunk.remote_addr;
                wr.wr.rdma.rkey = rkey;

                struct ibv_send_wr *bad_wr = nullptr;
                int ret = ibv_post_send(qp, &wr, &bad_wr);
                if (ret != 0) {
                    MPCOMM_LOG_ERROR("MPComm: Worker: ibv_post_send failed on NIC %zu: %d\n", best_nic, ret);
                    ctx.error_code.store(MPCOMM_ERR_TRANSFER);
                    ctx.end_time = std::chrono::steady_clock::now();
                    ctx.finished.store(true);  // Must be last write to ctx (release fence)
                    break;
                }

                // Update worker-level flow control counters
                worker_nic_posted[best_nic]++;
                worker_nic_qp_posted[best_nic][qp_index]++;
                lv.posted(best_nic);

                // Update per-context counters (for stats and per-context completion tracking)
                ctx.per_nic_posted[best_nic]++;
                ctx.per_nic_qp_posted[best_nic][qp_index]++;
                ctx.per_nic_bytes[best_nic] += chunk.length;
                ctx.next_chunk_idx.fetch_add(1);
                post_batch_count++;
                diag_iter_direct_posted++;

                // Record per-NIC first post time (for per-NIC BW measurement)
                if (ctx.per_nic_first_post[best_nic].time_since_epoch().count() == 0) {
                    ctx.per_nic_first_post[best_nic] = std::chrono::steady_clock::now();
                }

                if (!ac.first_post_recorded) {
                    ctx.first_post_time = std::chrono::steady_clock::now();
                    ac.first_post_recorded = true;
                }
            }
            if (ctx.pxn_enabled) {
                auto t_direct1 = std::chrono::steady_clock::now();
                ctx.pxn_diag_ns_direct_posting += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t_direct1 - t_direct0).count());
            }

#ifdef USE_CUDA
            // Final proxy posting pass: post any remaining proxy chunks after
            // the direct posting loop exits (direct NICs full or all direct
            // chunks done). This complements the interleaved proxy posting
            // inside the direct loop above.
            if (ctx.pxn_enabled && !ctx.finished.load()) {
                auto t_px0 = std::chrono::steady_clock::now();
                diag_iter_proxy_posted += post_pxn_proxy_chunks_fn(ctx, ac.cache, ac.first_post_recorded);
                auto t_px1 = std::chrono::steady_clock::now();
                ctx.pxn_diag_ns_proxy_posting += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t_px1 - t_px0).count());
            }
#endif  // USE_CUDA

            // Record all_posted_time when all direct chunks have been posted (only once)
            total_chunks = ctx.total_chunks.load();
            if (ctx.next_chunk_idx.load() >= total_chunks &&
                ctx.all_posted_time.time_since_epoch().count() == 0) {
                ctx.all_posted_time = std::chrono::steady_clock::now();
            }
        }

        // ---- PXN diagnostics: sample outstanding BEFORE CQ polling ----
        for (auto& ac : active_contexts) {
            TransferContext& ctx = *ac.ctx;
            if (!ctx.pxn_enabled) continue;
            for (size_t nic = 0; nic < num_nics; ++nic) {
                size_t outstanding = worker_nic_posted[nic] - worker_nic_completed[nic];
                if (ctx.pxn_is_proxy_nic[nic]) {
                    if (outstanding > ctx.pxn_diag_max_outstanding_proxy)
                        ctx.pxn_diag_max_outstanding_proxy = outstanding;
                } else if (ctx.per_nic_posted[nic] > 0) {
                    if (outstanding > ctx.pxn_diag_max_outstanding_direct)
                        ctx.pxn_diag_max_outstanding_direct = outstanding;
                }
            }
        }

        // ---- Phase 2b: Poll completions from all candidate NICs ----
        // CQ polling runs after each round of posting (both direct and proxy)
        // to free outstanding slots and keep all NICs' pipelines flowing.
        {
            active_ctx_info.clear();
            for (const auto& ac : active_contexts) {
                active_ctx_info.push_back({ac.ctx->handle, ac.ctx});
            }

            auto t_poll0 = std::chrono::steady_clock::now();
            int poll_ret = pollAllNicsForWorker(numa_id, poll_nic_indices,
                                                worker_nic_completed,
                                                worker_nic_qp_completed,
                                                active_ctx_info, lv);
            auto t_poll1 = std::chrono::steady_clock::now();
            for (auto& ac2 : active_contexts) {
                if (ac2.ctx->pxn_enabled) {
                    ac2.ctx->pxn_diag_ns_cq_polling += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t_poll1 - t_poll0).count());
                }
            }
            if (poll_ret != MPCOMM_SUCCESS) {
                // Fatal poll error — mark all active contexts as failed
                for (auto& ac : active_contexts) {
                    if (!ac.ctx->finished.load()) {
                        ac.ctx->error_code.store(poll_ret);
                        ac.ctx->end_time = std::chrono::steady_clock::now();
                        ac.ctx->finished.store(true);  // Must be last write to ctx (release fence)
                    }
                }
                active_contexts.clear();
                poll_nic_indices.clear();
                continue;
            }
        }

        // ---- Phase 2c: Post-CQ proxy RDMA posting ----
        // After CQ polling frees RDMA slots, aggressively post proxy chunks
        // whose NVLink copy is done. This is critical because:
        //   - The direct posting loop (Phase 2) may exit early (direct NIC full
        //     or all chunks assigned), limiting proxy posting opportunities.
        //   - CQ polling releases RDMA outstanding slots, creating room for
        //     more proxy RDMA posts that the Phase 2 tail pass couldn't fill.
        //   - Without this, proxy NIC RDMA outstanding stays very low (~7)
        //     because proxy posting only runs a few times per iteration.
        // We also advance copies here to pick up newly completed NVLink copies.
#ifdef USE_CUDA
        // Poll copy results again after CQ polling (new completions may have arrived)
        dispatch_pxn_copy_results_fn();

        for (auto& ac : active_contexts) {
            TransferContext& ctx = *ac.ctx;
            if (!ctx.pxn_enabled || ctx.finished.load()) continue;

            // Advance copies first to pick up new completions
            advance_pxn_copies_fn(ctx);

            // Post proxy RDMA aggressively
            auto t_px0 = std::chrono::steady_clock::now();
            size_t proxy_posted = post_pxn_proxy_chunks_fn(ctx, ac.cache, ac.first_post_recorded);
            auto t_px1 = std::chrono::steady_clock::now();
            ctx.pxn_diag_ns_proxy_posting += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t_px1 - t_px0).count());
            diag_iter_proxy_posted += proxy_posted;
        }

        // Sample proxy outstanding peak after Phase 2c posting
        for (auto& ac : active_contexts) {
            TransferContext& ctx = *ac.ctx;
            if (!ctx.pxn_enabled) continue;
            for (size_t nic = 0; nic < num_nics; ++nic) {
                if (!ctx.pxn_is_proxy_nic[nic]) continue;
                size_t outstanding = worker_nic_posted[nic] - worker_nic_completed[nic];
                if (outstanding > ctx.pxn_diag_max_outstanding_proxy)
                    ctx.pxn_diag_max_outstanding_proxy = outstanding;
            }
        }
#endif  // USE_CUDA

        // ---- PXN diagnostics: accumulate per-iteration stats ----
        for (auto& ac : active_contexts) {
            TransferContext& ctx = *ac.ctx;
            if (!ctx.pxn_enabled) continue;
            ctx.pxn_diag_loop_iterations++;
            ctx.pxn_diag_direct_break_nic_full += diag_iter_direct_break_nic_full;
            ctx.pxn_diag_direct_break_qp_full += diag_iter_direct_break_qp_full;
            if (diag_iter_direct_break_nic_full == 0 && diag_iter_direct_break_qp_full == 0 &&
                diag_iter_direct_posted == 0) {
                ctx.pxn_diag_direct_break_no_chunks++;
            }
            if (diag_iter_direct_posted > ctx.pxn_diag_max_direct_per_iter)
                ctx.pxn_diag_max_direct_per_iter = diag_iter_direct_posted;
            if (diag_iter_proxy_posted > ctx.pxn_diag_max_proxy_per_iter)
                ctx.pxn_diag_max_proxy_per_iter = diag_iter_proxy_posted;
            // Track CQ completions polled this iteration
            // (outstanding peak is now sampled before CQ polling above)
        }

        // ---- Phase 4: Remove completed contexts ----
        {
            bool any_removed = false;
            for (size_t i = 0; i < active_contexts.size(); ) {
                TransferContext& ctx = *active_contexts[i].ctx;
                size_t total_chunks = ctx.total_chunks.load();

                if (ctx.finished.load()) {
                    // Already finished (error case) — finalize stats and remove
                    cleanupPxnContextState(ctx);
                    finalizeTransferStats(ctx);
                    active_contexts.erase(active_contexts.begin() + i);
                    any_removed = true;
                    continue;
                }

                if (ctx.total_completed.load() >= total_chunks) {
                    // All chunks completed — mark finished
                    cleanupPxnContextState(ctx);
                    ctx.error_code.store(MPCOMM_SUCCESS);
                    ctx.end_time = std::chrono::steady_clock::now();
                    finalizeTransferStats(ctx);
                    ctx.finished.store(true);  // Must be last write to ctx (release fence)
                    active_contexts.erase(active_contexts.begin() + i);
                    any_removed = true;
                    continue;
                }
                ++i;
            }

            // Rebuild poll NIC indices if contexts were removed
            if (any_removed) {
                poll_nic_indices.clear();
                std::vector<bool> nic_seen(num_nics, false);
                for (const auto& actx : active_contexts) {
                    for (size_t nic : actx.ctx->candidate_nic_indices) {
                        if (!nic_seen[nic]) {
                            nic_seen[nic] = true;
                            poll_nic_indices.push_back(nic);
                        }
                    }
                }
            }
        }

        // ---- Phase 5: NIC quarantine probe (worker_id==0 only) ----
        if (worker_id == 0 && !nic_quarantine_.empty()) {
            probeQuarantinedNics(num_nics);
        }
    }

    MPCOMM_LOG_INFO("MPComm: Worker %zu (NUMA %d) exiting\n", worker_id, numa_id);
}

// Periodically checks quarantined NICs via ibv_query_port.
// If the port is back to ACTIVE and all QPs can be rebuilt
// RESET->INIT->RTR->RTS, the NIC is re-admitted to the pool.
// Probe interval ~10ms (kProbeIntervalLoops iterations).
// Called only from worker 0.
void MPComm::Impl::probeQuarantinedNics(size_t num_nics) {
    static const size_t kProbeIntervalLoops = []() -> size_t {
        const char* e = std::getenv("MPCOMM_NIC_PROBE_INTERVAL");
        return e ? static_cast<size_t>(std::atoi(e)) : 10000;
    }();
    static const double kQuarantineSec = []() -> double {
        const char* e = std::getenv("MPCOMM_NIC_QUARANTINE_SEC");
        return e ? std::atof(e) : 5.0;
    }();

    static size_t probe_counter = 0;
    if (++probe_counter < kProbeIntervalLoops) return;
    probe_counter = 0;

    auto now = std::chrono::steady_clock::now();
    for (size_t nic = 0; nic < num_nics; ++nic) {
        if (!nic_quarantine_[nic]) continue;
        auto& q = *nic_quarantine_[nic];
        if (!q.in_quarantine.load(std::memory_order_relaxed)) continue;
        double secs = std::chrono::duration<double>(
            now - q.quarantine_since).count();
        if (secs < kQuarantineSec) continue;  // still in quarantine period

        // Probe: check physical port state
        auto& nic_ctx = *nic_contexts_[nic];
        struct ibv_port_attr port_attr = {};
        if (ibv_query_port(nic_ctx.context, nic_ctx.port, &port_attr) != 0
            || port_attr.state != IBV_PORT_ACTIVE) {
            // Port still down — reset quarantine timer
            q.quarantine_since = now;
            MPCOMM_LOG_DEBUG("MPComm: [QUARANTINE] NIC %zu still down after %.1fs, "
                   "re-quarantined\n", nic, secs);
            continue;
        }

        // Port is UP — try to rebuild all QPs for this NIC
        bool all_ok = true;
        {
            std::lock_guard<std::mutex> lock(nic_ctx.qp_mutex);
            for (auto& [key, qps] : nic_ctx.qp_map) {
                // Parse "host_id:local_nic:remote_nic"
                size_t last_colon = key.rfind(':');
                size_t second_colon = key.rfind(':', last_colon - 1);
                size_t remote_nic_idx = std::stoul(key.substr(last_colon + 1));
                std::string host_id = key.substr(0, second_colon);

                for (size_t qi = 0; qi < qps.size(); ++qi) {
                    RemoteEndpointInfo rinfo = {};
                    {
                        std::lock_guard<std::mutex> conn_lock(connections_mutex_);
                        auto it = connections_.find(host_id);
                        if (it != connections_.end() &&
                            remote_nic_idx < it->second.nic_endpoints.size()) {
                            rinfo = it->second.nic_endpoints[remote_nic_idx];
                            struct ibv_qp_attr qattr = {};
                            struct ibv_qp_init_attr qiattr = {};
                            if (ibv_query_qp(qps[qi], &qattr, IBV_QP_DEST_QPN, &qiattr) == 0) {
                                rinfo.qp_num = qattr.dest_qp_num;
                            }
                        }
                    }
                    // RESET -> INIT -> RTR -> RTS
                    struct ibv_qp_attr rst = {};
                    rst.qp_state = IBV_QPS_RESET;
                    if (ibv_modify_qp(qps[qi], &rst, IBV_QP_STATE) != 0 ||
                        modifyQPToInit(nic_ctx, qps[qi]) != 0 ||
                        modifyQPToRTR(nic_ctx, qps[qi], rinfo) != 0 ||
                        modifyQPToRTS(qps[qi]) != 0) {
                        all_ok = false;
                        break;
                    }
                }
                if (!all_ok) break;
            }
        }

        if (all_ok) {
            q.consecutive_errors.store(0, std::memory_order_relaxed);
            q.in_quarantine.store(false, std::memory_order_release);
            MPCOMM_LOG_WARN("MPComm: [QUARANTINE] NIC %zu recovered after %.1fs, "
                   "re-admitted to pool\n", nic, secs);
        } else {
            q.quarantine_since = now;
            MPCOMM_LOG_WARN("MPComm: [QUARANTINE] NIC %zu port UP but QP rebuild "
                   "failed, re-quarantined\n", nic);
        }
    }
}

// ============================================================================
// Pipelined Worker Helper Functions
// ============================================================================

void MPComm::Impl::initContextCache(TransferContext& ctx, NicConnCache& cache) {
    // Pre-cache connection info, lkeys, and QP pointers for all hosts and all candidate NICs.
    // This eliminates mutex acquisitions and linear searches from the hot posting loop.
    std::lock_guard<std::mutex> lock(connections_mutex_);
    for (size_t host_idx = 0; host_idx < ctx.host_list.size(); ++host_idx) {
        const std::string& host_id = ctx.host_list[host_idx];
        auto conn_it = connections_.find(host_id);
        if (conn_it != connections_.end()) {
            for (size_t local_nic : ctx.candidate_nic_indices) {
                NicConnInfo info;
                info.remote_nic = local_nic;
                const auto& nic_map = conn_it->second.local_to_remote_nic_map;
                auto map_it = nic_map.find(local_nic);
                if (map_it != nic_map.end() && !map_it->second.empty()) {
                    info.remote_nic = map_it->second[0];
                }
                info.rkey = conn_it->second.getRkeyForAddr(ctx.remote_addrs[host_idx], info.remote_nic);

                // PXN: For proxy NICs, use the proxy buffer's lkey instead of
                // the source buffer's lkey, because RDMA will post from proxy buffer.
                // For direct NICs (no proxy), use the source buffer's lkey as usual.
#ifdef USE_CUDA
                if (ctx.pxn_enabled) {
                    int proxy_gpu = pxn_manager_.getProxyGpuForNic(
                        ctx.pxn_source_gpu, local_nic);
                    if (proxy_gpu >= 0) {
                        // This NIC goes through a proxy GPU — use proxy buffer lkey
                        info.lkey = pxn_manager_.getProxyLkey(proxy_gpu, local_nic);
                    } else {
                        // Direct NIC — use source buffer lkey
                        info.lkey = getLkey(local_nic, reinterpret_cast<void *>(ctx.local_addr));
                    }
                } else  // NOLINT(readability/braces) - else body is the block below the #endif
#endif
                {
                    info.lkey = getLkey(local_nic, reinterpret_cast<void *>(ctx.local_addr));
                }

                info.num_qps = qps_per_connection_;
                for (size_t qi = 0; qi < qps_per_connection_ && qi < NicConnInfo::kMaxQPsPerConn; ++qi) {
                    info.qps[qi] = getOrCreateQP(local_nic, host_id, info.remote_nic, qi);
                }
                for (size_t qi = qps_per_connection_; qi < NicConnInfo::kMaxQPsPerConn; ++qi) {
                    info.qps[qi] = nullptr;
                }
                uint64_t cache_key = (static_cast<uint64_t>(host_idx) << 16) | local_nic;
                cache[cache_key] = info;
            }
        }
    }
}

int MPComm::Impl::pollAllNicsForWorker(
    int numa_id,
    const std::vector<size_t>& poll_nic_indices,
    size_t (&worker_nic_completed)[kMaxNics],
    size_t (&worker_nic_qp_completed)[kMaxNics][kMaxQPsPerNic],
    const std::vector<ActiveContextInfo>& active_ctx_info,
    WorkerLimitView& limits) {
    const int poll_batch_size = static_cast<int>(poll_batch_size_);
    struct ibv_wc wc_array[256];
    size_t num_nics = nic_contexts_.size();
    NicLimitTraceBuffer* trace = nic_limit_trace_.load(std::memory_order_acquire);

    for (size_t nic : poll_nic_indices) {
        auto &nic_ctx = *nic_contexts_[nic];
        int n = ibv_poll_cq(nic_ctx.cq, poll_batch_size, wc_array);
        if (n < 0) {
            MPCOMM_LOG_ERROR("MPComm: ibv_poll_cq failed on NIC %zu in worker\n", nic);
            return MPCOMM_ERR_TRANSFER;
        }
        // One timestamp per CQ batch for the experimental limit trace.
        NicLimitTraceBuffer* nic_trace = (n > 0 && trace && trace->nic == nic) ? trace : nullptr;
        const int64_t batch_ns = nic_trace ? steadyNowNs() : 0;
        for (int i = 0; i < n; ++i) {
            size_t completed_nic = WrIdEncoding::decodeNic(wc_array[i].wr_id);
            size_t completed_qp = WrIdEncoding::decodeQp(wc_array[i].wr_id);
            TransferHandle wc_handle = WrIdEncoding::decodeHandle(wc_array[i].wr_id);

            // Update worker-level flow control counters (always, regardless of success)
            if (completed_nic < num_nics) {
                worker_nic_completed[completed_nic]++;
                if (completed_qp < qps_per_connection_) {
                    worker_nic_qp_completed[completed_nic][completed_qp]++;
                }
                // Experimental NIC limit: per-generation accounting (never underflows
                // even if another worker's completion lands on this CQ poll).
                uint32_t& tagged = limits.out_by_tag[completed_nic][WrIdEncoding::decodeTag(wc_array[i].wr_id)];
                if (tagged > 0) --tagged;
                advanceNicLimitDrain(limits, completed_nic, trace);
            }

            // Fast path: find target context in local active list (no lock needed)
            TransferContext* target_ctx = nullptr;
            for (const auto& aci : active_ctx_info) {
                if (aci.handle == wc_handle) {
                    target_ctx = aci.ctx;
                    break;
                }
            }

            // Slow path: handle belongs to another transfer on same NUMA (rare)
            if (!target_ctx) {
                auto& nts = per_numa_transfers_[numa_id];
                std::lock_guard<std::mutex> lock(nts.mutex);
                auto it = nts.active_transfers.find(wc_handle);
                if (it != nts.active_transfers.end()) {
                    target_ctx = it->second.get();
                }
            }

            if (!target_ctx) {
                // Orphaned completion (transfer already released) — ignore
                continue;
            }

            if (wc_array[i].status != IBV_WC_SUCCESS) {
                // Quarantine the NIC that produced this WC error
                if (nic < nic_quarantine_.size() && nic_quarantine_[nic]) {
                    auto& q = *nic_quarantine_[nic];
                    size_t prev_errors = q.consecutive_errors.fetch_add(1, std::memory_order_relaxed);
                    if (!q.in_quarantine.load(std::memory_order_relaxed)) {
                        q.quarantine_since = std::chrono::steady_clock::now();
                        q.in_quarantine.store(true, std::memory_order_release);
                        MPCOMM_LOG_WARN("MPComm: [QUARANTINE] NIC %zu quarantined after WC error "
                               "(status=%d, consecutive=%zu)\n",
                               nic, wc_array[i].status, prev_errors + 1);
                    }
                }

                // When MPCOMM_FAULT_INJECT_TOLERATE is set, treat WC errors as
                // successful completions to observe degraded throughput instead
                // of immediately failing the transfer. For testing only.
                static const bool fi_tolerate = (std::getenv("MPCOMM_FAULT_INJECT_TOLERATE") != nullptr);
                if (fi_tolerate) {
                    static std::atomic<size_t> fi_wc_error_count{0};
                    size_t cnt = fi_wc_error_count.fetch_add(1);
                    if (cnt < 5) {  // Only log first 5 errors to avoid spam
                        MPCOMM_LOG_WARN("MPComm: [FAULT TOLERATE] WC error on NIC %zu ignored "
                               "(status=%d, wr_id=0x%lx, total_errors=%zu)\n",
                               nic, wc_array[i].status, wc_array[i].wr_id, cnt + 1);
                    }
                    // Fall through to count as completed (degraded mode)
                } else {
                    MPCOMM_LOG_ERROR("MPComm: WC error on NIC %zu in worker: status=%d, wr_id=0x%lx\n",
                            nic, wc_array[i].status, wc_array[i].wr_id);
                    target_ctx->error_code.store(MPCOMM_ERR_TRANSFER);
                    target_ctx->finished.store(true);
                    target_ctx->end_time = std::chrono::steady_clock::now();
                    continue;
                }
            }

            // Route completion to the target context
            if (completed_nic < num_nics) {
                target_ctx->per_nic_completed[completed_nic]++;
                target_ctx->per_nic_last_completion[completed_nic] = std::chrono::steady_clock::now();
                if (completed_qp < qps_per_connection_) {
                    target_ctx->per_nic_qp_completed[completed_nic][completed_qp]++;
                }
                // Decrement assigned_outstanding for proxy NIC pipelines
                if (target_ctx->pxn_enabled && target_ctx->pxn_is_proxy_nic[completed_nic]) {
                    for (auto& pp : target_ctx->pxn_proxy_pipelines) {
                        if (pp.nic_index == completed_nic) {
                            if (pp.assigned_outstanding > 0) pp.assigned_outstanding--;
                            break;
                        }
                    }
                }
            }
            if (nic_trace) {
                const size_t chunk_idx = WrIdEncoding::decodeChunk(wc_array[i].wr_id);
                if (chunk_idx < target_ctx->total_chunks.load(std::memory_order_relaxed)) {
                    nic_trace->recordBytes(batch_ns, target_ctx->getChunk(chunk_idx).length);
                }
            }
            target_ctx->total_completed.fetch_add(1);
        }
    }
    return MPCOMM_SUCCESS;
}

// ==================== Experimental per-NIC outstanding limit ====================

void MPComm::Impl::resetNicLimits() {
    for (auto& packed : nic_limit_packed_) {
        packed.store(packNicLimit(0, MPCOMM_NIC_LIMIT_NONE), std::memory_order_relaxed);
    }
    nic_limit_version_.fetch_add(1, std::memory_order_release);
}

void MPComm::Impl::adoptNicLimits(size_t worker_id, WorkerLimitView& view, size_t num_nics,
                                  const size_t (&posted)[kMaxNics],
                                  const size_t (&completed)[kMaxNics]) {
    WorkerLimitShared& shared = *worker_limit_shared_[worker_id];
    // Publish counters BEFORE applied_generation (release), so a reader that sees a
    // new applied generation also sees outstanding values at least this fresh.
    for (size_t nic = 0; nic < num_nics; ++nic) {
        shared.posted[nic].store(posted[nic], std::memory_order_relaxed);
        shared.outstanding[nic].store(posted[nic] >= completed[nic] ? posted[nic] - completed[nic] : 0,
                                      std::memory_order_relaxed);
    }

    const uint64_t version = nic_limit_version_.load(std::memory_order_acquire);
    if (version == view.seen_version) return;
    view.seen_version = version;

    NicLimitTraceBuffer* trace = nic_limit_trace_.load(std::memory_order_acquire);
    for (size_t nic = 0; nic < num_nics; ++nic) {
        const uint64_t packed = nic_limit_packed_[nic].load(std::memory_order_acquire);
        const uint32_t generation = static_cast<uint32_t>(packed >> 32);
        if (generation == view.generation[nic]) continue;
        if (view.out_by_tag[nic][generation & 0xFF] != 0) {
            // Drain the old owner before reusing its tag; CQ polling must continue.
            view.limit[nic] = 0;
            view.seen_version = ~0ULL;
            continue;
        }

        const uint32_t limit = static_cast<uint32_t>(packed);
        const uint32_t previous = view.generation[nic];
        view.limit[nic] = (limit == MPCOMM_NIC_LIMIT_NONE) ? SIZE_MAX : static_cast<size_t>(limit);
        view.generation[nic] = generation;
        view.live_gens[nic].push_back(generation);
        if (trace && trace->nic == nic) {
            trace->mark(previous, generation, steadyNowNs(), true);
        }
        advanceNicLimitDrain(view, nic, trace);
        shared.applied_generation[nic].store(generation, std::memory_order_release);
    }
}

void MPComm::Impl::advanceNicLimitDrain(WorkerLimitView& view, size_t nic,
                                        NicLimitTraceBuffer* trace) {
    auto& live = view.live_gens[nic];
    if (live.size() < 2) return;
    const uint32_t oldest = live.front();
    // The newest (current) generation always stays; older ones leave once drained.
    while (live.size() > 1 && view.out_by_tag[nic][live.front() & 0xFF] == 0) {
        live.pop_front();
    }
    // Every generation <= new front now has no older WQE in flight on this worker.
    if (trace && trace->nic == nic && live.front() != oldest) {
        trace->mark(oldest, live.front(), steadyNowNs(), false);
    }
}

int MPComm::Impl::setNicOutstandingLimit(size_t nic_index, uint32_t limit) {
    const int64_t set_ns = steadyNowNs();
    if (!initialized_) {
        MPCOMM_LOG_ERROR("MPComm: setNicOutstandingLimit failed - not initialized\n");
        return MPCOMM_ERR_CONTEXT;
    }
    if (nic_index >= nic_contexts_.size()) {
        MPCOMM_LOG_ERROR("MPComm: setNicOutstandingLimit failed - NIC index %zu out of range\n",
                         nic_index);
        return MPCOMM_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> lock(nic_limit_mutex_);
    const uint32_t generation =
        static_cast<uint32_t>(nic_limit_packed_[nic_index].load(std::memory_order_relaxed) >> 32) + 1;
    NicLimitTraceBuffer* trace = nic_limit_trace_.load(std::memory_order_relaxed);
    if (trace && trace->nic == nic_index) {
        const uint32_t index = generation - trace->first_generation;
        if (index < trace->max_events) {
            auto& event = trace->events[index];
            event.generation = generation;
            event.limit = limit;
            event.set_ns = set_ns;
            trace->event_count.store(static_cast<size_t>(index) + 1, std::memory_order_release);
        } else {
            trace->dropped_events.fetch_add(1, std::memory_order_relaxed);
        }
    }
    nic_limit_packed_[nic_index].store(packNicLimit(generation, limit), std::memory_order_release);
    nic_limit_version_.fetch_add(1, std::memory_order_release);
    return MPCOMM_SUCCESS;
}

int MPComm::Impl::getNicLimitState(size_t nic_index, NicLimitState &out) const {
    if (!initialized_ || worker_limit_shared_.empty()) return MPCOMM_ERR_CONTEXT;
    if (nic_index >= nic_contexts_.size()) return MPCOMM_ERR_INVALID_ARG;

    const uint64_t packed = nic_limit_packed_[nic_index].load(std::memory_order_acquire);
    out.generation = static_cast<uint32_t>(packed >> 32);
    out.limit = static_cast<uint32_t>(packed);
    const uint64_t static_cap = static_cast<uint64_t>(max_outstanding_per_qp_) * qps_per_connection_;
    out.effective_limit = static_cast<uint32_t>(
        std::min<uint64_t>({static_cast<uint64_t>(out.limit), static_cap, 0xFFFFFFFFULL}));
    out.all_workers_applied = true;
    out.outstanding = 0;
    out.worker_posted.clear();
    for (const auto& worker : worker_limit_shared_) {
        const uint32_t applied = worker->applied_generation[nic_index].load(std::memory_order_acquire);
        if (static_cast<int32_t>(applied - out.generation) < 0) out.all_workers_applied = false;
        out.outstanding += worker->outstanding[nic_index].load(std::memory_order_relaxed);
        out.worker_posted.push_back(worker->posted[nic_index].load(std::memory_order_relaxed));
    }
    return MPCOMM_SUCCESS;
}

int MPComm::Impl::waitNicLimitApplied(size_t nic_index, int timeout_ms) {
    const auto start = std::chrono::steady_clock::now();
    NicLimitState state;
    for (;;) {
        const int rc = getNicLimitState(nic_index, state);
        if (rc != MPCOMM_SUCCESS) return rc;
        if (state.all_workers_applied && state.outstanding <= state.effective_limit) {
            return MPCOMM_SUCCESS;
        }
        if (timeout_ms >= 0 &&
            std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(timeout_ms)) {
            return MPCOMM_ERR_TIMEOUT;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

int MPComm::Impl::startNicLimitTrace(size_t nic_index, uint32_t bin_ns,
                                     size_t max_bins, size_t max_events) {
    if (!initialized_) return MPCOMM_ERR_CONTEXT;
    if (nic_index >= nic_contexts_.size() || bin_ns < 1000 ||
        max_bins == 0 || max_bins > (size_t{1} << 26) ||
        max_events == 0 || max_events > (size_t{1} << 20)) {
        MPCOMM_LOG_ERROR("MPComm: startNicLimitTrace - invalid arguments\n");
        return MPCOMM_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> lock(nic_limit_mutex_);
    if (nic_limit_trace_.load(std::memory_order_relaxed)) {
        MPCOMM_LOG_ERROR("MPComm: startNicLimitTrace - a trace is already active\n");
        return MPCOMM_ERR_PENDING;
    }
    auto trace = std::make_unique<NicLimitTraceBuffer>();
    trace->nic = nic_index;
    trace->bin_ns = bin_ns;
    trace->max_bins = max_bins;
    trace->max_events = max_events;
    trace->bins.reset(new std::atomic<uint64_t>[max_bins]());
    trace->events.reset(new NicLimitTraceBuffer::Event[max_events]);
    trace->first_generation =
        static_cast<uint32_t>(nic_limit_packed_[nic_index].load(std::memory_order_relaxed) >> 32) + 1;
    trace->origin_ns = steadyNowNs();
    nic_limit_trace_.store(trace.get(), std::memory_order_release);
    nic_limit_traces_.push_back(std::move(trace));
    return MPCOMM_SUCCESS;
}

int MPComm::Impl::stopNicLimitTrace(NicLimitTrace &out) {
    std::lock_guard<std::mutex> lock(nic_limit_mutex_);
    NicLimitTraceBuffer* trace = nic_limit_trace_.load(std::memory_order_relaxed);
    if (!trace) return MPCOMM_ERR_CONTEXT;
    nic_limit_trace_.store(nullptr, std::memory_order_release);

    out.nic_index = trace->nic;
    out.origin_ns = trace->origin_ns;
    out.bin_ns = trace->bin_ns;
    size_t used = trace->max_bins;
    while (used > 0 && trace->bins[used - 1].load(std::memory_order_relaxed) == 0) --used;
    out.completed_bytes.resize(used);
    for (size_t i = 0; i < used; ++i) {
        out.completed_bytes[i] = trace->bins[i].load(std::memory_order_relaxed);
    }
    const size_t count = trace->event_count.load(std::memory_order_acquire);
    out.events.clear();
    out.events.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& event = trace->events[i];
        out.events.push_back({event.generation, event.limit, event.set_ns,
                              event.applied_ns.load(std::memory_order_relaxed),
                              event.drained_ns.load(std::memory_order_relaxed)});
    }
    out.dropped_events = trace->dropped_events.load(std::memory_order_relaxed);
    out.dropped_completions = trace->dropped_completions.load(std::memory_order_relaxed);
    return MPCOMM_SUCCESS;
}

void MPComm::Impl::finalizeTransferStats(TransferContext& ctx) {
    if (mpcomm_get_log_level() < MPCOMM_LOG_LEVEL_DEBUG) return;

    // Sampling: only generate stats for every Nth transfer (0 = every transfer)
    size_t stats_seq = transfer_stats_counter_.fetch_add(1, std::memory_order_relaxed);
    if (transfer_stats_interval_ > 0 && (stats_seq % transfer_stats_interval_ != 0)) return;

    auto to_us = [](const std::chrono::steady_clock::time_point& start,
                    const std::chrono::steady_clock::time_point& end) -> double {
        return std::chrono::duration<double, std::micro>(end - start).count();
    };

    size_t total_bytes = 0;
    for (size_t i = 0; i < ctx.lengths.size(); ++i) {
        total_bytes += ctx.lengths[i];
    }
    size_t total_chunks = ctx.total_chunks.load();

    double total_time_us = to_us(ctx.start_time, ctx.end_time);
    double bandwidth_gbps = (total_bytes * 8.0) / (total_time_us * 1000.0);

    char buf[256];
    auto& s = ctx.timing_breakdown_str;
    s.reserve(2048);

    snprintf(buf, sizeof(buf), "MPComm: Transfer %lu timing breakdown (total=%.1f us, %.2f GB/s, %.1f Gbps):\n",
             ctx.handle, total_time_us, total_bytes / total_time_us / 1000.0, bandwidth_gbps);
    s += buf;
    snprintf(buf, sizeof(buf), "  [1] Preparation (start->queued):     %8.1f us\n",
             to_us(ctx.start_time, ctx.queued_time));
    s += buf;
    snprintf(buf, sizeof(buf), "  [2] Queue wait (queued->worker):     %8.1f us\n",
             to_us(ctx.queued_time, ctx.worker_start_time));
    s += buf;
    snprintf(buf, sizeof(buf), "  [3] Cache build (worker->cache):     %8.1f us\n",
             to_us(ctx.worker_start_time, ctx.cache_done_time));
    s += buf;
    snprintf(buf, sizeof(buf), "  [4] First post (cache->first_post):  %8.1f us\n",
             to_us(ctx.cache_done_time, ctx.first_post_time));
    s += buf;
    snprintf(buf, sizeof(buf), "  [5] Posting (first_post->all_posted):%8.1f us\n",
             to_us(ctx.first_post_time, ctx.all_posted_time));
    s += buf;
    snprintf(buf, sizeof(buf), "  [6] Completion (all_posted->end):    %8.1f us\n",
             to_us(ctx.all_posted_time, ctx.end_time));
    s += buf;
    snprintf(buf, sizeof(buf), "  Total chunks: %zu\n", total_chunks);
    s += buf;
    if (ctx.pxn_enabled) {
        snprintf(buf, sizeof(buf),
                 "  PXN per-chunk pipeline: direct_chunks=%zu, "
                 "proxy_chunks=%zu, proxy_copy_bytes=%zu\n",
                 ctx.pxn_diag_direct_chunks,
                 ctx.pxn_diag_proxy_chunks,
                 ctx.pxn_diag_proxy_copy_bytes);
        s += buf;

        // Per-proxy pipeline summary
        for (size_t pi = 0; pi < ctx.pxn_proxy_pipelines.size(); ++pi) {
            const auto& pp = ctx.pxn_proxy_pipelines[pi];
            snprintf(buf, sizeof(buf),
                     "    Pipeline[%zu]: gpu=%d, nic=%zu, "
                     "chunks_posted=%zu, copies_submitted=%zu, "
                     "bytes_assigned=%zu, bytes_copied=%zu\n",
                     pi, pp.proxy_gpu_id, pp.nic_index,
                     pp.total_chunks_posted, pp.total_copies_submitted,
                     pp.total_bytes_assigned, pp.total_bytes_copied);
            s += buf;
        }

        // Per-NIC completion timing relative to all_posted_time
        double direct_last_us = 0.0, proxy_last_us = 0.0;
        for (size_t nic = 0; nic < ctx.num_nics_used; ++nic) {
            if (ctx.per_nic_completed[nic] == 0) continue;
            double nic_us = to_us(ctx.all_posted_time, ctx.per_nic_last_completion[nic]);
            if (ctx.pxn_is_proxy_nic[nic]) {
                if (nic_us > proxy_last_us) proxy_last_us = nic_us;
            } else {
                if (nic_us > direct_last_us) direct_last_us = nic_us;
            }
        }
        snprintf(buf, sizeof(buf),
                 "  PXN completion: direct_last=%.1f us, proxy_last=%.1f us\n",
                 direct_last_us, proxy_last_us);
        s += buf;

        // Show proxy RDMA posting span relative to all_posted_time
        if (ctx.pxn_diag_first_rdma_post.time_since_epoch().count() != 0) {
            double first_post_rel = to_us(ctx.all_posted_time, ctx.pxn_diag_first_rdma_post);
            double last_post_rel = to_us(ctx.all_posted_time, ctx.pxn_diag_last_rdma_post);
            snprintf(buf, sizeof(buf),
                     "  Proxy RDMA post span (rel to all_posted): first=%.1f us, last=%.1f us\n",
                     first_post_rel, last_post_rel);
            s += buf;
        }

        // PXN posting loop diagnostics
        snprintf(buf, sizeof(buf),
                 "  PXN loop diag: iterations=%zu, break_nic_full=%zu, break_qp_full=%zu, "
                 "break_no_chunks=%zu\n",
                 ctx.pxn_diag_loop_iterations,
                 ctx.pxn_diag_direct_break_nic_full,
                 ctx.pxn_diag_direct_break_qp_full,
                 ctx.pxn_diag_direct_break_no_chunks);
        s += buf;
        snprintf(buf, sizeof(buf),
                 "  PXN posting: max_direct/iter=%zu, max_proxy/iter=%zu, "
                 "proxy_not_ready=%zu\n",
                 ctx.pxn_diag_max_direct_per_iter,
                 ctx.pxn_diag_max_proxy_per_iter,
                 ctx.pxn_diag_proxy_not_ready);
        s += buf;
        snprintf(buf, sizeof(buf),
                 "  PXN outstanding peak: direct=%zu, proxy=%zu (max_per_nic=%zu)\n",
                 ctx.pxn_diag_max_outstanding_direct,
                 ctx.pxn_diag_max_outstanding_proxy,
                 max_outstanding_per_qp_ * qps_per_connection_);
        s += buf;
        snprintf(buf, sizeof(buf),
                 "  PXN time breakdown: advance_copies=%.1f ms, direct_post=%.1f ms, "
                 "proxy_post=%.1f ms, cq_poll=%.1f ms\n",
                 ctx.pxn_diag_ns_advance_copies / 1e6,
                 ctx.pxn_diag_ns_direct_posting / 1e6,
                 ctx.pxn_diag_ns_proxy_posting / 1e6,
                 ctx.pxn_diag_ns_cq_polling / 1e6);
        s += buf;
    }
}

int MPComm::Impl::getNumaForAddr(uintptr_t addr) const {
    // Get NUMA node for the memory address
    int numa_node = getNumaNodeForAddr(reinterpret_cast<void*>(addr));

    // If NUMA node is valid and within range, use it
    if (numa_node >= 0 && static_cast<size_t>(numa_node) < num_numa_nodes_) {
        return numa_node;
    }

    // Fallback: use address hash to distribute across NUMA nodes
    return static_cast<int>((addr >> 12) % num_numa_nodes_);
}

size_t MPComm::Impl::selectWorkerForNuma(int numa_id) {
    // Ensure numa_id is valid
    if (numa_id < 0 || static_cast<size_t>(numa_id) >= num_numa_nodes_) {
        numa_id = 0;
    }

    // Round-robin selection among workers for this NUMA node
    size_t local_idx = numa_worker_rr_[numa_id]->fetch_add(1, std::memory_order_relaxed) % kWorkersPerNuma;

    // Convert to global worker ID
    return static_cast<size_t>(numa_id) * kWorkersPerNuma + local_idx;
}

// ==================== End Async Transfer Implementation ====================

// =====================================================================
// MPComm Public API - Delegates to Impl (Pimpl Pattern)
// =====================================================================

MPComm::MPComm() : impl_(std::make_unique<Impl>()) {}

MPComm::~MPComm() = default;

MPComm::MPComm(MPComm &&) noexcept = default;

MPComm &MPComm::operator=(MPComm &&) noexcept = default;

int MPComm::init(const std::string &local_host_id,
                 const std::string &device_names,
                 int tcp_port) {
    return impl_->init(local_host_id, device_names, tcp_port);
}

int MPComm::initSingleNicPair(const std::string &local_host_id,
                              const std::string &device_name,
                              int tcp_port) {
    return impl_->initSingleNicPair(local_host_id, device_name, tcp_port);
}

void MPComm::shutdown() {
    impl_->shutdown();
}

int MPComm::registerMemory(void *addr, size_t length) {
    return impl_->registerMemory(addr, length);
}

int MPComm::unregisterMemory(void *addr) {
    return impl_->unregisterMemory(addr);
}

int MPComm::connect(const std::string &remote_host_id,
                    const std::string &remote_tcp_addr,
                    int remote_tcp_port) {
    return impl_->connect(remote_host_id, remote_tcp_addr, remote_tcp_port);
}

int MPComm::startAcceptThread() {
    return impl_->startAcceptThread();
}

void MPComm::stopAcceptThread() {
    impl_->stopAcceptThread();
}

int MPComm::updateRemoteMemoryInfo(const std::string &remote_host_id,
                                   const std::vector<uint32_t> &rkeys) {
    return impl_->updateRemoteMemoryInfo(remote_host_id, rkeys);
}

int MPComm::publishBuffer(void *addr, size_t length, int numa_node) {
    return impl_->publishBuffer(addr, length, numa_node);
}

int MPComm::unpublishBuffer(void *addr) {
    return impl_->unpublishBuffer(addr);
}

void MPComm::unpublishAllBuffers() {
    impl_->unpublishAllBuffers();
}

int MPComm::queryRemoteBuffer(const std::string &remote_host_id,
                              const std::string &remote_tcp_addr,
                              int remote_tcp_port,
                              RemoteBufferInfo &out_info) {
    return impl_->queryRemoteBuffer(remote_host_id, remote_tcp_addr,
                                    remote_tcp_port, out_info);
}

int MPComm::queryRemoteBufferByNuma(const std::string &remote_host_id,
                                    const std::string &remote_tcp_addr,
                                    int remote_tcp_port,
                                    int numa_node,
                                    RemoteBufferEntry &out_entry) {
    return impl_->queryRemoteBufferByNuma(remote_host_id, remote_tcp_addr,
                                          remote_tcp_port, numa_node, out_entry);
}

const PublishedBufferInfo* MPComm::getPublishedBufferInfo() const {
    return impl_->getPublishedBufferInfo();
}

size_t MPComm::getPublishedBufferCount() const {
    return impl_->getPublishedBufferCount();
}

uint32_t MPComm::getRkey(size_t nic_index, void *addr) const {
    return impl_->getRkey(nic_index, addr);
}

TransferHandle MPComm::scatterAsync(uintptr_t local_addr,
                                    const std::vector<std::string> &host_list,
                                    const std::vector<uintptr_t> &remote_addrs,
                                    const std::vector<size_t> &lengths) {
    return impl_->scatterAsync(local_addr, host_list, remote_addrs, lengths);
}

TransferHandle MPComm::gatherAsync(uintptr_t local_addr,
                                   const std::vector<std::string> &host_list,
                                   const std::vector<uintptr_t> &remote_addrs,
                                   const std::vector<size_t> &lengths) {
    return impl_->gatherAsync(local_addr, host_list, remote_addrs, lengths);
}

TransferHandle MPComm::broadcastAsync(uintptr_t local_addr,
                                      size_t length,
                                      const std::vector<std::string> &host_list,
                                      const std::vector<uintptr_t> &remote_addrs) {
    return impl_->broadcastAsync(local_addr, length, host_list, remote_addrs);
}

TransferHandle MPComm::putAsync(uintptr_t local_addr,
                                const std::string &remote_host_id,
                                uintptr_t remote_addr,
                                size_t length) {
    return impl_->putAsync(local_addr, remote_host_id, remote_addr, length);
}

TransferHandle MPComm::getAsync(uintptr_t local_addr,
                                const std::string &remote_host_id,
                                uintptr_t remote_addr,
                                size_t length) {
    return impl_->getAsync(local_addr, remote_host_id, remote_addr, length);
}

bool MPComm::isTransferComplete(TransferHandle handle) {
    return impl_->isTransferComplete(handle);
}

int MPComm::waitTransfer(TransferHandle handle, int timeout_ms) {
    return impl_->waitTransfer(handle, timeout_ms);
}

TransferResult MPComm::getTransferResult(TransferHandle handle) {
    return impl_->getTransferResult(handle);
}

void MPComm::releaseTransfer(TransferHandle handle) {
    impl_->releaseTransfer(handle);
}

int MPComm::setNicOutstandingLimit(size_t nic_index, uint32_t limit) {
    return impl_->setNicOutstandingLimit(nic_index, limit);
}

int MPComm::clearNicOutstandingLimit(size_t nic_index) {
    return impl_->setNicOutstandingLimit(nic_index, MPCOMM_NIC_LIMIT_NONE);
}

int MPComm::getNicLimitState(size_t nic_index, NicLimitState &out) const {
    return impl_->getNicLimitState(nic_index, out);
}

int MPComm::waitNicLimitApplied(size_t nic_index, int timeout_ms) {
    return impl_->waitNicLimitApplied(nic_index, timeout_ms);
}

int MPComm::startNicLimitTrace(size_t nic_index, uint32_t bin_ns, size_t max_bins, size_t max_events) {
    return impl_->startNicLimitTrace(nic_index, bin_ns, max_bins, max_events);
}

int MPComm::stopNicLimitTrace(NicLimitTrace &out) {
    return impl_->stopNicLimitTrace(out);
}

size_t MPComm::getNumNics() const {
    return impl_->getNumNics();
}

const std::string &MPComm::getLocalHostId() const {
    return impl_->getLocalHostId();
}

int MPComm::getTcpPort() const {
    return impl_->getTcpPort();
}

std::string MPComm::getGid(size_t nic_index) const {
    return impl_->getGid(nic_index);
}

std::string MPComm::getDeviceName(size_t nic_index) const {
    return impl_->getDeviceName(nic_index);
}

std::vector<std::string> MPComm::getActiveDevices() const {
    return impl_->getActiveDevices();
}

const char* MPComm::getNicFilterEnvVarName() {
    return kNicFilterEnvVar;
}

size_t MPComm::getMaxRdmaTransferSize() const {
    return impl_->getMaxRdmaTransferSize();
}

size_t MPComm::getQpsPerConnection() const {
    return impl_->getQpsPerConnection();
}

const char* MPComm::getQpsPerConnectionEnvVarName() {
    return kQpsPerConnectionEnvVar;
}

const std::vector<NumaTopology>& MPComm::getNumaTopology() const {
    return impl_->getNumaTopology();
}

const std::vector<NicTopologyInfo>& MPComm::getNicTopology() const {
    return impl_->getNicTopology();
}

int MPComm::getNicNumaNode(const std::string& nic_name) const {
    return impl_->getNicNumaNode(nic_name);
}

int MPComm::getNumaNodeForAddr(void* addr) const {
    return impl_->getNumaNodeForAddr(addr);
}

std::vector<size_t> MPComm::getLocalNicIndicesForNuma(int numa_node) const {
    return impl_->getLocalNicIndicesForNuma(numa_node);
}

int MPComm::getGpuNumaNode(int gpu_device_id) const {
    return impl_->getGpuNumaNode(gpu_device_id);
}

// =====================================================================
// HBM-DRAM Mapping Implementation
// =====================================================================

uintptr_t MPComm::Impl::mapDRAMtoGPU(void *host_addr, size_t length) {
#ifdef MPCOMM_ENABLE_CUDA_KERNELS
    if (!host_addr || length == 0) {
        MPCOMM_LOG_ERROR("MPComm: mapDRAMtoGPU: invalid arguments (addr=%p, length=%zu)\n",
                         host_addr, length);
        return 0;
    }

    // Query NUMA node for logging
    int numa_node = -1;
    int status = -1;
    if (move_pages(0, 1, &host_addr, nullptr, &status, 0) == 0 && status >= 0) {
        numa_node = status;
    }
    MPCOMM_LOG_INFO("MPComm: mapDRAMtoGPU: registering DRAM buffer %p (%zu bytes) on NUMA node %d\n",
                    host_addr, length, numa_node);

    // Register as pinned + mapped memory
    cudaError_t err = cudaHostRegister(host_addr, length, cudaHostRegisterMapped);
    if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
        MPCOMM_LOG_ERROR("MPComm: mapDRAMtoGPU: cudaHostRegister failed: %s\n",
                         cudaGetErrorString(err));
        return 0;
    }

    // Get GPU-accessible device pointer
    void *device_ptr = nullptr;
    err = cudaHostGetDevicePointer(&device_ptr, host_addr, 0);
    if (err != cudaSuccess || !device_ptr) {
        MPCOMM_LOG_ERROR("MPComm: mapDRAMtoGPU: cudaHostGetDevicePointer failed: %s\n",
                         cudaGetErrorString(err));
        cudaHostUnregister(host_addr);
        return 0;
    }

    // Track the mapping
    {
        std::lock_guard<std::mutex> lock(dram_gpu_mappings_mutex_);
        DRAMGPUMapping mapping;
        mapping.host_addr = host_addr;
        mapping.length = length;
        mapping.device_ptr = reinterpret_cast<uintptr_t>(device_ptr);
        dram_gpu_mappings_[reinterpret_cast<uintptr_t>(host_addr)] = mapping;
    }

    MPCOMM_LOG_INFO("MPComm: mapDRAMtoGPU: DRAM %p mapped to GPU device pointer %p\n",
                    host_addr, device_ptr);
    return reinterpret_cast<uintptr_t>(device_ptr);
#else
    (void)host_addr;
    (void)length;
    MPCOMM_LOG_ERROR("MPComm: mapDRAMtoGPU: CUDA kernels not compiled (USE_CUDA_KERNELS=OFF)\n");
    return 0;
#endif
}

int MPComm::Impl::unmapDRAMfromGPU(void *host_addr) {
#ifdef MPCOMM_ENABLE_CUDA_KERNELS
    if (!host_addr) {
        return MPCOMM_ERR_INVALID_ARG;
    }

    // Remove from tracking
    {
        std::lock_guard<std::mutex> lock(dram_gpu_mappings_mutex_);
        auto it = dram_gpu_mappings_.find(reinterpret_cast<uintptr_t>(host_addr));
        if (it == dram_gpu_mappings_.end()) {
            MPCOMM_LOG_WARN("MPComm: unmapDRAMfromGPU: address %p not found in mapping table\n",
                            host_addr);
            return MPCOMM_ERR_INVALID_ARG;
        }
        dram_gpu_mappings_.erase(it);
    }

    cudaError_t err = cudaHostUnregister(host_addr);
    if (err != cudaSuccess) {
        MPCOMM_LOG_ERROR("MPComm: unmapDRAMfromGPU: cudaHostUnregister failed: %s\n",
                         cudaGetErrorString(err));
        return MPCOMM_ERR_MEMORY;
    }

    MPCOMM_LOG_INFO("MPComm: unmapDRAMfromGPU: unregistered DRAM buffer %p\n", host_addr);
    return MPCOMM_SUCCESS;
#else
    (void)host_addr;
    MPCOMM_LOG_ERROR("MPComm: unmapDRAMfromGPU: CUDA kernels not compiled (USE_CUDA_KERNELS=OFF)\n");
    return MPCOMM_ERR_DEVICE;
#endif
}

// =====================================================================
// TMA Transfer Implementation
// =====================================================================

// External TMA kernel declarations (defined in mpcomm_tma_kernels.cu)
#ifdef MPCOMM_ENABLE_CUDA_KERNELS
extern void launch_tma_gather_kernel(
    const char *src_base, const int64_t *indices, char *dst_base,
    int block_size_bytes, int total_tasks, int max_sm_count, int mode);

extern void launch_tma_scatter_kernel(
    const char *src_base, const int64_t *indices, char *dst_base,
    int block_size_bytes, int total_tasks, int max_sm_count, int mode);
#endif

int MPComm::Impl::tmaGather(uintptr_t dram_dev_ptr, const int64_t *indices,
                             void *gpu_dst, int num_blocks, int block_size,
                             int max_sm_count, int mode) {
#ifdef MPCOMM_ENABLE_CUDA_KERNELS
    if (!dram_dev_ptr || !indices || !gpu_dst || num_blocks <= 0 || block_size <= 0) {
        MPCOMM_LOG_ERROR("MPComm: tmaGather: invalid arguments\n");
        return MPCOMM_ERR_INVALID_ARG;
    }

    launch_tma_gather_kernel(
        reinterpret_cast<const char *>(dram_dev_ptr),
        indices,
        reinterpret_cast<char *>(gpu_dst),
        block_size,
        num_blocks,
        max_sm_count,
        mode);

    return MPCOMM_SUCCESS;
#else
    (void)dram_dev_ptr; (void)indices; (void)gpu_dst;
    (void)num_blocks; (void)block_size; (void)max_sm_count; (void)mode;
    MPCOMM_LOG_ERROR("MPComm: tmaGather: CUDA kernels not compiled (USE_CUDA_KERNELS=OFF)\n");
    return MPCOMM_ERR_DEVICE;
#endif
}

int MPComm::Impl::tmaScatter(void *gpu_src, const int64_t *indices,
                              uintptr_t dram_dev_ptr, int num_blocks,
                              int block_size, int max_sm_count, int mode) {
#ifdef MPCOMM_ENABLE_CUDA_KERNELS
    if (!gpu_src || !indices || !dram_dev_ptr || num_blocks <= 0 || block_size <= 0) {
        MPCOMM_LOG_ERROR("MPComm: tmaScatter: invalid arguments\n");
        return MPCOMM_ERR_INVALID_ARG;
    }

    launch_tma_scatter_kernel(
        reinterpret_cast<const char *>(gpu_src),
        indices,
        reinterpret_cast<char *>(dram_dev_ptr),
        block_size,
        num_blocks,
        max_sm_count,
        mode);

    return MPCOMM_SUCCESS;
#else
    (void)gpu_src; (void)indices; (void)dram_dev_ptr;
    (void)num_blocks; (void)block_size; (void)max_sm_count; (void)mode;
    MPCOMM_LOG_ERROR("MPComm: tmaScatter: CUDA kernels not compiled (USE_CUDA_KERNELS=OFF)\n");
    return MPCOMM_ERR_DEVICE;
#endif
}

// =====================================================================
// Public API Forwarding: HBM-DRAM Mapping & TMA
// =====================================================================

uintptr_t MPComm::mapDRAMtoGPU(void *host_addr, size_t length) {
    return impl_->mapDRAMtoGPU(host_addr, length);
}

int MPComm::unmapDRAMfromGPU(void *host_addr) {
    return impl_->unmapDRAMfromGPU(host_addr);
}

int MPComm::tmaGather(uintptr_t dram_dev_ptr, const int64_t *indices,
                      void *gpu_dst, int num_blocks, int block_size,
                      int max_sm_count, H2DMode mode) {
    return impl_->tmaGather(dram_dev_ptr, indices, gpu_dst, num_blocks,
                            block_size, max_sm_count, static_cast<int>(mode));
}

int MPComm::tmaScatter(void *gpu_src, const int64_t *indices,
                       uintptr_t dram_dev_ptr, int num_blocks,
                       int block_size, int max_sm_count, H2DMode mode) {
    return impl_->tmaScatter(gpu_src, indices, dram_dev_ptr, num_blocks,
                             block_size, max_sm_count, static_cast<int>(mode));
}

}  // namespace mpcomm