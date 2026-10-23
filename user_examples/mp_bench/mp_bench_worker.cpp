#include "mpcomm.h"

#ifndef MP_BENCH_WITH_CUDA
#define MP_BENCH_WITH_CUDA 1
#endif
#if MP_BENCH_WITH_CUDA
#include <cuda.h>
#endif
#include <numa.h>
#include <numaif.h>
#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern char** environ;

namespace {
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
constexpr const char* kServerHost = "mp-bench-server";
constexpr int64_t kNsPerSecond = 1000000000;

void onSignal(int signal) { interrupted = signal; }

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void checkSignal() {
    if (interrupted) fail("interrupted by signal " + std::to_string(interrupted));
}

void checkComm(int rc, const char* operation) {
    if (rc != mpcomm::MPCOMM_SUCCESS)
        fail(std::string(operation) + " failed, MPComm error " + std::to_string(rc));
}

#if MP_BENCH_WITH_CUDA
void checkCuda(CUresult rc, const char* operation) {
    if (rc == CUDA_SUCCESS) return;
    const char* description = nullptr;
    cuGetErrorString(rc, &description);
    fail(std::string(operation) + " failed: " +
         (description ? description : std::to_string(static_cast<int>(rc))));
}
#endif

std::string jsonString(const std::string& value) {
    std::ostringstream out;
    out << '"';
    const char* hex = "0123456789abcdef";
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 32) out << "\\u00" << hex[c >> 4] << hex[c & 15];
                else out << static_cast<char>(c);
        }
    }
    out << '"';
    return out.str();
}

void emit(const std::string& json) {
    std::cout << json << '\n' << std::flush;
    if (!std::cout) fail("controller stdout pipe closed");
}

uint64_t unsignedNumber(const std::string& value, const std::string& flag) {
    uint64_t result = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        fail(flag + " must be a nonnegative integer");
    return result;
}

int integerNumber(const std::string& value, const std::string& flag) {
    int result = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        fail(flag + " must be an integer in int range");
    return result;
}

int64_t secondsNs(const std::string& value, const std::string& flag, bool allow_zero) {
    char* end = nullptr;
    errno = 0;
    const double seconds = std::strtod(value.c_str(), &end);
    if (value.empty() || end != value.c_str() + value.size() || errno || !std::isfinite(seconds) ||
        seconds < 0 || (!allow_zero && seconds <= 0) || seconds > 86400.0)
        fail(flag + " must be finite seconds in " + (allow_zero ? "[0,86400]" : "(0,86400]"));
    int64_t ns = static_cast<int64_t>(seconds * kNsPerSecond);
    if (!allow_zero && ns == 0) fail(flag + " must be at least one nanosecond");
    return ns;
}

struct Options {
    std::string role, memory, op, nic, peer_ip, numa = "auto", verify = "full";
    int gpu = -1, stream = 0, listen_port = 0, peer_port = 0;
    size_t size = 4 * 1024 * 1024, depth = 16, chunk = 8 * 1024 * 1024;
    unsigned qps = 4, outstanding = 16;
    int64_t warmup_ns = 2 * kNsPerSecond, duration_ns = 10 * kNsPerSecond;
    int64_t interval_ns = kNsPerSecond, timeout_ns = 30 * kNsPerSecond;
    int64_t setup_timeout_ns = 120 * kNsPerSecond;
    size_t total = 0;

    static Options parse(int argc, char** argv) {
        const std::vector<std::string> allowed = {
            "--role", "--memory", "--op", "--nic", "--gpu", "--numa", "--listen-port",
            "--peer-ip", "--peer-port", "--stream", "--size", "--depth", "--warmup",
            "--duration", "--interval", "--timeout-ms", "--qps", "--chunk", "--outstanding",
            "--verify", "--setup-timeout"};
        std::unordered_map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            const std::string key = argv[i];
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                fail("unknown option: " + key);
            if (i + 1 >= argc) fail("missing value for " + key);
            if (!args.emplace(key, argv[i + 1]).second) fail("duplicate option: " + key);
        }
        auto get = [&](const std::string& key, const std::string& fallback) {
            auto it = args.find(key);
            return it == args.end() ? fallback : it->second;
        };
        auto required = [&](const std::string& key) {
            auto value = get(key, "");
            if (value.empty()) fail("required option: " + key);
            return value;
        };
        Options o;
        o.role = required("--role");
        o.memory = required("--memory");
        o.op = required("--op");
        o.nic = required("--nic");
        o.peer_ip = get("--peer-ip", "");
        o.numa = get("--numa", "auto");
        o.verify = get("--verify", "full");
        o.gpu = integerNumber(get("--gpu", "-1"), "--gpu");
        o.stream = integerNumber(get("--stream", o.gpu >= 0 ? std::to_string(o.gpu) : "0"), "--stream");
        o.listen_port = integerNumber(get("--listen-port", "0"), "--listen-port");
        o.peer_port = integerNumber(get("--peer-port", "0"), "--peer-port");
        auto size_value = [&](const char* flag, size_t fallback) {
            uint64_t n = unsignedNumber(get(flag, std::to_string(fallback)), flag);
            if (!n || n > std::numeric_limits<size_t>::max()) fail(std::string(flag) + " is out of size_t range");
            return static_cast<size_t>(n);
        };
        o.size = size_value("--size", o.size);
        o.depth = size_value("--depth", o.depth);
        o.chunk = size_value("--chunk", o.chunk);
        uint64_t qps = unsignedNumber(get("--qps", "4"), "--qps");
        uint64_t outstanding = unsignedNumber(get("--outstanding", "16"), "--outstanding");
        if (qps < 1 || qps > 16) fail("--qps must be in [1,16]");
        if (outstanding < 1 || outstanding > (o.op == "read" ? 16u : 128u))
            fail("--outstanding must be in [1,16] for READ or [1,128] for WRITE");
        if (qps * outstanding > 1024)
            fail("--qps * --outstanding must not exceed the single-NIC CQ capacity (1024)");
        o.qps = static_cast<unsigned>(qps);
        o.outstanding = static_cast<unsigned>(outstanding);
        o.warmup_ns = secondsNs(get("--warmup", "2"), "--warmup", true);
        o.duration_ns = secondsNs(get("--duration", "10"), "--duration", false);
        o.interval_ns = secondsNs(get("--interval", "1"), "--interval", false);
        o.setup_timeout_ns = secondsNs(get("--setup-timeout", "120"), "--setup-timeout", false);
        uint64_t timeout_ms = unsignedNumber(get("--timeout-ms", "30000"), "--timeout-ms");
        if (!timeout_ms || timeout_ms > 86400000) fail("--timeout-ms must be in [1,86400000]");
        o.timeout_ns = static_cast<int64_t>(timeout_ms) * 1000000;
        if (o.role != "server" && o.role != "client") fail("--role must be server or client");
        if (o.memory != "cpu" && o.memory != "gpu") fail("--memory must be cpu or gpu");
        if (o.op != "read" && o.op != "write") fail("--op must be read or write");
        if (o.verify != "none" && o.verify != "full") fail("--verify must be none or full");
        if (o.stream < 0 || o.gpu < -1) fail("--stream must be nonnegative and --gpu must be >= -1");
        if (o.nic.find_first_of(", \t\r\n") != std::string::npos) fail("--nic must name exactly one NIC");
        if (o.numa != "auto" && integerNumber(o.numa, "--numa") < 0)
            fail("--numa must be auto or a nonnegative node ID");
        if (o.listen_port < 0 || o.listen_port > 65535 || o.peer_port < 0 || o.peer_port > 65535)
            fail("TCP ports must be in [0,65535]");
        if (o.role == "client" && (o.peer_ip.empty() || o.peer_port == 0))
            fail("client requires --peer-ip and a nonzero --peer-port");
        const bool gpu_expected = (o.role == "server" && o.op == "read") ||
                                  (o.role == "client" && o.op == "write");
        if ((o.memory == "gpu") != gpu_expected)
            fail("READ requires GPU server/CPU client; WRITE requires CPU server/GPU client");
#if !MP_BENCH_WITH_CUDA
        if (o.memory == "gpu") fail("GPU memory is unavailable: rebuild with -DMP_BENCH_WITH_CUDA=ON");
#endif
        if (o.memory == "gpu" && o.gpu < 0) fail("GPU memory requires --gpu with a logical CUDA ordinal");
        if (o.memory == "cpu" && o.gpu != -1) fail("CPU workers require --gpu -1; use --stream for the paired GPU label");
        if (o.chunk > std::numeric_limits<uint32_t>::max()) fail("--chunk exceeds the RDMA SGE 32-bit length limit");
        if (o.depth > std::numeric_limits<size_t>::max() / o.size)
            fail("--size * --depth overflows size_t");
        o.total = o.size * o.depth;
        return o;
    }

    uint8_t pattern() const { return static_cast<uint8_t>(stream % 251 + 1); }
};

void setEnvironment(const char* key, const std::string& value) {
    if (::setenv(key, value.c_str(), 1) != 0)
        fail(std::string("setenv ") + key + ": " + std::strerror(errno));
}

void configureEnvironment(const Options& o) {
    const char* filter = std::getenv("MPCOMM_NIC_FILTER");
    if (filter && *filter && o.nic != filter)
        fail("existing MPCOMM_NIC_FILTER differs from --nic; explicitly remove or correct it");
    std::vector<std::string> remove;
    for (char** item = environ; item && *item; ++item) {
        std::string key(*item);
        key.resize(key.find('='));
        if (key.rfind("MPCOMM_PXN", 0) == 0 || key.rfind("MPCOMM_FAULT", 0) == 0)
            remove.push_back(key);
    }
    for (const auto& key : remove) {
        if (::unsetenv(key.c_str()) != 0) fail("unsetenv failed for " + key);
    }
    setEnvironment("MPCOMM_PXN_ENABLE", "0");
    setEnvironment("MPCOMM_LOG_LEVEL", "error");
    setEnvironment("MPCOMM_NIC_FILTER", o.nic);
    setEnvironment("MPCOMM_QPS_PER_CONNECTION", std::to_string(o.qps));
    setEnvironment("MPCOMM_MAX_RDMA_TRANSFER_SIZE", std::to_string(o.chunk));
    setEnvironment("MPCOMM_MAX_OUTSTANDING_PER_QP", std::to_string(o.outstanding));
    setEnvironment("MPCOMM_MAX_SEND_WR", "256");
}

void probeGpus() {
#if MP_BENCH_WITH_CUDA
    CUresult init = cuInit(0);
    if (init == CUDA_ERROR_NO_DEVICE) {
        emit("{\"event\":\"gpus\",\"devices\":[]}");
        return;
    }
    checkCuda(init, "cuInit");
    int count = 0;
    checkCuda(cuDeviceGetCount(&count), "cuDeviceGetCount");
    std::ostringstream out;
    out << "{\"event\":\"gpus\",\"devices\":[";
    for (int id = 0; id < count; ++id) {
        CUdevice device;
        char pci[64]{}, name[256]{};
        checkCuda(cuDeviceGet(&device, id), "cuDeviceGet");
        checkCuda(cuDeviceGetPCIBusId(pci, sizeof(pci), device), "cuDeviceGetPCIBusId");
        checkCuda(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
        if (id) out << ',';
        out << "{\"id\":" << id << ",\"pci\":" << jsonString(pci)
            << ",\"name\":" << jsonString(name) << '}';
    }
    out << "]}";
    emit(out.str());
#else
    fail("--probe-gpus is unavailable: rebuild with -DMP_BENCH_WITH_CUDA=ON");
#endif
}

class Memory {
public:
    ~Memory() {
#if MP_BENCH_WITH_CUDA
        if (context_) {
            cuCtxSetCurrent(context_);
            if (device_ptr_) cuMemFree(device_ptr_);
            cuDevicePrimaryCtxRelease(device_);
        }
#endif
        if (cpu_) numa_free(cpu_, allocated_);
    }

    void allocate(const Options& o, int node) {
        if (o.memory == "gpu") {
#if MP_BENCH_WITH_CUDA
            checkCuda(cuInit(0), "cuInit");
            checkCuda(cuDeviceGet(&device_, o.gpu), "cuDeviceGet");
            checkCuda(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
            checkCuda(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
            checkCuda(cuMemAlloc(&device_ptr_, o.total), "cuMemAlloc");
            checkCuda(cuMemsetD8(device_ptr_, o.pattern(), o.total), "cuMemsetD8");
            checkCuda(cuCtxSynchronize(), "cuCtxSynchronize");
            return;
#else
            fail("GPU memory is unavailable: rebuild with -DMP_BENCH_WITH_CUDA=ON");
#endif
        }
        if (numa_available() < 0) fail("NUMA is unavailable; CPU allocation cannot be strictly bound");
        if (node < 0 || node > numa_max_node() ||
            !numa_bitmask_isbitset(numa_all_nodes_ptr, static_cast<unsigned>(node)))
            fail("requested NUMA node is not an available memory node");
        long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size <= 0) fail("sysconf(_SC_PAGESIZE) failed");
        size_t page = static_cast<size_t>(page_size);
        if (o.total > std::numeric_limits<size_t>::max() - (page - 1))
            fail("page-rounded allocation size overflows size_t");
        allocated_ = ((o.total + page - 1) / page) * page;
        cpu_ = numa_alloc_onnode(allocated_, node);
        if (!cpu_) fail("numa_alloc_onnode failed");
        std::unique_ptr<bitmask, decltype(&numa_bitmask_free)> mask(
            numa_allocate_nodemask(), &numa_bitmask_free);
        if (!mask) fail("numa_allocate_nodemask failed");
        numa_bitmask_clearall(mask.get());
        numa_bitmask_setbit(mask.get(), static_cast<unsigned>(node));
        if (::mbind(cpu_, allocated_, MPOL_BIND, mask->maskp, mask->size, MPOL_MF_STRICT) != 0)
            fail(std::string("strict mbind(MPOL_BIND) failed: ") + std::strerror(errno));
        std::memset(cpu_, 0, allocated_);
    }

    uintptr_t address() const {
        if (cpu_) return reinterpret_cast<uintptr_t>(cpu_);
#if MP_BENCH_WITH_CUDA
        return static_cast<uintptr_t>(device_ptr_);
#else
        return 0;
#endif
    }

    void verify(const Options& o) const {
        if (!cpu_) fail("full receiver verification requires CPU memory");
        const volatile uint8_t* bytes = static_cast<const volatile uint8_t*>(cpu_);
        for (size_t i = 0; i < o.total; ++i) {
            if ((i & 65535u) == 0) checkSignal();
            if (bytes[i] != o.pattern())
                fail("verification mismatch at byte " + std::to_string(i) +
                     ", expected " + std::to_string(o.pattern()) +
                     ", got " + std::to_string(bytes[i]));
        }
    }

private:
    void* cpu_ = nullptr;
    size_t allocated_ = 0;
#if MP_BENCH_WITH_CUDA
    CUdevice device_ = 0;
    CUcontext context_ = nullptr;
    CUdeviceptr device_ptr_ = 0;
#endif
};

class Input {
public:
    void pump(int timeout_ms) {
        if (eof_) return;
        pollfd fd{STDIN_FILENO, POLLIN, 0};
        int rc = ::poll(&fd, 1, timeout_ms);
        if (rc < 0) {
            if (errno == EINTR) return;
            fail(std::string("stdin poll: ") + std::strerror(errno));
        }
        if (rc == 0) return;
        if (fd.revents & (POLLERR | POLLNVAL)) fail("stdin pipe failed");
        if (!(fd.revents & (POLLIN | POLLHUP))) return;
        char data[4096];
        ssize_t length = ::read(STDIN_FILENO, data, sizeof(data));
        if (length < 0) {
            if (errno == EINTR || errno == EAGAIN) return;
            fail(std::string("stdin read: ") + std::strerror(errno));
        }
        if (length == 0) {
            eof_ = true;
            if (!pending_.empty()) { lines_.push_back(pending_); pending_.clear(); }
            return;
        }
        pending_.append(data, static_cast<size_t>(length));
        size_t newline;
        while ((newline = pending_.find('\n')) != std::string::npos) {
            std::string line = pending_.substr(0, newline);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            lines_.push_back(std::move(line));
            pending_.erase(0, newline + 1);
        }
        if (pending_.size() > 4096 || lines_.size() > 64) fail("stdin protocol input exceeds its limit");
    }

    bool take(std::string& line) {
        if (lines_.empty()) return false;
        line = std::move(lines_.front());
        lines_.pop_front();
        return true;
    }

    bool eof() const { return eof_; }

private:
    std::string pending_;
    std::deque<std::string> lines_;
    bool eof_ = false;
};

struct Slot {
    mpcomm::TransferHandle handle = mpcomm::INVALID_TRANSFER_HANDLE;
    int64_t submitted_ns = 0;
    bool received = false;
};

struct Counts {
    uint64_t bytes = 0, operations = 0, drain_bytes = 0, drain_operations = 0;
};

void addCount(uint64_t& count, uint64_t amount) {
    if (count > std::numeric_limits<uint64_t>::max() - amount) fail("completion counter overflow");
    count += amount;
}

class Worker {
public:
    explicit Worker(Options options) : options_(std::move(options)) {
        configureEnvironment(options_);
        comm_ = std::make_unique<mpcomm::MPComm>();
    }

    ~Worker() { shutdown(); }

    void run() {
        checkSignal();
        const std::string host = options_.role == "server" ? kServerHost :
            "mp-bench-client-" + std::to_string(::getpid());
        checkComm(comm_->initSingleNicPair(host, options_.nic, options_.listen_port), "initSingleNicPair");
        if (comm_->getNumNics() != 1 || comm_->getDeviceName(0) != options_.nic)
            fail("initSingleNicPair did not initialize exactly the requested NIC");
        if (comm_->getQpsPerConnection() != options_.qps || comm_->getMaxRdmaTransferSize() != options_.chunk)
            fail("MPComm did not honor the requested QP/chunk environment");
        node_ = options_.memory == "gpu" ? -1 :
            (options_.numa == "auto" ? comm_->getNicNumaNode(options_.nic) :
             integerNumber(options_.numa, "--numa"));
        if (options_.memory == "cpu" && node_ < 0)
            fail("NIC NUMA node is unknown; explicitly pass --numa N instead of auto");
        memory_.allocate(options_, node_);
        if (memory_.address() > std::numeric_limits<uintptr_t>::max() - options_.total)
            fail("local buffer address range overflows uintptr_t");
        checkComm(comm_->registerMemory(reinterpret_cast<void*>(memory_.address()), options_.total), "registerMemory");
        if (options_.role == "server") runServer();
        else runClient();
    }

private:
    void shutdown() noexcept {
        if (!comm_ || shutdown_) return;
        comm_->shutdown();
        shutdown_ = true;
        for (auto& slot : slots_) {
            if (slot.handle != mpcomm::INVALID_TRANSFER_HANDLE) {
                comm_->releaseTransfer(slot.handle);
                slot.handle = mpcomm::INVALID_TRANSFER_HANDLE;
            }
        }
    }

    void ready() {
        std::ostringstream out;
        out << "{\"event\":\"ready\",\"role\":" << jsonString(options_.role)
            << ",\"stream\":" << options_.stream << ",\"port\":" << comm_->getTcpPort()
            << ",\"size\":" << options_.size << ",\"depth\":" << options_.depth
            << ",\"memory\":" << jsonString(options_.memory) << ",\"numa\":" << node_
            << ",\"nic\":" << jsonString(options_.nic) << ",\"gpu\":" << options_.gpu
            << ",\"op\":" << jsonString(options_.op) << ",\"pattern\":" << unsigned(options_.pattern())
            << ",\"qps\":" << options_.qps << ",\"chunk\":" << options_.chunk
            << ",\"outstanding\":" << options_.outstanding
            << ",\"clock_ns\":" << nowNs() << ",\"allocated_bytes\":" << options_.total
            << ",\"allocated\":true}";
        emit(out.str());
    }

    void report(const char* event, const Counts& counts, int64_t elapsed, const char* verified = nullptr) {
        std::ostringstream out;
        out << "{\"event\":" << jsonString(event) << ",\"role\":" << jsonString(options_.role)
            << ",\"stream\":" << options_.stream << ",\"bytes\":" << counts.bytes
            << ",\"operations\":" << counts.operations << ",\"elapsed_ns\":" << elapsed
            << ",\"sample_ns\":" << last_sample_ns_ << ",\"window\":" << sample_window_
            << ",\"sampling_skew_ns\":" << (last_sample_ns_ - elapsed) << ",\"errors\":0";
        if (std::strcmp(event, "result") == 0) {
            out << ",\"drain_bytes\":" << counts.drain_bytes
                << ",\"drain_operations\":" << counts.drain_operations
                << ",\"verified\":" << (verified ? verified : "null")
                << ",\"verification\":" << jsonString(options_.verify == "none" ? "none" :
                    (options_.op == "write" ? "server" :
                     (options_.role == "server" ? "input_initialized" : "client")));
        }
        out << '}';
        emit(out.str());
    }

    void runServer() {
        checkComm(comm_->publishBuffer(reinterpret_cast<void*>(memory_.address()), options_.total, node_), "publishBuffer");
        checkComm(comm_->startAcceptThread(), "startAcceptThread");
        ready();
        for (;;) {
            checkSignal();
            input_.pump(100);
            std::string command;
            if (input_.take(command)) {
                if (command != "stop") fail("server expected stdin command: stop");
                break;
            }
            if (input_.eof()) break;
        }
        shutdown();
        const char* verified = nullptr;
        if (options_.verify == "full") {
            if (options_.op == "write") memory_.verify(options_);
            verified = "true";
        }
        report("result", Counts{}, 0, verified);
    }

    std::string waitCommand() {
        const int64_t deadline = nowNs() + std::max(options_.setup_timeout_ns,
            options_.warmup_ns + options_.timeout_ns + 30 * kNsPerSecond);
        for (;;) {
            checkSignal();
            input_.pump(100);
            std::string command;
            if (input_.take(command)) return command;
            if (input_.eof()) fail("controller stdin EOF before client command");
            if (nowNs() >= deadline) fail("controller stdin command timed out");
        }
    }

    void checkController() {
        checkSignal();
        const int64_t now = nowNs();
        if (now < next_control_ns_) return;
        next_control_ns_ = now + 10000000;
        input_.pump(0);
        std::string command;
        if (input_.take(command)) fail("unexpected client command during transfer: " + command);
        if (input_.eof()) fail("controller stdin EOF during client transfer");
    }

    void submit(size_t index) {
        checkSignal();
        auto& slot = slots_[index];
        if (slot.handle != mpcomm::INVALID_TRANSFER_HANDLE) fail("attempt to reuse an active buffer slot");
        const size_t offset = index * options_.size;
        slot.submitted_ns = nowNs();
        if (options_.op == "read")
            slot.handle = comm_->getAsync(memory_.address() + offset, kServerHost, remote_.addr + offset, options_.size);
        else
            slot.handle = comm_->putAsync(memory_.address() + offset, kServerHost, remote_.addr + offset, options_.size);
        if (slot.handle == mpcomm::INVALID_TRANSFER_HANDLE) fail("async transfer submission failed");
    }

    bool complete(size_t index, Counts& counts, bool measured, int64_t deadline) {
        auto& slot = slots_[index];
        if (slot.handle == mpcomm::INVALID_TRANSFER_HANDLE) return false;
        if (nowNs() - slot.submitted_ns >= options_.timeout_ns)
            fail("transfer observation timeout for slot " + std::to_string(index));
        if (!comm_->isTransferComplete(slot.handle)) return false;
        const int64_t observed_ns = nowNs();
        if (measured) publishDueSamples(counts, observed_ns);
        auto result = comm_->getTransferResult(slot.handle);
        if (result.error_code != mpcomm::MPCOMM_SUCCESS || result.bytes_transferred != options_.size)
            fail("transfer result for slot " + std::to_string(index) + ": error=" +
                 std::to_string(result.error_code) + ", bytes=" + std::to_string(result.bytes_transferred) +
                 ", expected=" + std::to_string(options_.size));
        comm_->releaseTransfer(slot.handle);
        slot.handle = mpcomm::INVALID_TRANSFER_HANDLE;
        slot.received = true;
        if (measured && observed_ns >= deadline) {
            addCount(counts.drain_bytes, result.bytes_transferred);
            addCount(counts.drain_operations, 1);
        } else {
            addCount(counts.bytes, result.bytes_transferred);
            addCount(counts.operations, 1);
        }
        return true;
    }

    bool active() const {
        for (const auto& slot : slots_)
            if (slot.handle != mpcomm::INVALID_TRANSFER_HANDLE) return true;
        return false;
    }

    void drain(Counts& counts, bool measured, int64_t measurement_deadline) {
        const int64_t drain_deadline = nowNs() + options_.timeout_ns;
        while (active()) {
            checkController();
            bool progress = false;
            for (size_t i = 0; i < slots_.size(); ++i) {
                checkSignal();
                if (nowNs() >= drain_deadline) fail("drain deadline exceeded");
                progress = complete(i, counts, measured, measurement_deadline) || progress;
            }
            if (!progress) std::this_thread::yield();
        }
    }

    void publishDueSamples(const Counts& counts, int64_t observed_ns) {
        const int64_t observed_elapsed = observed_ns - measurement_start_ns_;
        while (next_sample_elapsed_ns_ <= options_.duration_ns &&
               observed_elapsed >= next_sample_elapsed_ns_) {
            checkSignal();
            ++sample_window_;
            last_sample_ns_ = observed_elapsed;
            report("sample", counts, next_sample_elapsed_ns_);
            if (next_sample_elapsed_ns_ == options_.duration_ns)
                next_sample_elapsed_ns_ = options_.duration_ns + 1;
            else
                next_sample_elapsed_ns_ += std::min(options_.interval_ns,
                    options_.duration_ns - next_sample_elapsed_ns_);
        }
    }

    void window(Counts& counts, int64_t start, int64_t deadline, bool measured) {
        if (measured) {
            measurement_start_ns_ = start;
            next_sample_elapsed_ns_ = std::min(options_.interval_ns, options_.duration_ns);
            sample_window_ = 0;
            last_sample_ns_ = 0;
        }
        size_t index = 0;
        while (nowNs() < deadline) {
            checkController();
            if (measured) publishDueSamples(counts, nowNs());
            if (nowNs() >= deadline) break;
            complete(index, counts, measured, deadline);
            if (slots_[index].handle == mpcomm::INVALID_TRANSFER_HANDLE && nowNs() < deadline)
                submit(index);
            index = (index + 1) % slots_.size();
        }
        if (measured) publishDueSamples(counts, nowNs());
    }

    void verifyRead() {
        if (options_.verify != "full" || options_.op != "read") return;
        for (const auto& slot : slots_)
            if (!slot.received) fail("READ verification found a slot without any successful completion");
        memory_.verify(options_);
    }

    void runClient() {
        checkComm(comm_->connect(kServerHost, options_.peer_ip, options_.peer_port), "connect");
        checkComm(comm_->queryRemoteBufferByNuma(kServerHost, options_.peer_ip, options_.peer_port, -1, remote_),
                  "queryRemoteBufferByNuma");
        if (remote_.length < options_.total || remote_.rkeys.size() != 1 || remote_.rkeys[0] == 0)
            fail("remote buffer is too small or does not advertise exactly one valid rkey");
        if (remote_.addr > std::numeric_limits<uintptr_t>::max() - options_.total)
            fail("remote buffer address range overflows uintptr_t");
        checkComm(comm_->updateRemoteMemoryInfo(kServerHost, remote_.rkeys), "updateRemoteMemoryInfo");
        slots_.resize(options_.depth);
        ready();
        if (waitCommand() != "start") fail("client expected stdin command: start");
        Counts warmup;
        const int64_t warmup_start = nowNs();
        for (size_t i = 0; i < slots_.size(); ++i) {
            checkController();
            submit(i);
        }
        window(warmup, warmup_start, warmup_start + options_.warmup_ns, false);
        drain(warmup, false, 0);
        verifyRead();
        emit("{\"event\":\"warmup_ready\",\"stream\":" + std::to_string(options_.stream) +
             ",\"operations\":" + std::to_string(warmup.operations) +
             ",\"verified\":" + (options_.verify == "full" && options_.op == "read" ? "true" : "null") + "}");
        const std::string command = waitCommand();
        const std::string prefix = "measure ";
        if (command.rfind(prefix, 0) != 0) fail("client expected stdin command: measure <start_ns>");
        uint64_t raw_start = unsignedNumber(command.substr(prefix.size()), "measure start_ns");
        const int64_t now = nowNs();
        if (raw_start > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - options_.duration_ns))
            fail("measurement deadline overflows int64 nanoseconds");
        const int64_t start = static_cast<int64_t>(raw_start);
        if (start <= now) fail("measurement start must be in the future on CLOCK_MONOTONIC");
        if (start - now > std::max(options_.setup_timeout_ns,
            options_.warmup_ns + options_.timeout_ns + 30 * kNsPerSecond))
            fail("measurement start is beyond the controller barrier timeout");
        next_control_ns_ = 0;
        while (nowNs() < start) {
            checkController();
            const int64_t remaining = start - nowNs();
            if (remaining > 2000000)
                std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<int64_t>(remaining - 1000000, 10000000)));
        }
        const int64_t deadline = start + options_.duration_ns;
        Counts measured;
        window(measured, start, deadline, true);
        drain(measured, true, deadline);
        if (measured.operations == 0) fail("measurement completed zero successful operations before its deadline");
        verifyRead();
        shutdown();
        report("result", measured, options_.duration_ns,
               options_.verify == "full" && options_.op == "read" ? "true" : nullptr);
    }

    Options options_;
    Memory memory_;
    std::unique_ptr<mpcomm::MPComm> comm_;
    std::vector<Slot> slots_;
    mpcomm::RemoteBufferEntry remote_{};
    Input input_;
    int node_ = -1;
    bool shutdown_ = false;
    int64_t next_control_ns_ = 0;
    int64_t measurement_start_ns_ = 0;
    int64_t next_sample_elapsed_ns_ = 0;
    int64_t last_sample_ns_ = 0;
    uint64_t sample_window_ = 0;
};
}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGTERM, onSignal);
    std::signal(SIGINT, onSignal);
    std::signal(SIGPIPE, SIG_IGN);
    try {
        if (argc == 2 && std::strcmp(argv[1], "--probe-gpus") == 0) {
            probeGpus();
            return 0;
        }
        Worker worker(Options::parse(argc, argv));
        worker.run();
        return 0;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        std::cerr << message << '\n';
        std::cout << "{\"event\":\"error\",\"errors\":1,\"message\":"
                  << jsonString(message) << "}\n" << std::flush;
        return 1;
    }
}
