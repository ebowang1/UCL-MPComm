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
#include <atomic>
#include <cctype>
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
constexpr size_t kMaxStreams = 16;

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

bool validNicName(const std::string& name) {
    return !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_' || c == '.' || c == '-';
    });
}

int64_t millisNs(const std::string& value, const std::string& flag) {
    char* end = nullptr;
    errno = 0;
    const double ms = std::strtod(value.c_str(), &end);
    if (value.empty() || end != value.c_str() + value.size() || errno || !std::isfinite(ms) ||
        ms <= 0 || ms > 60000.0)
        fail(flag + " must be milliseconds in (0,60000]");
    const int64_t ns = static_cast<int64_t>(std::llround(ms * 1e6));
    if (ns < 1000) fail(flag + " must be at least 1 microsecond");
    return ns;
}

// Comma-separated outstanding-WQE limits, e.g. "1,4,16,64".
std::vector<uint32_t> parseOqList(const std::string& text) {
    std::vector<uint32_t> values;
    size_t begin = 0;
    for (;;) {
        const size_t end = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const uint64_t value = unsignedNumber(item, "--oq-list");
        if (value == 0 || value >= mpcomm::MPCOMM_NIC_LIMIT_NONE) fail("--oq-list values must be positive");
        values.push_back(static_cast<uint32_t>(value));
        if (values.size() > 64) fail("--oq-list supports at most 64 values");
        if (end == std::string::npos) return values;
        begin = end + 1;
    }
}

// One READ stream of the single receiver process: GPU label, local NIC and source data port.
struct StreamSpec {
    int gpu = 0;
    std::string nic;
    int port = 0;
};

std::vector<StreamSpec> parseStreams(const std::string& text) {
    std::vector<StreamSpec> streams;
    size_t begin = 0;
    for (;;) {
        const size_t end = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const size_t first = item.find(':'), last = item.rfind(':');
        if (first == std::string::npos || first == last) fail("--streams entries must be GPU:NIC:PORT");
        StreamSpec spec;
        spec.gpu = integerNumber(item.substr(0, first), "--streams GPU");
        spec.nic = item.substr(first + 1, last - first - 1);
        spec.port = integerNumber(item.substr(last + 1), "--streams PORT");
        if (spec.gpu < 0 || !validNicName(spec.nic) || spec.port < 1 || spec.port > 65535)
            fail("--streams entries require GPU >= 0, a valid NIC name and PORT in [1,65535]");
        for (const auto& other : streams)
            if (other.gpu == spec.gpu) fail("--streams contains a duplicate GPU");
        streams.push_back(std::move(spec));
        if (streams.size() > kMaxStreams) fail("--streams supports at most 16 GPU streams");
        if (end == std::string::npos) return streams;
        begin = end + 1;
    }
}

struct Options {
    std::string role, memory, op, nic, peer_ip, numa = "auto", verify = "full";
    int gpu = -1, stream = 0, listen_port = 0, peer_port = 0;
    size_t size = 4 * 1024 * 1024, depth = 16, chunk = 8 * 1024 * 1024;
    unsigned qps = 4, outstanding = 16;
    int64_t warmup_ns = 2 * kNsPerSecond, duration_ns = 10 * kNsPerSecond;
    int64_t interval_ns = kNsPerSecond, timeout_ns = 30 * kNsPerSecond;
    int64_t setup_timeout_ns = 120 * kNsPerSecond;
    bool read_rounds = false;
    size_t total = 0;
    std::vector<StreamSpec> streams;
    // READ receiver test mode: rounds (synchronized rounds), sweep (OQ -> bandwidth),
    // switch (limit switching cost). sweep/switch read total_bytes split across GPUs.
    std::string mode = "rounds";
    uint64_t total_bytes = 0;
    std::vector<uint32_t> oq_list;
    unsigned repeat = 3, switches = 100;
    uint32_t high_oq = 64, low_oq = 1, bin_ns = 100000;
    int64_t high_ns = 2000000, low_ns = 2000000;

    static Options parse(int argc, char** argv) {
        const std::vector<std::string> allowed = {
            "--role", "--memory", "--op", "--nic", "--gpu", "--numa", "--listen-port",
            "--peer-ip", "--peer-port", "--stream", "--size", "--depth", "--warmup",
            "--duration", "--interval", "--timeout-ms", "--qps", "--chunk", "--outstanding",
            "--verify", "--setup-timeout", "--read-rounds", "--streams", "--mode", "--total-bytes",
            "--oq-list", "--repeat", "--high-oq", "--low-oq", "--high-ms", "--low-ms",
            "--switches", "--bin-us"};
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
        o.nic = get("--nic", "");
        o.peer_ip = get("--peer-ip", "");
        o.numa = get("--numa", "auto");
        o.verify = get("--verify", "full");
        const std::string read_rounds = get("--read-rounds", "0");
        if (read_rounds != "0" && read_rounds != "1") fail("--read-rounds must be 0 or 1");
        o.read_rounds = read_rounds == "1";
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
        if (args.count("--streams")) {
            if (o.role != "client" || o.memory != "cpu" || o.op != "read")
                fail("--streams is only valid for the CPU READ client");
            if (args.count("--nic") || args.count("--peer-port") || args.count("--stream"))
                fail("--streams replaces --nic, --peer-port and --stream");
            if (!o.read_rounds || o.depth != 1) fail("--streams requires --read-rounds 1 and --depth 1");
            o.streams = parseStreams(args.at("--streams"));
        } else {
            if (o.nic.empty()) fail("required option: --nic");
            if (o.nic.find_first_of(", \t\r\n") != std::string::npos) fail("--nic must name exactly one NIC");
            if (o.read_rounds) fail("--read-rounds 1 requires --streams");
            if (o.role == "client" && o.op == "read")
                fail("the CPU READ client requires --streams: one process serves all GPU streams");
        }
        if (o.verify != "none" && o.verify != "full") fail("--verify must be none or full");
        if (o.stream < 0 || o.gpu < -1) fail("--stream must be nonnegative and --gpu must be >= -1");
        if (o.numa != "auto" && integerNumber(o.numa, "--numa") < 0)
            fail("--numa must be auto or a nonnegative node ID");
        if (o.listen_port < 0 || o.listen_port > 65535 || o.peer_port < 0 || o.peer_port > 65535)
            fail("TCP ports must be in [0,65535]");
        if (o.role == "client" && (o.peer_ip.empty() || (o.streams.empty() && o.peer_port == 0)))
            fail("client requires --peer-ip and a nonzero --peer-port (or --streams)");
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
        o.parseLimitMode(args);
        return o;
    }

    // Shares of total_bytes per stream (sweep/switch): remainder bytes go to the first streams.
    std::vector<size_t> shares() const {
        std::vector<size_t> result(streams.size(), size);
        if (mode == "rounds") return result;
        const uint64_t base = total_bytes / streams.size(), remainder = total_bytes % streams.size();
        for (size_t i = 0; i < result.size(); ++i) result[i] = static_cast<size_t>(base + (i < remainder ? 1 : 0));
        return result;
    }

    void parseLimitMode(const std::unordered_map<std::string, std::string>& args) {
        auto has = [&](const char* key) { return args.count(key) != 0; };
        auto get = [&](const char* key, const std::string& fallback) {
            auto it = args.find(key);
            return it == args.end() ? fallback : it->second;
        };
        mode = get("--mode", "rounds");
        if (mode != "rounds" && mode != "sweep" && mode != "switch") fail("--mode must be rounds, sweep or switch");
        if (mode == "rounds") {
            for (const char* key : {"--total-bytes", "--oq-list", "--repeat", "--high-oq", "--low-oq",
                                    "--high-ms", "--low-ms", "--switches", "--bin-us"})
                if (has(key)) fail(std::string(key) + " is only valid with --mode sweep or switch");
            return;
        }
        if (streams.empty()) fail("--mode sweep/switch requires the CPU READ receiver (--streams)");
        for (const auto& spec : streams)
            if (spec.nic != streams[0].nic) fail("--mode sweep/switch requires exactly one A NIC for all streams");
        if (!has("--total-bytes")) fail("--mode sweep/switch requires --total-bytes");
        total_bytes = unsignedNumber(get("--total-bytes", ""), "--total-bytes");
        if (total_bytes < streams.size()) fail("--total-bytes must give every GPU stream at least one byte");
        const uint64_t cap = static_cast<uint64_t>(qps) * outstanding;
        oq_list = parseOqList(get("--oq-list", "1,4,16,64"));
        for (uint32_t oq : oq_list)
            if (oq > (mode == "sweep" ? cap : 256))
                fail("--oq-list values must not exceed --qps * --outstanding (" + std::to_string(cap) + ")");
        const uint64_t rep = unsignedNumber(get("--repeat", "3"), "--repeat");
        if (rep < 1 || rep > 100) fail("--repeat must be in [1,100]");
        repeat = static_cast<unsigned>(rep);
        const uint64_t high = unsignedNumber(get("--high-oq", "64"), "--high-oq");
        const uint64_t low = unsignedNumber(get("--low-oq", "1"), "--low-oq");
        if (high < 1 || high > (mode == "switch" ? cap : 256))
            fail("--high-oq must be in [1, --qps * --outstanding = " + std::to_string(cap) + "]");
        if (low >= high) fail("--low-oq must be smaller than --high-oq (0 pauses posting)");
        high_oq = static_cast<uint32_t>(high);
        low_oq = static_cast<uint32_t>(low);
        high_ns = millisNs(get("--high-ms", "2"), "--high-ms");
        low_ns = millisNs(get("--low-ms", "2"), "--low-ms");
        const uint64_t count = unsignedNumber(get("--switches", "100"), "--switches");
        if (count < 1 || count > 100000) fail("--switches must be in [1,100000]");
        switches = static_cast<unsigned>(count);
        const uint64_t bin_us = unsignedNumber(get("--bin-us", "100"), "--bin-us");
        if (bin_us < 1 || bin_us > 10000) fail("--bin-us must be in [1,10000]");
        bin_ns = static_cast<uint32_t>(bin_us * 1000);
        for (size_t share : shares())
            if ((share + chunk - 1) / chunk > std::numeric_limits<uint32_t>::max())
                fail("a GPU share needs more than 2^32 chunks; increase --chunk");
    }

    uint8_t pattern() const { return static_cast<uint8_t>(stream % 251 + 1); }
};

void setEnvironment(const char* key, const std::string& value) {
    if (::setenv(key, value.c_str(), 1) != 0)
        fail(std::string("setenv ") + key + ": " + std::strerror(errno));
}

void configureEnvironment(const Options& o, const std::vector<std::string>& nics) {
    // Must run before any MPComm object exists: constructors read these variables and
    // no library thread may be running while the environment is modified.
    const char* filter = std::getenv("MPCOMM_NIC_FILTER");
    if (filter && *filter && (nics.size() != 1 || nics[0] != filter))
        fail("existing MPCOMM_NIC_FILTER differs from the selected NIC(s); explicitly remove or correct it");
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
    if (nics.size() == 1) setEnvironment("MPCOMM_NIC_FILTER", nics[0]);
    else if (::unsetenv("MPCOMM_NIC_FILTER") != 0) fail("unsetenv failed for MPCOMM_NIC_FILTER");
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

    // Zero a CPU receive buffer so a later verify() proves fresh READ data arrived.
    void clear(const Options& o) {
        if (!cpu_) fail("clearing requires CPU memory");
        std::memset(cpu_, 0, o.total);
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
    int64_t completed_ns = 0;
    bool received = false;
};

struct Counts {
    uint64_t bytes = 0, operations = 0, drain_bytes = 0, drain_operations = 0;
};

void addCount(uint64_t& count, uint64_t amount) {
    if (count > std::numeric_limits<uint64_t>::max() - amount) fail("completion counter overflow");
    count += amount;
}

// Line-based stdin commands from the Python controller.
class Controller {
public:
    std::string wait(int64_t timeout_ns, int pump_ms) {
        const int64_t deadline = nowNs() + timeout_ns;
        for (;;) {
            checkSignal();
            input.pump(pump_ms);
            std::string command;
            if (input.take(command)) return command;
            if (input.eof()) fail("controller stdin EOF before the next command");
            if (nowNs() >= deadline) fail("controller stdin command timed out");
        }
    }

    // Rate-limited check during transfers: no command may arrive and stdin must stay open.
    void check() {
        checkSignal();
        const int64_t now = nowNs();
        if (now < next_check_ns_) return;
        next_check_ns_ = now + 10000000;
        input.pump(0);
        std::string command;
        if (input.take(command)) fail("unexpected controller command during transfer: " + command);
        if (input.eof()) fail("controller stdin EOF during transfer");
    }

    void checkSoon() { next_check_ns_ = 0; }

    Input input;

private:
    int64_t next_check_ns_ = 0;
};

int64_t commandTimeout(const Options& o) {
    return std::max(o.setup_timeout_ns, o.warmup_ns + o.timeout_ns + 30 * kNsPerSecond);
}

int64_t measurementStart(const std::string& command, const Options& o) {
    const std::string prefix = "measure ";
    if (command.rfind(prefix, 0) != 0) fail("expected stdin command: measure <start_ns>");
    const uint64_t raw_start = unsignedNumber(command.substr(prefix.size()), "measure start_ns");
    if (raw_start > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - o.duration_ns))
        fail("measurement deadline overflows int64 nanoseconds");
    const int64_t start = static_cast<int64_t>(raw_start);
    const int64_t now = nowNs();
    if (start <= now) fail("measurement start must be in the future on CLOCK_MONOTONIC");
    if (start - now > commandTimeout(o)) fail("measurement start is beyond the controller barrier timeout");
    return start;
}

class Worker {
public:
    explicit Worker(Options options) : options_(std::move(options)) {
        configureEnvironment(options_, {options_.nic});
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
            << ",\"window\":" << sample_window_ << ",\"errors\":0";
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
            controller_.input.pump(100);
            std::string command;
            if (controller_.input.take(command)) {
                if (command != "stop") fail("server expected stdin command: stop");
                break;
            }
            if (controller_.input.eof()) break;
        }
        shutdown();
        const char* verified = nullptr;
        if (options_.verify == "full") {
            if (options_.op == "write") memory_.verify(options_);
            verified = "true";
        }
        report("result", Counts{}, 0, verified);
    }

    std::string waitCommand() { return controller_.wait(commandTimeout(options_), 100); }

    void submit(size_t index) {
        checkSignal();
        auto& slot = slots_[index];
        if (slot.handle != mpcomm::INVALID_TRANSFER_HANDLE) fail("attempt to reuse an active buffer slot");
        const size_t offset = index * options_.size;
        slot.submitted_ns = nowNs();
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
        slot.completed_ns = observed_ns;
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
            controller_.check();
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
            controller_.check();
            if (measured) publishDueSamples(counts, nowNs());
            if (nowNs() >= deadline) break;
            complete(index, counts, measured, deadline);
            if (slots_[index].handle == mpcomm::INVALID_TRANSFER_HANDLE && nowNs() < deadline)
                submit(index);
            index = (index + 1) % slots_.size();
        }
        if (measured) publishDueSamples(counts, nowNs());
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
            controller_.check();
            submit(i);
        }
        window(warmup, warmup_start, warmup_start + options_.warmup_ns, false);
        drain(warmup, false, 0);
        emit("{\"event\":\"warmup_ready\",\"stream\":" + std::to_string(options_.stream) +
             ",\"operations\":" + std::to_string(warmup.operations) + ",\"verified\":null}");
        const int64_t start = measurementStart(waitCommand(), options_);
        controller_.checkSoon();
        while (nowNs() < start) {
            controller_.check();
            const int64_t remaining = start - nowNs();
            if (remaining > 2000000)
                std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<int64_t>(remaining - 1000000, 10000000)));
        }
        const int64_t deadline = start + options_.duration_ns;
        Counts measured;
        window(measured, start, deadline, true);
        drain(measured, true, deadline);
        if (measured.operations == 0) fail("measurement completed zero successful operations before its deadline");
        shutdown();
        report("result", measured, options_.duration_ns);
    }

    Options options_;
    Memory memory_;
    std::unique_ptr<mpcomm::MPComm> comm_;
    std::vector<Slot> slots_;
    mpcomm::RemoteBufferEntry remote_{};
    Controller controller_;
    int node_ = -1;
    bool shutdown_ = false;
    int64_t measurement_start_ns_ = 0;
    int64_t next_sample_elapsed_ns_ = 0;
    int64_t last_sample_ns_ = 0;
    uint64_t sample_window_ = 0;
};

// Toggles NIC 0's outstanding limit between --high-oq and --low-oq from a dedicated
// thread, so the switch instants are not disturbed by controller IPC. The schedule
// starts at high, ends with a final high phase and then clears the limit.
class LimitToggler {
public:
    LimitToggler(mpcomm::MPComm& comm, const Options& options) : comm_(comm), options_(options) {
        thread_ = std::thread([this] { run(); });
    }

    ~LimitToggler() {
        stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable()) thread_.join();
    }

    LimitToggler(const LimitToggler&) = delete;
    LimitToggler& operator=(const LimitToggler&) = delete;

    bool done() const { return done_.load(std::memory_order_acquire); }

    // Join and rethrow a failure of the toggling thread.
    void finish() {
        if (thread_.joinable()) thread_.join();
        if (!error_.empty()) fail("limit toggling failed: " + error_);
    }

private:
    void set(uint32_t limit) { checkComm(comm_.setNicOutstandingLimit(0, limit), "setNicOutstandingLimit"); }

    bool sleepUntil(Clock::time_point when) {
        std::this_thread::sleep_until(when);
        return !stop_.load(std::memory_order_relaxed);
    }

    void run() noexcept {
        try {
            auto next = Clock::now();
            set(options_.high_oq);
            unsigned down = 0, up = 0;
            while (down < options_.switches || up < options_.switches) {
                next += std::chrono::nanoseconds(options_.high_ns);
                if (!sleepUntil(next)) break;
                set(options_.low_oq);
                ++down;
                next += std::chrono::nanoseconds(options_.low_ns);
                if (!sleepUntil(next)) break;
                set(options_.high_oq);
                ++up;
            }
            // Keep the last recovery a full high phase before clearing.
            sleepUntil(next + std::chrono::nanoseconds(options_.high_ns));
        } catch (const std::exception& error) {
            error_ = error.what();
        }
        comm_.clearNicOutstandingLimit(0);
        done_.store(true, std::memory_order_release);
    }

    mpcomm::MPComm& comm_;
    const Options& options_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> done_{false};
    std::string error_;
    std::thread thread_;
};

template <typename T, typename Write>
void emitChunks(const char* event, const char* field, const std::vector<T>& items, size_t per_line, Write write) {
    for (size_t offset = 0; offset < items.size(); offset += per_line) {
        std::ostringstream out;
        out << "{\"event\":\"" << event << "\",\"offset\":" << offset << ",\"" << field << "\":[";
        const size_t end = std::min(items.size(), offset + per_line);
        for (size_t i = offset; i < end; ++i) {
            if (i > offset) out << ',';
            write(out, items[i]);
        }
        out << "]}";
        emit(out.str());
    }
}

// CPU READ receiver: ONE process for all GPU streams, with one MPComm instance per
// local NIC (normally exactly one). Streams that share a NIC therefore share that
// instance's connection set, NUMA worker and outstanding window, as in a Decode node
// that pulls KV from several Prefill GPUs through one NIC.
class ReadReceiver {
public:
    explicit ReadReceiver(Options options) : options_(std::move(options)) {
        for (const auto& spec : options_.streams)
            if (std::find(nics_.begin(), nics_.end(), spec.nic) == nics_.end()) nics_.push_back(spec.nic);
        configureEnvironment(options_, nics_);
        for (size_t i = 0; i < nics_.size(); ++i) comms_.push_back(std::make_unique<mpcomm::MPComm>());
        nodes_.assign(nics_.size(), -1);
    }

    ~ReadReceiver() { shutdown(); }

    void run() {
        checkSignal();
        const std::string local = "mp-bench-client-" + std::to_string(::getpid());
        for (size_t i = 0; i < nics_.size(); ++i) {
            auto& instance = *comms_[i];
            checkComm(instance.initSingleNicPair(local + "-" + nics_[i], nics_[i], 0), "initSingleNicPair");
            if (instance.getNumNics() != 1 || instance.getDeviceName(0) != nics_[i])
                fail("initSingleNicPair did not initialize exactly NIC " + nics_[i]);
            if (instance.getQpsPerConnection() != options_.qps || instance.getMaxRdmaTransferSize() != options_.chunk)
                fail("MPComm did not honor the requested QP/chunk environment");
            nodes_[i] = options_.numa == "auto" ? instance.getNicNumaNode(nics_[i]) :
                integerNumber(options_.numa, "--numa");
            if (nodes_[i] < 0)
                fail("NIC " + nics_[i] + " NUMA node is unknown; explicitly pass --numa N instead of auto");
        }
        const std::vector<size_t> shares = options_.shares();
        for (size_t i = 0; i < options_.streams.size(); ++i) {
            const auto& spec = options_.streams[i];
            auto stream = std::make_unique<Stream>();
            stream->options = options_;
            stream->options.stream = spec.gpu;
            stream->options.nic = spec.nic;
            stream->options.peer_port = spec.port;
            // Each stream reads exactly its share: one request of `size` bytes.
            stream->options.size = shares[i];
            stream->options.depth = 1;
            stream->options.total = shares[i];
            stream->comm_index = static_cast<size_t>(
                std::find(nics_.begin(), nics_.end(), spec.nic) - nics_.begin());
            // Distinct logical IDs: every source GPU process is a separate connection.
            stream->host = std::string(kServerHost) + "-gpu" + std::to_string(spec.gpu);
            stream->memory.allocate(stream->options, nodes_[stream->comm_index]);
            if (stream->memory.address() > std::numeric_limits<uintptr_t>::max() - stream->options.total)
                fail("local buffer address range overflows uintptr_t");
            checkComm(comm(*stream).registerMemory(reinterpret_cast<void*>(stream->memory.address()),
                                                    stream->options.total), "registerMemory");
            streams_.push_back(std::move(stream));
        }
        for (auto& stream : streams_) connect(*stream);
        ready();
        if (options_.mode == "sweep") runSweep();
        else if (options_.mode == "switch") runSwitch();
        else runRounds();
    }

private:
    struct Stream {
        Options options;
        size_t comm_index = 0;
        std::string host;
        Memory memory;
        mpcomm::RemoteBufferEntry remote{};
        Slot slot;
        Counts counts;
    };

    mpcomm::MPComm& comm(const Stream& stream) { return *comms_[stream.comm_index]; }

    std::string label(const Stream& stream) const { return "GPU " + std::to_string(stream.options.stream); }

    void shutdown() noexcept {
        if (shutdown_) return;
        shutdown_ = true;
        for (auto& c : comms_) c->shutdown();
        for (auto& stream : streams_) {
            if (stream->slot.handle != mpcomm::INVALID_TRANSFER_HANDLE) {
                comm(*stream).releaseTransfer(stream->slot.handle);
                stream->slot.handle = mpcomm::INVALID_TRANSFER_HANDLE;
            }
        }
    }

    void connect(Stream& stream) {
        auto& c = comm(stream);
        const int port = stream.options.peer_port;
        checkComm(c.connect(stream.host, options_.peer_ip, port), "connect");
        checkComm(c.queryRemoteBufferByNuma(stream.host, options_.peer_ip, port, -1, stream.remote),
                  "queryRemoteBufferByNuma");
        if (stream.remote.length < stream.options.total || stream.remote.rkeys.size() != 1 || stream.remote.rkeys[0] == 0)
            fail(label(stream) + ": remote buffer is too small or does not advertise exactly one valid rkey");
        if (stream.remote.addr > std::numeric_limits<uintptr_t>::max() - stream.options.total)
            fail(label(stream) + ": remote buffer address range overflows uintptr_t");
        checkComm(c.updateRemoteMemoryInfo(stream.host, stream.remote.rkeys), "updateRemoteMemoryInfo");
    }

    void ready() {
        std::ostringstream out;
        out << "{\"event\":\"ready\",\"role\":\"client\",\"op\":\"read\",\"memory\":\"cpu\""
            << ",\"mode\":" << jsonString(options_.mode)
            << ",\"processes\":1,\"instances\":" << comms_.size() << ",\"size\":" << options_.size
            << ",\"qps\":" << options_.qps << ",\"chunk\":" << options_.chunk
            << ",\"outstanding\":" << options_.outstanding << ",\"clock_ns\":" << nowNs() << ",\"streams\":[";
        for (size_t i = 0; i < streams_.size(); ++i) {
            const Stream& s = *streams_[i];
            if (i) out << ',';
            out << "{\"stream\":" << s.options.stream << ",\"nic\":" << jsonString(s.options.nic)
                << ",\"numa\":" << nodes_[s.comm_index] << ",\"pattern\":" << unsigned(s.options.pattern())
                << ",\"bytes\":" << s.options.size << ",\"allocated_bytes\":" << s.options.total << '}';
        }
        out << "]}";
        emit(out.str());
    }

    std::string waitCommand() { return controller_.wait(commandTimeout(options_), 10); }

    void submit(Stream& stream) {
        checkSignal();
        if (stream.slot.handle != mpcomm::INVALID_TRANSFER_HANDLE) fail(label(stream) + ": READ slot is still active");
        stream.slot.submitted_ns = nowNs();
        stream.slot.handle = comm(stream).getAsync(stream.memory.address(), stream.host,
                                                   stream.remote.addr, stream.options.size);
        if (stream.slot.handle == mpcomm::INVALID_TRANSFER_HANDLE) fail(label(stream) + ": READ submission failed");
    }

    bool complete(Stream& stream) {
        auto& slot = stream.slot;
        if (slot.handle == mpcomm::INVALID_TRANSFER_HANDLE) return false;
        if (nowNs() - slot.submitted_ns >= options_.timeout_ns) fail(label(stream) + ": READ observation timeout");
        auto& c = comm(stream);
        if (!c.isTransferComplete(slot.handle)) return false;
        slot.completed_ns = nowNs();
        auto result = c.getTransferResult(slot.handle);
        if (result.error_code != mpcomm::MPCOMM_SUCCESS || result.bytes_transferred != stream.options.size)
            fail(label(stream) + ": READ result error=" + std::to_string(result.error_code) +
                 ", bytes=" + std::to_string(result.bytes_transferred) +
                 ", expected=" + std::to_string(stream.options.size));
        c.releaseTransfer(slot.handle);
        slot.handle = mpcomm::INVALID_TRANSFER_HANDLE;
        slot.received = true;
        addCount(stream.counts.bytes, result.bytes_transferred);
        addCount(stream.counts.operations, 1);
        return true;
    }

    // One round: one getAsync per GPU stream from this process, then wait for all of them.
    void executeRound() {
        controller_.checkSoon();
        for (auto& stream : streams_) submit(*stream);
        size_t pending = streams_.size();
        while (pending) {
            controller_.check();
            for (auto& stream : streams_)
                if (complete(*stream)) --pending;
        }
    }

    void emitRound(uint64_t round) {
        std::ostringstream out;
        out << "{\"event\":\"round_done\",\"round\":" << round << ",\"streams\":[";
        for (size_t i = 0; i < streams_.size(); ++i) {
            const Stream& s = *streams_[i];
            if (i) out << ',';
            out << "{\"stream\":" << s.options.stream << ",\"submitted_ns\":" << s.slot.submitted_ns
                << ",\"completed_ns\":" << s.slot.completed_ns << ",\"bytes\":" << s.options.size
                << ",\"errors\":0}";
        }
        out << "]}";
        emit(out.str());
    }

    void verify() {
        if (options_.verify != "full") return;
        for (const auto& stream : streams_) {
            if (!stream->slot.received) fail(label(*stream) + ": READ verification found no successful completion");
            stream->memory.verify(stream->options);
        }
    }

    void reportResult(uint64_t rounds) {
        const char* verified = options_.verify == "full" ? "true" : "null";
        std::ostringstream out;
        out << "{\"event\":\"result\",\"role\":\"client\",\"read_rounds\":true,\"rounds\":" << rounds
            << ",\"results\":[";
        for (size_t i = 0; i < streams_.size(); ++i) {
            const Stream& s = *streams_[i];
            if (i) out << ',';
            out << "{\"stream\":" << s.options.stream << ",\"bytes\":" << s.counts.bytes
                << ",\"operations\":" << s.counts.operations << ",\"elapsed_ns\":" << options_.duration_ns
                << ",\"window\":0,\"errors\":0,\"read_rounds\":true,\"drain_bytes\":0,\"drain_operations\":0"
                << ",\"verified\":" << verified << ",\"verification\":"
                << jsonString(options_.verify == "none" ? "none" : "client") << '}';
        }
        out << "]}";
        emit(out.str());
    }

    void runRounds() {
        if (waitCommand() != "start") fail("READ rounds expected stdin command: start");
        emit("{\"event\":\"rounds_ready\"}");
        enum class Phase { Warmup, AwaitMeasure, Measured };
        Phase phase = Phase::Warmup;
        uint64_t rounds = 0, last_round = 0;
        int64_t measurement_start = 0;
        const std::string round_prefix = "round ";
        for (;;) {
            const std::string command = waitCommand();
            if (command.rfind(round_prefix, 0) == 0) {
                if (phase == Phase::AwaitMeasure) fail("READ rounds expected measure before the next round");
                const uint64_t round = unsignedNumber(command.substr(round_prefix.size()), "round id");
                if (last_round == std::numeric_limits<uint64_t>::max() || round != last_round + 1)
                    fail("READ round IDs must be positive and consecutive from 1");
                if (phase == Phase::Measured && nowNs() < measurement_start)
                    fail("READ measurement round arrived before measurement start");
                executeRound();
                last_round = round;
                addCount(rounds, 1);
                emitRound(round);
            } else if (command == "warmup_done") {
                if (phase != Phase::Warmup) fail("READ warmup_done is only valid during warmup");
                if (rounds == 0) fail("READ warmup requires at least one completed round");
                verify();
                phase = Phase::AwaitMeasure;
                emit("{\"event\":\"warmup_ready\",\"rounds\":" + std::to_string(rounds) +
                     ",\"verified\":" + (options_.verify == "full" ? "true" : "null") + "}");
            } else if (command.rfind("measure ", 0) == 0) {
                if (phase != Phase::AwaitMeasure) fail("READ measure is only valid once, after warmup_done");
                measurement_start = measurementStart(command, options_);
                rounds = 0;
                for (auto& stream : streams_) stream->counts = Counts{};
                phase = Phase::Measured;
                emit("{\"event\":\"measurement_ready\",\"start_ns\":" + std::to_string(measurement_start) + "}");
            } else if (command == "finish") {
                if (phase != Phase::Measured) fail("READ finish requires measurement phase");
                if (nowNs() < measurement_start + options_.duration_ns)
                    fail("READ finish arrived before the measurement deadline");
                if (rounds == 0) fail("READ measurement requires at least one completed round");
                verify();
                shutdown();
                reportResult(rounds);
                return;
            } else {
                fail("unexpected READ rounds command: " + command);
            }
        }
    }

    // ---- sweep / switch: whole total_bytes read once per measurement ----

    struct TransferTiming {
        int64_t submitted_ns = 0;  // common submit instant (before the first getAsync)
        int64_t completed_ns = 0;  // last stream observed complete
    };

    // Submit one getAsync per stream (its whole share) at once and wait for all of them.
    TransferTiming transferOnce() {
        controller_.checkSoon();
        TransferTiming timing;
        timing.submitted_ns = nowNs();
        for (auto& stream : streams_) submit(*stream);
        size_t pending = streams_.size();
        while (pending) {
            controller_.check();
            bool progress = false;
            for (auto& stream : streams_) {
                if (complete(*stream)) {
                    --pending;
                    progress = true;
                }
            }
            if (pending && !progress) std::this_thread::yield();
        }
        for (const auto& stream : streams_)
            timing.completed_ns = std::max(timing.completed_ns, stream->slot.completed_ns);
        return timing;
    }

    int limitTimeoutMs() const {
        return static_cast<int>(std::min<int64_t>(options_.timeout_ns / 1000000, std::numeric_limits<int>::max()));
    }

    void applyLimit(mpcomm::MPComm& instance, uint32_t limit) {
        checkComm(instance.setNicOutstandingLimit(0, limit), "setNicOutstandingLimit");
        checkComm(instance.waitNicLimitApplied(0, limitTimeoutMs()), "waitNicLimitApplied");
    }

    // The OQ limit is enforced per worker thread: results are only valid if one worker posted.
    unsigned postingWorkers(mpcomm::MPComm& instance) {
        mpcomm::NicLimitState state{};
        checkComm(instance.getNicLimitState(0, state), "getNicLimitState");
        unsigned count = 0;
        for (uint64_t posted : state.worker_posted)
            if (posted) ++count;
        if (count != 1)
            fail(std::to_string(count) + " MPComm workers posted on the NIC; the OQ limit is per worker, so "
                 "results are invalid. Keep every receive buffer on the NIC NUMA node (--numa auto).");
        return count;
    }

    void clearBuffers() {
        if (options_.verify != "full") return;
        for (auto& stream : streams_) stream->memory.clear(stream->options);
    }

    void emitWarmup(const TransferTiming& timing) {
        emit("{\"event\":\"warmup_done\",\"latency_ns\":" +
             std::to_string(timing.completed_ns - timing.submitted_ns) + "}");
    }

    void emitSweepResult(uint32_t oq, unsigned repeat, const TransferTiming& timing) {
        std::ostringstream out;
        out << "{\"event\":\"sweep_result\",\"oq\":" << oq << ",\"repeat\":" << repeat
            << ",\"submitted_ns\":" << timing.submitted_ns << ",\"completed_ns\":" << timing.completed_ns
            << ",\"latency_ns\":" << (timing.completed_ns - timing.submitted_ns) << ",\"streams\":[";
        for (size_t i = 0; i < streams_.size(); ++i) {
            const Stream& s = *streams_[i];
            if (i) out << ',';
            out << "{\"stream\":" << s.options.stream << ",\"bytes\":" << s.options.size
                << ",\"completed_ns\":" << s.slot.completed_ns << '}';
        }
        out << "]}";
        emit(out.str());
    }

    // Test 1: for every OQ, repeat: set limit, wait until applied, read total_bytes once.
    void runSweep() {
        if (waitCommand() != "start") fail("sweep expected stdin command: start");
        auto& instance = *comms_[0];
        checkComm(instance.clearNicOutstandingLimit(0), "clearNicOutstandingLimit");
        emitWarmup(transferOnce());
        const size_t runs = options_.oq_list.size() * options_.repeat;
        size_t run = 0;
        for (uint32_t oq : options_.oq_list) {
            for (unsigned repeat = 1; repeat <= options_.repeat; ++repeat) {
                if (++run == runs) clearBuffers();  // the last run must rewrite every verified byte
                applyLimit(instance, oq);
                emitSweepResult(oq, repeat, transferOnce());
            }
        }
        checkComm(instance.clearNicOutstandingLimit(0), "clearNicOutstandingLimit");
        const unsigned posting = postingWorkers(instance);
        verify();
        shutdown();
        emit("{\"event\":\"sweep_done\",\"runs\":" + std::to_string(runs) + ",\"posting_workers\":" +
             std::to_string(posting) + ",\"verified\":" + (options_.verify == "full" ? "true" : "null") + "}");
    }

    // Test 2: toggle the limit between high and low while reading total_bytes repeatedly,
    // and export the limit events, completed bytes per bin and transfer boundaries.
    void runSwitch() {
        if (waitCommand() != "start") fail("switch expected stdin command: start");
        auto& instance = *comms_[0];
        checkComm(instance.clearNicOutstandingLimit(0), "clearNicOutstandingLimit");
        emitWarmup(transferOnce());
        clearBuffers();
        const int64_t planned = static_cast<int64_t>(options_.switches) * (options_.high_ns + options_.low_ns) +
                                options_.high_ns;
        const int64_t span = planned + options_.timeout_ns + 10 * kNsPerSecond;
        const int64_t needed_bins = span / options_.bin_ns + 1;
        if (needed_bins > (int64_t{1} << 26))
            fail("switch trace exceeds capacity; increase --bin-us or reduce --switches/phase durations");
        const size_t max_bins = static_cast<size_t>(needed_bins);
        const size_t max_events = 2 * static_cast<size_t>(options_.switches) + 16;
        checkComm(instance.startNicLimitTrace(0, options_.bin_ns, max_bins, max_events), "startNicLimitTrace");
        std::vector<TransferTiming> transfers;
        {
            LimitToggler toggler(instance, options_);
            do {
                transfers.push_back(transferOnce());
            } while (!toggler.done());
            toggler.finish();
        }
        mpcomm::NicLimitTrace trace;
        checkComm(instance.stopNicLimitTrace(trace), "stopNicLimitTrace");
        const unsigned posting = postingWorkers(instance);

        std::ostringstream meta;
        meta << "{\"event\":\"trace_meta\",\"origin_ns\":" << trace.origin_ns << ",\"bin_ns\":" << trace.bin_ns
             << ",\"bins\":" << trace.completed_bytes.size() << ",\"events\":" << trace.events.size()
             << ",\"transfers\":" << transfers.size() << ",\"dropped_events\":" << trace.dropped_events
             << ",\"dropped_completions\":" << trace.dropped_completions << '}';
        emit(meta.str());
        emitChunks("trace_transfers", "transfers", transfers, 512, [](std::ostream& out, const TransferTiming& t) {
            out << '[' << t.submitted_ns << ',' << t.completed_ns << ']';
        });
        emitChunks("trace_events", "events", trace.events, 512, [](std::ostream& out, const mpcomm::NicLimitEvent& e) {
            out << '[' << e.generation << ',' << e.limit << ',' << e.set_ns << ',' << e.applied_ns << ','
                << e.drained_ns << ']';
        });
        emitChunks("trace_bins", "bytes", trace.completed_bytes, 4096, [](std::ostream& out, uint64_t bytes) {
            out << bytes;
        });
        verify();
        shutdown();
        emit("{\"event\":\"switch_done\",\"posting_workers\":" + std::to_string(posting) +
             ",\"verified\":" + (options_.verify == "full" ? "true" : "null") + "}");
    }

    Options options_;
    std::vector<std::string> nics_;
    std::vector<int> nodes_;
    std::vector<std::unique_ptr<Stream>> streams_;  // declared before comms_: buffers outlive MPComm
    std::vector<std::unique_ptr<mpcomm::MPComm>> comms_;
    Controller controller_;
    bool shutdown_ = false;
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
        Options options = Options::parse(argc, argv);
        if (!options.streams.empty()) {
            ReadReceiver receiver(std::move(options));
            receiver.run();
        } else {
            Worker worker(std::move(options));
            worker.run();
        }
        return 0;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        std::cerr << message << '\n';
        std::cout << "{\"event\":\"error\",\"errors\":1,\"message\":"
                  << jsonString(message) << "}\n" << std::flush;
        return 1;
    }
}
