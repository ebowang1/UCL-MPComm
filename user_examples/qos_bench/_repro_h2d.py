#!/usr/bin/env python3
"""Minimal repro: is a single tma_gather really100x slower under a background GET?

Isolates the question with NO NIC sampler, NO gap, NO QoS. Prints per-trial
CUDA-event latency for:
  phase A: H2D alone (no background GET)
  phase B: H2D while a background RDMA GET stream runs
If phase B is ~100x A, the H2D kernel itself is being serialized by the NIC
contention (not a timing artifact). We also print the MPComm native TMA backend
line (via MPCOMM_LOG_LEVEL) to see if the kernel path/backend changed.
"""
import os, sys, time, threading
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
for c in (os.path.join(ROOT, "mpcomm-install", "lib", "python"), os.path.join(ROOT, "build")):
    if os.path.isdir(os.path.join(c, "mpcomm")) and c not in sys.path:
        sys.path.insert(0, c)
import mpcomm

HOST, IP, PORT = sys.argv[1].split(":")
PORT = int(PORT)
GPU = int(os.environ.get("GPU", "0"))
DEV = os.environ.get("DEV", "mlx5_bond_1")
BLOCK = int(os.environ.get("BLK", "1024"))
NBLK = int(os.environ.get("NBLK", "10240"))
POOLX = int(os.environ.get("POOLX", "4"))
GET_SIZE = int(os.environ.get("GET_SIZE", str(1 << 30)))
N = int(os.environ.get("N", "20"))
MODE = int(os.environ.get("MODE", "2"))

torch.cuda.set_device(GPU)
comm = mpcomm.MPComm()
assert comm.init("repro", DEV, 0) == 0
assert comm.connect(HOST, IP, PORT) == 0
info = comm.query_remote_buffers(HOST, IP, PORT)
remote = info["buffers"][0] if info and info.get("buffers") else comm.query_remote_buffer(HOST, IP, PORT)
assert remote and comm.update_remote_memory_info(HOST, remote["rkeys"]) == 0

gpu_buf = torch.empty(GET_SIZE, device=f"cuda:{GPU}", dtype=torch.uint8)
assert comm.register_memory(gpu_buf.data_ptr(), gpu_buf.numel()) == 0

pool_blocks = NBLK * POOLX
import ctypes, ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
p = ctypes.c_void_p()
libc.posix_memalign(ctypes.byref(p), 1024, pool_blocks * BLOCK)
raw = (ctypes.c_char * (pool_blocks * BLOCK)).from_address(p.value)
cpu_pool = torch.frombuffer(raw, dtype=torch.uint8)
cpu_pool.random_(0, 255)
dram = comm.map_dram_to_gpu(cpu_pool.data_ptr(), cpu_pool.numel())
assert dram != 0
idx = torch.randperm(pool_blocks, device=f"cuda:{GPU}", dtype=torch.int64)[:NBLK]
dst = torch.empty(NBLK * BLOCK, device=f"cuda:{GPU}", dtype=torch.uint8)

def one():
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    rc = comm.tma_gather(dram, idx.data_ptr(), dst.data_ptr(), NBLK, BLOCK, 0, MODE)
    assert rc == 0
    e.record(); e.synchronize()
    return s.elapsed_time(e) * 1000.0

print(f"[repro] H2D {NBLK}x{BLOCK}B = {NBLK*BLOCK/1048576:.2f} MiB, GET_SIZE={GET_SIZE}")
for _ in range(5):
    one()
a = [one() for _ in range(N)]
print(f"[A] H2D alone      : avg={sum(a)/len(a):8.2f}us min={min(a):8.2f} max={max(a):8.2f}")

stop = threading.Event()
err = []
def bg():
    try:
        while not stop.is_set():
            h = comm.get_async(gpu_buf.data_ptr(), HOST, remote["addr"], GET_SIZE)
            comm.wait_transfer(h, -1)
            comm.get_transfer_result(h)
            comm.release_transfer(h)
    except Exception as ex:
        err.append(ex); stop.set()
t = threading.Thread(target=bg, daemon=True); t.start()
time.sleep(3)
b = [one() for _ in range(N)]
stop.set(); t.join(timeout=10)
if err: print("[bg-error]", err[0])
print(f"[B] H2D + bg GET   : avg={sum(b)/len(b):8.2f}us min={min(b):8.2f} max={max(b):8.2f}")
print(f"[ratio] B/A = {(sum(b)/len(b))/(sum(a)/len(a)):.1f}x")

comm.disable_h2d_qos() if hasattr(comm, "disable_h2d_qos") else None
comm.unmap_dram_from_gpu(cpu_pool.data_ptr())
comm.unregister_memory(gpu_buf.data_ptr())
comm.shutdown()
