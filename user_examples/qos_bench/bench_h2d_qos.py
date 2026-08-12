#!/usr/bin/env python3
"""Validate H2D-triggered NIC QoS with a SINGLE-SHOT H2D model.

The QoS logic lives inside MPComm (C++): once ``enable_h2d_qos`` is called,
each ``tma_gather`` automatically throttles the background RDMA while its H2D
kernel is in flight and restores the baseline once the kernel completes. This
script contains NO QoS logic of its own -- it only flips the switch and calls
the native ``tma_gather``.

Real-business model (NOT a continuous H2D stream):

  background : one continuous RDMA GET stream keeps the NIC->GPU PCIe busy
  foreground : issue ONE tma_gather, then observe -- within that single H2D
               window [T0, T1] -- both

                 H2D bandwidth = H2D_bytes / (T1 - T0)
                 NIC bandwidth = (nic_bytes(T1) - nic_bytes(T0)) / (T1 - T0)

  The NIC byte counter comes from comm.get_nic_completed_bytes(), which counts
  RDMA CQEs at chunk granularity, so it reflects bytes actually delivered to
  GPU HBM inside the exact H2D window (not whole 1GiB transfers).

Each trial is an independent single H2D. We repeat it with QoS disabled and
with QoS enabled, then report the two bandwidths side by side.
"""

import argparse
import ctypes
import ctypes.util
import os
import statistics
import sys
import threading
import time

import torch

ROOT = os.path.normpath(
	os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
for candidate in (
	os.path.join(ROOT, "mpcomm-install", "lib", "python"),
	os.path.join(ROOT, "build"),
):
	if os.path.isdir(os.path.join(candidate, "mpcomm")) and candidate not in sys.path:
		sys.path.insert(0, candidate)

try:
	import mpcomm
except ImportError as error:
	raise SystemExit(f"Cannot import mpcomm: {error}") from error


def parse_size(value):
	text = str(value).strip().upper()
	units = {"K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2,
			"G": 1024**3, "GB": 1024**3}
	for suffix in sorted(units, key=len, reverse=True):
		if text.endswith(suffix):
			return int(float(text[:-len(suffix)]) * units[suffix])
	return int(text)


def parse_target(value):
	parts = str(value).split(":")
	if len(parts) != 3:
		raise ValueError("target must be <host_id>:<ip>:<port>")
	return parts[0], parts[1], int(parts[2])


def allocate_aligned(size_bytes):
	"""Allocate a 1024-byte-aligned CPU DRAM pool, matching bench_h2d_nic2gpu."""
	libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
	pointer = ctypes.c_void_p()
	status = libc.posix_memalign(ctypes.byref(pointer), 1024, size_bytes)
	if status != 0:
		raise MemoryError(f"posix_memalign failed: {status}")
	raw = (ctypes.c_char * size_bytes).from_address(pointer.value)
	tensor = torch.frombuffer(raw, dtype=torch.uint8)
	# Keep the ctypes-owned storage and raw pointer alive for the tensor lifetime.
	tensor._mpcomm_raw = raw
	tensor._mpcomm_pointer = pointer
	return tensor


class GetStream:
	"""Runs back-to-back RDMA GET transfers until stopped (background NIC load)."""

	def __init__(self, comm, local_address, host_id, remote_addr, size):
		self.comm = comm
		self.local_address = local_address
		self.host_id = host_id
		self.remote_addr = remote_addr
		self.size = size
		self.stop_event = threading.Event()
		self.error = None
		self.lock = threading.Lock()
		self.completed_bytes = 0
		self.completed_ops = 0
		self.thread = threading.Thread(target=self._run, daemon=True)

	def _run(self):
		try:
			while not self.stop_event.is_set():
				handle = self.comm.get_async(self.local_address, self.host_id,
											self.remote_addr, self.size)
				if handle == mpcomm.INVALID_TRANSFER_HANDLE:
					raise RuntimeError("get_async returned invalid handle")
				status = self.comm.wait_transfer(handle, 10000)
				result = self.comm.get_transfer_result(handle)
				self.comm.release_transfer(handle)
				if status != 0 or result["error_code"] != 0:
					raise RuntimeError(
						f"GET transfer failed: wait_status={status}, "
						f"error_code={result['error_code']}")
				with self.lock:
					self.completed_bytes += self.size
					self.completed_ops += 1
		except Exception as exc:  # noqa: BLE001 - surfaced to main thread
			self.error = exc
			self.stop_event.set()

	def start(self):
		self.thread.start()

	def snapshot(self):
		with self.lock:
			return self.completed_bytes, self.completed_ops

	def assert_healthy(self, require_progress=False):
		if self.error is not None:
			raise RuntimeError(f"background RDMA GET failed: {self.error}") from self.error
		if not self.thread.is_alive():
			raise RuntimeError("background RDMA GET thread exited unexpectedly")
		if require_progress and self.snapshot()[1] == 0:
			raise RuntimeError(
				"background RDMA GET made no completed transfers; "
				"the concurrent QoS result is invalid")

	def stop(self):
		self.stop_event.set()
		self.thread.join(timeout=30)
		if self.thread.is_alive():
			raise RuntimeError("background RDMA GET did not stop within 30 seconds")


def measure_steady_nic_bw(comm, stream, seconds):
	"""Steady-state NIC CQE bandwidth, rejecting an inactive background stream."""
	start_bytes = comm.get_nic_completed_bytes()
	_, start_ops = stream.snapshot()
	started = time.perf_counter()
	time.sleep(seconds)
	elapsed = time.perf_counter() - started
	end_bytes = comm.get_nic_completed_bytes()
	_, end_ops = stream.snapshot()
	stream.assert_healthy()
	if end_ops == start_ops:
		raise RuntimeError(
			"background RDMA GET completed no transfers during NIC baseline; "
			"refusing to report a false zero-bandwidth concurrent test")
	if end_bytes == start_bytes:
		raise RuntimeError(
			"native NIC CQE byte counter did not advance despite completed GETs")
	return (end_bytes - start_bytes) / elapsed / 1e9


class NicSampler:
	"""Samples the cumulative NIC CQE byte counter at a target us interval.

	Runs in a background thread and busy-waits (time.sleep is far too coarse
	for us-scale intervals). Because Python + the GIL cannot honor a strict
	few-us cadence, the per-interval bandwidth is always computed from the
	MEASURED timestamp delta between adjacent samples, not from the nominal
	interval. Each sample is (perf_counter_seconds, nic_completed_bytes).
	"""

	def __init__(self, comm, interval_us=10):
		self.comm = comm
		self.interval = interval_us * 1e-6
		self.samples = []
		self._stop = threading.Event()
		self._thread = None

	def _run(self):
		# Take a first sample immediately, then spin to the next target.
		self.samples.append((time.perf_counter(),
							 self.comm.get_nic_completed_bytes()))
		while not self._stop.is_set():
			target = time.perf_counter() + self.interval
			while time.perf_counter() < target:
				if self._stop.is_set():
					break
			self.samples.append((time.perf_counter(),
								 self.comm.get_nic_completed_bytes()))

	def start(self):
		self.samples = []
		self._stop.clear()
		self._thread = threading.Thread(target=self._run, daemon=True)
		self._thread.start()

	def stop(self):
		self._stop.set()
		if self._thread is not None:
			self._thread.join(timeout=5)

	def windows(self):
		"""Return per-interval windows as (t_start, t_end, bw_gbps).

		bw is computed from the measured timestamp delta, so uneven spacing
		(inevitable under the GIL) does not bias the bandwidth.
		"""
		out = []
		for (t0, b0), (t1, b1) in zip(self.samples, self.samples[1:]):
			dt = t1 - t0
			if dt > 0:
				out.append((t0, t1, (b1 - b0) / dt / 1e9))
		return out



def run_single_h2d(comm, dram_dev_ptr, indices_ptr, gpu_dst_ptr,
				   num_blocks, block_size, mode, h2d_bytes):
	"""Issue ONE tma_gather and measure H2D plus NIC CQE boundary bandwidth.

	H2D latency uses CUDA events. Background NIC bandwidth uses the native,
	lock-free CQE-byte counter at the two host-side H2D boundaries, so measuring
	the NIC does not perturb the CUDA critical path.
	"""
	start_ev = torch.cuda.Event(enable_timing=True)
	end_ev = torch.cuda.Event(enable_timing=True)

	# Clean starting point: drain any work already queued on the default stream
	# (including anything the previous trial or other CUDA calls left behind) so
	# the start event is not preceded by unrelated queued work. Without this,
	# under a busy background the [start_ev, end_ev] interval can absorb work
	# that does not belong to this single H2D and blow the latency up.
	torch.cuda.synchronize()

	# tma_gather asynchronously publishes GPU-specific worker post pacing. It
	# never waits for a remote ACK or for already-posted WQEs to drain, so H2D
	# enters its CUDA timing window immediately.

	# Do not start a Python busy-wait sampler in this critical window. Such a
	# thread holds the GIL and can delay the host's kernel launch/end-event record
	# by one CPython scheduling quantum (about 5 ms), which masquerades as H2D
	# latency. Read the lock-free native CQE counter at the two boundaries instead.
	t0 = time.perf_counter()
	nic_start_bytes = comm.get_nic_completed_bytes()
	start_ev.record()

	rc = comm.tma_gather(dram_dev_ptr, indices_ptr, gpu_dst_ptr,
						 num_blocks, block_size, 0, mode)
	if rc != 0:
		raise RuntimeError(f"tma_gather failed rc={rc}")

	end_ev.record()
	# Wait only for THIS H2D's end event, not the whole device, so we don't
	# fold unrelated device work into the measured window.
	end_ev.synchronize()
	t_end_host = time.perf_counter()   # host-side H2D completion boundary
	nic_end_bytes = comm.get_nic_completed_bytes()

	h2d_latency_us = start_ev.elapsed_time(end_ev) * 1000.0     # ms -> us
	h2d_bw = h2d_bytes / (h2d_latency_us * 1e-6) / 1e9

	window_seconds = t_end_host - t0
	nic_bw_active = (
		(nic_end_bytes - nic_start_bytes) / window_seconds / 1e9
		if window_seconds > 0 else 0.0)
	active_windows = []


	return h2d_latency_us, h2d_bw, nic_bw_active, active_windows


def run_phase(comm, stream, label, repeat, gap_s, dram_dev_ptr, indices_ptr,
			gpu_dst_ptr, num_blocks, block_size, mode, h2d_bytes):
	"""Run `repeat` independent single-shot H2D trials for one QoS state."""
	lat, h2d_bw, nic_bw = [], [], []
	for i in range(repeat):
		stream.assert_healthy(require_progress=True)
		l, hb, nb, _ = run_single_h2d(
			comm, dram_dev_ptr, indices_ptr, gpu_dst_ptr,
			num_blocks, block_size, mode, h2d_bytes)
		lat.append(l)
		h2d_bw.append(hb)
		nic_bw.append(nb)
		print(f"[{label}] trial {i + 1:>3}: "
			  f"h2d_lat={l:8.2f}us h2d_bw={hb:6.2f}GB/s "
			  f"nic_bw_active={nb:6.2f}GB/s")
		if gap_s > 0:
			time.sleep(gap_s)   # let the NIC return to steady state between trials
	return lat, h2d_bw, nic_bw


def stats(values):
	ordered = sorted(values)
	p99 = ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))]
	return (statistics.mean(ordered), statistics.median(ordered), p99,
			ordered[0], ordered[-1])


def bandwidth_from_average_latency(payload_bytes, latencies_us):
	"""Match bench_h2d_nic2gpu.py: payload / mean(CUDA-event latency)."""
	return payload_bytes / (statistics.mean(latencies_us) * 1e-6) / 1e9


def main():
	parser = argparse.ArgumentParser(description=__doc__,
									 formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--target", required=True,
						help="<host_id>:<ip>:<port> of the RDMA target")
	parser.add_argument("--gpu", type=int, default=0)
	parser.add_argument("--device", default="mlx5_bond_1")
	parser.add_argument("--size", default="1G",
						help="per-transfer size for the background GET stream")
	parser.add_argument("--block-size", type=int, default=1024,
						help="H2D gather block size in bytes (16B aligned)")
	parser.add_argument("--num-blocks", type=int, default=10240,
						help="H2D gather block count")
	parser.add_argument("--pool-multiplier", type=int, default=4,
						help="source DRAM pool size / requested H2D block count (default: 4; matches run-bench-h2d.sh)")
	parser.add_argument("--mode", type=int, default=2,
						help="H2D mode: 0=AUTO, 1=SM, 2=TMA (default)")
	parser.add_argument("--qos-rate-bps", type=int, default=0,
						help="total matching-RDMA post rate while H2D is active; 0 pauses new posts")
	parser.add_argument("--qos-burst-bytes", type=int, default=0,
						help="token-bucket burst; 0 derives a small default from rate")
	parser.add_argument("--warmup-seconds", type=float, default=3.0)
	parser.add_argument("--measure-seconds", type=float, default=3.0)
	parser.add_argument("--repeat", type=int, default=30,
						help="number of independent single-shot H2D trials per phase")
	parser.add_argument("--h2d-only-warmup", type=int, default=100,
						help="warmup iterations for the isolated H2D baseline "
							 "(match run-bench-h2d.sh so the GPU reaches boost clocks)")
	parser.add_argument("--h2d-only-iters", type=int, default=1000,
						help="measured iterations for the isolated H2D baseline "
							 "(back-to-back, no gap, no NIC sampler)")
	parser.add_argument("--gap-seconds", type=float, default=0.05,
						help="idle gap between concurrent trials so the NIC returns to steady state")

	args = parser.parse_args()

	host_id, address, port = parse_target(args.target)
	size = parse_size(args.size)
	if args.qos_rate_bps < 0 or args.qos_burst_bytes < 0:
		raise SystemExit("qos-rate-bps/qos-burst-bytes must be non-negative")
	if args.num_blocks <= 0 or args.block_size <= 0 or args.block_size % 16 != 0:
		raise SystemExit("num-blocks/block-size must be positive and block-size 16B aligned")
	if args.pool_multiplier <= 0:
		raise SystemExit("pool-multiplier must be positive")
	h2d_bytes = args.num_blocks * args.block_size
	pool_blocks = max(args.num_blocks * args.pool_multiplier, args.num_blocks)
	pool_bytes = pool_blocks * args.block_size

	torch.cuda.set_device(args.gpu)
	comm = mpcomm.MPComm()
	if comm.init("h2d-qos-bench", args.device, 0) != 0:
		raise SystemExit("MPComm init failed")
	if comm.connect(host_id, address, port) != 0:
		raise SystemExit("RDMA connection failed")
	info = comm.query_remote_buffers(host_id, address, port)
	remote = info["buffers"][0] if info and info.get("buffers") else \
		comm.query_remote_buffer(host_id, address, port)
	if not remote or comm.update_remote_memory_info(host_id, remote["rkeys"]) != 0:
		raise SystemExit("remote buffer or rkey setup failed")
	if remote.get("length", 0) < size:
		raise SystemExit(
			f"remote buffer is too small for background GET: "
			f"need={size} bytes, published={remote.get('length', 0)} bytes")

	# GPU buffer used as the background GET destination.
	gpu_buffer = torch.empty(size, device=f"cuda:{args.gpu}", dtype=torch.uint8)
	local_address = gpu_buffer.data_ptr()
	if comm.register_memory(local_address, gpu_buffer.numel()) != 0:
		raise SystemExit("GPU memory registration failed; check nvidia-peermem")

	# Match the standalone H2D benchmark: a 1024B-aligned, pre-touched DRAM
	# source pool that is 4x larger than the request by default. This keeps the
	# source alignment, NUMA placement and sparse-index working set comparable.
	cpu_pool = allocate_aligned(pool_bytes)
	cpu_pool.random_(0, 255)
	dram_dev_ptr = comm.map_dram_to_gpu(cpu_pool.data_ptr(), cpu_pool.numel())
	if dram_dev_ptr == 0:
		raise SystemExit("map_dram_to_gpu failed")
	indices = torch.randperm(pool_blocks, device=f"cuda:{args.gpu}",
							 dtype=torch.int64)[:args.num_blocks]
	gpu_dst = torch.empty(h2d_bytes, device=f"cuda:{args.gpu}", dtype=torch.uint8)

	print(f"[h2d-qos] qps_per_connection={comm.get_qps_per_connection()}, "
		  f"baseline chunk from MPCOMM_MAX_RDMA_TRANSFER_SIZE="
		  f"{os.environ.get('MPCOMM_MAX_RDMA_TRANSFER_SIZE', 'default')}")
	print(f"[h2d-qos] single-shot H2D: block_size={args.block_size} "
		  f"num_blocks={args.num_blocks} (={h2d_bytes / (1024 * 1024):.2f} MiB), "
		  f"pool_blocks={pool_blocks} (={pool_bytes / (1024 * 1024):.2f} MiB, align=1024B), "
		  f"mode={args.mode}, qos_rate_bps={args.qos_rate_bps}, "
		  f"qos_burst_bytes={args.qos_burst_bytes}, repeat={args.repeat}")

	stream = GetStream(comm, local_address, host_id, remote["addr"], size)
	stream_started = False


	try:
		# ---- Baseline 1: H2D alone (no RDMA GET, no gap, no NIC sampler) ----
		# Matches run-bench-h2d.sh: many warmup iters so the GPU reaches boost
		# clocks, then back-to-back measured iters. No sleep between iters and
		# no sampler thread, so the small-kernel bandwidth is comparable to the
		# standalone H2D benchmark.
		comm.disable_h2d_qos()
		print(f"[h2d-qos] isolated H2D baseline: warmup={args.h2d_only_warmup} "
			  f"iters={args.h2d_only_iters} (back-to-back, no NIC sampler)")
		for _ in range(args.h2d_only_warmup):
			run_single_h2d(comm, dram_dev_ptr, indices.data_ptr(),
						   gpu_dst.data_ptr(), args.num_blocks,
						   args.block_size, args.mode, h2d_bytes)
		a_lat = []
		for _ in range(args.h2d_only_iters):
			l, _, _, _ = run_single_h2d(
				comm, dram_dev_ptr, indices.data_ptr(), gpu_dst.data_ptr(),
				args.num_blocks, args.block_size, args.mode, h2d_bytes)
			a_lat.append(l)

		# Start the continuous GET only after the isolated H2D baseline is done.
		stream.start()
		stream_started = True
		time.sleep(args.warmup_seconds)
		stream.assert_healthy(require_progress=True)

		# ---- Baseline 2: NIC alone (no H2D during this measurement window) ----
		steady_nic_bw = measure_steady_nic_bw(comm, stream, args.measure_seconds)
		print(f"[h2d-qos] isolated NIC baseline (no H2D) = "
			  f"{steady_nic_bw:.2f} GB/s")

		# ---- Phase 1: one H2D + background NIC, QoS disabled ----
		comm.disable_h2d_qos()
		d_lat, _, d_nic_bw = run_phase(
			comm, stream, "qos_off", args.repeat, args.gap_seconds,
			dram_dev_ptr, indices.data_ptr(), gpu_dst.data_ptr(),
			args.num_blocks, args.block_size, args.mode, h2d_bytes)

		# ---- Phase 2: one H2D + background NIC, post-rate QoS enabled ----
		comm.enable_h2d_qos(args.qos_rate_bps, args.qos_burst_bytes)
		e_lat, _, e_nic_bw = run_phase(
			comm, stream, "qos_on", args.repeat, args.gap_seconds,
			dram_dev_ptr, indices.data_ptr(), gpu_dst.data_ptr(),
			args.num_blocks, args.block_size, args.mode, h2d_bytes)
		comm.disable_h2d_qos()
	finally:
		comm.disable_h2d_qos()
		if stream_started:
			stream.stop()
		comm.unmap_dram_from_gpu(cpu_pool.data_ptr())
		comm.unregister_memory(local_address)
		comm.shutdown()

	def row(name, values, unit):
		a, p50, p99, mn, mx = stats(values)
		return (f"  {name:<26} avg={a:8.2f} p50={p50:8.2f} p99={p99:8.2f} "
				f"min={mn:8.2f} max={mx:8.2f} {unit}")

	print("\n" + "=" * 86)
	print("Single-shot H2D QoS: isolated baselines and concurrent-window comparison")
	print("=" * 86)
	print(f"single H2D size: {h2d_bytes / (1024 * 1024):.2f} MiB\n")

	a_h2d_avg_bw = bandwidth_from_average_latency(h2d_bytes, a_lat)
	d_h2d_avg_bw = bandwidth_from_average_latency(h2d_bytes, d_lat)
	e_h2d_avg_bw = bandwidth_from_average_latency(h2d_bytes, e_lat)

	print("Isolated baselines:")
	print(row("H2D only (no NIC)", a_lat, "us"))
	print(f"  {'H2D only bandwidth':<26} avg={a_h2d_avg_bw:8.2f} GB/s "
		  "(payload / avg CUDA-event latency)")
	print(f"  {'NIC only (no H2D)':<26} avg={steady_nic_bw:8.2f} GB/s")

	print("\nConcurrent single-H2D latency:")
	print(row("QoS disabled", d_lat, "us"))
	print(row("QoS enabled", e_lat, "us"))
	print("\nConcurrent H2D bandwidth (payload / avg CUDA-event latency):")
	print(f"  {'QoS disabled':<26} avg={d_h2d_avg_bw:8.2f} GB/s")
	print(f"  {'QoS enabled':<26} avg={e_h2d_avg_bw:8.2f} GB/s")
	print("\nConcurrent background NIC bandwidth "
		  "(native CQE counter across the H2D boundary):")
	print(row("QoS disabled", d_nic_bw, "GB/s"))
	print(row("QoS enabled", e_nic_bw, "GB/s"))

	print("\nSummary (avg over trials):")
	print(f"  {'scenario':<30} {'H2D_BW(GB/s)':>14} {'NIC_active(GB/s)':>16} "
		  f"{'H2D_lat(us)':>14}")
	print(f"  {'H2D only':<30} {a_h2d_avg_bw:>14.2f} "
		  f"{'-':>16} {statistics.mean(a_lat):>14.2f}")
	print(f"  {'NIC only':<30} {'-':>14} {steady_nic_bw:>16.2f} {'-':>14}")
	print(f"  {'Concurrent, QoS disabled':<30} {d_h2d_avg_bw:>14.2f} "
		  f"{statistics.mean(d_nic_bw):>16.2f} {statistics.mean(d_lat):>14.2f}")
	print(f"  {'Concurrent, QoS enabled':<30} {e_h2d_avg_bw:>14.2f} "
		  f"{statistics.mean(e_nic_bw):>16.2f} {statistics.mean(e_lat):>14.2f}")
	print("=" * 86)


if __name__ == "__main__":
	main()
