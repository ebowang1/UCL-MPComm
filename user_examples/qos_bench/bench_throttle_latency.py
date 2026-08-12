#!/usr/bin/env python3
"""Measure how fast a runtime NIC throttle takes effect.

The benchmark keeps a continuous RDMA GET stream running in a background
thread, then applies ``set_nic_throttle`` from the main thread and reports:

  worker_us- throttle published -> worker observed the new policy
  drain_us      - worker observed    -> all pre-throttle WQEs completed
  effective_us  - throttle published -> pre-throttle WQEs completed
  bandwidth     - measured before and after the throttle

Only the local initiator path is exercised, so no control-plane signalling
latency is included;``effective_us`` is the local component of the feedback
delay.
"""

import argparse
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


class GetStream:
	"""Runs back-to-back RDMA GET transfers until stopped."""

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
				status = self.comm.wait_transfer(handle, -1)
				result = self.comm.get_transfer_result(handle)
				self.comm.release_transfer(handle)
				if status != 0 or result["error_code"] != 0:
					raise RuntimeError("GET transfer failed")
				with self.lock:
					self.completed_bytes += self.size
					self.completed_ops += 1
		except Exception as exc:  # noqa: BLE001 - surfaced to main thread
			self.error = exc
			self.stop_event.set()

	def start(self):
		self.thread.start()

	def stop(self):
		self.stop_event.set()
		self.thread.join(timeout=30)

	def snapshot(self):
		with self.lock:
			return self.completed_bytes, self.completed_ops


def measure_bandwidth(stream, seconds):
	"""Average bandwidth (GB/s) observed over a wall-clock window."""
	start_bytes, start_ops = stream.snapshot()
	started = time.perf_counter()
	time.sleep(seconds)
	end_bytes, end_ops = stream.snapshot()
	elapsed = time.perf_counter() - started
	if stream.error:
		raise stream.error
	delta = end_bytes - start_bytes
	return (delta / elapsed / 1e9, end_ops - start_ops)


def main():
	parser = argparse.ArgumentParser(description=__doc__,
									 formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--target", required=True,
						help="<host_id>:<ip>:<port> of the RDMA target")
	parser.add_argument("--gpu", type=int, default=0)
	parser.add_argument("--device", default="mlx5_bond_1")
	parser.add_argument("--size", default="1G",
						help="per-transfer size for the background GET stream")
	parser.add_argument("--throttle-outstanding", type=int, default=1,
						help="max outstanding per QP while throttled")
	parser.add_argument("--throttle-chunk", default="0",
						help="chunk size while throttled (0 = keep baseline)")
	parser.add_argument("--warmup-seconds", type=float, default=3.0)
	parser.add_argument("--measure-seconds", type=float, default=3.0)
	parser.add_argument("--repeat", type=int, default=5,
						help="number of throttle/release cycles")
	parser.add_argument("--poll-timeout", type=float, default=5.0,
						help="seconds to wait for the drain watermark")
	parser.add_argument("--min-snapshot-wqes", type=int, default=0,
						help="optional filter: retain only samples with at least this many t3 outstanding WQEs (default: 0, retain all real samples)")


	args = parser.parse_args()

	host_id, address, port = parse_target(args.target)
	size = parse_size(args.size)
	throttle_chunk = parse_size(args.throttle_chunk)

	torch.cuda.set_device(args.gpu)
	comm = mpcomm.MPComm()
	if comm.init("throttle-bench", args.device, 0) != 0:
		raise SystemExit("MPComm init failed")

	if comm.connect(host_id, address, port) != 0:
		raise SystemExit("RDMA connection failed")
	info = comm.query_remote_buffers(host_id, address, port)
	remote = info["buffers"][0] if info and info.get("buffers") else \
		comm.query_remote_buffer(host_id, address, port)
	if not remote or comm.update_remote_memory_info(host_id, remote["rkeys"]) != 0:
		raise SystemExit("remote buffer or rkey setup failed")

	gpu_buffer = torch.empty(size, device=f"cuda:{args.gpu}", dtype=torch.uint8)
	local_address = gpu_buffer.data_ptr()
	if comm.register_memory(local_address, gpu_buffer.numel()) != 0:
		raise SystemExit("GPU memory registration failed; check nvidia-peermem")

	print(f"[throttle] qps_per_connection={comm.get_qps_per_connection()}, "
		  f"baseline chunk from MPCOMM_MAX_RDMA_TRANSFER_SIZE="
		  f"{os.environ.get('MPCOMM_MAX_RDMA_TRANSFER_SIZE', 'default')}")
	print(f"[throttle] transfer size={size} bytes, throttle_outstanding="
		  f"{args.throttle_outstanding}, throttle_chunk={throttle_chunk}, "
		  f"min_snapshot_wqes={args.min_snapshot_wqes}")



	stream = GetStream(comm, local_address, host_id, remote["addr"], size)
	stream.start()

	worker_samples = []
	drain_samples = []
	effective_samples = []
	snapshot_samples = []


	try:
		time.sleep(args.warmup_seconds)
		baseline_bw, baseline_ops = measure_bandwidth(stream, args.measure_seconds)
		print(f"[throttle] baseline bandwidth={baseline_bw:.2f} GB/s "
			  f"({baseline_ops} transfers)")

		for cycle in range(1, args.repeat + 1):
			comm.set_nic_throttle(args.throttle_outstanding, throttle_chunk)
			deadline = time.perf_counter() + args.poll_timeout
			stats = comm.get_throttle_latency()
			while not stats["drained"] and time.perf_counter() < deadline:
				time.sleep(0.0002)
				stats = comm.get_throttle_latency()
			if not stats["drained"]:
				print(f"[throttle] cycle {cycle}: drain watermark not reached "
					  f"within {args.poll_timeout}s")
			else:
				snapshot_wqes = stats["snapshot_outstanding_wqes"]
				is_valid = snapshot_wqes >= args.min_snapshot_wqes
				print(f"[throttle] cycle {cycle}: snapshot_wqes={snapshot_wqes} "
					  f"worker_us={stats['worker_us']:.1f} "
					  f"drain_us={stats['drain_us']:.1f} "
					  f"effective_us={stats['effective_us']:.1f} "
					  f"sample={'accepted' if is_valid else 'rejected'}")
				if is_valid:
					snapshot_samples.append(snapshot_wqes)
					worker_samples.append(stats["worker_us"])
					drain_samples.append(stats["drain_us"])
					effective_samples.append(stats["effective_us"])
				else:
					print(f"[throttle] cycle {cycle}: filtered because snapshot_wqes "
						  f"<{args.min_snapshot_wqes}; this filter is intended only "
						  "for a controlled high-in-flight stress experiment")



			throttled_bw, throttled_ops = measure_bandwidth(stream, args.measure_seconds)
			comm.clear_nic_throttle()
			restored_bw, _ = measure_bandwidth(stream, args.measure_seconds)
			print(f"[throttle] cycle {cycle}: throttled={throttled_bw:.2f} GB/s "
				  f"({throttled_ops} transfers), restored={restored_bw:.2f} GB/s")
	finally:
		comm.clear_nic_throttle()
		stream.stop()
		comm.unregister_memory(local_address)
		comm.shutdown()

	if effective_samples:
		print("\n" + "=" * 72)
		print("Runtime throttle latency summary (local GET path)")
		print("=" * 72)
		print(f"{'metric':>16} {'avg':>10} {'min':>10} {'max':>10}")
		print(f"{'snapshot_wqes':>16} {statistics.mean(snapshot_samples):>10.1f} "
			  f"{min(snapshot_samples):>10d} {max(snapshot_samples):>10d}")
		for name, samples in (("worker_us", worker_samples),
							  ("drain_us", drain_samples),
							  ("effective_us", effective_samples)):
			print(f"{name:>16} {statistics.mean(samples):>10.1f} "
				  f"{min(samples):>10.1f} {max(samples):>10.1f}")

		print("=" * 72)


if __name__ == "__main__":
	main()
