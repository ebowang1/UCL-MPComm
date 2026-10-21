#!/usr/bin/env python3
"""MPComm QoS benchmark: fixed H2D TMA and NIC-to-GPU RDMA."""

import argparse
import csv
import ctypes
import ctypes.util
import os
import statistics
import sys
import time

import torch

# The H2D launcher can silence native C/CUDA stdout while retaining Python
# progress and summary messages on the terminal.
if os.environ.get("MPCOMM_STATUS_STDERR") == "1":
	sys.stdout = sys.stderr

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

MODE_VALUES = {"auto": 0, "sm": 1, "tma": 2}
MODE_NAMES = {0: "AUTO", 1: "SM(int4)", 2: "TMA(cp.async.bulk)"}


def parse_size(value):
	text = str(value).strip().upper()
	units = {"K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2,
			 "G": 1024**3, "GB": 1024**3}
	for suffix in sorted(units, key=len, reverse=True):
		if text.endswith(suffix):
			return int(float(text[:-len(suffix)]) * units[suffix])
	return int(text)


def format_size(value):
	size = float(value)
	for unit in ("B", "KiB", "MiB", "GiB"):
		if size < 1024 or unit == "GiB":
			return f"{int(size)}B" if unit == "B" else f"{size:.2f}{unit}"
		size /= 1024
	return f"{size:.2f}GiB"


def print_pcie_domain_stats(comm):
	"""Print per-PCIe-domain in-flight byte accounting, if instrumented.

	in_flight = posted - completed, i.e. bytes the NIC was authorized to move
	but has not finished moving. A posted WQE cannot be revoked, so this bounds
	how long a competing H2D window must wait for the PCIe uplink:
	    T_residual <= in_flight / BW_pcie
	inflight_max is the value that sizes a future admission budget.

	No-op unless MPCOMM_PCIE_STAT_ENABLE=1.
	"""
	getter = getattr(comm, "get_pcie_domain_stats", None)
	if getter is None:
		return
	try:
		stats = getter()
	except Exception:
		return
	for entry in stats:
		print(f"[PCIE-DOMAIN] gpu={entry['gpu']} "
			  f"inflight={format_size(entry['in_flight_bytes'])} "
			  f"inflight_max={format_size(entry['in_flight_bytes_max'])} "
			  f"posted={format_size(entry['posted_bytes'])} "
			  f"completed={format_size(entry['completed_bytes'])} "
			  f"posts={entry['post_count']} "
			  f"defers={entry['defer_count']}", flush=True)


def allocate_aligned(size_bytes):
	libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
	pointer = ctypes.c_void_p()
	status = libc.posix_memalign(ctypes.byref(pointer), 1024, size_bytes)
	if status != 0:
		raise MemoryError(f"posix_memalign failed: {status}")
	raw = (ctypes.c_char * size_bytes).from_address(pointer.value)
	tensor = torch.frombuffer(raw, dtype=torch.uint8)
	tensor._mpcomm_raw = raw
	tensor._mpcomm_pointer = pointer
	return tensor


def summarize(latencies, payload):
	ordered = sorted(latencies)
	average = statistics.mean(ordered)
	return {
		"average_ms": average * 1000.0,
		"p50_ms": ordered[round((len(ordered) - 1) * 0.50)] * 1000.0,
		"p99_ms": ordered[round((len(ordered) - 1) * 0.99)] * 1000.0,
		"minimum_ms": ordered[0] * 1000.0,
		"maximum_ms": ordered[-1] * 1000.0,
		"average_bandwidth_gbs": payload / average / 1e9,
		"peak_bandwidth_gbs": payload / ordered[0] / 1e9,
	}


def print_results(title, rows):
	print("\n" + "=" * 128)
	print(title)
	print("=" * 128)
	print(f"{'workload':>16} {'op':>8} {'backend':>32} {'avg_ms':>10} {'p50_ms':>10} {'p99_ms':>10} {'avg_GB/s':>11} {'peak_GB/s':>12}")
	print("-" * 128)
	for row in rows:
		print(f"{row['workload']:>16} {row['operation']:>8} {row['backend']:>32} {row['average_ms']:>10.4f} {row['p50_ms']:>10.4f} {row['p99_ms']:>10.4f} {row['average_bandwidth_gbs']:>11.2f} {row['peak_bandwidth_gbs']:>12.2f}")
	print("=" * 128)


def write_csv(path, rows):
	if not path:
		return
	fields = ["path", "operation", "backend", "block_size", "num_blocks",
			  "bytes_per_iteration", "iterations", "average_ms", "p50_ms", "p99_ms",
			  "minimum_ms", "maximum_ms", "average_bandwidth_gbs", "peak_bandwidth_gbs"]
	new_file = not os.path.exists(path)
	with open(path, "a", newline="", encoding="utf-8") as output:
		writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
		if new_file:
			writer.writeheader()
		writer.writerows(rows)
	print(f"[csv] wrote {path}")


def choose_h2d_mode(args):
	requested = MODE_VALUES[args.h2d_mode]
	if requested == 1:
		return requested
	try:
		shared_limit = torch.cuda.get_device_properties(args.gpu).shared_memory_per_block_optin
	except AttributeError:
		shared_limit = 227 * 1024
	max_tma_block = ((shared_limit // 128) - 8) // 16 * 16
	if args.block_size > max_tma_block:
		print(f"[h2d] {args.block_size}B exceeds TMA limit {max_tma_block}B; using SM(int4).")
		return 1
	return 2 if requested == 0 else requested


def run_h2d(comm, args):
	if args.num_blocks <= 0 or args.block_size <= 0:
		raise ValueError("num-blocks and block-size must be positive")
	if args.block_size % 16 != 0:
		raise ValueError("block-size must be a multiple of 16")
	mode = choose_h2d_mode(args)
	payload = args.num_blocks * args.block_size
	pool_blocks = max(args.num_blocks * args.pool_multiplier, args.num_blocks)
	workload = f"{args.num_blocks}x{args.block_size}B"
	print(f"[h2d] fixed sparse workload={workload}, payload={format_size(payload)}, pool_blocks={pool_blocks}, backend={MODE_NAMES[mode]}")
	operations = ["gather", "scatter"] if args.op == "both" else [args.op]
	operations = ["gather" if item in ("gather", "get") else "scatter" for item in operations]
	device = torch.device(f"cuda:{args.gpu}")
	rows = []
	for operation in operations:
		cpu_pool = allocate_aligned(pool_blocks * args.block_size)
		mapped = False
		try:
			if operation == "gather":
				cpu_pool.random_(0, 255)
			else:
				cpu_pool.zero_()
			dram_address = comm.map_dram_to_gpu(cpu_pool.data_ptr(), cpu_pool.numel())
			if dram_address == 0:
				raise RuntimeError("map_dram_to_gpu failed")
			mapped = True
			indices = torch.randperm(pool_blocks, device=device, dtype=torch.int64)[:args.num_blocks]
			if operation == "gather":
				gpu_buffer = torch.empty(payload, device=device, dtype=torch.uint8)
			else:
				gpu_buffer = torch.randint(0, 255, (payload,), device=device, dtype=torch.uint8)
			def submit():
				if operation == "gather":
					return comm.tma_gather(dram_address, indices.data_ptr(), gpu_buffer.data_ptr(), args.num_blocks, args.block_size, args.max_sm_count, mode)
				return comm.tma_scatter(gpu_buffer.data_ptr(), indices.data_ptr(), dram_address, args.num_blocks, args.block_size, args.max_sm_count, mode)
			for _ in range(args.warmup):
				if submit() != 0:
					raise RuntimeError(f"{operation} submission failed")
			torch.cuda.synchronize()
			latencies = []
			window_latencies = []
			start_event = torch.cuda.Event(enable_timing=True)
			end_event = torch.cuda.Event(enable_timing=True)
			for iteration in range(args.iterations):
				start_event.record()
				if submit() != 0:
					raise RuntimeError(f"{operation} submission failed")
				end_event.record()
				torch.cuda.synchronize()
				latency = start_event.elapsed_time(end_event) / 1000.0
				latencies.append(latency)
				window_latencies.append(latency)
				if args.report_every > 0 and (iteration + 1) % args.report_every == 0:
					window = summarize(window_latencies, payload)
					print(f"[H2D-LIVE] iter={iteration + 1}/{args.iterations} "
						  f"window={len(window_latencies)} avg_ms={window['average_ms']:.4f} "
						  f"avg_GB/s={window['average_bandwidth_gbs']:.2f} "
						  f"p99_ms={window['p99_ms']:.4f}", flush=True)
					window_latencies.clear()
			if args.report_every > 0 and window_latencies:
				window = summarize(window_latencies, payload)
				print(f"[H2D-LIVE] iter={args.iterations}/{args.iterations} "
					  f"window={len(window_latencies)} avg_ms={window['average_ms']:.4f} "
					  f"avg_GB/s={window['average_bandwidth_gbs']:.2f} "
					  f"p99_ms={window['p99_ms']:.4f}", flush=True)
			row = {"path": "h2d", "operation": operation, "backend": MODE_NAMES[mode], "block_size": args.block_size, "num_blocks": args.num_blocks, "bytes_per_iteration": payload, "iterations": args.iterations, "workload": workload}
			row.update(summarize(latencies, payload))
			rows.append(row)
		finally:
			if mapped:
				comm.unmap_dram_from_gpu(cpu_pool.data_ptr())
	if args.cuda_copy_baseline:
		rows.extend(run_cuda_copy_baseline(args))
	print_results("Fixed-shape H2D", rows)
	return rows


def run_cuda_copy_baseline(args):
	"""Measure contiguous pinned-memory cudaMemcpyAsync equivalents via PyTorch copy_.

	The source/destination buffer is contiguous and pre-packed outside the timed
	region. This isolates the CUDA copy-engine DMA cost and is intentionally not
	a full equivalent of a sparse gather/scatter operation.
	"""
	payload = args.num_blocks * args.block_size
	workload = f"{args.num_blocks}x{args.block_size}B"
	operations = ["gather", "scatter"] if args.op == "both" else [args.op]
	operations = ["gather" if item in ("gather", "get") else "scatter"
			  for item in operations]
	rows = []

	for operation in operations:
		host_buffer = torch.empty(payload, dtype=torch.uint8, pin_memory=True)
		if operation == "gather":
			host_buffer.random_(0, 255)
			gpu_buffer = torch.empty(payload, device=f"cuda:{args.gpu}", dtype=torch.uint8)

			def submit_copy():
				gpu_buffer.copy_(host_buffer, non_blocking=True)

			backend = "cudaMemcpyAsync H2D (contiguous pinned)"
		else:
			host_buffer.zero_()
			gpu_buffer = torch.randint(
				0, 255, (payload,), device=f"cuda:{args.gpu}", dtype=torch.uint8
			)

			def submit_copy():
				host_buffer.copy_(gpu_buffer, non_blocking=True)

			backend = "cudaMemcpyAsync D2H (contiguous pinned)"

		for _ in range(args.warmup):
			submit_copy()
		torch.cuda.synchronize()

		latencies = []
		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
		for _ in range(args.iterations):
			start_event.record()
			submit_copy()
			end_event.record()
			torch.cuda.synchronize()
			latencies.append(start_event.elapsed_time(end_event) / 1000.0)

		row = {
			"path": "cuda_copy_baseline",
			"operation": operation,
			"backend": backend,
			"block_size": args.block_size,
			"num_blocks": args.num_blocks,
			"bytes_per_iteration": payload,
			"iterations": args.iterations,
			"workload": workload,
		}
		row.update(summarize(latencies, payload))
		rows.append(row)

	return rows


def parse_target(value):
	parts = value.split(":")
	if len(parts) != 3:
		raise ValueError("target must be host_id:tcp_address:tcp_port")
	return parts[0], parts[1], int(parts[2])


def run_nic2gpu(comm, args):
	host_id, address, port = parse_target(args.target)
	if comm.connect(host_id, address, port) != 0:
		raise RuntimeError("RDMA connection failed")
	info = comm.query_remote_buffers(host_id, address, port)
	remote = info["buffers"][0] if info and info.get("buffers") else comm.query_remote_buffer(host_id, address, port)
	if not remote or comm.update_remote_memory_info(host_id, remote["rkeys"]) != 0:
		raise RuntimeError("remote buffer or rkey setup failed")
	sizes = [parse_size(value) for value in args.sizes.split(",") if value.strip()]
	# Receive buffer. "gpu" (default) is GPUDirect RDMA into local HBM; "cpu"
	# registers pinned host DRAM instead, so a READ initiated here lands in local
	# CPU memory (remote GPU -> local CPU). The source is still whatever the target
	# publishes, so pair --dest cpu with a --gpu target for "remote GPU -> local CPU".
	if args.dest == "cpu":
		local_buffer = torch.empty(max(sizes), dtype=torch.uint8, pin_memory=True)
	else:
		local_buffer = torch.empty(max(sizes), device=f"cuda:{args.gpu}", dtype=torch.uint8)
	local_address = local_buffer.data_ptr()
	if comm.register_memory(local_address, local_buffer.numel()) != 0:
		hint = "; check nvidia-peermem" if args.dest == "gpu" else ""
		raise RuntimeError(f"{args.dest} memory registration failed{hint}")
	rows = []
	try:
		operations = ["get", "put"] if args.op == "both" else [args.op]
		operations = ["get" if item in ("get", "gather") else "put" for item in operations]
		for operation in operations:
			for size in sizes:
				def submit():
					handle = comm.get_async(local_address, host_id, remote["addr"], size) if operation == "get" else comm.put_async(local_address, host_id, remote["addr"], size)
					if handle == mpcomm.INVALID_TRANSFER_HANDLE:
						raise RuntimeError(f"{operation} returned invalid handle")
					status = comm.wait_transfer(handle, -1)
					result = comm.get_transfer_result(handle)
					comm.release_transfer(handle)
					if status != 0 or result["error_code"] != 0:
						raise RuntimeError(f"{operation} transfer failed")
				for _ in range(args.warmup):
					submit()
				latencies = []
				window_latencies = []
				for iteration in range(args.iterations):
					started = time.perf_counter()
					submit()
					latency = time.perf_counter() - started
					latencies.append(latency)
					window_latencies.append(latency)
					if args.report_every > 0 and (iteration + 1) % args.report_every == 0:
						window = summarize(window_latencies, size)
						print(f"[NIC-LIVE] op={operation} size={format_size(size)} "
							  f"iter={iteration + 1}/{args.iterations} "
							  f"window={len(window_latencies)} avg_ms={window['average_ms']:.4f} "
							  f"avg_GB/s={window['average_bandwidth_gbs']:.2f} "
							  f"p99_ms={window['p99_ms']:.4f}", flush=True)
						print_pcie_domain_stats(comm)
						window_latencies.clear()
				if args.report_every > 0 and window_latencies:
					window = summarize(window_latencies, size)
					print(f"[NIC-LIVE] op={operation} size={format_size(size)} "
						  f"iter={args.iterations}/{args.iterations} "
						  f"window={len(window_latencies)} avg_ms={window['average_ms']:.4f} "
						  f"avg_GB/s={window['average_bandwidth_gbs']:.2f} "
						  f"p99_ms={window['p99_ms']:.4f}", flush=True)
					print_pcie_domain_stats(comm)
				backend = "GPUDirect-RDMA" if args.dest == "gpu" else "RDMA-to-host(pinned)"
				row = {"path": "nic2gpu", "operation": operation, "backend": backend, "block_size": size, "num_blocks": 1, "bytes_per_iteration": size, "iterations": args.iterations, "workload": format_size(size)}
				row.update(summarize(latencies, size))
				rows.append(row)
	finally:
		comm.unregister_memory(local_address)
	title = ("NIC-to-GPU GPUDirect RDMA" if args.dest == "gpu"
			 else "NIC-to-CPU RDMA (remote GPU -> local host)")
	print_results(title, rows)
	return rows


def main():
	parser = argparse.ArgumentParser(description="MPComm QoS benchmark")
	parser.add_argument("--path", required=True, choices=("h2d", "nic2gpu"))
	parser.add_argument("--gpu", type=int, default=0)
	parser.add_argument("--op", default="both")
	parser.add_argument("--iterations", "--iters", dest="iterations", type=int, default=50)
	parser.add_argument("--warmup", type=int, default=10)
	parser.add_argument(
		"--single",
		action="store_true",
		help="single-shot cold run: force warmup=0 and iterations=1 to measure exactly ONE call (business-like)",
	)
	parser.add_argument(
		"--report-every",
		type=int,
		default=0,
		help="for H2D, print a live per-window result every N timed iterations (0 disables)",
	)
	parser.add_argument("--csv", default="")
	parser.add_argument("--h2d-mode", choices=("auto", "sm", "tma"), default="tma")
	parser.add_argument("--num-blocks", type=int, default=2048)
	parser.add_argument("--block-size", type=int, default=272)
	parser.add_argument("--pool-multiplier", type=int, default=4)
	parser.add_argument("--max-sm-count", type=int, default=0)
	parser.add_argument(
		"--cuda-copy-baseline",
		action="store_true",
		help="also measure contiguous pinned-memory cudaMemcpyAsync H2D/D2H baselines",
	)
	parser.add_argument("--sizes", default="512K,1M,4M,16M,64M,256M,1G")
	parser.add_argument("--target", default="")
	parser.add_argument("--host-id", default=os.getenv("MPCOMM_HOST_ID", "initiator:0"))
	parser.add_argument("--device", default=os.getenv("MPCOMM_DEVICE", ""))
	parser.add_argument(
		"--dest",
		choices=("gpu", "cpu"),
		default="gpu",
		help="local receive buffer for --path nic2gpu: gpu (default, GPUDirect "
			 "into HBM) or cpu (pinned host DRAM, i.e. remote -> local CPU)",
	)
	args = parser.parse_args()
	if args.iterations <= 0 or args.warmup < 0 or args.report_every < 0:
		raise ValueError("iterations must be positive; warmup and report-every cannot be negative")
	if args.single:
		args.warmup = 0
		args.iterations = 1
		print("[benchmark] SINGLE-SHOT COLD RUN: warmup=0, iterations=1 (measures one uncached call)")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is unavailable")
	torch.cuda.set_device(args.gpu)
	properties = torch.cuda.get_device_properties(args.gpu)
	print(f"[benchmark] path={args.path}, gpu={properties.name}, cc={properties.major}.{properties.minor}, iterations={args.iterations}, warmup={args.warmup}")
	comm = mpcomm.MPComm()
	status = comm.init(f"h2d-bench:{args.gpu}", "", 0) if args.path == "h2d" else comm.init(args.host_id, args.device, 0)
	if status != 0:
		raise RuntimeError(f"MPComm initialization failed: {status}")
	try:
		if args.path == "h2d":
			rows = run_h2d(comm, args)
		else:
			if not args.target:
				raise ValueError("--target is required for nic2gpu")
			rows = run_nic2gpu(comm, args)
	finally:
		comm.shutdown()
	write_csv(args.csv, rows)


if __name__ == "__main__":
	main()
