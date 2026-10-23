#!/usr/bin/env python3
"""Two-host GPU-HBM -> CPU-DRAM sustained RDMA benchmark.

Build on BOTH Linux hosts, from user_examples/mp_bench:
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
A without CUDA can configure with -DMP_BENCH_WITH_CUDA=OFF; B requires CUDA.

Start B (GPU source) first:
  python3 mp_bench.py --role source --bind 0.0.0.0 --port 18515 --gpus 0,1,2,3
Start A (CPU receiver):
  python3 mp_bench.py --role receiver --peer <B-IP> --port 18515 \
      --nics mlx5_bond_1 --op read --request-size 16M --inflight 1 \
      --duration 60 --warmup 5 --interval 1 --numa auto
Use --op write on A for B-initiated writes into A CPU memory (same data direction).
B --gpu-count N selects the first N visible CUDA devices; --gpus lists exact IDs.
B --nics optionally restricts automatic PCIe-nearest NIC selection.
A --data-ip can advertise a different address for B to reach A in WRITE mode.
A --json PATH saves samples and final results. READ uses synchronized rounds:
one request per GPU, all requests complete before the next round (inflight=1).
Round latency = latest native completion observation minus earliest native
submission start, in microseconds on A's CLOCK_MONOTONIC; it is not the maximum
of independently started request durations. Output latency is the mean across
completed rounds in each sample window, with latency_max also reported. Warmup
and a final round completing after the deadline are excluded. Python IPC/barrier
coordination affects throughput; timing excludes result delivery after native
completion but includes staggered submissions. These are software-observed
completion times, not hardware CQE timestamps. WRITE retains continuous refill
and does not report synchronized READ latency. Verification is outside measurement. No SSH, installation, or system network
changes are performed. With Mellanox NICs, optionally build with -DUSE_MLNX=ON.
Use Python >=3.8, CMake >=3.18 and a C++17 compiler. GPU B also needs CUDA Driver
headers/libraries and working GPUDirect memory registration (nvidia-peermem).
CPU NUMA binding uses mbind; denied syscalls/memory placement fail explicitly.

Ports: B control --port plus B --data-port+i (READ); A --data-port+i (WRITE).
Default data base is 18516 on B and 19516 on A. Open these TCP ports for metadata,
and ensure RDMA reachability between every selected pair. --timeout-ms bounds
per-request drain; setup uses the library's single-NIC handshake timeout.
The control service is unauthenticated: bind it only on a trusted test network.
The native worker uses the library built from this checkout, not an old wheel.
"""

import argparse
import json
import math
import os
from pathlib import Path
import queue
import re
import select
import socket
import subprocess
import sys
import threading
import time

MAX_FRAME = 1024 * 1024
MAX_STREAMS = 16


def size_arg(text):
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT]?(?:i?B)?)?", str(text), re.I)
    if not match:
        raise argparse.ArgumentTypeError("size must be positive bytes or e.g. 64K, 16MiB")
    suffix = (match.group(2) or "").upper().replace("IB", "").replace("B", "")
    value = int(match.group(1)) * 1024 ** ("KMGT".index(suffix) + 1 if suffix else 0)
    if value > (1 << 40):
        raise argparse.ArgumentTypeError("size must not exceed 1 TiB")
    return value


def csv_arg(text):
    values = [part.strip() for part in text.split(",")]
    if not all(values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("list must be nonempty and contain no duplicates")
    return values


def gpu_arg(text):
    values = csv_arg(text)
    if any(not v.isdigit() or int(v) >= 1024 for v in values):
        raise argparse.ArgumentTypeError("GPU IDs must be nonnegative CUDA ordinals")
    ids = [int(v) for v in values]
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("duplicate GPU IDs")
    return ids


def normalize_pci(text):
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", text)
    if not match:
        raise ValueError(f"invalid PCI address: {text}")
    return f"{int(match[1], 16):04x}:{match[2].lower()}:{match[3].lower()}.{match[4]}"


def nearest_nic(pci, allowed=None, sysfs=Path("/sys")):
    gpu_path = (sysfs / "bus/pci/devices" / normalize_pci(pci)).resolve(strict=True)
    nic_root = sysfs / "class/infiniband"
    names = list(allowed) if allowed else sorted(p.name for p in nic_root.iterdir())
    if not names:
        raise ValueError("no RDMA NICs found")
    ranked = []
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid NIC name: {name}")
        device = (nic_root / name / "device").resolve(strict=True)
        # Common PCIe ancestors, not name suffixes or NUMA IDs, define affinity.
        common = 0
        for left, right in zip(gpu_path.parts, device.parts):
            if left != right:
                break
            common += 1
        distance = len(gpu_path.parts) + len(device.parts) - 2 * common
        ranked.append((-common, distance, name))
    ranked.sort()
    return ranked[0][2]


def probe_gpus(worker):
    result = subprocess.run([str(worker), "--probe-gpus"], capture_output=True,
                            text=True, timeout=30, check=False)
    for line in reversed(result.stdout.splitlines()):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if data.get("event") == "gpus" and result.returncode == 0:
            return data["devices"]
    raise RuntimeError(f"CUDA probe failed: {result.stderr.strip()} {result.stdout.strip()}")


def selected_gpus(args):
    devices = probe_gpus(args.worker)
    available = {int(d["id"]): d for d in devices}
    ids = args.gpus if args.gpus is not None else list(available)[:args.gpu_count]
    if not ids or len(ids) > MAX_STREAMS or (args.gpu_count and len(ids) != args.gpu_count):
        raise ValueError("requested GPU count is unavailable or exceeds 16")
    streams = []
    for gpu in ids:
        if gpu not in available:
            raise ValueError(f"CUDA GPU {gpu} is not visible")
        device = available[gpu]
        streams.append({"gpu": gpu, "gpu_nic": nearest_nic(device["pci"], args.nics),
                        "pci": normalize_pci(device["pci"]), "name": device["name"]})
    return streams


class Control:
    def __init__(self, sock):
        self.sock = sock
        self.buffer = bytearray()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(10)

    def send(self, data):
        encoded = (json.dumps(data, allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > MAX_FRAME:
            raise ValueError("control frame too large")
        self.sock.sendall(encoded)

    def receive(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                frame = bytes(self.buffer[:newline])
                del self.buffer[:newline + 1]
                data = json.loads(frame)
                if not isinstance(data, dict) or not isinstance(data.get("event"), str):
                    raise ValueError("invalid control message")
                if data["event"] == "error":
                    raise RuntimeError(f"peer failed: {data.get('message', 'unknown error')}")
                return data
            if len(self.buffer) >= MAX_FRAME:
                raise ValueError("control frame exceeds size limit")
            wait = max(0, deadline - time.monotonic())
            if not select.select([self.sock], [], [], wait)[0]:
                raise TimeoutError("control receive timed out")
            chunk = self.sock.recv(min(65536, MAX_FRAME - len(self.buffer)))
            if not chunk:
                raise ConnectionError("peer disconnected")
            self.buffer.extend(chunk)

    def expect(self, event, timeout):
        data = self.receive(timeout)
        if data["event"] != event:
            raise ValueError(f"expected {event}, got {data['event']}")
        return data


class Worker:
    def __init__(self, args, stream, config, server, gpu_memory, peer_ip=None, peer_port=None):
        self.stream = stream["gpu"]
        self.events = queue.Queue()
        nic = stream["gpu_nic"] if gpu_memory else stream["cpu_nic"]
        command = [str(args.worker), "--role", "server" if server else "client",
                   "--memory", "gpu" if gpu_memory else "cpu", "--op", config["op"],
                   "--nic", nic, "--gpu", str(self.stream if gpu_memory else -1),
                   "--stream", str(self.stream), "--numa", str(config["numa"]),
                   "--size", str(config["size"]), "--depth", str(config["depth"]),
                   "--warmup", str(config["warmup"]), "--duration", str(config["duration"]),
                   "--interval", str(config["interval"]), "--timeout-ms", str(config["timeout_ms"]),
                   "--setup-timeout", str(args.setup_timeout),
                   "--qps", str(config["qps"]), "--chunk", str(config["chunk"]),
                   "--outstanding", str(config["outstanding"]), "--verify", config["verify"],
                   "--read-rounds", "1" if config["op"] == "read" else "0"]
        if server:
            command += ["--listen-port", str(stream["port"])]
        else:
            command += ["--peer-ip", peer_ip, "--peer-port", str(peer_port)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.err_reader = threading.Thread(target=self._read_errors, daemon=True)
        self.reader.start()
        self.err_reader.start()

    def _read(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_FRAME + 1)
                if not line:
                    break
                if len(line) > MAX_FRAME:
                    self.events.put({"event": "error", "message": "worker output too long"})
                    break
                try:
                    event = json.loads(line)
                except ValueError:
                    print(f"[GPU {self.stream}] {line.rstrip()}", file=sys.stderr)
                    continue
                if isinstance(event, dict) and "event" in event:
                    self.events.put(event)
        finally:
            self.events.put({"event": "eof"})

    def _read_errors(self):
        for line in self.process.stderr:
            print(f"[GPU {self.stream}] {line.rstrip()}", file=sys.stderr)

    def command(self, text):
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

    def event(self):
        try:
            data = self.events.get_nowait()
        except queue.Empty:
            return None
        if data["event"] == "error":
            raise RuntimeError(f"GPU {self.stream}: {data.get('message', data)}")
        if data["event"] == "eof":
            raise RuntimeError(f"GPU {self.stream}: worker exited before expected event")
        if "stream" in data and int(data["stream"]) != self.stream:
            raise ValueError("worker stream identity mismatch")
        return data

    def successful_exit(self, timeout=15):
        code = self.process.wait(timeout=timeout)
        if code:
            raise RuntimeError(f"GPU {self.stream}: worker exit code {code}")

    def close(self):
        if self.process.poll() is None:
            try:
                self.command("stop")
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        self.reader.join(timeout=1)
        self.err_reader.join(timeout=1)
        for handle in (self.process.stdin, self.process.stdout, self.process.stderr):
            handle.close()


def wait_events(workers, expected, timeout, check=None, pause=0.002):
    pending = {w.stream: w for w in workers}
    results = {}
    deadline = time.monotonic() + timeout
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"waiting for {expected} from GPUs {list(pending)}")
        if check:
            check()
        for stream, worker in list(pending.items()):
            data = worker.event()
            if data is None:
                continue
            if data["event"] != expected:
                raise ValueError(f"GPU {stream}: expected {expected}, got {data['event']}")
            results[stream] = data
            del pending[stream]
        if pending:
            time.sleep(pause)
    return [results[w.stream] for w in workers]


def check_servers(workers):
    for worker in workers:
        data = worker.event()
        if data is not None:
            raise RuntimeError(f"GPU {worker.stream}: server stopped unexpectedly: {data}")


class Samples:
    def __init__(self, streams):
        self.streams = streams
        self.pending = {}
        self.previous = {s["gpu"]: (0, 0) for s in streams}
        self.last_window = 0

    def add(self, event):
        stream = int(event["stream"])
        window = int(event["window"])
        if stream not in self.previous or window <= self.last_window:
            raise ValueError("unexpected/duplicate sample")
        bucket = self.pending.setdefault(window, {})
        if stream in bucket:
            raise ValueError("duplicate GPU sample")
        bucket[stream] = event
        if len(bucket) != len(self.streams):
            return None
        if window != self.last_window + 1:
            raise ValueError("missing sample window")
        rows, elapsed = [], []
        for spec in self.streams:
            gpu = spec["gpu"]
            data = bucket[gpu]
            total, ns = int(data["bytes"]), int(data["elapsed_ns"])
            old_bytes, old_ns = self.previous[gpu]
            if total < old_bytes or ns <= old_ns or int(data.get("errors", 0)):
                raise ValueError("invalid completion counters")
            bw = (total - old_bytes) / (ns - old_ns)
            rows.append({"gpu": gpu, "GB_s": bw, "completed_bytes": total,
                         "window_bytes": total - old_bytes, "window_ns": ns - old_ns})
            elapsed.append(ns)
            self.previous[gpu] = (total, ns)
        self.last_window = window
        del self.pending[window]
        return {"event": "bandwidth", "window": window, "elapsed_s": max(elapsed) / 1e9,
                "streams": rows, "total_GB_s": sum(r["GB_s"] for r in rows)}


def round_timing(events, streams, round_id, request_size, dispatched_ns):
    expected = {s["gpu"] for s in streams}
    if len(events) != len(expected) or {e.get("stream") for e in events} != expected:
        raise ValueError("READ round does not contain every configured GPU exactly once")
    starts, ends = [], []
    for event in events:
        if event.get("event") != "round_done" or event.get("round") != round_id:
            raise ValueError("incorrect READ round identity")
        if type(event.get("bytes")) is not int or event["bytes"] != request_size or event.get("errors", 0):
            raise ValueError("READ round transfer failed or has incorrect byte count")
        submitted, completed = event.get("submitted_ns"), event.get("completed_ns")
        if type(submitted) is not int or type(completed) is not int or submitted < dispatched_ns or completed <= submitted:
            raise ValueError("invalid native READ round timestamps")
        starts.append(submitted)
        ends.append(completed)
    return {"submitted_ns": min(starts), "completed_ns": max(ends),
            "latency_ns": max(ends) - min(starts)}


class ReadRoundSamples:
    def __init__(self, streams, config, start_ns, emit):
        self.streams, self.config, self.start_ns, self.emit = streams, config, start_ns, emit
        self.duration_ns = int(config["duration"] * 1e9)
        self.interval_ns = int(config["interval"] * 1e9)
        self.next_boundary = min(self.interval_ns, self.duration_ns)
        self.last_boundary = 0
        self.window = 0
        self.rounds = self.latency_sum_ns = self.latency_max_ns = 0
        self.window_rounds = self.window_sum_ns = self.window_max_ns = 0

    def flush(self, elapsed_ns):
        while self.next_boundary <= self.duration_ns and elapsed_ns >= self.next_boundary:
            self.window += 1
            window_ns = self.next_boundary - self.last_boundary
            window_bytes = self.window_rounds * self.config["size"]
            rows = [{"gpu": s["gpu"], "GB_s": window_bytes / window_ns,
                     "completed_bytes": self.rounds * self.config["size"],
                     "window_bytes": window_bytes, "window_ns": window_ns} for s in self.streams]
            self.emit({"event": "bandwidth", "window": self.window,
                       "elapsed_s": self.next_boundary / 1e9, "streams": rows,
                       "total_GB_s": sum(r["GB_s"] for r in rows),
                       "rounds": self.window_rounds,
                       "latency_us": self.window_sum_ns / self.window_rounds / 1000 if self.window_rounds else None,
                       "latency_max_us": self.window_max_ns / 1000 if self.window_rounds else None})
            self.last_boundary = self.next_boundary
            self.window_rounds = self.window_sum_ns = self.window_max_ns = 0
            if self.next_boundary == self.duration_ns:
                self.next_boundary = self.duration_ns + 1
            else:
                self.next_boundary += min(self.interval_ns, self.duration_ns - self.next_boundary)

    def add(self, timing):
        elapsed = timing["completed_ns"] - self.start_ns
        if timing["submitted_ns"] < self.start_ns or elapsed < 0:
            raise ValueError("READ round started before measurement")
        self.flush(min(elapsed, self.duration_ns))
        if elapsed >= self.duration_ns:
            return False
        self.rounds += 1
        self.window_rounds += 1
        latency = timing["latency_ns"]
        self.latency_sum_ns += latency
        self.window_sum_ns += latency
        self.latency_max_ns = max(self.latency_max_ns, latency)
        self.window_max_ns = max(self.window_max_ns, latency)
        return True

    def result(self):
        return {"rounds": self.rounds,
                "latency_us": self.latency_sum_ns / self.rounds / 1000 if self.rounds else None,
                "latency_max_us": self.latency_max_ns / 1000 if self.rounds else None,
                "latency_definition": "max(native completed_ns) - min(native submitted_ns)",
                "mode": "synchronized_read"}


def run_synchronized_read(workers, config, streams, emit, check=None):
    if config["depth"] != 1:
        raise ValueError("synchronized READ requires --inflight 1")
    for worker in workers:
        worker.command("start")
    wait_events(workers, "rounds_ready", config["timeout_ms"] / 1000 + 30, check)
    sequence = 0

    def execute_round():
        nonlocal sequence
        sequence += 1
        dispatched = time.monotonic_ns()
        for worker in workers:
            worker.command(f"round {sequence}")
        events = wait_events(workers, "round_done", config["timeout_ms"] / 1000 + 10, check, pause=0)
        timing = round_timing(events, streams, sequence, config["size"], dispatched)
        if timing["completed_ns"] > time.monotonic_ns():
            raise ValueError("native READ round timestamp is ahead of the controller clock")
        return timing

    warmup_deadline = time.monotonic_ns() + int(config["warmup"] * 1e9)
    execute_round()
    while time.monotonic_ns() < warmup_deadline:
        execute_round()
    for worker in workers:
        worker.command("warmup_done")
    wait_events(workers, "warmup_ready", config["timeout_ms"] / 1000 + 30, check)
    start_ns = time.monotonic_ns() + 200_000_000
    for worker in workers:
        worker.command(f"measure {start_ns}")
    wait_events(workers, "measurement_ready", config["timeout_ms"] / 1000 + 30, check)
    emit({"event": "measurement_start", "message": "warmup complete"})
    while time.monotonic_ns() < start_ns:
        if check:
            check()
        time.sleep(min(0.001, max(0, start_ns - time.monotonic_ns()) / 1e9))
    reporter = ReadRoundSamples(streams, config, start_ns, emit)
    deadline = start_ns + reporter.duration_ns
    drained_rounds = 0
    while time.monotonic_ns() < deadline:
        timing = execute_round()
        if not reporter.add(timing):
            drained_rounds += 1
    reporter.flush(reporter.duration_ns)
    if reporter.rounds == 0:
        raise RuntimeError("no READ round completed on all GPUs before the deadline")
    for worker in workers:
        worker.command("finish")
    results = wait_events(workers, "result", config["timeout_ms"] / 1000 + 30, check)
    for worker in workers:
        worker.successful_exit()
    expected_ops = reporter.rounds + drained_rounds
    for result in results:
        if result.get("read_rounds") is not True or result.get("errors", 0):
            raise RuntimeError("native worker did not complete synchronized READ")
        if result.get("operations") != expected_ops or result.get("bytes") != expected_ops * config["size"]:
            raise RuntimeError("native completion counters disagree with synchronized READ rounds")
        result["bytes"] = reporter.rounds * config["size"]
        result["operations"] = reporter.rounds
        result["drain_bytes"] = drained_rounds * config["size"]
        result["drain_operations"] = drained_rounds
        result["elapsed_ns"] = reporter.duration_ns
    config["round_summary"] = reporter.result()
    return results


def run_clients(workers, config, streams, emit, check=None):
    if config["op"] == "read":
        return run_synchronized_read(workers, config, streams, emit, check)
    for worker in workers:
        worker.command("start")
    wait_events(workers, "warmup_ready", config["warmup"] + config["timeout_ms"] / 1000 + 30, check)
    emit({"event": "measurement_start", "message": "warmup complete"})
    start_ns = time.monotonic_ns() + 200_000_000
    for worker in workers:
        worker.command(f"measure {start_ns}")
    aggregator = Samples(streams)
    results = {}
    deadline = time.monotonic() + config["duration"] + config["timeout_ms"] / 1000 + 30
    while len(results) != len(workers):
        if time.monotonic() >= deadline:
            raise TimeoutError("measurement/drain deadline exceeded")
        if check:
            check()
        for worker in workers:
            if worker.stream in results:
                continue
            data = worker.event()
            if data is None:
                continue
            if data["event"] == "sample":
                row = aggregator.add(data)
                if row:
                    emit(row)
            elif data["event"] == "result":
                if int(data.get("errors", 0)) or int(data.get("bytes", 0)) <= 0:
                    raise RuntimeError(f"GPU {worker.stream}: no successful measured traffic")
                results[worker.stream] = data
            else:
                raise ValueError(f"unexpected client event: {data['event']}")
        time.sleep(0.001)
    if aggregator.pending:
        raise RuntimeError("incomplete cross-GPU sample window")
    for worker in workers:
        worker.successful_exit()
    return [results[w.stream] for w in workers]


def stop_servers(workers, config):
    for worker in workers:
        worker.command("stop")
    results = wait_events(workers, "result", config["timeout_ms"] / 1000 + 30)
    for worker in workers:
        worker.successful_exit()
    if config["verify"] == "full" and any(r.get("verified") is not True for r in results):
        raise RuntimeError("CPU target verification failed")
    return results


def validate_config(config, count):
    if not isinstance(config, dict) or config.get("op") not in ("read", "write"):
        raise ValueError("invalid operation")
    for name, low, high in (("size", 1, 1 << 40), ("depth", 1, 1024),
                           ("qps", 1, 16), ("chunk", 1, 1 << 30),
                           ("outstanding", 1, 16 if config["op"] == "read" else 128),
                           ("timeout_ms", 100, 600000)):
        value = config.get(name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"invalid {name}")
    if config["op"] == "read" and config["depth"] != 1:
        raise ValueError("synchronized READ requires --inflight 1")
    if config["qps"] * config["outstanding"] > 1024:
        raise ValueError("qps * outstanding exceeds the single-NIC CQ capacity (1024)")
    for name, low, high in (("duration", 0.1, 86400), ("warmup", 0, 3600), ("interval", 0.01, 3600)):
        value = config.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"invalid {name}")
    if config["interval"] > config["duration"]:
        raise ValueError("interval must not exceed duration")
    if config.get("verify") not in ("none", "full"):
        raise ValueError("invalid verification mode")
    if config.get("numa") != "auto" and not str(config.get("numa", "")).isdigit():
        raise ValueError("NUMA must be auto or a nonnegative node")
    streams = config.get("streams")
    if not isinstance(streams, list) or not 1 <= count <= MAX_STREAMS or len(streams) != count:
        raise ValueError("invalid stream count")
    ids = set()
    for stream in streams:
        gpu = stream.get("gpu")
        if type(gpu) is not int or gpu < 0 or gpu in ids:
            raise ValueError("invalid/duplicate GPU stream")
        ids.add(gpu)
        for field in ("gpu_nic", "cpu_nic"):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(stream.get(field, ""))):
                raise ValueError(f"invalid {field}")


def make_config(args, streams):
    if not isinstance(streams, list) or not streams:
        raise ValueError("source offered no GPU streams")
    if len(args.nics) > len(streams):
        raise ValueError("A has more selected NICs than GPU streams; some NICs would be unused")
    for i, stream in enumerate(streams):
        stream["cpu_nic"] = args.nics[i % len(args.nics)]
    config = {"op": args.op, "size": args.request_size, "depth": args.inflight,
              "duration": args.duration, "warmup": args.warmup, "interval": args.interval,
              "numa": args.numa, "qps": args.qps, "chunk": args.chunk,
              "outstanding": args.outstanding, "timeout_ms": args.timeout_ms,
              "verify": args.verify, "streams": streams}
    validate_config(config, len(streams))
    return config


def print_mapping(config):
    print(f"\n{config['op'].upper()} | GPU HBM (B) -> CPU DRAM (A) | "
          f"request={config['size']} B, inflight/GPU={config['depth']}, "
          f"warmup={config['warmup']}s, duration={config['duration']}s", flush=True)
    for s in config["streams"]:
        print(f"  GPU{s['gpu']} [{s['pci']}] -> B:{s['gpu_nic']} -> A:{s['cpu_nic']}", flush=True)
    shared = len({s["gpu_nic"] for s in config["streams"]}) < len(config["streams"])
    if shared:
        print("  NOTE: some GPUs share the same nearest NIC; this is not one NIC per GPU.", flush=True)
    print("  Completion-based payload bandwidth (GB/s); TOTAL=sum(per-GPU rates).", flush=True)
    if config["op"] == "read":
        print("  Synchronized READ: one request/GPU/round; latency=min submit to max completion. "
              "Python IPC/barrier overhead affects throughput.\n", flush=True)


def report_sample(data, samples):
    if data["event"] == "measurement_start":
        print("Warmup complete; measurement starting.", flush=True)
        return
    if data["event"] != "bandwidth":
        raise ValueError("unexpected report")
    if samples is not None:
        samples.append(data)
    detail = "  ".join(f"GPU{r['gpu']}={r['GB_s']:.3f}" for r in data["streams"])
    timing = ""
    if "latency_us" in data:
        if data["latency_us"] is None:
            timing = "  latency=-- latency_max=-- rounds=0"
        else:
            timing = (f"  latency={data['latency_us']:.3f}us "
                      f"latency_max={data['latency_max_us']:.3f}us rounds={data['rounds']}")
    print(f"[{data['elapsed_s']:8.3f}s] {detail}  TOTAL={data['total_GB_s']:.3f} GB/s{timing}", flush=True)


def summary(config, samples, results, server_results, output):
    expected = {s["gpu"] for s in config["streams"]}
    if {r["stream"] for r in results} != expected or len(results) != len(expected):
        raise ValueError("not all configured GPU streams completed")
    if {r["stream"] for r in server_results} != expected or len(server_results) != len(expected):
        raise ValueError("not all GPU streams were verified/stopped")
    if config["verify"] == "full":
        verify_results = results if config["op"] == "read" else server_results
        if any(r.get("verified") is not True for r in verify_results):
            raise RuntimeError("data verification did not pass on all CPU streams")
    total = 0.0
    print("\nFinal summary (warmup and post-deadline drain excluded):", flush=True)
    for r in results:
        elapsed = int(r["elapsed_ns"])
        if elapsed <= 0 or int(r["bytes"]) <= 0 or int(r.get("errors", 0)):
            raise ValueError("invalid final measured counters")
        bw = int(r["bytes"]) / elapsed
        total += bw
        print(f"  GPU{r['stream']}: {bw:.3f} GB/s, bytes={r['bytes']}, "
              f"ops={r['operations']}, drain_bytes={r.get('drain_bytes', 0)}", flush=True)
    print(f"  TOTAL: {total:.3f} GB/s; verification={config['verify']} "
          f"({'PASS' if config['verify'] == 'full' else 'DISABLED'})", flush=True)
    round_summary = config.get("round_summary")
    if config["op"] == "read":
        if not round_summary or round_summary["rounds"] <= 0:
            raise RuntimeError("synchronized READ has no completed round latency samples")
        if any(r["operations"] != round_summary["rounds"] for r in results):
            raise RuntimeError("per-GPU operations disagree with completed READ rounds")
        print(f"  READ rounds={round_summary['rounds']}, latency={round_summary['latency_us']:.3f}us "
              f"latency_max={round_summary['latency_max_us']:.3f}us", flush=True)
    if output:
        Path(output).write_text(json.dumps({"config": config, "samples": samples,
                                           "results": results, "server_results": server_results,
                                           "round_summary": round_summary,
                                           "total_GB_s": total}, indent=2, allow_nan=False), encoding="utf-8")


def source_session(args, control, peer_ip, workers):
    streams = selected_gpus(args)
    control.send({"event": "hello", "version": 2, "streams": streams})
    config = control.expect("config", args.setup_timeout)["config"]
    validate_config(config, len(streams))
    for chosen, offered in zip(config["streams"], streams):
        if chosen["gpu"] != offered["gpu"] or chosen["gpu_nic"] != offered["gpu_nic"]:
            raise ValueError("receiver attempted to change GPU affinity selection")
    print_mapping(config)
    timeout = config["duration"] + config["warmup"] + 2 * config["timeout_ms"] / 1000 + args.setup_timeout + 60
    if config["op"] == "read":
        if args.data_port + len(streams) - 1 > 65535:
            raise ValueError("source data port range exceeds 65535")
        for i, stream in enumerate(config["streams"]):
            stream["port"] = args.data_port + i
            workers.append(Worker(args, stream, config, True, True))
        ready = wait_events(workers, "ready", args.setup_timeout)
        for stream, item in zip(config["streams"], ready):
            stream["port"] = int(item["port"])
        control.send({"event": "prepared", "streams": config["streams"]})
        deadline = time.monotonic() + timeout
        while True:
            check_servers(workers)
            if time.monotonic() >= deadline:
                raise TimeoutError("receiver did not finish")
            try:
                data = control.receive(0.1)
            except TimeoutError:
                continue
            if data["event"] != "finish":
                raise ValueError("expected receiver finish")
            break
        server_results = stop_servers(workers, config)
        control.send({"event": "server_results", "results": server_results})
    else:
        prepared = control.expect("prepared", args.setup_timeout)
        data_ip = prepared.get("data_ip") or peer_ip
        peer_streams = prepared["streams"]
        if len(peer_streams) != len(streams):
            raise ValueError("incorrect target stream count")
        for stream, target in zip(config["streams"], peer_streams):
            if stream["gpu"] != target.get("gpu") or not 1 <= int(target["port"]) <= 65535:
                raise ValueError("incorrect target identity/port")
            workers.append(Worker(args, stream, config, False, True, data_ip, int(target["port"])))
        wait_events(workers, "ready", args.setup_timeout)
        control.send({"event": "clients_ready"})
        control.expect("start", args.setup_timeout)
        results = run_clients(workers, config, config["streams"], control.send)
        control.send({"event": "clients_finished", "results": results})
        control.expect("finish", timeout)
        control.send({"event": "finished"})
    print("Source session completed.", flush=True)


def receiver_session(args, control, workers):
    hello = control.expect("hello", args.setup_timeout)
    if hello.get("version") != 2:
        raise ValueError("benchmark control version mismatch; update both hosts")
    config = make_config(args, hello["streams"])
    print_mapping(config)
    control.send({"event": "config", "config": config})
    samples = [] if args.json else None
    if config["op"] == "read":
        prepared = control.expect("prepared", args.setup_timeout)
        if len(prepared["streams"]) != len(config["streams"]):
            raise ValueError("incorrect source stream count")
        for stream, remote in zip(config["streams"], prepared["streams"]):
            if stream["gpu"] != remote.get("gpu") or not 1 <= int(remote["port"]) <= 65535:
                raise ValueError("incorrect source identity/port")
            workers.append(Worker(args, stream, config, False, False, args.peer, int(remote["port"])))
        ready = wait_events(workers, "ready", args.setup_timeout)
        for stream, item in zip(config["streams"], ready):
            if "numa" in item:
                stream["cpu_numa"] = item["numa"]
                print(f"  GPU{stream['gpu']} CPU receive pool: A:{stream['cpu_nic']} "
                      f"NUMA={item['numa']}", flush=True)

        def check_peer():
            try:
                event = control.receive(0)
            except TimeoutError:
                return
            raise RuntimeError(f"unexpected peer event: {event}")

        results = run_clients(workers, config, config["streams"],
                              lambda d: report_sample(d, samples), check_peer)
        control.send({"event": "finish"})
        server_results = control.expect("server_results", config["timeout_ms"] / 1000 + 30)["results"]
    else:
        if args.data_port + len(config["streams"]) - 1 > 65535:
            raise ValueError("receiver data port range exceeds 65535")
        for i, stream in enumerate(config["streams"]):
            stream["port"] = args.data_port + i
            workers.append(Worker(args, stream, config, True, False))
        ready = wait_events(workers, "ready", args.setup_timeout)
        for stream, item in zip(config["streams"], ready):
            stream["port"] = int(item["port"])
            if "numa" in item:
                stream["cpu_numa"] = item["numa"]
                print(f"  GPU{stream['gpu']} CPU receive pool: A:{stream['cpu_nic']} "
                      f"NUMA={item['numa']}", flush=True)
        control.send({"event": "prepared", "streams": config["streams"],
                      "data_ip": args.data_ip or control.sock.getsockname()[0]})
        control.expect("clients_ready", args.setup_timeout)
        control.send({"event": "start"})
        deadline = time.monotonic() + args.warmup + args.duration + 2 * args.timeout_ms / 1000 + 60
        while True:
            check_servers(workers)
            if time.monotonic() >= deadline:
                raise TimeoutError("source did not complete")
            try:
                data = control.receive(0.1)
            except TimeoutError:
                continue
            if data["event"] == "clients_finished":
                results = data["results"]
                break
            report_sample(data, samples)
        server_results = stop_servers(workers, config)
        control.send({"event": "finish"})
        control.expect("finished", args.setup_timeout)
    summary(config, samples, results, server_results, args.json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("source", "receiver"), required=True)
    parser.add_argument("--worker", type=Path, default=Path(__file__).resolve().parent / "build/mp_bench_worker")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--peer", help="B control IP/address (receiver only)")
    parser.add_argument("--port", type=int, default=18515)
    parser.add_argument("--data-port", type=int, default=None)
    parser.add_argument("--data-ip", help="A address advertised for WRITE metadata connections")
    parser.add_argument("--nics", type=csv_arg, help="A required receive NICs; B optional candidate NICs")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--gpus", type=gpu_arg)
    selection.add_argument("--gpu-count", type=int)
    parser.add_argument("--op", choices=("read", "write"), default="read", help="selected on A")
    parser.add_argument("--request-size", type=size_arg, default=16 * 1024 * 1024)
    parser.add_argument("--inflight", type=int, default=None,
                        help="requests per GPU: READ requires 1 (default); WRITE defaults to 4")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--numa", default="auto", help="A CPU buffer NUMA: auto or node ID")
    parser.add_argument("--qps", type=int, default=4)
    parser.add_argument("--chunk", type=size_arg, default=65536)
    parser.add_argument("--outstanding", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--setup-timeout", type=float, default=120)
    parser.add_argument("--verify", choices=("full", "none"), default="full")
    parser.add_argument("--json", help="A optional result JSON path")
    args = parser.parse_args(argv)
    if args.inflight is None:
        args.inflight = 1 if args.op == "read" else 4
    if args.role == "receiver" and args.op == "read" and args.inflight != 1:
        parser.error("synchronized READ requires --inflight 1")
    if not 1 <= args.port <= 65535 or not math.isfinite(args.setup_timeout) or not 1 <= args.setup_timeout <= 3600:
        parser.error("invalid port/setup timeout")
    if args.data_port is None:
        args.data_port = 18516 if args.role == "source" else 19516
    if not 1 <= args.data_port <= 65535:
        parser.error("invalid data port")
    if args.role == "receiver":
        if not args.peer or not args.nics:
            parser.error("receiver requires --peer and --nics")
        if args.gpus is not None or args.gpu_count is not None:
            parser.error("select GPUs on source B, not receiver A")
    else:
        if args.gpus is None and args.gpu_count is None:
            args.gpus = [0]
        if args.gpu_count is not None and not 1 <= args.gpu_count <= MAX_STREAMS:
            parser.error("gpu-count must be 1..16")
        if args.gpus is not None and len(args.gpus) > MAX_STREAMS:
            parser.error("at most 16 GPU streams")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.worker.is_file() or not os.access(args.worker, os.X_OK):
        print(f"Native worker missing/not executable: {args.worker}. Build with cmake first.", file=sys.stderr)
        return 1
    workers, control, listener = [], None, None
    try:
        if args.role == "source":
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.bind, args.port))
            listener.listen(1)
            print(f"GPU source B waiting on {args.bind}:{args.port} (trusted network only)", flush=True)
            sock, address = listener.accept()
            control = Control(sock)
            source_session(args, control, address[0], workers)
        else:
            sock = socket.create_connection((args.peer, args.port), timeout=args.setup_timeout)
            control = Control(sock)
            receiver_session(args, control, workers)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"Benchmark FAILED: {error}", file=sys.stderr, flush=True)
        if control:
            try:
                control.send({"event": "error", "message": str(error)})
            except (OSError, ValueError):
                pass
        return 1
    finally:
        for worker in workers:
            worker.close()
        if control:
            control.sock.close()
        if listener:
            listener.close()


if __name__ == "__main__":
    sys.exit(main())
