"""Hardware-free controller tests; run: python3 -B -m unittest discover -s ."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import queue
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

import mp_bench as bench


class FakeWorker:
    records = []

    def __init__(self, args, stream, config, server, gpu_memory, peer_ip=None, peer_port=None):
        self.stream = stream["gpu"]
        self.label = f"GPU {self.stream}"
        self.config = config
        self.server = server
        self.events = queue.Queue()
        self.closed = False
        self.records.append((args.role, server, gpu_memory, self.stream))
        self.events.put({"event": "ready", "stream": self.stream,
                         "port": stream.get("port", peer_port)})

    def command(self, text):
        if text == "start":
            self.events.put({"event": "warmup_ready", "stream": self.stream})
        elif text.startswith("measure "):
            for window in (1, 2):
                self.events.put({"event": "sample", "stream": self.stream, "window": window,
                                 "bytes": 1000000000 * window, "elapsed_ns": 1000000000 * window,
                                 "operations": 10 * window, "errors": 0})
            self.events.put({"event": "result", "stream": self.stream, "bytes": 2000000000,
                             "elapsed_ns": 2000000000, "operations": 20, "errors": 0,
                             "drain_bytes": 123, "verified": True})
        elif text == "stop":
            self.events.put({"event": "result", "stream": self.stream, "verified": True})
        else:
            raise ValueError(text)

    def event(self):
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

    def successful_exit(self, timeout=15):
        pass

    def close(self):
        self.closed = True


class FakeReadClient:
    """Simulates the single A READ process serving every GPU stream."""
    instances = []

    def __init__(self, args, config, peer_ip, ports):
        self.label = "A READ receiver"
        self.config = config
        self.gpus = [s["gpu"] for s in config["streams"]]
        self.ports = list(ports)
        self.events = queue.Queue()
        self.closed = False
        self.measuring = False
        self.measured_rounds = 0
        self.round_id = 0
        self.instances.append(self)
        self.events.put({"event": "ready", "instances": 1,
                         "streams": [{"stream": g, "numa": 0} for g in self.gpus]})

    def round_streams(self):
        items = []
        for gpu in self.gpus:
            submitted = time.monotonic_ns()
            completed = time.monotonic_ns()
            while completed <= submitted:
                completed = time.monotonic_ns()
            items.append({"stream": gpu, "submitted_ns": submitted, "completed_ns": completed,
                          "bytes": self.config["size"], "errors": 0})
        return items

    def command(self, text):
        mode = self.config.get("mode", "rounds")
        if text == "start" and mode == "sweep":
            self.simulate_sweep()
        elif text == "start" and mode == "switch":
            self.simulate_switch()
        elif text == "start":
            self.events.put({"event": "rounds_ready"})
        elif text.startswith("round "):
            round_id = int(text.split()[1])
            if round_id != self.round_id + 1:
                raise ValueError("nonsequential simulated READ round")
            self.round_id = round_id
            self.events.put({"event": "round_done", "round": round_id, "streams": self.round_streams()})
            if self.measuring:
                self.measured_rounds += 1
        elif text == "warmup_done":
            self.events.put({"event": "warmup_ready"})
        elif text.startswith("measure "):
            self.measuring = True
            self.events.put({"event": "measurement_ready"})
        elif text == "finish":
            ops = self.measured_rounds
            self.events.put({"event": "result", "rounds": ops, "results": [
                {"stream": g, "bytes": ops * self.config["size"], "operations": ops, "errors": 0,
                 "elapsed_ns": int(self.config["duration"] * 1e9), "read_rounds": True,
                 "verified": True} for g in self.gpus]})
        else:
            raise ValueError(text)

    def simulate_sweep(self):
        shares = bench.transfer_shares(self.config["total_bytes"], len(self.gpus))
        self.events.put({"event": "warmup_done", "latency_ns": 1000})
        runs = 0
        for oq in self.config["oq_list"]:
            for repeat in range(1, self.config["repeat"] + 1):
                runs += 1
                # Smaller OQ -> slower; GPUs complete one after another.
                latency = 1_000_000_000 // oq
                streams = [{"stream": g, "bytes": share, "completed_ns": 100 + latency * (i + 1) // len(self.gpus)}
                           for i, (g, share) in enumerate(zip(self.gpus, shares))]
                self.events.put({"event": "sweep_result", "oq": oq, "repeat": repeat, "submitted_ns": 100,
                                 "completed_ns": 100 + latency, "latency_ns": latency, "streams": streams})
        self.events.put({"event": "sweep_done", "runs": runs, "posting_workers": 1, "verified": True})

    def simulate_switch(self):
        origin, bin_ns, bins, events, transfers = synthetic_switch_trace(
            self.config["high_oq"], self.config["low_oq"])
        self.events.put({"event": "warmup_done", "latency_ns": 1000})
        self.events.put({"event": "trace_meta", "origin_ns": origin, "bin_ns": bin_ns, "bins": len(bins),
                         "events": len(events), "transfers": len(transfers),
                         "dropped_events": 0, "dropped_completions": 0})
        self.events.put({"event": "trace_transfers", "offset": 0, "transfers": transfers})
        self.events.put({"event": "trace_events", "offset": 0, "events": events[:2]})
        self.events.put({"event": "trace_events", "offset": 2, "events": events[2:]})
        self.events.put({"event": "trace_bins", "offset": 0, "bytes": bins[:50]})
        self.events.put({"event": "trace_bins", "offset": 50, "bytes": bins[50:]})
        self.events.put({"event": "switch_done", "posting_workers": 1, "verified": True})

    def event(self):
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

    def successful_exit(self, timeout=15):
        pass

    def close(self):
        self.closed = True


def synthetic_switch_trace(high=64, low=1):
    """2 ms phases, 100 us bins: high 50 GB/s, low 5 GB/s (0 when paused).
    Each drop settles 300 us after its set call, each rise 200 us after."""
    bin_ns, high_bin, low_bin = 100_000, 5_000_000, 0 if low == 0 else 500_000
    events = [[1, high, 0, 5_000, 6_000],
              [2, low, 2_000_000, 2_005_000, 2_150_000],
              [3, high, 4_000_000, 4_005_000, 4_020_000],
              [4, low, 6_000_000, 6_005_000, 6_150_000],
              [5, bench.NIC_LIMIT_NONE, 8_000_000, 8_005_000, 8_006_000]]
    bins = [high_bin if k < 23 or 42 <= k < 63 else low_bin for k in range(80)]
    return 0, bin_ns, bins, events, [[-1_000_000, 9_000_000]]


def control_pair():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    left = socket.create_connection(listener.getsockname())
    right, _ = listener.accept()
    listener.close()
    return bench.Control(left), bench.Control(right)


class ControllerTests(unittest.TestCase):
    def test_sizes(self):
        for text, expected in (("16M", 16777216), ("64KiB", 65536), ("42", 42), ("1GB", 1073741824)):
            self.assertEqual(bench.size_arg(text), expected)
        for text in ("0", "-1", "nan", "1.5M", "999999T"):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench.size_arg(text)

    def test_cli(self):
        args = bench.parse_args(["--role", "source", "--gpu-count", "4"])
        self.assertEqual(args.data_port, 18516)
        args = bench.parse_args(["--role", "receiver", "--peer", "127.0.0.1", "--nics", "mlx5_0"])
        self.assertEqual(args.numa, "auto")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(["--role", "receiver", "--peer", "127.0.0.1"])
        with self.assertRaises(argparse.ArgumentTypeError):
            bench.gpu_arg("0,00")

    def test_native_setup_timeout_is_forwarded(self):
        # Per-GPU processes remain only for B READ sources and both WRITE roles.
        for operation, server in (("read", True), ("write", True), ("write", False)):
            for timeout in (120, 300):
                gpu_memory = server if operation == "read" else not server
                role = "source" if gpu_memory else "receiver"
                argv = ["--role", role, "--setup-timeout", str(timeout)]
                if role == "receiver":
                    argv += ["--peer", "127.0.0.1", "--nics", "mlx5_0"]
                args = bench.parse_args(argv)
                stream = {"gpu": 0, "gpu_nic": "mlx5_7", "cpu_nic": "mlx5_0", "port": 18516}
                config = {"op": operation, "numa": "auto", "size": 4096, "depth": 4,
                          "warmup": 5, "duration": 60, "interval": 1, "timeout_ms": 30000,
                          "qps": 4, "chunk": 65536, "outstanding": 8, "verify": "full"}
                with self.subTest(operation=operation, server=server, timeout=timeout), \
                        mock.patch.object(bench.subprocess, "Popen") as popen, \
                        mock.patch.object(bench.threading, "Thread"):
                    popen.return_value.poll.return_value = 0
                    worker = bench.Worker(args, stream, config, server, gpu_memory,
                                          "127.0.0.1", 18516)
                    try:
                        command = popen.call_args.args[0]
                        self.assertEqual(command.count("--setup-timeout"), 1)
                        value = command[command.index("--setup-timeout") + 1]
                        self.assertEqual(float(value), timeout)
                        self.assertNotIn("--read-rounds", command)
                    finally:
                        worker.close()

    def test_read_client_is_one_process_for_all_streams(self):
        args = bench.parse_args(["--role", "receiver", "--peer", "10.0.0.2", "--nics", "mlx5_0",
                                 "--setup-timeout", "300"])
        streams = [{"gpu": g, "gpu_nic": f"mlx5_{g}"} for g in (0, 3, 5)]
        config = bench.make_config(args, streams)
        with mock.patch.object(bench.subprocess, "Popen") as popen, \
                mock.patch.object(bench.threading, "Thread"):
            popen.return_value.poll.return_value = 0
            client = bench.ReadClient(args, config, "10.0.0.2", [18516, 18517, 18518])
            try:
                self.assertEqual(popen.call_count, 1)
                command = popen.call_args.args[0]
                value = lambda flag: command[command.index(flag) + 1]
                self.assertEqual(value("--streams"), "0:mlx5_0:18516,3:mlx5_0:18517,5:mlx5_0:18518")
                self.assertEqual(value("--peer-ip"), "10.0.0.2")
                self.assertEqual((value("--op"), value("--memory"), value("--role")), ("read", "cpu", "client"))
                self.assertEqual((value("--depth"), value("--read-rounds")), ("1", "1"))
                self.assertEqual(float(value("--setup-timeout")), 300)
                for flag in ("--nic", "--peer-port", "--stream"):
                    self.assertNotIn(flag, command)
            finally:
                client.close()
        with self.assertRaises(ValueError):
            bench.ReadClient(args, config, "10.0.0.2", [18516])

    def test_invalid_data_port_is_not_defaulted(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(["--role", "source", "--data-port", "0"])

    def test_pci_normalization(self):
        self.assertEqual(bench.normalize_pci("00000000:AB:00.0"), "0000:ab:00.0")
        with self.assertRaises(ValueError):
            bench.normalize_pci("../../etc/passwd")

    def test_nearest_nic(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            sysfs = Path(root)
            gpu = sysfs / "bus/pci/devices/0000:01:00.0"
            for name in ("mlx5_7", "mlx5_0"):
                (sysfs / "class/infiniband" / name).mkdir(parents=True)
            # Symlink support is not guaranteed on Windows. Stub realpath with topology paths.
            paths = {str(gpu): Path("/sys/devices/pci0000:00/bridge/gpu"),
                     str(sysfs / "class/infiniband/mlx5_7/device"): Path("/sys/devices/pci0000:00/bridge/nic"),
                     str(sysfs / "class/infiniband/mlx5_0/device"): Path("/sys/devices/pci0001:00/nic")}
            def resolve(path, strict=False):
                if str(path) not in paths:
                    raise FileNotFoundError(str(path))
                return paths[str(path)]
            with mock.patch.object(Path, "resolve", resolve):
                self.assertEqual(bench.nearest_nic("0000:01:00.0", sysfs=sysfs), "mlx5_7")
                self.assertEqual(bench.nearest_nic("0000:01:00.0", ["mlx5_0"], sysfs), "mlx5_0")
                with self.assertRaises(FileNotFoundError):
                    bench.nearest_nic("0000:01:00.0", ["missing"], sysfs)

    def test_samples(self):
        aggregator = bench.Samples([{"gpu": 0}, {"gpu": 3}])
        def sample(gpu, window, size, ns):
            return {"event": "sample", "stream": gpu, "window": window, "bytes": size,
                    "elapsed_ns": ns, "errors": 0}
        self.assertIsNone(aggregator.add(sample(0, 1, 1000000000, 1000000000)))
        row = aggregator.add(sample(3, 1, 2000000000, 1000000000))
        self.assertEqual(row["total_GB_s"], 3)
        self.assertNotIn("skew_ns", row)
        self.assertNotIn("max_lag_ns", row)
        aggregator.add(sample(3, 2, 4000000000, 2000000000))
        row = aggregator.add(sample(0, 2, 2000000000, 2000000000))
        self.assertEqual(row["total_GB_s"], 3)
        with self.assertRaises(ValueError):
            aggregator.add(sample(0, 2, 2000000000, 2000000000))

    def test_counter_reversal(self):
        aggregator = bench.Samples([{"gpu": 0}])
        aggregator.add({"stream": 0, "window": 1, "bytes": 10, "elapsed_ns": 100})
        with self.assertRaises(ValueError):
            aggregator.add({"stream": 0, "window": 2, "bytes": 9, "elapsed_ns": 200})

    def test_control_fragmentation_and_disconnect(self):
        left, right = control_pair()
        try:
            left.sock.sendall(b'{"event":')
            with self.assertRaises(TimeoutError):
                right.receive(0.01)
            left.sock.sendall(b'"test"}\n')
            self.assertEqual(right.expect("test", 1)["event"], "test")
            left.sock.close()
            with self.assertRaises(ConnectionError):
                right.receive(1)
        finally:
            left.sock.close()
            right.sock.close()

    def test_control_peer_error(self):
        left, right = control_pair()
        try:
            left.send({"event": "error", "message": "bad GPU"})
            with self.assertRaisesRegex(RuntimeError, "bad GPU"):
                right.receive(1)
        finally:
            left.sock.close()
            right.sock.close()

    def test_invalid_load_config(self):
        args = bench.parse_args(["--role", "receiver", "--peer", "127.0.0.1", "--nics", "mlx5_0"])
        config = bench.make_config(args, [{"gpu": 0, "gpu_nic": "mlx5_1"}])
        for name, value in (("duration", float("nan")), ("outstanding", 17), ("depth", 0), ("qps", 17)):
            invalid = dict(config, **{name: value})
            with self.assertRaises(ValueError):
                bench.validate_config(invalid, 1)

    def test_write_outstanding_cq_limit(self):
        args = bench.parse_args(["--role", "receiver", "--peer", "127.0.0.1",
                                 "--nics", "mlx5_0", "--op", "write"])
        config = bench.make_config(args, [{"gpu": 0, "gpu_nic": "mlx5_1"}])
        with self.assertRaisesRegex(ValueError, "CQ capacity"):
            bench.validate_config(dict(config, qps=16, outstanding=128), 1)

    def test_all_gpu_streams_required(self):
        config = {"streams": [{"gpu": 0}, {"gpu": 1}], "op": "read", "verify": "full"}
        with self.assertRaises(ValueError):
            bench.summary(config, [], [{"stream": 0}], [{"stream": 0}], None)

    def test_reject_unused_receiver_nics(self):
        args = bench.parse_args(["--role", "receiver", "--peer", "127.0.0.1",
                                 "--nics", "mlx5_0,mlx5_1"])
        with self.assertRaisesRegex(ValueError, "unused"):
            bench.make_config(args, [{"gpu": 0, "gpu_nic": "mlx5_7"}])

    def test_verification_failure_is_fatal(self):
        config = {"streams": [{"gpu": 0}], "op": "write", "verify": "full"}
        with self.assertRaisesRegex(RuntimeError, "verification"):
            bench.summary(config, [], [{"stream": 0}],
                          [{"stream": 0, "verified": False}], None)

    def test_synchronized_read_defaults_and_rejects_parallel_rounds(self):
        argv = ["--role", "receiver", "--peer", "127.0.0.1", "--nics", "mlx5_0"]
        self.assertEqual(bench.parse_args(argv).inflight, 1)
        self.assertEqual(bench.parse_args(argv + ["--op", "write"]).inflight, 4)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(argv + ["--inflight", "4"])

    def test_bandwidth_output_has_no_skew_or_lag(self):
        row = {"event": "bandwidth", "elapsed_s": 1, "total_GB_s": 2,
               "streams": [{"gpu": 0, "GB_s": 1}, {"gpu": 1, "GB_s": 1}],
               "latency_us": 100, "latency_max_us": 150, "rounds": 8}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            bench.report_sample(row, None)
        text = output.getvalue()
        self.assertIn("latency=100.000us", text)
        self.assertIn("latency_max=150.000us", text)
        self.assertNotIn("skew", text)
        self.assertNotIn("lag", text)

    def test_completed_counters_only_exclude_drain(self):
        config = {"streams": [{"gpu": 0}], "op": "read", "verify": "none",
                  "round_summary": {"rounds": 100, "latency_us": 100, "latency_max_us": 150}}
        result = {"stream": 0, "bytes": 1000000000, "elapsed_ns": 1000000000,
                  "operations": 100, "drain_bytes": 9000000000}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            output = Path(root) / "result.json"
            with contextlib.redirect_stdout(io.StringIO()):
                bench.summary(config, [], [result], [{"stream": 0}], output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["total_GB_s"], 1)

    def test_read_write_sessions(self):
        for operation in ("read", "write"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
                FakeWorker.records = []
                FakeReadClient.instances = []
                left, right = control_pair()
                source = bench.parse_args(["--role", "source", "--gpus", "0,3", "--setup-timeout", "2"])
                output = str(Path(root) / "result.json")
                receiver = bench.parse_args(["--role", "receiver", "--peer", "127.0.0.1",
                                             "--nics", "mlx5_0", "--op", operation,
                                             "--duration", "2", "--warmup", "0", "--json", output,
                                             "--setup-timeout", "2"])
                offered = [{"gpu": 0, "gpu_nic": "mlx5_1", "pci": "0000:01:00.0", "name": "fake0"},
                           {"gpu": 3, "gpu_nic": "mlx5_7", "pci": "0000:02:00.0", "name": "fake3"}]
                errors, source_workers, receiver_workers = [], [], []
                def run_source():
                    try:
                        bench.source_session(source, right, "127.0.0.1", source_workers)
                    except BaseException as exc:
                        errors.append(exc)
                        right.sock.shutdown(socket.SHUT_RDWR)
                try:
                    with mock.patch.object(bench, "Worker", FakeWorker), \
                            mock.patch.object(bench, "ReadClient", FakeReadClient), \
                            mock.patch.object(bench, "selected_gpus", return_value=offered), \
                            contextlib.redirect_stdout(io.StringIO()):
                        thread = threading.Thread(target=run_source, daemon=True)
                        thread.start()
                        bench.receiver_session(receiver, left, receiver_workers)
                        thread.join(timeout=5)
                        self.assertFalse(thread.is_alive())
                    self.assertFalse(errors, errors)
                    saved = json.loads(Path(output).read_text(encoding="utf-8"))
                    if operation == "read":
                        self.assertGreater(saved["round_summary"]["rounds"], 0)
                        self.assertGreater(saved["round_summary"]["latency_us"], 0)
                        self.assertEqual({r["operations"] for r in saved["results"]},
                                         {saved["round_summary"]["rounds"]})
                    else:
                        self.assertEqual(saved["total_GB_s"], 2)
                        self.assertIsNone(saved["round_summary"])
                    self.assertEqual(len(saved["samples"]), 2)
                    self.assertEqual({r["stream"] for r in saved["results"]}, {0, 3})
                    gpu_roles = [r for r in FakeWorker.records if r[2]]
                    cpu_roles = [r for r in FakeWorker.records if not r[2]]
                    self.assertTrue(all(r[0] == "source" for r in gpu_roles))
                    self.assertEqual(sorted(r[3] for r in gpu_roles), [0, 3])
                    self.assertTrue(all(r[1] == (operation == "read") for r in gpu_roles))
                    if operation == "read":
                        # A: exactly one READ process, with one source port per GPU stream.
                        self.assertEqual(cpu_roles, [])
                        self.assertEqual(len(FakeReadClient.instances), 1)
                        self.assertEqual(FakeReadClient.instances[0].gpus, [0, 3])
                        self.assertEqual(len(set(FakeReadClient.instances[0].ports)), 2)
                    else:
                        self.assertEqual(FakeReadClient.instances, [])
                        self.assertTrue(all(r[0] == "receiver" and r[1] for r in cpu_roles))
                finally:
                    for worker in source_workers + receiver_workers:
                        worker.close()
                    left.sock.close()
                    right.sock.close()


class ReadRoundTests(unittest.TestCase):
    @staticmethod
    def event(gpu, started, completed, round_id=1, size=16):
        return {"event": "round_done", "stream": gpu, "round": round_id,
                "submitted_ns": started, "completed_ns": completed, "bytes": size, "errors": 0}

    def test_round_latency_is_full_span_not_max_request_duration(self):
        events = [self.event(0, 100, 180), self.event(1, 140, 200)]
        timing = bench.round_timing(events, [{"gpu": 0}, {"gpu": 1}], 1, 16, 90)
        self.assertEqual(timing["latency_ns"], 100)
        self.assertNotEqual(timing["latency_ns"], max(180 - 100, 200 - 140))

    def test_one_and_eight_gpu_rounds(self):
        for count in (1, 8):
            streams = [{"gpu": i} for i in range(count)]
            events = [self.event(i, 100 + i * 10, 200 + i * 10) for i in range(count)]
            timing = bench.round_timing(events, streams, 1, 16, 90)
            self.assertEqual(timing["latency_ns"], 100 + (count - 1) * 10)
            self.assertEqual(sum(e["bytes"] for e in events), count * 16)

    def test_missing_duplicate_wrong_round_and_errors(self):
        streams = [{"gpu": 0}, {"gpu": 1}]
        valid = [self.event(0, 100, 200), self.event(1, 110, 210)]
        variants = [valid[:1], [valid[0], valid[0]],
                    [valid[0], dict(valid[1], round=2)],
                    [valid[0], dict(valid[1], bytes=15)],
                    [valid[0], dict(valid[1], errors=1)],
                    [valid[0], dict(valid[1], completed_ns=100)],
                    [valid[0], dict(valid[1], submitted_ns=80)]]
        for events in variants:
            with self.assertRaises(ValueError):
                bench.round_timing(events, streams, 1, 16, 90)

    def test_windows_mean_max_and_partial_last_window(self):
        output = []
        config = {"duration": 2.5, "interval": 1, "size": 16}
        reporter = bench.ReadRoundSamples([{"gpu": 0}, {"gpu": 1}], config, 100, output.append)
        for end, latency in ((500000000, 100000), (900000000, 300000), (2000000000, 200000)):
            self.assertTrue(reporter.add({"submitted_ns": 100 + end - latency,
                                          "completed_ns": 100 + end, "latency_ns": latency}))
        reporter.flush(reporter.duration_ns)
        self.assertEqual([s["elapsed_s"] for s in output], [1, 2, 2.5])
        self.assertEqual(output[0]["latency_us"], 200)
        self.assertEqual(output[0]["latency_max_us"], 300)
        self.assertEqual(output[1]["rounds"], 0)
        self.assertIsNone(output[1]["latency_us"])
        self.assertEqual(output[2]["latency_us"], 200)
        self.assertEqual(output[2]["streams"][0]["window_ns"], 500000000)
        self.assertEqual(reporter.result()["rounds"], 3)
        self.assertEqual(reporter.result()["latency_us"], 200)

    def test_late_round_is_excluded_for_every_gpu(self):
        output = []
        reporter = bench.ReadRoundSamples([{"gpu": i} for i in range(8)],
                                          {"duration": 1, "interval": 1, "size": 16}, 100, output.append)
        self.assertTrue(reporter.add({"submitted_ns": 200, "completed_ns": 400, "latency_ns": 200}))
        self.assertFalse(reporter.add({"submitted_ns": 500, "completed_ns": 1000000100,
                                       "latency_ns": 999999600}))
        reporter.flush(reporter.duration_ns)
        self.assertEqual(reporter.result()["rounds"], 1)
        self.assertEqual(len(output), 1)
        self.assertTrue(all(s["completed_bytes"] == 16 for s in output[0]["streams"]))

    def test_no_completed_round_outputs_no_fake_zero_latency(self):
        output = []
        reporter = bench.ReadRoundSamples([{"gpu": 0}],
                                          {"duration": 1, "interval": 1, "size": 16}, 100, output.append)
        reporter.flush(reporter.duration_ns)
        self.assertIsNone(output[0]["latency_us"])
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            bench.report_sample(output[0], None)
        self.assertIn("latency=--", text.getvalue())

    def test_single_process_round_covers_every_gpu(self):
        args = bench.parse_args(["--role", "receiver", "--peer", "127.0.0.1", "--nics", "mlx5_0",
                                 "--duration", "0.1", "--interval", "0.1", "--warmup", "0"])
        streams = [{"gpu": i, "gpu_nic": "mlx5_1"} for i in (0, 1)]
        config = bench.make_config(args, streams)
        client = FakeReadClient(args, config, "127.0.0.1", [1, 2])
        client.events.get_nowait()
        result = bench.run_synchronized_read(client, config, streams, lambda e: None)
        self.assertGreater(config["round_summary"]["rounds"], 0)
        self.assertEqual([r["stream"] for r in result], [0, 1])
        self.assertEqual(result[0]["operations"], result[1]["operations"])

        class MissingGpuClient(FakeReadClient):
            def round_streams(self):
                return super().round_streams()[:1]
        config = bench.make_config(args, [dict(s) for s in streams])
        client = MissingGpuClient(args, config, "127.0.0.1", [1, 2])
        client.events.get_nowait()
        with self.assertRaisesRegex(ValueError, "every configured GPU"):
            bench.run_synchronized_read(client, config, config["streams"], lambda e: None)


class LimitModeTests(unittest.TestCase):
    RECEIVER = ["--role", "receiver", "--peer", "127.0.0.1", "--nics", "mlx5_0"]

    def test_decimal_sizes_and_shares(self):
        for text, expected in (("5G", 5 * 10 ** 9), ("5GB", 5 * 10 ** 9), ("640M", 640 * 10 ** 6),
                               ("5GiB", 5 << 30), ("42", 42), ("1k", 1000)):
            self.assertEqual(bench.decimal_size_arg(text), expected)
        for text in ("0", "iB", "5iB", "1.5G", "-1", "99999T"):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench.decimal_size_arg(text)
        self.assertEqual(bench.transfer_shares(10, 3), [4, 3, 3])
        for count in (1, 4, 8):
            shares = bench.transfer_shares(5 * 10 ** 9, count)
            self.assertEqual(sum(shares), 5 * 10 ** 9)
            self.assertEqual(len(set(shares)), 1)

    def test_oq_list_and_percentile(self):
        self.assertEqual(bench.oq_list_arg("1, 4,16,64"), [1, 4, 16, 64])
        for text in ("", "0", "1,x", ",".join(["1"] * 65)):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench.oq_list_arg(text)
        self.assertIsNone(bench.percentile([], 50))
        values = list(range(1, 101))
        self.assertEqual(bench.percentile(values, 50), 50)
        self.assertEqual(bench.percentile(values, 99), 99)
        self.assertEqual(bench.percentile([7], 99), 7)

    def test_cli_defaults_and_validation(self):
        args = bench.parse_args(self.RECEIVER + ["--mode", "sweep"])
        self.assertEqual((args.outstanding, args.total_bytes, args.oq_list), (16, 5 * 10 ** 9, [1, 4, 16, 64]))
        self.assertEqual(bench.parse_args(self.RECEIVER).outstanding, 8)
        args = bench.parse_args(self.RECEIVER + ["--mode", "switch", "--low-oq", "0", "--high-ms", "0.5"])
        self.assertEqual((args.high_oq, args.low_oq, args.high_ms), (64, 0, 0.5))
        bad = (["--mode", "sweep", "--op", "write"],
               ["--mode", "sweep", "--oq-list", "1,128"],              # beyond qps * outstanding = 64
               ["--mode", "sweep", "--qps", "1", "--oq-list", "32"],   # 1 * 16
               ["--mode", "switch", "--low-oq", "64"],                 # low must be < high
               ["--mode", "switch", "--high-oq", "65"],
               ["--mode", "switch", "--high-ms", "0"],
               ["--mode", "sweep", "--repeat", "0"])
        for extra in bad:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                bench.parse_args(self.RECEIVER + extra)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(["--role", "receiver", "--peer", "127.0.0.1", "--nics", "mlx5_0,mlx5_1",
                              "--mode", "sweep"])

    def test_limit_modes_with_one_qp(self):
        for mode, extra in (("sweep", ["--oq-list", "1,4,16"]),
                            ("switch", ["--high-oq", "16", "--low-oq", "0"])):
            with self.subTest(mode=mode):
                args = bench.parse_args(self.RECEIVER + ["--mode", mode, "--qps", "1"] + extra)
                config = bench.make_config(args, [{"gpu": 0, "gpu_nic": "mlx5_0"}])
                bench.validate_config(config, 1)

    def test_switch_rejects_dropped_completions(self):
        def collect(client, config, done_event, handle, check):
            handle({"event": "trace_meta", "dropped_events": 0, "dropped_completions": 1})
            return {"posting_workers": 1, "verified": True}

        with mock.patch.object(bench, "collect_until", side_effect=collect), \
                self.assertRaisesRegex(RuntimeError, "statistics are invalid"):
            bench.run_switch(mock.Mock(), {"verify": "full"})

    def test_config_and_native_command(self):
        args = bench.parse_args(self.RECEIVER + ["--mode", "sweep", "--total-bytes", "10", "--chunk", "4K",
                                                 "--oq-list", "1,64", "--repeat", "2"])
        streams = [{"gpu": g, "gpu_nic": f"mlx5_{g}"} for g in (0, 1, 2)]
        config = bench.make_config(args, streams)
        self.assertEqual((config["mode"], config["size"], config["total_bytes"]), ("sweep", 4, 10))
        bench.validate_config(json.loads(json.dumps(config)), 3)  # what B checks after transport
        for name, value in (("size", 3), ("oq_list", [65]), ("low_oq", 64), ("mode", "other"),
                            ("total_bytes", 2)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                bench.validate_config(dict(config, **{name: value}), 3)
        with mock.patch.object(bench.subprocess, "Popen") as popen, mock.patch.object(bench.threading, "Thread"):
            popen.return_value.poll.return_value = 0
            client = bench.ReadClient(args, config, "10.0.0.2", [1, 2, 3])
            command = popen.call_args.args[0]
            client.close()
        value = lambda flag: command[command.index(flag) + 1]
        self.assertEqual((value("--mode"), value("--total-bytes"), value("--oq-list"), value("--repeat")),
                         ("sweep", "10", "1,64", "2"))
        self.assertEqual((value("--chunk"), value("--outstanding"), value("--size")), ("4096", "16", "4"))
        rounds = bench.make_config(bench.parse_args(self.RECEIVER), [dict(s) for s in streams])
        with mock.patch.object(bench.subprocess, "Popen") as popen, mock.patch.object(bench.threading, "Thread"):
            popen.return_value.poll.return_value = 0
            bench.ReadClient(args, rounds, "10.0.0.2", [1, 2, 3]).close()
            command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--mode") + 1], "rounds")
        self.assertNotIn("--total-bytes", command)

    def test_sweep_row(self):
        streams = [{"gpu": 0}, {"gpu": 3}]
        event = {"event": "sweep_result", "oq": 4, "repeat": 1, "submitted_ns": 100, "completed_ns": 2100,
                 "latency_ns": 2000, "streams": [{"stream": 0, "bytes": 1000, "completed_ns": 1100},
                                                {"stream": 3, "bytes": 1000, "completed_ns": 2100}]}
        row = bench.sweep_row(event, streams, [1000, 1000])
        self.assertEqual(row["total_GB_s"], 1.0)
        self.assertEqual([s["GB_s"] for s in row["streams"]], [1.0, 0.5])
        self.assertEqual(row["latency_ms"], 0.002)
        for broken in (dict(event, latency_ns=1), dict(event, streams=event["streams"][:1]),
                       dict(event, streams=[dict(event["streams"][0], bytes=1), event["streams"][1]])):
            with self.assertRaises(ValueError):
                bench.sweep_row(broken, streams, [1000, 1000])

    def test_switch_analysis(self):
        origin, bin_ns, bins, events, transfers = synthetic_switch_trace()
        result = bench.analyze_switch(origin, bin_ns, bins, events, transfers, 64, 1)
        down, up = result["down"], result["up"]
        self.assertEqual((len(down["samples"]), down["excluded"]), (2, 0))
        self.assertEqual((len(up["samples"]), up["excluded"]), (1, 0))
        self.assertEqual(down["stats"]["T_90_us"]["p50"], 300)
        self.assertEqual(down["stats"]["T_down_us"]["p50"], 150)
        self.assertEqual(down["stats"]["T1_us"]["p99"], 5)
        self.assertEqual(up["stats"]["T_up_us"]["p50"], 200)
        self.assertEqual(down["stats"]["before_GB_s"]["p50"], 50)
        self.assertEqual(down["stats"]["after_GB_s"]["p50"], 5)
        # Pause (low = 0): the drop is still measured.
        origin, bin_ns, bins, events, transfers = synthetic_switch_trace(64, 0)
        paused = bench.analyze_switch(origin, bin_ns, bins, events, transfers, 64, 0)
        self.assertEqual(paused["down"]["stats"]["T_90_us"]["p50"], 300)
        self.assertEqual(paused["down"]["stats"]["after_GB_s"]["p50"], 0)
        # Transfer gaps exclude the windows that cross them.
        gapped = bench.analyze_switch(0, bin_ns, bins, events, [[0, 5_000_000], [5_100_000, 9_000_000]], 64, 0)
        self.assertEqual((len(gapped["down"]["samples"]), gapped["down"]["excluded"]), (1, 1))
        self.assertEqual(gapped["up"]["excluded"], 1)

    def test_switch_without_significant_drop(self):
        events = [[1, 64, 0, 1, 1], [2, 1, 2_000_000, 2_000_100, 2_000_200], [3, 64, 4_000_000, 4_000_100, 0]]
        result = bench.analyze_switch(0, 100_000, [5_000_000] * 40, events, [[0, 4_000_000]], 64, 1)
        sample = result["down"]["samples"][0]
        self.assertIsNone(sample["T_90_us"])  # limit did not reduce the rate
        self.assertEqual(sample["T_down_us"], 0.2)

    def run_session(self, mode, extra):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            FakeWorker.records, FakeReadClient.instances = [], []
            left, right = control_pair()
            output = str(Path(root) / "result.json")
            source = bench.parse_args(["--role", "source", "--gpus", "0,3", "--setup-timeout", "2"])
            receiver = bench.parse_args(self.RECEIVER + ["--mode", mode, "--json", output,
                                                         "--setup-timeout", "2"] + extra)
            offered = [{"gpu": 0, "gpu_nic": "mlx5_1", "pci": "0000:01:00.0", "name": "fake0"},
                       {"gpu": 3, "gpu_nic": "mlx5_7", "pci": "0000:02:00.0", "name": "fake3"}]
            errors, source_workers, receiver_workers = [], [], []

            def run_source():
                try:
                    bench.source_session(source, right, "127.0.0.1", source_workers)
                except BaseException as exc:
                    errors.append(exc)
                    right.sock.shutdown(socket.SHUT_RDWR)
            try:
                with mock.patch.object(bench, "Worker", FakeWorker), \
                        mock.patch.object(bench, "ReadClient", FakeReadClient), \
                        mock.patch.object(bench, "selected_gpus", return_value=offered), \
                        contextlib.redirect_stdout(io.StringIO()) as printed:
                    thread = threading.Thread(target=run_source, daemon=True)
                    thread.start()
                    bench.receiver_session(receiver, left, receiver_workers)
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
                self.assertFalse(errors, errors)
                self.assertEqual(len(FakeReadClient.instances), 1)
                return json.loads(Path(output).read_text(encoding="utf-8")), printed.getvalue()
            finally:
                for worker in source_workers + receiver_workers:
                    worker.close()
                left.sock.close()
                right.sock.close()

    def test_sweep_session(self):
        saved, printed = self.run_session("sweep", ["--total-bytes", "5G", "--oq-list", "1,4", "--repeat", "2"])
        result = saved["result"]
        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual([s["oq"] for s in result["summary"]], [1, 4])
        self.assertEqual(saved["config"]["size"], 2_500_000_000)
        self.assertEqual(result["summary"][0]["latency_ms"], 1000)
        self.assertEqual(result["summary"][0]["total_GB_s"], 5)
        self.assertEqual(result["summary"][1]["per_gpu_GB_s"]["3"], 2.5 * 4)
        self.assertIn("TOTAL(GB/s)", printed)

    def test_switch_session(self):
        saved, printed = self.run_session("switch", ["--total-bytes", "1G"])
        analysis = saved["result"]["analysis"]
        self.assertEqual(analysis["down"]["stats"]["T_90_us"]["p50"], 300)
        self.assertEqual(analysis["up"]["stats"]["T_up_us"]["p50"], 200)
        self.assertEqual(len(saved["result"]["bins"]), 80)
        self.assertIn("T_down", printed)


if __name__ == "__main__":
    unittest.main()
