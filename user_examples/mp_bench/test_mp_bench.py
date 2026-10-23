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
        self.config = config
        self.server = server
        self.events = queue.Queue()
        self.closed = False
        self.measuring = False
        self.measured_operations = 0
        self.round_id = 0
        self.records.append((args.role, server, gpu_memory, self.stream))
        self.events.put({"event": "ready", "stream": self.stream,
                         "port": stream.get("port", peer_port)})

    def command(self, text):
        if self.config["op"] == "read" and not self.server:
            if text == "start":
                self.events.put({"event": "rounds_ready", "stream": self.stream})
            elif text.startswith("round "):
                round_id = int(text.split()[1])
                if round_id != self.round_id + 1:
                    raise ValueError("nonsequential simulated READ round")
                self.round_id = round_id
                submitted = time.monotonic_ns()
                completed = time.monotonic_ns()
                while completed <= submitted:
                    completed = time.monotonic_ns()
                self.events.put({"event": "round_done", "round": round_id, "stream": self.stream,
                                 "submitted_ns": submitted, "completed_ns": completed,
                                 "bytes": self.config["size"], "errors": 0})
                if self.measuring:
                    self.measured_operations += 1
            elif text == "warmup_done":
                self.events.put({"event": "warmup_ready", "stream": self.stream})
            elif text.startswith("measure "):
                self.measuring = True
                self.events.put({"event": "measurement_ready", "stream": self.stream})
            elif text == "finish":
                self.events.put({"event": "result", "stream": self.stream,
                                 "bytes": self.measured_operations * self.config["size"],
                                 "operations": self.measured_operations, "errors": 0,
                                 "elapsed_ns": int(self.config["duration"] * 1e9),
                                 "read_rounds": True, "verified": True})
            else:
                raise ValueError(text)
            return
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
        for operation in ("read", "write"):
            for server in (True, False):
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
                        finally:
                            worker.close()

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
                    self.assertTrue(all(r[0] == "receiver" for r in cpu_roles))
                    self.assertTrue(all(r[1] == (operation == "read") for r in gpu_roles))
                    self.assertTrue(all(r[1] == (operation == "write") for r in cpu_roles))
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

    def test_barrier_waits_for_slowest_gpu_before_next_round(self):
        records = {"submitted": {}, "completed": {}, "next": 1}
        class BarrierWorker(FakeWorker):
            def command(self, text):
                if text.startswith("round "):
                    round_id = int(text.split()[1])
                    if round_id > 1:
                        if records["completed"].get(round_id - 1) != {0, 1}:
                            raise AssertionError("next round started before all GPU completions")
                    records["submitted"].setdefault(round_id, set()).add(self.stream)
                super().command(text)

            def event(self):
                data = super().event()
                if data and data["event"] == "round_done":
                    # GPU1 has no completion visible until both requests were submitted.
                    if self.stream == 1 and records["submitted"][data["round"]] != {0, 1}:
                        self.events.put(data)
                        return None
                    records["completed"].setdefault(data["round"], set()).add(self.stream)
                return data
        args = bench.parse_args(["--role", "receiver", "--peer", "127.0.0.1", "--nics", "mlx5_0",
                                 "--duration", "0.1", "--interval", "0.1", "--warmup", "0"])
        streams = [{"gpu": i, "gpu_nic": "mlx5_1"} for i in (0, 1)]
        config = bench.make_config(args, streams)
        workers = [BarrierWorker(args, s, config, False, False) for s in streams]
        for worker in workers:
            worker.events.get_nowait()
        result = bench.run_synchronized_read(workers, config, streams, lambda e: None)
        self.assertGreater(config["round_summary"]["rounds"], 0)
        self.assertEqual(result[0]["operations"], result[1]["operations"])
        self.assertTrue(all(group == {0, 1} for group in records["completed"].values()))


if __name__ == "__main__":
    unittest.main()
