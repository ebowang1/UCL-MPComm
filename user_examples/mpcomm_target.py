#!/usr/bin/env python3
# Copyright (C) 2026 Tencent. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MPComm Target Server

Remote helper that keeps a DDR buffer online for mp_replicate tests.
This script runs on the target host and waits for incoming RDMA connections.

Usage:
    python mpcomm_target.py --host-id target1 --tcp-port 12345 --device mlx5_0

    # Multi-NUMA mode (allocate separate buffer on specified NUMA nodes)
    python mpcomm_target.py --host-id target1 --tcp-port 12345 --num-numas 0,1

Environment variables:
    MPCOMM_HOST_ID: Local host identifier (default: 127.0.0.1:12345)
    MPCOMM_TCP_PORT: TCP port for metadata exchange (default: 12345)
    MPCOMM_DEVICE: RDMA device name (default: auto-detect)
    MPCOMM_BUFFER_SIZE: Buffer size in bytes (default: 2GB)
"""

import argparse
import ipaddress
import os
import signal
import socket
import sys
import time
from typing import List, Optional

import torch

# Import mpcomm module (built from C++)
try:
    import mpcomm
except ImportError:
    print("Error: mpcomm module not found. Please build it first:", file=sys.stderr)
    print("  cd mpcomm && mkdir build && cd build", file=sys.stderr)
    print("  cmake .. -DBUILD_MPCOMM_PYTHON=ON && make", file=sys.stderr)
    sys.exit(1)


DEFAULT_BUFFER_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB
DEFAULT_TCP_PORT = 12345
PATTERN_PREVIEW_BYTES = 16


def _build_ipv4_pattern(ip_bytes: List[int], total_length: int) -> bytes:
    """Build a repeating pattern based on IPv4 address bytes."""
    if total_length <= 0:
        return b""
    pattern = bytearray(total_length)
    chunk = ip_bytes or [0, 0, 0, 0]
    for idx in range(total_length):
        pattern[idx] = chunk[idx % 4]
    return bytes(pattern)


def _resolve_ipv4_bytes(host: str) -> List[int]:
    """Resolve hostname to IPv4 bytes."""
    host_only = host.split(":", 1)[0]
    try:
        ip_obj = ipaddress.ip_address(socket.gethostbyname(host_only))
        return list(ip_obj.packed)
    except OSError:
        print(f"Failed to resolve host {host_only}, using zeros for pattern", file=sys.stderr)
        return [0, 0, 0, 0]


def _preview_hex(payload: bytes, limit: int = PATTERN_PREVIEW_BYTES) -> str:
    """Preview bytes as space-separated decimal values."""
    if not payload:
        return "<empty>"
    preview = payload[:limit]
    return " ".join(str(byte) for byte in preview)


def _format_bytes(num_bytes: int) -> str:
    """Format bytes with human-readable units (network convention, base 1000)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    idx = 0
    while value >= 1000 and idx < len(units) - 1:
        value /= 1000
        idx += 1
    return f"{value:.2f} {units[idx]}"


class MPCommTargetServer:
    """Target server that publishes either a CPU-DRAM or GPU-HBM RDMA buffer."""

    def __init__(
        self,
        *,
        host_id: str,
        tcp_port: int,
        device_name: str,
        buffer_size: int,
        gpu_id: Optional[int],
        verbose: bool,
    ) -> None:
        self.host_id = host_id
        self.tcp_port = tcp_port
        self.device_name = device_name
        self.buffer_size = buffer_size
        self.gpu_id = gpu_id
        self.verbose = verbose
        self._stop_requested = False

        if self.gpu_id is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("--gpu requires CUDA, but CUDA is unavailable")
            torch.cuda.set_device(self.gpu_id)

        # Initialize MPComm
        self.comm = mpcomm.MPComm()
        ret = self.comm.init(host_id, device_name, tcp_port)
        if ret != 0:
            raise RuntimeError(f"MPComm initialization failed with code {ret}")

        # Get actual TCP port (in case 0 was passed for auto-select)
        self.tcp_port = self.comm.get_tcp_port()

        print(f"[target] Initialized MPComm with {self.comm.get_num_nics()} NICs")
        for i in range(self.comm.get_num_nics()):
            print(f"  NIC {i}: GID={self.comm.get_gid(i)}")

        # Allocate and register buffer.
        self._tensor: Optional[torch.Tensor] = None
        self.buffer_addr = self._allocate_buffer()
        self._ipv4_bytes = _resolve_ipv4_bytes(host_id)

        # The verification seed uses a CPU memcpy and is therefore valid only
        # for a DRAM target. A GPU target is intended for remote RDMA WRITE and
        # is initialized by CUDA instead.
        if self.gpu_id is None:
            self._initialize_buffer_pattern()
        else:
            self._tensor.zero_()

        # Publish buffer for remote queries
        ret = self.comm.publish_buffer(self.buffer_addr, self.buffer_size)
        if ret != 0:
            raise RuntimeError(f"Failed to publish buffer: {ret}")

    def _allocate_buffer(self) -> int:
        """Allocate a CPU-DRAM or GPU-HBM tensor and register it for RDMA."""
        device = "cpu" if self.gpu_id is None else f"cuda:{self.gpu_id}"
        tensor = torch.empty(self.buffer_size, dtype=torch.uint8, device=device)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        addr = tensor.data_ptr()

        ret = self.comm.register_memory(addr, self.buffer_size)
        if ret != 0:
            memory_kind = "GPU HBM" if self.gpu_id is not None else "CPU DRAM"
            raise RuntimeError(
                f"Failed to register {memory_kind} with RDMA: {ret}; "
                "check nvidia-peermem for GPU memory")

        self._tensor = tensor
        memory_kind = "CPU DRAM" if self.gpu_id is None else f"GPU {self.gpu_id} HBM"
        print(f"[target] Registered {memory_kind} at 0x{addr:x} "
              f"({_format_bytes(self.buffer_size)})")
        return addr

    def _initialize_buffer_pattern(self) -> None:
        """Initialize buffer with IPv4 pattern for verification."""
        pattern_size = min(self.buffer_size, 1024 * 1024)  # Initialize first 1MB
        payload = _build_ipv4_pattern(self._ipv4_bytes, pattern_size)

        ret = self.comm.write_bytes_to_buffer(self.buffer_addr, payload, len(payload))
        if ret != 0:
            print("[target] Warning: failed to seed buffer with IPv4 pattern", file=sys.stderr)
            return

        print(f"[target] Seeded buffer using IPv4 bytes {self._ipv4_bytes} preview={_preview_hex(payload)}")

    def _unpublish_buffer(self) -> None:
        """Unpublish the buffer before cleanup."""
        if self.buffer_addr:
            self.comm.unpublish_buffer(self.buffer_addr)

    def get_buffer_info(self) -> dict:
        """Get buffer information for exchange with initiator."""
        return {
            "host_id": self.host_id,
            "tcp_addr": self.host_id.split(":")[0],
            "tcp_port": self.tcp_port,
            "buffer_addr": self.buffer_addr,
            "buffer_size": self.buffer_size,
            "rkeys": self.comm.get_all_rkeys(self.buffer_addr),
            "num_nics": self.comm.get_num_nics(),
        }

    def stop(self) -> None:
        """Request the server to stop."""
        self._stop_requested = True

    def close(self) -> None:
        """Cleanup resources."""
        if self.buffer_addr and self._tensor is not None:
            self._unpublish_buffer()
            self.comm.unregister_memory(self.buffer_addr)
            self.buffer_addr = 0
            self._tensor = None
        self.comm.stop_accept_thread()
        self.comm.shutdown()

    def run(self) -> None:
        """Run the target server, accepting connections and waiting."""
        # Start accept thread for incoming connections
        ret = self.comm.start_accept_thread()
        if ret != 0:
            raise RuntimeError(f"Failed to start accept thread: {ret}")

        # Print connection info
        print("\n" + "=" * 60)
        print("MPComm Target Server Ready")
        print("=" * 60)
        info = self.get_buffer_info()
        print(f"  Host ID:      {info['host_id']}")
        print(f"  TCP Address:  {info['tcp_addr']}:{info['tcp_port']}")
        print(f"  Buffer Addr:  0x{info['buffer_addr']:x}")
        print(f"  Buffer Size:  {_format_bytes(info['buffer_size'])}")
        print(f"  Buffer Type:  {'CPU DRAM' if self.gpu_id is None else f'GPU {self.gpu_id} HBM'}")
        print(f"  Num NICs:     {info['num_nics']}")
        print(f"  RKeys:        {info['rkeys']}")
        print("=" * 60)
        print("\nBuffer is published and accessible via TCP metadata exchange.")
        print("Initiator can use query_remote_buffer() to get buffer info automatically.")
        print("\nWaiting for connections... (Press Ctrl+C to stop)\n")

        # Main loop - just wait and periodically print status
        poll_interval = 5.0
        while not self._stop_requested:
            time.sleep(poll_interval)
            if self.verbose:
                # Read and print buffer preview
                preview_data = self.comm.read_bytes_from_buffer(self.buffer_addr, 64)
                print(f"[target] Buffer preview: {_preview_hex(preview_data)}")

        print("[target] Stopping server...")


class MultiNumaTargetServer:
    """Target server that maintains buffers with NUMA regions.

    This class allocates a buffer for each NUMA node and publishes each buffer
    with its NUMA info. Initiator can query all buffers and match NUMA-to-NUMA:
    - initiator NUMA 0 -> target buffer with numa_node=0
    - initiator NUMA 1 -> target buffer with numa_node=1
    - etc.

    Each buffer is published separately with publish_buffer(addr, length, numa_node).
    The new multi-buffer API allows querying all buffers with their NUMA info.
    """

    def __init__(
        self,
        *,
        host_id: str,
        tcp_port: int,
        device_name: str,
        buffer_size: int,
        numa_nodes: List[int],
        verbose: bool,
    ) -> None:
        self.host_id = host_id
        self.tcp_port = tcp_port
        self.device_name = device_name
        self.buffer_size = buffer_size
        self.numa_nodes = numa_nodes  # List of specific NUMA node IDs
        self.num_numas = len(numa_nodes)  # For compatibility
        self.verbose = verbose
        self._stop_requested = False

        if not numa_nodes:
            raise ValueError("numa_nodes must not be empty")

        # Initialize MPComm
        self.comm = mpcomm.MPComm()
        ret = self.comm.init(host_id, device_name, tcp_port)
        if ret != 0:
            raise RuntimeError(f"MPComm initialization failed with code {ret}")

        # Get actual TCP port
        self.tcp_port = self.comm.get_tcp_port()

        print(f"[multi-numa target] Initialized MPComm with {self.comm.get_num_nics()} NICs")
        print(f"[multi-numa target] Using NUMA nodes: {numa_nodes}")
        for i in range(self.comm.get_num_nics()):
            print(f"  NIC {i}: GID={self.comm.get_gid(i)}")

        # Allocate and register buffers for each NUMA node
        self._tensors: List[Optional[torch.Tensor]] = []
        self.numa_buffer_addrs: List[int] = []
        self._ipv4_bytes = _resolve_ipv4_bytes(host_id)
        self._libnuma = None  # Will be set by _allocate_numa_buffers
        self._allocate_numa_buffers()

        # Initialize each NUMA buffer with IPv4 pattern
        for idx, numa_node in enumerate(numa_nodes):
            self._initialize_buffer_pattern(idx, self.numa_buffer_addrs[idx])

        # Publish each buffer with its NUMA info
        for idx, numa_node in enumerate(numa_nodes):
            ret = self.comm.publish_buffer(
                self.numa_buffer_addrs[idx],
                self.buffer_size,
                numa_node  # Pass actual NUMA node ID
            )
            if ret != 0:
                raise RuntimeError(f"Failed to publish buffer for NUMA {numa_node}: {ret}")
            print(f"[multi-numa target] Published buffer for NUMA node {numa_node}")

    def _allocate_numa_buffers(self) -> None:
        """Allocate tensor buffers on different NUMA nodes using libnuma."""
        import ctypes

        # Try to load libnuma for explicit NUMA allocation
        try:
            libnuma = ctypes.CDLL("libnuma.so.1", mode=ctypes.RTLD_GLOBAL)
            libnuma.numa_available.restype = ctypes.c_int
            libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
            libnuma.numa_alloc_onnode.restype = ctypes.c_void_p
            libnuma.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libnuma.numa_max_node.restype = ctypes.c_int

            if libnuma.numa_available() < 0:
                libnuma = None
                print("[multi-numa target] WARNING: libnuma not available, falling back to default allocation")
        except OSError:
            libnuma = None
            print("[multi-numa target] WARNING: libnuma.so.1 not found, falling back to default allocation")

        self._libnuma = libnuma  # Store for cleanup

        for idx, target_numa in enumerate(self.numa_nodes):
            try:
                if libnuma is not None:
                    # Validate target NUMA node
                    max_node = libnuma.numa_max_node()
                    if target_numa > max_node:
                        raise RuntimeError(f"NUMA node {target_numa} exceeds max node {max_node}")

                    # Allocate aligned to page size for better performance
                    page_size = 4096
                    alloc_size = ((self.buffer_size + page_size - 1) // page_size) * page_size

                    ptr = libnuma.numa_alloc_onnode(alloc_size, target_numa)
                    if not ptr:
                        raise RuntimeError(f"numa_alloc_onnode failed for NUMA {target_numa}")

                    # Touch all pages to ensure they are allocated
                    ctypes.memset(ptr, 0, alloc_size)

                    addr = ptr
                    ret = self.comm.register_memory(addr, self.buffer_size)
                    if ret != 0:
                        libnuma.numa_free(ptr, alloc_size)
                        raise RuntimeError(f"Failed to register memory for NUMA {target_numa}: {ret}")

                    # Store as tuple: (ptr, alloc_size, is_numa_alloc)
                    self._tensors.append((ptr, alloc_size, True))
                    self.numa_buffer_addrs.append(addr)
                    print(f"[multi-numa target] Buffer {idx}: Allocated {_format_bytes(alloc_size)} "
                          f"on NUMA node {target_numa} at 0x{addr:x}")
                else:
                    # Fallback: use PyTorch tensor
                    tensor = torch.empty(self.buffer_size, dtype=torch.uint8)
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    tensor.fill_(0)

                    addr = tensor.data_ptr()
                    ret = self.comm.register_memory(addr, self.buffer_size)
                    if ret != 0:
                        raise RuntimeError(f"Failed to register memory for NUMA {target_numa}: {ret}")

                    self._tensors.append((tensor, 0, False))
                    self.numa_buffer_addrs.append(addr)
                    print(f"[multi-numa target] Buffer {idx}: Registered buffer at 0x{addr:x} "
                          f"({_format_bytes(self.buffer_size)}) [WARNING: may not be on NUMA {target_numa}]")
            except Exception as e:
                raise RuntimeError(f"Failed to allocate buffer for NUMA {target_numa}: {e}")

    def _initialize_buffer_pattern(self, numa_id: int, addr: int) -> None:
        """Initialize buffer with IPv4 pattern for verification."""
        pattern_size = min(self.buffer_size, 1024 * 1024)  # Initialize first 1MB
        payload = _build_ipv4_pattern(self._ipv4_bytes, pattern_size)

        ret = self.comm.write_bytes_to_buffer(addr, payload, len(payload))
        if ret != 0:
            print(f"[multi-numa target] Warning: failed to seed buffer {numa_id} with IPv4 pattern",
                  file=sys.stderr)
            return

        print(f"[multi-numa target] NUMA {numa_id}: Seeded buffer with IPv4 pattern "
              f"{self._ipv4_bytes} preview={_preview_hex(payload)}")

    def _unpublish_buffers(self) -> None:
        """Unpublish all buffers before cleanup."""
        self.comm.unpublish_all_buffers()

    def get_buffer_info(self) -> dict:
        """Get buffer information for exchange with initiator."""
        # Return info for all NUMA buffers
        buffers_info = []
        for numa_node, addr in zip(self.numa_nodes, self.numa_buffer_addrs):
            buffers_info.append({
                "numa_node": numa_node,  # Use actual NUMA node ID
                "addr": addr,
                "length": self.buffer_size,
                "rkeys": self.comm.get_all_rkeys(addr),
            })

        return {
            "host_id": self.host_id,
            "tcp_addr": self.host_id.split(":")[0],
            "tcp_port": self.tcp_port,
            "numa_nodes": self.numa_nodes,  # List of actual NUMA node IDs
            "num_numas": self.num_numas,
            "num_nics": self.comm.get_num_nics(),
            "buffers": buffers_info,
        }

    def stop(self) -> None:
        """Request the server to stop."""
        self._stop_requested = True

    def close(self) -> None:
        """Cleanup resources."""
        self._unpublish_buffers()
        for addr, tensor_info in zip(self.numa_buffer_addrs, self._tensors):
            if addr:
                self.comm.unregister_memory(addr)
                # Check if this was a numa allocation
                if isinstance(tensor_info, tuple) and len(tensor_info) == 3:
                    ptr, alloc_size, is_numa_alloc = tensor_info
                    if is_numa_alloc and self._libnuma is not None:
                        self._libnuma.numa_free(ptr, alloc_size)
        self.numa_buffer_addrs = []
        self._tensors = []
        self.comm.stop_accept_thread()
        self.comm.shutdown()

    def run(self) -> None:
        """Run the target server, accepting connections and waiting."""
        # Start accept thread for incoming connections
        ret = self.comm.start_accept_thread()
        if ret != 0:
            raise RuntimeError(f"Failed to start accept thread: {ret}")

        # Print connection info
        print("\n" + "=" * 70)
        print("MPComm Multi-NUMA Target Server Ready")
        print("=" * 70)
        info = self.get_buffer_info()
        print(f"  Host ID:      {info['host_id']}")
        print(f"  TCP Address:  {info['tcp_addr']}:{info['tcp_port']}")
        print(f"  Num NICs:     {info['num_nics']}")
        print(f"  Num NUMAs:    {info['num_numas']}")
        print()
        print("  Published Buffers (each with NUMA info):")
        for buf in info['buffers']:
            print(f"    NUMA {buf['numa_node']}: addr=0x{buf['addr']:x}, "
                  f"length={_format_bytes(buf['length'])}, rkeys={buf['rkeys']}")
        print("=" * 70)
        print("\nBuffers are published with NUMA info and accessible via TCP metadata exchange.")
        print("Initiator can use query_remote_buffers() to get all buffer info automatically,")
        print("or use query_remote_buffer_by_numa(numa_node) to get a specific NUMA buffer.")
        print("\nWaiting for connections... (Press Ctrl+C to stop)\n")

        # Main loop
        poll_interval = 5.0
        while not self._stop_requested:
            time.sleep(poll_interval)
            if self.verbose:
                for numa_id, addr in enumerate(self.numa_buffer_addrs):
                    preview_data = self.comm.read_bytes_from_buffer(addr, 64)
                    print(f"[multi-numa target] NUMA {numa_id} buffer preview: {_preview_hex(preview_data)}")

        print("[multi-numa target] Stopping server...")


def _install_signal_handlers(target) -> None:
    """Install signal handlers for graceful shutdown."""
    def _handler(signum, _frame):
        print(f"\n[target] Received signal {signum}, stopping...")
        target.stop()

    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MPComm Target Server - maintains DDR buffer for RDMA tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start target server with auto-detected device
    python mpcomm_target.py --host-id target1:12345

    # Start target server with specific device
    python mpcomm_target.py --host-id <target_ip>:12345 --device mlx5_0

    # Start target server with larger buffer
    python mpcomm_target.py --host-id target1:12345 --buffer-size 20GB

    # Multi-NUMA mode (use NUMA nodes 0 and 1, each with separate buffer)
    python mpcomm_target.py --host-id target1:12345 --num-numas 0,1

    # Multi-NUMA with custom buffer size per NUMA
    python mpcomm_target.py --host-id target1:12345 --num-numas 0,1 --buffer-size 10GB
""",
    )

    parser.add_argument(
        "--host-id",
        default=os.getenv("MPCOMM_HOST_ID", "127.0.0.1:12345"),
        help="Local host identifier (default: %(default)s)",
    )

    parser.add_argument(
        "--tcp-port",
        type=int,
        default=int(os.getenv("MPCOMM_TCP_PORT", DEFAULT_TCP_PORT)),
        help="TCP port for metadata exchange (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        default=os.getenv("MPCOMM_DEVICE", ""),
        help="RDMA device name (comma-separated for multiple). Empty = auto-detect",
    )

    parser.add_argument(
        "--buffer-size",
        default=os.getenv("MPCOMM_BUFFER_SIZE", str(DEFAULT_BUFFER_SIZE)),
        help="Buffer size with optional suffix (K/M/G/T) (default: 10GB)",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="Publish a GPU-HBM buffer on this CUDA device; -1 publishes CPU DRAM (default)",
    )

    parser.add_argument(
        "--num-numas",
        type=str,
        default="0",
        help="NUMA nodes to use, e.g. '0', '1', '0,1' (default: '0' = single buffer on NUMA 0)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print periodic buffer status",
    )

    return parser.parse_args()


def parse_size(size_str: str) -> int:
    """Parse size string with optional suffix (K/M/G/T)."""
    size_str = size_str.strip().upper()
    multipliers = {
        "K": 1024,
        "KB": 1024,
        "KIB": 1024,
        "M": 1024 ** 2,
        "MB": 1024 ** 2,
        "MIB": 1024 ** 2,
        "G": 1024 ** 3,
        "GB": 1024 ** 3,
        "GIB": 1024 ** 3,
        "T": 1024 ** 4,
        "TB": 1024 ** 4,
        "TIB": 1024 ** 4,
    }

    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            return int(float(size_str[:-len(suffix)]) * mult)

    return int(size_str)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    buffer_size = parse_size(args.buffer_size)

    # Parse NUMA nodes (e.g. "0", "1", "0,1")
    numa_nodes_str = args.num_numas.strip()
    numa_nodes = [int(n.strip()) for n in numa_nodes_str.split(",") if n.strip()]

    if args.gpu >= 0 and len(numa_nodes) > 1:
        raise SystemExit("--gpu cannot be combined with multi-NUMA target buffers")

    if len(numa_nodes) > 1:
        print(f"[main] Starting Multi-NUMA target server with NUMA nodes: {numa_nodes}")
        target = MultiNumaTargetServer(
            host_id=args.host_id,
            tcp_port=args.tcp_port,
            device_name=args.device,
            buffer_size=buffer_size,
            numa_nodes=numa_nodes,
            verbose=args.verbose,
        )
    else:
        target = MPCommTargetServer(
            host_id=args.host_id,
            tcp_port=args.tcp_port,
            device_name=args.device,
            buffer_size=buffer_size,
            gpu_id=args.gpu if args.gpu >= 0 else None,
            verbose=args.verbose,
        )

    _install_signal_handlers(target)

    try:
        target.run()
    except KeyboardInterrupt:
        target.stop()
    finally:
        target.close()


if __name__ == "__main__":
    main()
