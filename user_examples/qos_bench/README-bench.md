# MPComm 带宽/时延基准测试 (H2D + NIC->GPU)

本目录提供一个统一脚本 `bench_h2d_nic2gpu.py`，测试 MPComm 的两种通信路径：

| 模式 | 参数 | 数据流 | 底层 API | 运行环境 |
|------|------|--------|----------|----------|
| **H2D** | `--path h2d` | 本机 DRAM ↔ GPU HBM (PCIe) | `map_dram_to_gpu` + `tma_gather`/`tma_scatter` | 单机 1 GPU |
| **NIC→GPU** | `--path nic2gpu` | 远端 → 本地 GPU 显存 (GPUDirect RDMA) | GPU HBM 作 buffer + `get_async`/`put_async` | 2 台机器+ RDMA 网卡 |

指标：平均带宽 (GB/s)、峰值带宽、平均/P50/P99 时延、min/max 时延；可 `--csv` 导出。

---

## 0. 前置：先编译库

```bash
cd trmt-mpcomm && sh build.sh     # 需 USE_CUDA=ON
# 产物: mpcomm-install/lib/python/mpcomm  (脚本会自动加入 sys.path)
```

---

## 1. H2D（本机，单机即可）

```bash
cd user_examples/qos_bench

# 一键：GPU0，固定 2048 个 272B 离散块，TMA 后端
bash run-bench-h2d.sh

# 参数：GPU_ID / NUM_BLOCKS / BLOCK_SIZE / MODE
bash run-bench-h2d.sh 0 2048 272 tma

# 直接调脚本，只测 gather（DRAM->GPU）并导出 CSV
python3 bench_h2d_nic2gpu.py --path h2d --gpu 0 --op gather \
    --num-blocks 2048 --block-size 272 --h2d-mode tma --csv h2d.csv
```

- H2D 默认模拟 `2048 × 272B` 的随机离散块访问；每次迭代有效负载为 `557,056B`（约 `544KiB`）。
- `--num-blocks`：离散数据块数量；默认 `2048`。
- `--block-size`：每个离散块的字节数，必须为 16 的倍数；默认 `272`。
- `--h2d-mode`：`tma`（默认，需 Hopper sm_90+，H20可用）/ `sm`（int4 Zero-Copy）/ `auto`。
- `--op`：`gather`(DRAM→GPU) / `scatter`(GPU→DRAM) / `both`。随机索引无重复，可避免 scatter 写冲突。
- `run-bench-h2d.sh` 默认附带 `--cuda-copy-baseline`：测预先整理好的连续 pinned host buffer 与 HBM 之间的 PyTorch `copy_(non_blocking=True)`，即 `cudaMemcpyAsync` H2D/D2H 等价基线。该基线只计 DMA copy，不包含 CPU 对离散块的 pack/scatter，不能与离散 TMA 路径视为完全等价的端到端实现。

---

## 2. NIC→GPU（跨机，需 2 台服务器）

**步骤 A — 目标机（被访问方，不必有 GPU）：**

```bash
cd user_examples/qos_bench
bash run-bench-nic2gpu-target.sh 12345 mlx5_bond_1,mlx5_bond_2 4G
# 输出里会打印 initiator 应使用的 --target 值, 例如:<ip>:<ip>:12345
```

**步骤 B — 发起机（本机，有 GPU）：**

```bash
# 先确认 GPUDirect RDMA 依赖已加载
lsmod | grep nvidia_peermem || sudo modprobe nvidia-peermem

cd user_examples/qos_bench
#<TARGET> 用上一步 target 打印的值
bash run-bench-nic2gpu-initiator.sh <ip>:<ip>:12345 0 both

# 或直接调脚本
python3 bench_h2d_nic2gpu.py --path nic2gpu --gpu 0 \
    --target <ip>:<ip>:12345 --op get \
    --sizes 512K,1M,4M,16M,64M,256M,1G --device mlx5_bond_1,mlx5_bond_2 \
    --csv nic2gpu.csv
```

- `--op get`：远端 NIC → 本地 GPU（读，RDMA READ）
- `--op put`：本地 GPU → 远端（写，RDMA WRITE）
- `--op both`：两者都测
- target发布的 buffer 需≥ initiator 扫描的最大块（`--sizes` 最大值）

---

## 3. 常用环境变量

脚本级：
- `MPCOMM_BENCH_ITERS`（默认 50）、`MPCOMM_BENCH_WARMUP`（默认 10）
- `MPCOMM_BENCH_TOTAL`（H2D 每次总量，默认 256M）
- `MPCOMM_BENCH_CSV`（CSV 输出路径）
- `MPCOMM_HOST_ID` / `MPCOMM_DEVICE`

库级（详见 `include/mpcomm.h`）：
- `MPCOMM_QPS_PER_CONNECTION`、`MPCOMM_NIC_FILTER`、`MPCOMM_GID_INDEX`、`MPCOMM_LOG_LEVEL` 等

---

## 4. 注意事项

1. **NIC→GPU 必须跨机**：单机无法测出真实网卡带宽。
2. **TMA 需 Hopper**：非 sm_90+ 的 GPU，H2D 请用 `--h2d-mode sm`；脚本会自动降级并告警。
3. **GPUDirect RDMA 依赖 `nvidia-peermem`**：未加载时GPU 内存注册会失败。
4. **对齐**：H2D 的 `block_size` 必须是 16 字节的倍数（TMA 建议 1024B 对齐，脚本已按1024B 分配 DRAM 池）。
5. **带宽口径**：GB/s 采用网络惯例base 1e9；时延用 CUDA event（H2D）或 wall clock（NIC→GPU）。
