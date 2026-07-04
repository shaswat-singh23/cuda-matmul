# cuda-matmul

Progressive optimization of CUDA matrix multiplication as a self-study project through Kirk & Hwu's *Programming Massively Parallel Processors*. Each baseline represents a stage of learning.

## Hardware
Developed on NVIDIA RTX 4070 Laptop GPU (Ada Lovelace, sm_89), WSL2 Ubuntu.

## Baselines
- **baseline1-3:** CPU implementations
- **baseline4:** Naive GPU kernel
- **baseline5:** Tiled shared-memory kernel with thread coarsening

## Build
```bash
nvcc -arch=sm_89 -o baseline5 baseline5.cu
nvcc -arch=sm_89 -lcublas -o testbench testbench.cu
```

## Benchmarks
Benchmarking harness in `benchmark.cu` measures GFLOPS across square matrices of size N = 256, 512, 1024, 2048, 4096, and 8192 with 3 GPU warmpu iterations + 20 timed and averaged iterations. See `results.csv` for raw data.
Peak performance at N=8192: coarsened kernel achieves ~1600 GFLOPS = ~11% of cuBLAS.

## Status
Work in progress. Rigorous benchmarking harness, Nsight Compute profiling analysis, and detailed performance notes coming in follow-up commits.