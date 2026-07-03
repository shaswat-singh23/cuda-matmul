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
## Status
Work in progress. Rigorous benchmarking harness, Nsight Compute profiling analysis, and detailed performance notes coming in follow-up commits.