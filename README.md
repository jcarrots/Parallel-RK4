# Parallel RK4

This repository contains several implementations of a fourth-order Runge–Kutta (RK4) solver for the linear ODE \(\dot{r} = L r\). Variants include serial, OpenMP, MPI+OpenMP, CUDA, and MPI+CUDA executables that all share a common interface for choosing the problem size and time-step parameters.

## Repository layout
- `rk4_serial.cpp`: Baseline single-process implementation.
- `rk4_omp.cpp`: Shared-memory OpenMP implementation.
- `rk4_mpi_omp_1d.cpp`: 1D block-row domain decomposition using MPI with OpenMP on each rank.
- `rk4_mpi_omp_2d.cpp`: 2D block distribution of the matrix and vectors with MPI and OpenMP.
- `rk4_cuda.cu`: Single-GPU CUDA implementation.
- `rk4_mpi_cuda_1d.cu`: Hybrid MPI + CUDA version with a 1D decomposition.
- `rk4_mpi_cuda_2d.cu`: Hybrid MPI + CUDA version with a 2D grid of ranks/GPUs.
- `slurm/`: Example SLURM sweep scripts for MPI + OpenMP and MPI + CUDA runs.
- `runs/`: Example output directories (timings, solution dumps) from experiments.
- `parallel_rk4_report.pdf`: Project report summarizing performance results.

## Building
The source files are standalone; you can build only the variants you need.

### Serial
```bash
g++ -O3 -std=c++17 rk4_serial.cpp -o rk4_serial
```

### OpenMP
```bash
g++ -O3 -std=c++17 -fopenmp rk4_omp.cpp -o rk4_omp
```

### MPI + OpenMP
Use your MPI compiler wrapper with OpenMP enabled.
```bash
mpicxx -O3 -std=c++17 -fopenmp rk4_mpi_omp_1d.cpp -o rk4_mpi_omp_1d
mpicxx -O3 -std=c++17 -fopenmp rk4_mpi_omp_2d.cpp -o rk4_mpi_omp_2d
```

### CUDA
```bash
nvcc -O3 -std=c++17 rk4_cuda.cu -o rk4_cuda
nvcc -O3 -std=c++17 rk4_mpi_cuda_1d.cu -o rk4_mpi_cuda_1d
nvcc -O3 -std=c++17 rk4_mpi_cuda_2d.cu -o rk4_mpi_cuda_2d
```

## Runtime options
All executables accept a consistent set of long options:
- `--D <int>`: Dimension of the system (size of `r` and the matrix `L`).
- `--T <float>`: Final time to integrate to. Overrides `--nsteps` if both are provided.
- `--dt <float>`: Time-step size.
- `--nsteps <int>`: Number of time steps to take. If set, `T = nsteps * dt`.
- `--rho-out <path>`: Optional path to write the final solution vector.
- `--timing-out <path>`: Optional path to write a CSV-style timing and error summary.

### Running examples
- Serial or OpenMP (single node):
  ```bash
  ./rk4_serial --D 1024 --T 10 --dt 1e-3 --timing-out timing_serial.csv
  OMP_NUM_THREADS=8 ./rk4_omp --D 4096 --nsteps 5000
  ```
- MPI + OpenMP (1D block rows):
  ```bash
  mpirun -np 4 ./rk4_mpi_omp_1d --D 8192 --T 1 --dt 1e-3
  ```
- MPI + CUDA (2D process grid):
  ```bash
  mpirun -np 4 ./rk4_mpi_cuda_2d --D 16384 --nsteps 2000 --timing-out timing_mpi_cuda.csv
  ```

The diagonal test matrix defined in each program uses entries \(L_{ii} = -1 - 0.1 i\) so the exact solution is known for accuracy checks. Optional output files include the wall time, error norm, and final state.

## SLURM batch scripts
The `slurm/` directory contains ready-to-edit job scripts for larger sweeps. Update the module loads, partition names, and executable paths for your cluster, then submit with `sbatch slurm/sweep_mpi_openmp.sbatch` or `sbatch slurm/sweep_mpi_cuda.sbatch`.
