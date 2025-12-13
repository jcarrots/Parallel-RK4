# Parallel RK4 implementations

This repository showcases several implementations of the classical fourth‑order Runge–Kutta (RK4) method for solving ordinary differential equations. Each version explores a different parallelization strategy—ranging from single-threaded C++ to hybrid MPI + CUDA—to highlight how algorithm structure and hardware capabilities affect performance.

## Implementations
- `rk4_serial.cpp` — baseline single-threaded solver for correctness and reference timing.
- `rk4_omp.cpp` — OpenMP implementation for shared-memory parallelism on multi-core CPUs.
- `rk4_mpi_omp_1d.cpp` / `rk4_mpi_omp_2d.cpp` — hybrid MPI + OpenMP solvers for 1D and 2D domain decompositions.
- `rk4_cuda.cu` — CUDA-only implementation targeting a single GPU.
- `rk4_mpi_cuda_1d.cu` / `rk4_mpi_cuda_2d.cu` — hybrid MPI + CUDA versions that distribute work across GPUs.
- `slurm/` — example batch scripts for cluster queues.
- `runs/` — sample inputs or job configurations (if present).
- `parallel_rk4_report.pdf` — summary of design choices and performance observations.

## Build
Install the toolchain you need for your target (GNU C++ with OpenMP, MPI library, and/or CUDA Toolkit). Adjust include and library paths to match your environment.

```bash
# Serial baseline
g++ -O3 rk4_serial.cpp -o rk4_serial

# OpenMP
g++ -O3 -fopenmp rk4_omp.cpp -o rk4_omp

# MPI + OpenMP
mpicxx -O3 -fopenmp rk4_mpi_omp_1d.cpp -o rk4_mpi_omp_1d
mpicxx -O3 -fopenmp rk4_mpi_omp_2d.cpp -o rk4_mpi_omp_2d

# CUDA
nvcc -O3 rk4_cuda.cu -o rk4_cuda

# MPI + CUDA (edit CUDA library paths as needed)
mpicxx -O3 rk4_mpi_cuda_1d.cu -lcudart -L/usr/local/cuda/lib64 -o rk4_mpi_cuda_1d
mpicxx -O3 rk4_mpi_cuda_2d.cu -lcudart -L/usr/local/cuda/lib64 -o rk4_mpi_cuda_2d
```

## Run
Command-line options depend on how you configure the problem (grid dimensions, time step, iteration count, output files). The examples below illustrate typical launch patterns:

```bash
# Serial or OpenMP
./rk4_serial [args]
./rk4_omp [args]

# MPI + OpenMP
mpirun -np <ranks> ./rk4_mpi_omp_1d [args]
mpirun -np <ranks> ./rk4_mpi_omp_2d [args]

# MPI + CUDA
mpirun -np <ranks> ./rk4_mpi_cuda_1d [args]
mpirun -np <ranks> ./rk4_mpi_cuda_2d [args]

# CUDA-only
./rk4_cuda [args]
```

## Performance and scaling notes
- OpenMP variants should scale within a node; tune `OMP_NUM_THREADS` for your core count and observe NUMA placement.
- MPI introduces communication overhead; larger domains and fewer halo exchanges per step generally improve scaling.
- CUDA versions benefit from larger work batches per kernel launch and minimizing host–device transfers.
- See `parallel_rk4_report.pdf` for detailed benchmarks and methodology.

## Validation
Compare outputs between implementations (e.g., serial vs. OpenMP, MPI vs. CUDA) on small domains to confirm numerical agreement before running large jobs.

## SLURM usage
Templates in `slurm/` provide a starting point for cluster runs. Edit node counts, tasks per node, GPU requests, module loads, and environment variables to match your scheduler setup before submission.

## Configuration tips
- Use environment variables like `OMP_NUM_THREADS`, `CUDA_VISIBLE_DEVICES`, and MPI rank placement flags to match hardware topology.
- Adjust grid sizes, time step, and iteration counts in code or via command-line options (if implemented) to balance accuracy and runtime.

## License
Add your preferred license if you plan to distribute or publish results.
