# Final Projects

## Project Reports
Report: [CSE_6230_final_project.pdf](./CSE_6230_final_project.pdf)
## Problem: 
Runge Kutta 4th order - row-wise data decomposition parallelism.
## Implementation: 
MPI+OPENMP(rk4_mpi_omp.cpp); MPI+CUDA(rk4_mpi_cuda.cu)
## Experiments: 
Problem parameters: 
L is diagonal with L_ii=1-0.1i, which has exact solution; 
D=2^{4,5,6,7,8,9,10,11,12,13,14,15}; dt=0.001; nstep=20;
Simulation parameters:
MPI+OPENMP: sweep p={1,2,4,8,16}, t={1,2,4}; 
MPI+CUDA: sweep p={1,2,4}, g={1}; Not enough gpus to see meaningful acceleration. 
## Results: 
under /runs
