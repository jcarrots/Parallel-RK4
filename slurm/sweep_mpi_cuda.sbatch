#!/bin/bash
#SBATCH -J rk4-mpi-cuda-sweep
#SBATCH -N 1                       # 1 node
#SBATCH --ntasks-per-node=4        # up to 4 MPI ranks per node
#SBATCH --gres=gpu:h100:4          # up to 4 H100 GPUs on this node
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --output=logs/rk4-mpi-cuda-sweep-%j.out

set -euo pipefail

module purge
module load gcc cuda openmpi

cd "$SLURM_SUBMIT_DIR"

mkdir -p runs/sweep_mpi_cuda logs

# Build MPI+CUDA executable if needed
if [ ! -x ./rk4_mpi_cuda ]; then
    echo "Building rk4_mpi_cuda..."
    nvcc -ccbin mpicxx -O3 -std=c++17 -o rk4_mpi_cuda rk4_mpi_cuda.cu -lmpi
fi

# Time-stepping parameters
DT=1e-3
NSTEPS=20               # T = NSTEPS * DT; adjust if you want

# GPUs to test
GPU_LIST="1 2 4"

# Problem sizes: 2^4 ... 2^16
EXP_LIST="4 5 6 7 8 9 10 11 12 13 14 15 16"

for g in $GPU_LIST; do
    echo "========== Testing with $g GPU(s) =========="
    for e in $EXP_LIST; do
        D=$((1 << e))
        OUTFILE="runs/sweep_mpi_cuda/rk4_mpi_cuda_g${g}_D${D}.csv"

        echo "Running: GPUs=${g}, D=${D} -> ${OUTFILE}"

        # One MPI rank per GPU
        export OMP_NUM_THREADS=1   # just to be safe; no OpenMP here

        srun -n "$g" --gpus-per-task=1 ./rk4_mpi_cuda \
            --D "$D" --dt "$DT" --nsteps "$NSTEPS" \
            --timing-out "$OUTFILE"

        echo "Done: GPUs=${g}, D=${D}"
    done
done

echo "All runs finished."
