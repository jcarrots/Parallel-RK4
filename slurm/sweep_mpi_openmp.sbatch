#!/bin/bash
#SBATCH -J rk4-mpi-omp-sweep
#SBATCH -N 2                       # up to 2 nodes
#SBATCH --ntasks-per-node=8        # 8 MPI ranks per node max
#SBATCH -c 4                       # 4 OpenMP threads per rank
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --output=logs/rk4-mpi-omp-sweep-%j.out

set -euo pipefail

module purge
module load gcc openmpi

# Run from the directory where you did sbatch
cd "$SLURM_SUBMIT_DIR"

mkdir -p runs/sweep_mpi_omp logs

# Build MPI+OpenMP code if needed
# Assuming your source is rk4_mpi_omp.cpp
if [ ! -x ./rk4_mpi_omp ]; then
    echo "Building rk4_mpi_omp..."
    mpicxx -O3 -fopenmp -std=c++17 -o rk4_mpi_omp rk4_mpi_omp.cpp
fi

# Time-stepping parameters
DT=1e-3
NSTEPS=20              # T = 0.02

# MPI ranks
MPI_RANKS_LIST="1 2 4 8 16"

# OpenMP threads per rank to sweep
OMP_THREADS_LIST="1 2 4"

# Problem sizes: 2^4 ... 2^16
EXP_LIST="4 5 6 7 8 9 10 11 12 13 14 15"

for ranks in $MPI_RANKS_LIST; do
    for t in $OMP_THREADS_LIST; do
        echo "========== MPI ranks = $ranks, OMP_NUM_THREADS = $t =========="
        export OMP_NUM_THREADS=$t

        for e in $EXP_LIST; do
            D=$((1 << e))
            OUTFILE="runs/sweep_mpi_omp/rk4_mpi_omp_r${ranks}_t${t}_D${D}.csv"

            echo "Running: ranks=${ranks}, threads=${t}, D=${D}  ->  ${OUTFILE}"

            srun -n "$ranks" --cpus-per-task="$t" ./rk4_mpi_omp \
                --D "$D" --dt "$DT" --nsteps "$NSTEPS" \
                --timing-out "$OUTFILE"

            echo "Done: ranks=${ranks}, threads=${t}, D=${D}"
        done
    done
done

echo "All MPI+OpenMP sweep runs finished."
