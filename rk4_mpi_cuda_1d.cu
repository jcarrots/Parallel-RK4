#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

__global__ void matvec_local(const double *L, const double *x, double *y,
                             int M_loc, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M_loc) {
        const double *row = L + (size_t)i * D;
        double sum = 0.0;
        for (int j = 0; j < D; ++j) sum += row[j] * x[j];
        y[i] = sum;
    }
}

__global__ void copy_vec(const double *x, double *y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = x[i];
}

__global__ void axpy_vec(double alpha, const double *x, double *y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] += alpha * x[i];
}

__global__ void rk4_update(double *r, const double *k1, const double *k2,
                           const double *k3, const double *k4,
                           double dt, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        r[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // --------------------------
    // Multi-GPU per node setup
    // --------------------------
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &node_comm);

    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);

    int ngpus = 0;
    cudaGetDeviceCount(&ngpus);
    if (ngpus == 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: no CUDA devices found on this node.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dev_id = local_rank % ngpus;
    cudaSetDevice(dev_id);

    if (rank == 0) {
        printf("Total MPI ranks = %d, GPUs per node (visible to rank 0) = %d\n",
               nranks, ngpus);
    }
    printf("Rank %d: local_rank=%d uses device %d\n", rank, local_rank, dev_id);
    fflush(stdout);

    // --------------------------
    // Argument parsing & setup
    // --------------------------
    int D = 1 << 12;
    double T = 1.0, dt = 1e-3;
    int nsteps = -1;
    const char *rho_out = nullptr;
    const char *timing_out = nullptr;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--D") == 0 && i + 1 < argc) {
                D = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) {
                T = atof(argv[++i]);
            } else if (strcmp(argv[i], "--nsteps") == 0 && i + 1 < argc) {
                nsteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc) {
                dt = atof(argv[++i]);
            } else if (strcmp(argv[i], "--rho-out") == 0 && i + 1 < argc) {
                rho_out = argv[++i];
            } else if (strcmp(argv[i], "--timing-out") == 0 && i + 1 < argc) {
                timing_out = argv[++i];
            } else {
                fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            }
        }
    }

    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nsteps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (nsteps > 0) {
        T = nsteps * dt;
    } else {
        nsteps = int(std::round(T / dt));
    }

    if (D < nranks) {
        if (rank == 0) {
            fprintf(stderr, "Error: D (%d) must be >= nranks (%d)\n", D, nranks);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --------------------------
    // Row partitioning
    // --------------------------
    int base = D / nranks;
    int rem  = D % nranks;
    int offset = 0;
    std::vector<int> rows(nranks), displs(nranks);
    for (int p = 0; p < nranks; ++p) {
        rows[p]   = base + (p < rem ? 1 : 0);
        displs[p] = offset;
        offset   += rows[p];
    }
    int M_loc = rows[rank];

    size_t Lbytes      = (size_t)M_loc * D * sizeof(double);
    size_t vloc_bytes  = (size_t)M_loc * sizeof(double);
    size_t vglob_bytes = (size_t)D * sizeof(double);

    // --------------------------
    // Host buffers
    // --------------------------
    std::vector<double> L_local(Lbytes / sizeof(double));
    std::vector<double> r_local(M_loc, 1.0);
    std::vector<double> tmp_local(M_loc);
    std::vector<double> r_global(D);

    // Init diagonal L
    for (int i = 0; i < M_loc; ++i) {
        int gi = displs[rank] + i;
        double *row = L_local.data() + (size_t)i * D;
        for (int j = 0; j < D; ++j) {
            row[j] = (gi == j) ? (-1.0 - 0.1 * gi) : 0.0;
        }
    }

    // --------------------------
    // Device buffers
    // --------------------------
    double *dL         = nullptr;
    double *dr_global  = nullptr;
    double *dk1        = nullptr;
    double *dk2        = nullptr;
    double *dk3        = nullptr;
    double *dk4        = nullptr;
    double *dtmp       = nullptr;
    double *dr_local   = nullptr;

    cudaMalloc(&dL,        Lbytes);
    cudaMalloc(&dr_global, vglob_bytes);
    cudaMalloc(&dk1,       vloc_bytes);
    cudaMalloc(&dk2,       vloc_bytes);
    cudaMalloc(&dk3,       vloc_bytes);
    cudaMalloc(&dk4,       vloc_bytes);
    cudaMalloc(&dtmp,      vloc_bytes);
    cudaMalloc(&dr_local,  vloc_bytes);

    cudaMemcpy(dL,        L_local.data(), Lbytes,      cudaMemcpyHostToDevice);
    cudaMemcpy(dr_local,  r_local.data(), vloc_bytes,  cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid_loc ((M_loc + block.x - 1) / block.x);
    dim3 grid_glob((D      + block.x - 1) / block.x);  // (currently unused; kept for clarity)

    // --------------------------
    // Time stepping
    // --------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int n = 0; n < nsteps; ++n) {
        // k1
        MPI_Allgatherv(r_local.data(), M_loc, MPI_DOUBLE,
                       r_global.data(), rows.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        cudaMemcpy(dr_global, r_global.data(), vglob_bytes, cudaMemcpyHostToDevice);
        matvec_local<<<grid_loc, block>>>(dL, dr_global, dk1, M_loc, D);

        // k2
        copy_vec<<<grid_loc, block>>>(dr_local, dtmp, M_loc);
        axpy_vec <<<grid_loc, block>>>(0.5 * dt, dk1, dtmp, M_loc);
        cudaMemcpy(tmp_local.data(), dtmp, vloc_bytes, cudaMemcpyDeviceToHost);

        MPI_Allgatherv(tmp_local.data(), M_loc, MPI_DOUBLE,
                       r_global.data(), rows.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        cudaMemcpy(dr_global, r_global.data(), vglob_bytes, cudaMemcpyHostToDevice);
        matvec_local<<<grid_loc, block>>>(dL, dr_global, dk2, M_loc, D);

        // k3
        copy_vec<<<grid_loc, block>>>(dr_local, dtmp, M_loc);
        axpy_vec <<<grid_loc, block>>>(0.5 * dt, dk2, dtmp, M_loc);
        cudaMemcpy(tmp_local.data(), dtmp, vloc_bytes, cudaMemcpyDeviceToHost);

        MPI_Allgatherv(tmp_local.data(), M_loc, MPI_DOUBLE,
                       r_global.data(), rows.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        cudaMemcpy(dr_global, r_global.data(), vglob_bytes, cudaMemcpyHostToDevice);
        matvec_local<<<grid_loc, block>>>(dL, dr_global, dk3, M_loc, D);

        // k4
        copy_vec<<<grid_loc, block>>>(dr_local, dtmp, M_loc);
        axpy_vec <<<grid_loc, block>>>(dt, dk3, dtmp, M_loc);
        cudaMemcpy(tmp_local.data(), dtmp, vloc_bytes, cudaMemcpyDeviceToHost);

        MPI_Allgatherv(tmp_local.data(), M_loc, MPI_DOUBLE,
                       r_global.data(), rows.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        cudaMemcpy(dr_global, r_global.data(), vglob_bytes, cudaMemcpyHostToDevice);
        matvec_local<<<grid_loc, block>>>(dL, dr_global, dk4, M_loc, D);

        // Update r
        rk4_update<<<grid_loc, block>>>(dr_local, dk1, dk2, dk3, dk4, dt, M_loc);
        cudaMemcpy(r_local.data(), dr_local, vloc_bytes, cudaMemcpyDeviceToHost);
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time   = 0.0;

    double *r_final = nullptr;
    if (rank == 0) {
        r_final = (double *)malloc((size_t)D * sizeof(double));
    }

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gatherv(r_local.data(), M_loc, MPI_DOUBLE,
                r_final, rows.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // --------------------------
    // Output & cleanup
    // --------------------------
    if (rank == 0) {
        printf("Max wall time over ranks for %d steps: %.6f s\n",
               nsteps, max_time);

        double err2 = 0.0;
        for (int i = 0; i < D; ++i) {
            double lambda_i = -1.0 - 0.1 * i;
            double exact    = std::exp(lambda_i * T);
            double diff     = r_final[i] - exact;
            err2 += diff * diff;
        }
        double l2err = std::sqrt(err2);
        printf("L2 error at T = %g is %e\n", T, l2err);

        if (timing_out) {
            FILE *ft = fopen(timing_out, "w");
            if (ft) {
                fprintf(ft,
                        "D,%d\nT,%.17g\ndt,%.17g\nnsteps,%d\nmax_time_s,%.9f\nl2_error,%e\n",
                        D, T, dt, nsteps, max_time, l2err);
                fclose(ft);
            } else {
                fprintf(stderr, "Warning: could not open timing file %s\n",
                        timing_out);
            }
        }
        if (rho_out) {
            FILE *fr = fopen(rho_out, "w");
            if (fr) {
                for (int i = 0; i < D; ++i) {
                    fprintf(fr, "%.17g\n", r_final[i]);
                }
                fclose(fr);
            } else {
                fprintf(stderr, "Warning: could not open rho file %s\n",
                        rho_out);
            }
        }
        free(r_final);
    }

    cudaFree(dL);
    cudaFree(dr_global);
    cudaFree(dk1);
    cudaFree(dk2);
    cudaFree(dk3);
    cudaFree(dk4);
    cudaFree(dtmp);
    cudaFree(dr_local);

    MPI_Finalize();
    return 0;
}
