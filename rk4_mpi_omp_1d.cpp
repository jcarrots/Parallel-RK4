#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include <omp.h>

void matvec_omp_mpi_local(const double *L,
                          const double *x,
                          double *y,
                          int M_loc,
                          int D) {
    #pragma omp for schedule(static)
    for (int i = 0; i < M_loc; ++i) {
        double sum = 0.0;
        const double *row = L + (size_t)i * D;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < D; ++j) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

void vec_axpy_omp(double alpha,
                  const double *x,
                  double *y,
                  int D) {
    #pragma omp for simd schedule(static)
    for (int i = 0; i < D; ++i) {
        y[i] += alpha * x[i];
    }
}

void vec_copy_omp(const double *x,
                  double *y,
                  int D) {
    #pragma omp for simd schedule(static)
    for (int i = 0; i < D; ++i) {
        y[i] = x[i];
    }
}

void rk4_step_mpi_omp(const double *L_local,
                      double *r_local,
                      double *k1_local,
                      double *k2_local,
                      double *k3_local,
                      double *k4_local,
                      double *tmp_local,
                      double *r_global,
                      int D,
                      double dt,
                      int M_loc,
                      const int *recvcounts,
                      const int *displs,
                      MPI_Comm comm) {
    #pragma omp parallel
    {
    // k1 = L r
    #pragma omp single
    MPI_Allgatherv(r_local, M_loc, MPI_DOUBLE,
                   r_global, recvcounts, displs, MPI_DOUBLE,
                   comm);
    matvec_omp_mpi_local(L_local, r_global, k1_local, M_loc, D);

    // tmp = r + dt/2 * k1
    vec_copy_omp(r_local, tmp_local, M_loc);
    vec_axpy_omp(0.5 * dt, k1_local, tmp_local, M_loc);

    // k2 = L (r + dt/2 k1)
    #pragma omp single
    MPI_Allgatherv(tmp_local, M_loc, MPI_DOUBLE,
                   r_global, recvcounts, displs, MPI_DOUBLE,
                   comm);
   #pragma omp barrier 
    matvec_omp_mpi_local(L_local, r_global, k2_local, M_loc, D);
   
    // tmp = r + dt/2 * k2
    vec_copy_omp(r_local, tmp_local, M_loc);
    vec_axpy_omp(0.5 * dt, k2_local, tmp_local, M_loc);

    // k3 = L (r + dt/2 k2)
    #pragma omp single
    MPI_Allgatherv(tmp_local, M_loc, MPI_DOUBLE,
                   r_global, recvcounts, displs, MPI_DOUBLE,
                   comm);
   #pragma omp barrier 
    matvec_omp_mpi_local(L_local, r_global, k3_local, M_loc, D);

    // tmp = r + dt * k3
    vec_copy_omp(r_local, tmp_local, M_loc);
    vec_axpy_omp(dt, k3_local, tmp_local, M_loc);

    // k4 = L (r + dt k3)
    #pragma omp single
    MPI_Allgatherv(tmp_local, M_loc, MPI_DOUBLE,
                   r_global, recvcounts, displs, MPI_DOUBLE,
                   comm);
   #pragma omp barrier 
    matvec_omp_mpi_local(L_local, r_global, k4_local, M_loc, D);

    // r = r + dt/6 (k1 + 2 k2 + 2 k3 + k4)
    #pragma omp for simd schedule(static)
    for (int i = 0; i < M_loc; ++i) {
        r_local[i] += (dt / 6.0) * (k1_local[i]
                                    + 2.0 * k2_local[i]
                                    + 2.0 * k3_local[i]
                                    + k4_local[i]);
    }
   }
}

int main(int argc, char **argv) {
    int provided;
    int required = MPI_THREAD_SERIALIZED;   // safer with omp single + MPI
    MPI_Init_thread(&argc, &argv, required, &provided);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (provided < MPI_THREAD_SERIALIZED) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI does not support required threading level (SERIALIZED)\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int    D       = 1 << 12;
    double T       = 1.0;
    double dt      = 1e-3;
    int    nsteps  = -1;
    const char *rho_out    = nullptr;
    const char *timing_out = nullptr;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--D") == 0 && i + 1 < argc) {
                D = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--T") == 0 && i + 1 < argc) {
                T = std::atof(argv[++i]);
            } else if (std::strcmp(argv[i], "--nsteps") == 0 && i + 1 < argc) {
                nsteps = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--dt") == 0 && i + 1 < argc) {
                dt = std::atof(argv[++i]);
            } else if (std::strcmp(argv[i], "--rho-out") == 0 && i + 1 < argc) {
                rho_out = argv[++i];
            } else if (std::strcmp(argv[i], "--timing-out") == 0 && i + 1 < argc) {
                timing_out = argv[++i];
            } else {
                std::fprintf(stderr,
                             "Unknown or incomplete argument: %s\n", argv[i]);
            }
        }
    }

    // derive nsteps / T
    if (nsteps > 0) {
        T = nsteps * dt;
    } else {
        nsteps = (int)std::round(T / dt);
    }

    // broadcast parameters
    MPI_Bcast(&D,       1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&T,       1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nsteps,  1, MPI_INT,    0, MPI_COMM_WORLD);

    if (D < nranks) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Error: D (%d) must be >= nranks (%d)\n", D, nranks);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int base   = D / nranks;
    int rem    = D % nranks;
    int offset = 0;

    int *rows    = (int*)std::malloc(nranks * sizeof(int));
    int *rdispls = (int*)std::malloc(nranks * sizeof(int));

    for (int p = 0; p < nranks; ++p) {
        rows[p]    = base + (p < rem ? 1 : 0);
        rdispls[p] = offset;
        offset    += rows[p];
    }

    int M_loc = rows[rank];
    MPI_LONG_LONG_INT *sendcounts_L = (MPI_Count*)std::malloc(nranks * sizeof(MPI_Count));
    MPI_Aint  *displs_L     = (MPI_Aint*) std::malloc(nranks * sizeof(MPI_Aint));

    if (!sendcounts_L || !displs_L) {
         std::fprintf(stderr, "Allocation failed for sendcounts_L/displs_L on rank %d\n", rank);
         MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int p = 0; p < nranks; ++p) {
          long long count_ll = (long long)rows[p] * (long long)D;
          long long disp_ll  = (long long)rdispls[p] * (long long)D;

        if (count_ll < 0 || disp_ll < 0) {
            if (rank == 0) {
                std::fprintf(stderr,
                            "Negative count/disp at D=%d: p=%d count=%lld disp=%lld\n",
                             D, p, count_ll, disp_ll);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

           sendcounts_L[p] = (MPI_LONG_LONG_INT)count_ll;
           displs_L[p]     = (MPI_Aint) disp_ll;
    }
	

    double *L_global      = nullptr;
    double *r_global_root = nullptr;

    double *L_local   = (double*)std::malloc((size_t)M_loc * D * sizeof(double));
    double *r_local   = (double*)std::malloc((size_t)M_loc * sizeof(double));
    double *k1_local  = (double*)std::malloc((size_t)M_loc * sizeof(double));
    double *k2_local  = (double*)std::malloc((size_t)M_loc * sizeof(double));
    double *k3_local  = (double*)std::malloc((size_t)M_loc * sizeof(double));
    double *k4_local  = (double*)std::malloc((size_t)M_loc * sizeof(double));
    double *tmp_local = (double*)std::malloc((size_t)M_loc * sizeof(double));
    double *r_global  = (double*)std::malloc((size_t)D * sizeof(double));

    if (rank == 0) {
        L_global      = (double*)std::malloc((size_t)D * D * sizeof(double));
        r_global_root = (double*)std::malloc((size_t)D * sizeof(double));

        if (!L_global || !r_global_root) {
            std::fprintf(stderr, "Initial allocation failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                L_global[(size_t)i * D + j] =
                    (i == j) ? (-1.0 - 0.1 * i) : 0.0;
            }
            r_global_root[i] = 1.0;
        }
        std::printf("MPI initialized with %d ranks, thread support level %d\n",
                    nranks, provided);
    }

    if (!L_local || !r_local || !k1_local || !k2_local ||
        !k3_local || !k4_local || !tmp_local || !r_global) {
        std::fprintf(stderr, "Initial allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatter matrix rows
 //   MPI_Scatterv(L_global, sendcounts_L, displs_L, MPI_DOUBLE,
 //                L_local,  M_loc * D,    MPI_DOUBLE,
 //                0, MPI_COMM_WORLD);

/* === SCATTER MATRIX USING LARGE COUNTS === */
int err = MPI_Scatterv_c(
    L_global,
    sendcounts_L, displs_L, MPI_DOUBLE,
    L_local,
    recvcount_L, MPI_DOUBLE,
    0, MPI_COMM_WORLD
);

if (err != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Scatterv_c for L_global failed on rank %d\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

    // Scatter initial r
    MPI_Scatterv(r_global_root, rows, rdispls, MPI_DOUBLE,
                 r_local,       M_loc,        MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::free(L_global);
        std::free(r_global_root);
    }

    double t = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    MPI_Comm comm = MPI_COMM_WORLD;

    #pragma omp parallel default(none) \
        shared(L_local, r_local, k1_local, k2_local, k3_local, k4_local, \
               tmp_local, r_global, D, dt, M_loc, rows, rdispls, nsteps, comm) \
        reduction(+:t)
    {
        for (int n = 0; n < nsteps; ++n) {
            rk4_step_mpi_omp(L_local,
                             r_local,
                             k1_local,
                             k2_local,
                             k3_local,
                             k4_local,
                             tmp_local,
                             r_global,
                             D,
                             dt,
                             M_loc,
                             rows,      // recvcounts
                             rdispls,   // displs
                             comm);
            #pragma omp single
            t += dt;
        }
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time   = 0.0;

    double *r_final = nullptr;
    if (rank == 0) {
        r_final = (double*)std::malloc((size_t)D * sizeof(double));
    }

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Gatherv(r_local, M_loc, MPI_DOUBLE,
                r_final, rows, rdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Max wall time over ranks for %d steps: %.6f s\n",
                    nsteps, max_time);

        if (timing_out) {
            FILE *ft = std::fopen(timing_out, "w");
            if (ft) {
                std::fprintf(ft,
                             "D,%d\nT,%.17g\ndt,%.17g\nnsteps,%d\nmax_time_s,%.9f\n",
                             D, T, dt, nsteps, max_time);
                if (r_final) {
                    double err2 = 0.0;
                    for (int i = 0; i < D; ++i) {
                        double lambda_i = -1.0 - 0.1 * i;
                        double exact    = std::exp(lambda_i * T);
                        double diff     = r_final[i] - exact;
                        err2 += diff * diff;
                    }
                    std::fprintf(ft, "l2_error,%e\n", std::sqrt(err2));
                }
                std::fclose(ft);
            } else {
                std::fprintf(stderr,
                             "Warning: could not open timing file %s\n",
                             timing_out);
            }
        }

        double err2 = 0.0;
        for (int i = 0; i < D; ++i) {
            double lambda_i = -1.0 - 0.1 * i;
            double exact    = std::exp(lambda_i * T);
            double diff     = r_final[i] - exact;
            err2 += diff * diff;
        }
        std::printf("L2 error at T = %g is %e\n", T, std::sqrt(err2));

        if (rho_out) {
            FILE *fr = std::fopen(rho_out, "w");
            if (fr) {
                for (int i = 0; i < D; ++i) {
                    std::fprintf(fr, "%.17g\n", r_final[i]);
                }
                std::fclose(fr);
            } else {
                std::fprintf(stderr,
                             "Warning: could not open rho file %s\n",
                             rho_out);
            }
        }

        std::free(r_final);
    }

    std::free(L_local);
    std::free(r_local);
    std::free(k1_local);
    std::free(k2_local);
    std::free(k3_local);
    std::free(k4_local);
    std::free(tmp_local);
    std::free(r_global);
    std::free(rows);
    std::free(rdispls);
    std::free(sendcounts_L);
    std::free(displs_L);

    MPI_Finalize();
    return 0;
}
