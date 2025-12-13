#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static inline int block_len(int N, int P, int coord) {
    int base = N / P;
    int rem  = N % P;
    return base + (coord < rem ? 1 : 0);
}
static inline int block_start(int N, int P, int coord) {
    int base = N / P;
    int rem  = N % P;
    return coord * base + std::min(coord, rem);
}

// Same overlap-based transpose used in CUDA version
static void redistribute_row_to_col(
    MPI_Comm cart_comm,
    int pr, int pc,
    int pr_dim, int pc_dim,
    const std::vector<int>& rowStarts,
    const std::vector<int>& rowLens,
    const std::vector<int>& colStarts,
    const std::vector<int>& colLens,
    const std::vector<int>& rowRootRanks,
    const std::vector<int>& colOwnerRanks,
    const double* y_row,  // valid on (pr=i,pc=0)
    double* y_col,        // valid on (pr=0,pc=j)
    int tag_base
) {
    std::vector<MPI_Request> reqs;

    if (pr == 0) {
        int j = pc;
        int cStart = colStarts[j];
        int cEnd   = cStart + colLens[j];

        for (int i = 0; i < pr_dim; ++i) {
            int rStart = rowStarts[i];
            int rEnd   = rStart + rowLens[i];
            int s = std::max(rStart, cStart);
            int e = std::min(rEnd, cEnd);
            int len = e - s;
            if (len > 0) {
                int src = rowRootRanks[i];
                int dst_off = s - cStart;
                int tag = tag_base + i * pc_dim + j;
                MPI_Request rq;
                MPI_Irecv(y_col + dst_off, len, MPI_DOUBLE, src, tag, cart_comm, &rq);
                reqs.push_back(rq);
            }
        }
    }

    if (pc == 0) {
        int i = pr;
        int rStart = rowStarts[i];
        int rEnd   = rStart + rowLens[i];

        for (int j = 0; j < pc_dim; ++j) {
            int cStart = colStarts[j];
            int cEnd   = cStart + colLens[j];
            int s = std::max(rStart, cStart);
            int e = std::min(rEnd, cEnd);
            int len = e - s;
            if (len > 0) {
                int dest = colOwnerRanks[j];
                int src_off = s - rStart;
                int tag = tag_base + i * pc_dim + j;
                MPI_Request rq;
                MPI_Isend(y_row + src_off, len, MPI_DOUBLE, dest, tag, cart_comm, &rq);
                reqs.push_back(rq);
            }
        }
    }

    if (!reqs.empty()) {
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }
}

// OpenMP local matvec for a block A(m x n) times x(n) -> y(m)
static void matvec_block_omp(const double* A, const double* x, double* y, int m, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const double* row = A + (size_t)i * n;
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; ++j) sum += row[j] * x[j];
        y[i] = sum;
    }
}

static void matvec_2d_omp(
    MPI_Comm cart_comm, MPI_Comm row_comm, MPI_Comm col_comm,
    int pr, int pc,
    int pr_dim, int pc_dim,
    int D,
    const std::vector<int>& rowStarts,
    const std::vector<int>& rowLens,
    const std::vector<int>& colStarts,
    const std::vector<int>& colLens,
    const std::vector<int>& rowRootRanks,
    const std::vector<int>& colOwnerRanks,
    const double* x_col_owner, // pr==0 only, length n_local
    double* y_col_owner,       // pr==0 only, length n_local
    const double* A, int m_local, int n_local,
    std::vector<double>& x_buf,           // n_local
    std::vector<double>& y_partial,       // m_local
    std::vector<double>& y_row_root,      // m_local on pc==0
    std::vector<double>& y_col_recv,      // n_local on pr==0
    int tag_base
) {
    // 1) Column bcast
    if (pr == 0) std::copy(x_col_owner, x_col_owner + n_local, x_buf.begin());
    MPI_Bcast(x_buf.data(), n_local, MPI_DOUBLE, 0, col_comm);

    // 2) Local compute
    matvec_block_omp(A, x_buf.data(), y_partial.data(), m_local, n_local);

    // 3) Row reduce to pc==0
    if (pc == 0) {
        MPI_Reduce(y_partial.data(), y_row_root.data(), m_local, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    } else {
        MPI_Reduce(y_partial.data(), nullptr,        m_local, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    }

    // 4) Redistribute row-block to col owners
    if (pr == 0) std::fill(y_col_recv.begin(), y_col_recv.end(), 0.0);
    const double* y_row_ptr = (pc == 0) ? y_row_root.data() : nullptr;
    double* y_col_ptr = (pr == 0) ? y_col_recv.data() : nullptr;

    redistribute_row_to_col(cart_comm, pr, pc, pr_dim, pc_dim,
                            rowStarts, rowLens, colStarts, colLens,
                            rowRootRanks, colOwnerRanks,
                            y_row_ptr, y_col_ptr, tag_base);

    if (pr == 0) std::copy(y_col_recv.begin(), y_col_recv.end(), y_col_owner);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int D = 1 << 12;
    double T = 1.0, dt = 1e-3;
    int nsteps = -1;
    const char* rho_out = nullptr;
    const char* timing_out = nullptr;

    if (world_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--T") && i + 1 < argc) T = atof(argv[++i]);
            else if (!strcmp(argv[i], "--dt") && i + 1 < argc) dt = atof(argv[++i]);
            else if (!strcmp(argv[i], "--nsteps") && i + 1 < argc) nsteps = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--rho-out") && i + 1 < argc) rho_out = argv[++i];
            else if (!strcmp(argv[i], "--timing-out") && i + 1 < argc) timing_out = argv[++i];
            else fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
        }
    }

    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nsteps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (nsteps > 0) T = nsteps * dt;
    else nsteps = (int)llround(T / dt);

    // 2D grid
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    int pr_dim = dims[0], pc_dim = dims[1];

    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/0, &cart_comm);

    int coords[2] = {0, 0};
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);
    int pr = coords[0], pc = coords[1];

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart_comm, pr, pc, &row_comm);
    MPI_Comm_split(cart_comm, pc, pr, &col_comm);

    std::vector<int> rowStarts(pr_dim), rowLens(pr_dim);
    std::vector<int> colStarts(pc_dim), colLens(pc_dim);
    for (int r = 0; r < pr_dim; ++r) {
        rowStarts[r] = block_start(D, pr_dim, r);
        rowLens[r]   = block_len(D, pr_dim, r);
    }
    for (int c = 0; c < pc_dim; ++c) {
        colStarts[c] = block_start(D, pc_dim, c);
        colLens[c]   = block_len(D, pc_dim, c);
    }

    int m_local = rowLens[pr];
    int n_local = colLens[pc];
    int rowStart = rowStarts[pr];
    int colStart = colStarts[pc];

    std::vector<int> rowRootRanks(pr_dim), colOwnerRanks(pc_dim);
    for (int r = 0; r < pr_dim; ++r) {
        int cc[2] = {r, 0};
        MPI_Cart_rank(cart_comm, cc, &rowRootRanks[r]);
    }
    for (int c = 0; c < pc_dim; ++c) {
        int cc[2] = {0, c};
        MPI_Cart_rank(cart_comm, cc, &colOwnerRanks[c]);
    }

    // Local matrix block (row-major) initialized as diagonal-only
    std::vector<double> A((size_t)m_local * n_local, 0.0);
    for (int i = 0; i < m_local; ++i) {
        int gi = rowStart + i;
        if (gi >= colStart && gi < colStart + n_local) {
            int j = gi - colStart;
            A[(size_t)i * n_local + j] = -1.0 - 0.1 * (double)gi;
        }
    }

    // Buffers used by all ranks
    std::vector<double> x_buf(n_local, 0.0);
    std::vector<double> y_partial(m_local, 0.0);
    std::vector<double> y_row_root((pc == 0) ? m_local : 1, 0.0);
    std::vector<double> y_col_recv((pr == 0) ? n_local : 1, 0.0);

    // RK4 vectors only on pr==0
    std::vector<double> r_seg((pr == 0) ? n_local : 1, 1.0);
    std::vector<double> tmp_seg((pr == 0) ? n_local : 1, 0.0);
    std::vector<double> k1_seg((pr == 0) ? n_local : 1, 0.0);
    std::vector<double> k2_seg((pr == 0) ? n_local : 1, 0.0);
    std::vector<double> k3_seg((pr == 0) ? n_local : 1, 0.0);
    std::vector<double> k4_seg((pr == 0) ? n_local : 1, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int step = 0; step < nsteps; ++step) {
        matvec_2d_omp(cart_comm, row_comm, col_comm, pr, pc,
                      pr_dim, pc_dim, D,
                      rowStarts, rowLens, colStarts, colLens,
                      rowRootRanks, colOwnerRanks,
                      (pr == 0 ? r_seg.data() : nullptr),
                      (pr == 0 ? k1_seg.data() : nullptr),
                      A.data(), m_local, n_local,
                      x_buf, y_partial, y_row_root, y_col_recv,
                      1000);

        if (pr == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_local; ++i) tmp_seg[i] = r_seg[i] + 0.5 * dt * k1_seg[i];
        }

        matvec_2d_omp(cart_comm, row_comm, col_comm, pr, pc,
                      pr_dim, pc_dim, D,
                      rowStarts, rowLens, colStarts, colLens,
                      rowRootRanks, colOwnerRanks,
                      (pr == 0 ? tmp_seg.data() : nullptr),
                      (pr == 0 ? k2_seg.data() : nullptr),
                      A.data(), m_local, n_local,
                      x_buf, y_partial, y_row_root, y_col_recv,
                      2000);

        if (pr == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_local; ++i) tmp_seg[i] = r_seg[i] + 0.5 * dt * k2_seg[i];
        }

        matvec_2d_omp(cart_comm, row_comm, col_comm, pr, pc,
                      pr_dim, pc_dim, D,
                      rowStarts, rowLens, colStarts, colLens,
                      rowRootRanks, colOwnerRanks,
                      (pr == 0 ? tmp_seg.data() : nullptr),
                      (pr == 0 ? k3_seg.data() : nullptr),
                      A.data(), m_local, n_local,
                      x_buf, y_partial, y_row_root, y_col_recv,
                      3000);

        if (pr == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_local; ++i) tmp_seg[i] = r_seg[i] + dt * k3_seg[i];
        }

        matvec_2d_omp(cart_comm, row_comm, col_comm, pr, pc,
                      pr_dim, pc_dim, D,
                      rowStarts, rowLens, colStarts, colLens,
                      rowRootRanks, colOwnerRanks,
                      (pr == 0 ? tmp_seg.data() : nullptr),
                      (pr == 0 ? k4_seg.data() : nullptr),
                      A.data(), m_local, n_local,
                      x_buf, y_partial, y_row_root, y_col_recv,
                      4000);

        if (pr == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_local; ++i) {
                r_seg[i] += (dt / 6.0) * (k1_seg[i] + 2.0*k2_seg[i] + 2.0*k3_seg[i] + k4_seg[i]);
            }
        }
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;

    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int root_cart_rank = 0;
    {
        int cc[2] = {0, 0};
        MPI_Cart_rank(cart_comm, cc, &root_cart_rank);
    }

    // Gather r on (0,0)
    MPI_Comm row0_comm = MPI_COMM_NULL;
    if (pr == 0) MPI_Comm_split(cart_comm, 0, pc, &row0_comm);
    else MPI_Comm_split(cart_comm, MPI_UNDEFINED, pc, &row0_comm);

    std::vector<double> r_final;
    if (world_rank == root_cart_rank) r_final.assign((size_t)D, 0.0);

    if (pr == 0) {
        std::vector<int> counts(pc_dim), displs(pc_dim);
        for (int c = 0; c < pc_dim; ++c) {
            counts[c] = colLens[c];
            displs[c] = colStarts[c];
        }
        int row0_root = 0; // pc==0
        double* recvbuf = (world_rank == root_cart_rank) ? r_final.data() : nullptr;

        MPI_Gatherv(r_seg.data(), n_local, MPI_DOUBLE,
                    recvbuf, counts.data(), displs.data(), MPI_DOUBLE,
                    row0_root, row0_comm);
        MPI_Comm_free(&row0_comm);
    }

    if (world_rank == root_cart_rank) {
        double err2 = 0.0;
        for (int i = 0; i < D; ++i) {
            double lambda_i = -1.0 - 0.1 * i;
            double exact = exp(lambda_i * T);
            double diff  = r_final[i] - exact;
            err2 += diff * diff;
        }
        double l2err = sqrt(err2);

        printf("2D MPI+OpenMP grid %dx%d, D=%d, steps=%d, OMP=%d\n",
               pr_dim, pc_dim, D, nsteps, omp_get_max_threads());
        printf("Max wall time over ranks: %.6f s\n", max_time);
        printf("L2 error at T = %g is %e\n", T, l2err);

        if (timing_out) {
            FILE* ft = fopen(timing_out, "w");
            if (ft) {
                fprintf(ft,
                        "D,%d\nT,%.17g\ndt,%.17g\nnsteps,%d\nmax_time_s,%.9f\nl2_error,%e\n",
                        D, T, dt, nsteps, max_time, l2err);
                fclose(ft);
            } else {
                fprintf(stderr, "Warning: could not open timing file %s\n", timing_out);
            }
        }

        if (rho_out) {
            FILE* fr = fopen(rho_out, "w");
            if (fr) {
                for (int i = 0; i < D; ++i) fprintf(fr, "%.17g\n", r_final[i]);
                fclose(fr);
            } else {
                fprintf(stderr, "Warning: could not open rho file %s\n", rho_out);
            }
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
