#include <mpi.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t _e = (call);                                  \
    if (_e != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(_e));  \
        MPI_Abort(MPI_COMM_WORLD, 1);                         \
    }                                                         \
} while(0)

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

__global__ void init_diag_block(double* A, int m, int n, int rowStart, int colStart) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    int gi = rowStart + i;
    if (gi >= colStart && gi < colStart + n) {
        int j = gi - colStart;
        A[(size_t)i * n + j] = -1.0 - 0.1 * (double)gi;
    }
}

__global__ void matvec_block(const double* __restrict__ A,
                             const double* __restrict__ x,
                             double* __restrict__ y,
                             int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const double* row = A + (size_t)i * n;
    double sum = 0.0;
    for (int j = 0; j < n; ++j) sum += row[j] * x[j];
    y[i] = sum;
}

// Transpose-redistribute:
//   row roots are ranks (pr=i, pc=0) holding y_row block [rowStart[i], rowEnd[i])
//   col owners are ranks (pr=0, pc=j) needing y_col block [colStart[j], colEnd[j])
// Sends only overlaps.
static void redistribute_row_to_col(
    MPI_Comm cart_comm,
    int pr, int pc,
    int pr_dim, int pc_dim,
    const std::vector<int>& rowStarts,
    const std::vector<int>& rowLens,
    const std::vector<int>& colStarts,
    const std::vector<int>& colLens,
    const std::vector<int>& rowRootRanks,   // size pr_dim: rank of (i,0)
    const std::vector<int>& colOwnerRanks,  // size pc_dim: rank of (0,j)
    const double* y_row,                    // valid on (pr=i,pc=0), length rowLens[i]
    double* y_col,                          // valid on (pr=0,pc=j), length colLens[j]
    int tag_base
) {
    std::vector<MPI_Request> reqs;

    // Post receives on column owners (pr==0)
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

    // Post sends on row roots (pc==0)
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

// One 2D matvec y = L x
// x is owned on (pr==0) as column blocks; y is returned on (pr==0) as column blocks.
static void matvec_2d_cuda(
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
    const double* x_col_owner, // valid only when pr==0 (length n_local)
    double* y_col_owner,       // valid only when pr==0 (length n_local)
    // local matrix block on GPU:
    const double* dA, int m_local, int n_local,
    // reusable device/host buffers:
    double* dX, double* dY,
    std::vector<double>& x_buf,           // length n_local on all ranks
    std::vector<double>& y_partial_host,  // length m_local on all ranks
    std::vector<double>& y_row_root,      // length m_local on row roots (pc==0), dummy otherwise
    std::vector<double>& y_col_recv,      // length n_local on col owners (pr==0), dummy otherwise
    int tag_base
) {
    // 1) Broadcast x segment down each column (root is pr==0 within that column)
    if (pr == 0) {
        // copy x_col_owner -> x_buf
        std::copy(x_col_owner, x_col_owner + n_local, x_buf.begin());
    }
    MPI_Bcast(x_buf.data(), n_local, MPI_DOUBLE, 0, col_comm);

    // 2) Local GPU compute: y_partial = A_block * x_buf
    CUDA_CHECK(cudaMemcpy(dX, x_buf.data(), (size_t)n_local * sizeof(double), cudaMemcpyHostToDevice));

    int block = 256;
    int grid  = (m_local + block - 1) / block;
    matvec_block<<<grid, block>>>(dA, dX, dY, m_local, n_local);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(y_partial_host.data(), dY, (size_t)m_local * sizeof(double), cudaMemcpyDeviceToHost));

    // 3) Reduce across row to root pc==0: y_row_root
    if (pc == 0) {
        MPI_Reduce(y_partial_host.data(), y_row_root.data(), m_local, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    } else {
        MPI_Reduce(y_partial_host.data(), nullptr,        m_local, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    }

    // 4) Redistribute row-block y to column-block owners on pr==0
    if (pr == 0) {
        // receive buffer y_col_recv already sized
        std::fill(y_col_recv.begin(), y_col_recv.end(), 0.0);
    }
    const double* y_row_ptr = (pc == 0) ? y_row_root.data() : nullptr;
    double* y_col_ptr = (pr == 0) ? y_col_recv.data() : nullptr;

    redistribute_row_to_col(
        cart_comm,
        pr, pc,
        pr_dim, pc_dim,
        rowStarts, rowLens,
        colStarts, colLens,
        rowRootRanks, colOwnerRanks,
        y_row_ptr, y_col_ptr,
        tag_base
    );

    // 5) On pr==0, copy recv -> output
    if (pr == 0) {
        std::copy(y_col_recv.begin(), y_col_recv.end(), y_col_owner);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Pick GPU by local rank on node (safe for multi-GPU nodes)
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (ngpu > 0) CUDA_CHECK(cudaSetDevice(local_rank % ngpu));
    MPI_Comm_free(&node_comm);

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

    // Row/col communicators
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart_comm, pr, pc, &row_comm);
    MPI_Comm_split(cart_comm, pc, pr, &col_comm);

    // Row/col partitions
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

    // Precompute ranks of row roots (i,0) and col owners (0,j)
    std::vector<int> rowRootRanks(pr_dim), colOwnerRanks(pc_dim);
    for (int r = 0; r < pr_dim; ++r) {
        int cc[2] = {r, 0};
        MPI_Cart_rank(cart_comm, cc, &rowRootRanks[r]);
    }
    for (int c = 0; c < pc_dim; ++c) {
        int cc[2] = {0, c};
        MPI_Cart_rank(cart_comm, cc, &colOwnerRanks[c]);
    }

    // Allocate local matrix block on GPU and initialize diagonal entries
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)m_local * n_local * sizeof(double)));
    CUDA_CHECK(cudaMemset(dA, 0, (size_t)m_local * n_local * sizeof(double)));
    {
        int block = 256;
        int grid  = (m_local + block - 1) / block;
        init_diag_block<<<grid, block>>>(dA, m_local, n_local, rowStart, colStart);
        CUDA_CHECK(cudaGetLastError());
    }

    // Device buffers for x and y_partial
    double* dX = nullptr;
    double* dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, (size_t)n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dY, (size_t)m_local * sizeof(double)));

    // Host buffers used by all ranks during matvec
    std::vector<double> x_buf(n_local, 0.0);
    std::vector<double> y_partial_host(m_local, 0.0);

    // y_row_root only meaningful on row roots (pc==0)
    std::vector<double> y_row_root((pc == 0) ? m_local : 1, 0.0);

    // y_col_recv only meaningful on column owners (pr==0)
    std::vector<double> y_col_recv((pr == 0) ? n_local : 1, 0.0);

    // RK4 vectors: ONLY stored on column owners (pr==0)
    std::vector<double> r_seg((pr == 0) ? n_local : 1, 1.0);
    std::vector<double> tmp_seg((pr == 0) ? n_local : 1, 0.0);
    std::vector<double> k1_seg((pr == 0) ? n_local : 1, 0.0);
    std::vector<double> k2_seg((pr == 0) ? n_local : 1, 0.0);
    std::vector<double> k3_seg((pr == 0) ? n_local : 1, 0.0);
    std::vector<double> k4_seg((pr == 0) ? n_local : 1, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // RK4 time stepping
    for (int step = 0; step < nsteps; ++step) {
        // k1 = L r
        matvec_2d_cuda(cart_comm, row_comm, col_comm, pr, pc,
                       pr_dim, pc_dim, D,
                       rowStarts, rowLens, colStarts, colLens,
                       rowRootRanks, colOwnerRanks,
                       (pr == 0 ? r_seg.data() : nullptr),
                       (pr == 0 ? k1_seg.data() : nullptr),
                       dA, m_local, n_local,
                       dX, dY,
                       x_buf, y_partial_host, y_row_root, y_col_recv,
                       1000);

        if (pr == 0) {
            for (int i = 0; i < n_local; ++i) tmp_seg[i] = r_seg[i] + 0.5 * dt * k1_seg[i];
        }

        // k2 = L (r + dt/2 k1)
        matvec_2d_cuda(cart_comm, row_comm, col_comm, pr, pc,
                       pr_dim, pc_dim, D,
                       rowStarts, rowLens, colStarts, colLens,
                       rowRootRanks, colOwnerRanks,
                       (pr == 0 ? tmp_seg.data() : nullptr),
                       (pr == 0 ? k2_seg.data() : nullptr),
                       dA, m_local, n_local,
                       dX, dY,
                       x_buf, y_partial_host, y_row_root, y_col_recv,
                       2000);

        if (pr == 0) {
            for (int i = 0; i < n_local; ++i) tmp_seg[i] = r_seg[i] + 0.5 * dt * k2_seg[i];
        }

        // k3 = L (r + dt/2 k2)
        matvec_2d_cuda(cart_comm, row_comm, col_comm, pr, pc,
                       pr_dim, pc_dim, D,
                       rowStarts, rowLens, colStarts, colLens,
                       rowRootRanks, colOwnerRanks,
                       (pr == 0 ? tmp_seg.data() : nullptr),
                       (pr == 0 ? k3_seg.data() : nullptr),
                       dA, m_local, n_local,
                       dX, dY,
                       x_buf, y_partial_host, y_row_root, y_col_recv,
                       3000);

        if (pr == 0) {
            for (int i = 0; i < n_local; ++i) tmp_seg[i] = r_seg[i] + dt * k3_seg[i];
        }

        // k4 = L (r + dt k3)
        matvec_2d_cuda(cart_comm, row_comm, col_comm, pr, pc,
                       pr_dim, pc_dim, D,
                       rowStarts, rowLens, colStarts, colLens,
                       rowRootRanks, colOwnerRanks,
                       (pr == 0 ? tmp_seg.data() : nullptr),
                       (pr == 0 ? k4_seg.data() : nullptr),
                       dA, m_local, n_local,
                       dX, dY,
                       x_buf, y_partial_host, y_row_root, y_col_recv,
                       4000);

        // r = r + dt/6*(k1+2k2+2k3+k4)
        if (pr == 0) {
            for (int i = 0; i < n_local; ++i) {
                r_seg[i] += (dt / 6.0) * (k1_seg[i] + 2.0*k2_seg[i] + 2.0*k3_seg[i] + k4_seg[i]);
            }
        }
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;

    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather final r onto rank (0,0) in the cart grid
    int root_cart_rank = 0;
    {
        int cc[2] = {0, 0};
        MPI_Cart_rank(cart_comm, cc, &root_cart_rank);
    }

    // Build communicator for row0 (pr==0)
    MPI_Comm row0_comm = MPI_COMM_NULL;
    if (pr == 0) {
        MPI_Comm_split(cart_comm, 0, pc, &row0_comm);
    } else {
        MPI_Comm_split(cart_comm, MPI_UNDEFINED, pc, &row0_comm);
    }

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

    double l2err = 0.0;
    if (world_rank == root_cart_rank) {
        double err2 = 0.0;
        for (int i = 0; i < D; ++i) {
            double lambda_i = -1.0 - 0.1 * i;
            double exact = exp(lambda_i * T);
            double diff = r_final[i] - exact;
            err2 += diff * diff;
        }
        l2err = sqrt(err2);

        printf("2D MPI+CUDA grid %dx%d, D=%d, steps=%d\n", pr_dim, pc_dim, D, nsteps);
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

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
