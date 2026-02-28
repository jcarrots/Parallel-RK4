// rk4_cuda.cu
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

static inline void cuda_check(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

// Dense matvec with shared-memory tiling over x:
// each block computes ROWS_PER_BLOCK output rows (one warp per row),
// while x tiles are loaded once and reused across all rows in the block.
constexpr int WARP_SIZE = 32;
#ifndef RK4_X_TILE
#define RK4_X_TILE 128
#endif
constexpr int X_TILE = RK4_X_TILE;
constexpr int ROWS_PER_BLOCK = 16; // 32 x 16 = 512 threads per block

__global__ void matvec_dense_tiled(const double *__restrict__ L,
                                   const double *__restrict__ x,
                                   double *__restrict__ y,
                                   int D) {
    __shared__ double x_tile[X_TILE];

    const int lane = threadIdx.x;      // 0..31
    const int local_row = threadIdx.y; // 0..ROWS_PER_BLOCK-1
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;
    const unsigned mask = 0xffffffffu;

    double sum = 0.0;
    for (int base = 0; base < D; base += X_TILE) {
        if (local_row == 0) {
            for (int t = lane; t < X_TILE; t += WARP_SIZE) {
                const int j = base + t;
                x_tile[t] = (j < D) ? x[j] : 0.0;
            }
        }
        __syncthreads();

        if (row < D) {
            const double *row_ptr = L + static_cast<size_t>(row) * static_cast<size_t>(D);
            for (int t = lane; t < X_TILE; t += WARP_SIZE) {
                const int j = base + t;
                if (j < D) {
                    sum = fma(row_ptr[j], x_tile[t], sum);
                }
            }
        }
        __syncthreads();
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    if (lane == 0 && row < D) {
        y[row] = sum;
    }
}

__global__ void vec_linear_combo(const double *__restrict__ x,
                                 const double *__restrict__ y,
                                 double alpha,
                                 double *__restrict__ out,
                                 int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        out[i] = x[i] + alpha * y[i];
    }
}

__global__ void rk4_update(double *__restrict__ r,
                           const double *__restrict__ k1,
                           const double *__restrict__ k2,
                           const double *__restrict__ k3,
                           const double *__restrict__ k4,
                           double dt,
                           int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        r[i] += (dt / 6.0) * (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]);
    }
}

// Dense, general RK4 time loop using a custom dense matvec kernel.
// L is stored row-major on host/device.
void rk4_time_loop(const double *dL,
                   double *dr,
                   double *dk1,
                   double *dk2,
                   double *dk3,
                   double *dk4,
                   double *dtmp,
                   int D,
                   double dt,
                   int nsteps,
                   dim3 vec_grid,
                   dim3 vec_block,
                   dim3 matvec_grid,
                   dim3 matvec_block) {
    const double half_dt = 0.5 * dt;

    for (int n = 0; n < nsteps; ++n) {
        matvec_dense_tiled<<<matvec_grid, matvec_block>>>(dL, dr, dk1, D);
        CUDA_CHECK(cudaGetLastError());

        vec_linear_combo<<<vec_grid, vec_block>>>(dr, dk1, half_dt, dtmp, D);
        CUDA_CHECK(cudaGetLastError());
        matvec_dense_tiled<<<matvec_grid, matvec_block>>>(dL, dtmp, dk2, D);
        CUDA_CHECK(cudaGetLastError());

        vec_linear_combo<<<vec_grid, vec_block>>>(dr, dk2, half_dt, dtmp, D);
        CUDA_CHECK(cudaGetLastError());
        matvec_dense_tiled<<<matvec_grid, matvec_block>>>(dL, dtmp, dk3, D);
        CUDA_CHECK(cudaGetLastError());

        vec_linear_combo<<<vec_grid, vec_block>>>(dr, dk3, dt, dtmp, D);
        CUDA_CHECK(cudaGetLastError());
        matvec_dense_tiled<<<matvec_grid, matvec_block>>>(dL, dtmp, dk4, D);
        CUDA_CHECK(cudaGetLastError());

        rk4_update<<<vec_grid, vec_block>>>(dr, dk1, dk2, dk3, dk4, dt, D);
        CUDA_CHECK(cudaGetLastError());
    }
}

int main(int argc, char **argv) {
    int D = 1 << 12;
    double T = 1.0;
    double dt = 1e-3;
    int nsteps = -1;
    const char *rho_out = nullptr;
    const char *timing_out = nullptr;

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
            printf("Unknown or incomplete argument: %s\n", argv[i]);
        }
    }

    if (nsteps > 0) {
        T = nsteps * dt;
    } else {
        nsteps = int(round(T / dt));
    }

    const size_t vbytes = static_cast<size_t>(D) * sizeof(double);
    const size_t Lbytes = static_cast<size_t>(D) * static_cast<size_t>(D) * sizeof(double);

    // Example initialization in row-major layout. The solver remains dense/general.
    std::vector<double> hL(static_cast<size_t>(D) * static_cast<size_t>(D));
    std::vector<double> hr(static_cast<size_t>(D), 1.0);
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            hL[static_cast<size_t>(i) * static_cast<size_t>(D) + static_cast<size_t>(j)] =
                (i == j) ? (-1.0 - 0.1 * i) : 0.0;
        }
    }

    double *dL = nullptr;
    double *dr = nullptr;
    double *dk1 = nullptr;
    double *dk2 = nullptr;
    double *dk3 = nullptr;
    double *dk4 = nullptr;
    double *dtmp = nullptr;

    CUDA_CHECK(cudaMalloc(&dL, Lbytes));
    CUDA_CHECK(cudaMalloc(&dr, vbytes));
    CUDA_CHECK(cudaMalloc(&dk1, vbytes));
    CUDA_CHECK(cudaMalloc(&dk2, vbytes));
    CUDA_CHECK(cudaMalloc(&dk3, vbytes));
    CUDA_CHECK(cudaMalloc(&dk4, vbytes));
    CUDA_CHECK(cudaMalloc(&dtmp, vbytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(dL, hL.data(), Lbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dr, hr.data(), vbytes, cudaMemcpyHostToDevice));

    dim3 vec_block(256);
    dim3 vec_grid((D + vec_block.x - 1) / vec_block.x);
    dim3 matvec_block(WARP_SIZE, ROWS_PER_BLOCK);
    dim3 matvec_grid((D + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    rk4_time_loop(dL, dr, dk1, dk2, dk3, dk4, dtmp, D, dt, nsteps,
                  vec_grid, vec_block, matvec_grid, matvec_block);

    CUDA_CHECK(cudaMemcpy(hr.data(), dr, vbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("End-to-end time (copies + kernels) for %d steps: %.3f ms\n", nsteps, ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double err2 = 0.0;
    for (int i = 0; i < D; ++i) {
        const double lambda_i = -1.0 - 0.1 * i;
        const double exact = exp(lambda_i * T);
        const double diff = hr[static_cast<size_t>(i)] - exact;
        err2 += diff * diff;
    }
    const double l2err = sqrt(err2);
    printf("L2 error = %e\n", l2err);

    if (timing_out) {
        FILE *ft = fopen(timing_out, "w");
        if (ft) {
            fprintf(ft, "D,%d\nT,%.17g\ndt,%.17g\nnsteps,%d\nwall_time_s,%.9f\n",
                    D, T, dt, nsteps, ms / 1000.0);
            fprintf(ft, "l2_error,%e\n", l2err);
            fclose(ft);
        } else {
            printf("Warning: could not open timing file %s\n", timing_out);
        }
    }

    if (rho_out) {
        FILE *fr = fopen(rho_out, "w");
        if (fr) {
            for (int i = 0; i < D; ++i) {
                fprintf(fr, "%.17g\n", hr[static_cast<size_t>(i)]);
            }
            fclose(fr);
        } else {
            printf("Warning: could not open rho file %s\n", rho_out);
        }
    }

    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dr));
    CUDA_CHECK(cudaFree(dk1));
    CUDA_CHECK(cudaFree(dk2));
    CUDA_CHECK(cudaFree(dk3));
    CUDA_CHECK(cudaFree(dk4));
    CUDA_CHECK(cudaFree(dtmp));
    return 0;
}
