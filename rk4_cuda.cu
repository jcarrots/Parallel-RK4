// rk4_cuda.cu
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

static inline void cuda_check(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

static inline void cublas_check(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d (status=%d)\n", file, line, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)
#define CUBLAS_CHECK(call) cublas_check((call), __FILE__, __LINE__)

enum class BackendMode {
    Custom,
    Cublas,
    Both
};

struct RunResult {
    double kernel_ms = 0.0;
    double total_ms = 0.0;
    double l2err = 0.0;
    std::vector<double> state;
};

constexpr int WARP_SIZE = 32;
#ifndef RK4_X_TILE
#define RK4_X_TILE 128
#endif
constexpr int X_TILE = RK4_X_TILE;
constexpr int ROWS_PER_BLOCK = 16; // 32 x 16 = 512 threads per block

const char *backend_name(BackendMode mode) {
    if (mode == BackendMode::Custom) return "custom";
    if (mode == BackendMode::Cublas) return "cublas";
    return "both";
}

BackendMode parse_backend(const char *value) {
    if (strcmp(value, "custom") == 0) return BackendMode::Custom;
    if (strcmp(value, "cublas") == 0) return BackendMode::Cublas;
    if (strcmp(value, "both") == 0) return BackendMode::Both;
    fprintf(stderr, "Unknown backend '%s'. Expected custom|cublas|both.\n", value);
    std::exit(EXIT_FAILURE);
}

void init_diagonal_matrix(std::vector<double> &L, int D, bool col_major) {
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            const double value = (i == j) ? (-1.0 - 0.1 * i) : 0.0;
            if (col_major) {
                L[static_cast<size_t>(j) * static_cast<size_t>(D) + static_cast<size_t>(i)] = value;
            } else {
                L[static_cast<size_t>(i) * static_cast<size_t>(D) + static_cast<size_t>(j)] = value;
            }
        }
    }
}

double compute_l2_error(const std::vector<double> &r, int D, double T) {
    double err2 = 0.0;
    for (int i = 0; i < D; ++i) {
        const double lambda_i = -1.0 - 0.1 * i;
        const double exact = exp(lambda_i * T);
        const double diff = r[static_cast<size_t>(i)] - exact;
        err2 += diff * diff;
    }
    return sqrt(err2);
}

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

void rk4_time_loop_custom(const double *dL,
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

void rk4_time_loop_cublas(const double *dL,
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
                          cublasHandle_t blas_handle) {
    const double half_dt = 0.5 * dt;
    const double one = 1.0;
    const double zero = 0.0;

    for (int n = 0; n < nsteps; ++n) {
        CUBLAS_CHECK(cublasDgemv(blas_handle, CUBLAS_OP_N, D, D, &one, dL, D, dr, 1, &zero, dk1, 1));

        vec_linear_combo<<<vec_grid, vec_block>>>(dr, dk1, half_dt, dtmp, D);
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasDgemv(blas_handle, CUBLAS_OP_N, D, D, &one, dL, D, dtmp, 1, &zero, dk2, 1));

        vec_linear_combo<<<vec_grid, vec_block>>>(dr, dk2, half_dt, dtmp, D);
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasDgemv(blas_handle, CUBLAS_OP_N, D, D, &one, dL, D, dtmp, 1, &zero, dk3, 1));

        vec_linear_combo<<<vec_grid, vec_block>>>(dr, dk3, dt, dtmp, D);
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasDgemv(blas_handle, CUBLAS_OP_N, D, D, &one, dL, D, dtmp, 1, &zero, dk4, 1));

        rk4_update<<<vec_grid, vec_block>>>(dr, dk1, dk2, dk3, dk4, dt, D);
        CUDA_CHECK(cudaGetLastError());
    }
}

RunResult run_backend(BackendMode mode, int D, double T, double dt, int nsteps, int warmup) {
    const bool use_cublas = (mode == BackendMode::Cublas);
    const size_t vbytes = static_cast<size_t>(D) * sizeof(double);
    const size_t Lbytes = static_cast<size_t>(D) * static_cast<size_t>(D) * sizeof(double);

    std::vector<double> hL(static_cast<size_t>(D) * static_cast<size_t>(D));
    std::vector<double> h0(static_cast<size_t>(D), 1.0);
    init_diagonal_matrix(hL, D, use_cublas);

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

    dim3 vec_block(256);
    dim3 vec_grid((D + vec_block.x - 1) / vec_block.x);
    dim3 matvec_block(WARP_SIZE, ROWS_PER_BLOCK);
    dim3 matvec_grid((D + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    cublasHandle_t blas_handle = nullptr;
    if (use_cublas) {
        CUBLAS_CHECK(cublasCreate(&blas_handle));
    }

    CUDA_CHECK(cudaMemcpy(dL, hL.data(), Lbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dr, h0.data(), vbytes, cudaMemcpyHostToDevice));

    for (int w = 0; w < warmup; ++w) {
        if (use_cublas) {
            rk4_time_loop_cublas(dL, dr, dk1, dk2, dk3, dk4, dtmp, D, dt, nsteps, vec_grid, vec_block, blas_handle);
        } else {
            rk4_time_loop_custom(dL, dr, dk1, dk2, dk3, dk4, dtmp, D, dt, nsteps, vec_grid, vec_block, matvec_grid, matvec_block);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(dr, h0.data(), vbytes, cudaMemcpyHostToDevice));
    }

    cudaEvent_t total_start, kernel_start, kernel_stop, total_stop;
    CUDA_CHECK(cudaEventCreate(&total_start));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventCreate(&total_stop));

    CUDA_CHECK(cudaEventRecord(total_start));
    CUDA_CHECK(cudaMemcpy(dL, hL.data(), Lbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dr, h0.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(kernel_start));

    if (use_cublas) {
        rk4_time_loop_cublas(dL, dr, dk1, dk2, dk3, dk4, dtmp, D, dt, nsteps, vec_grid, vec_block, blas_handle);
    } else {
        rk4_time_loop_custom(dL, dr, dk1, dk2, dk3, dk4, dtmp, D, dt, nsteps, vec_grid, vec_block, matvec_grid, matvec_block);
    }

    CUDA_CHECK(cudaEventRecord(kernel_stop));

    RunResult result;
    result.state.resize(static_cast<size_t>(D));
    CUDA_CHECK(cudaMemcpy(result.state.data(), dr, vbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(total_stop));
    CUDA_CHECK(cudaEventSynchronize(total_stop));

    float kernel_ms = 0.0f;
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));
    result.kernel_ms = static_cast<double>(kernel_ms);
    result.total_ms = static_cast<double>(total_ms);
    result.l2err = compute_l2_error(result.state, D, T);

    CUDA_CHECK(cudaEventDestroy(total_start));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(total_stop));

    if (use_cublas) {
        CUBLAS_CHECK(cublasDestroy(blas_handle));
    }

    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dr));
    CUDA_CHECK(cudaFree(dk1));
    CUDA_CHECK(cudaFree(dk2));
    CUDA_CHECK(cudaFree(dk3));
    CUDA_CHECK(cudaFree(dk4));
    CUDA_CHECK(cudaFree(dtmp));

    return result;
}

void write_timing_file(const char *timing_out,
                       BackendMode mode,
                       int D,
                       double T,
                       double dt,
                       int nsteps,
                       const RunResult &result) {
    FILE *ft = fopen(timing_out, "w");
    if (!ft) {
        printf("Warning: could not open timing file %s\n", timing_out);
        return;
    }

    fprintf(ft, "backend,%s\n", backend_name(mode));
    fprintf(ft, "D,%d\nT,%.17g\ndt,%.17g\nnsteps,%d\n", D, T, dt, nsteps);
    fprintf(ft, "kernel_time_s,%.9f\n", result.kernel_ms / 1000.0);
    fprintf(ft, "wall_time_s,%.9f\n", result.total_ms / 1000.0);
    fprintf(ft, "l2_error,%e\n", result.l2err);
    fclose(ft);
}

void write_state_file(const char *rho_out, const std::vector<double> &state) {
    FILE *fr = fopen(rho_out, "w");
    if (!fr) {
        printf("Warning: could not open rho file %s\n", rho_out);
        return;
    }

    for (double value : state) {
        fprintf(fr, "%.17g\n", value);
    }
    fclose(fr);
}

int main(int argc, char **argv) {
    int D = 1 << 12;
    double T = 1.0;
    double dt = 1e-3;
    int nsteps = -1;
    int warmup = 1;
    BackendMode backend = BackendMode::Custom;
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
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = parse_backend(argv[++i]);
        } else if (strcmp(argv[i], "--rho-out") == 0 && i + 1 < argc) {
            rho_out = argv[++i];
        } else if (strcmp(argv[i], "--timing-out") == 0 && i + 1 < argc) {
            timing_out = argv[++i];
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            return 1;
        }
    }

    if (nsteps > 0) {
        T = nsteps * dt;
    } else {
        nsteps = static_cast<int>(round(T / dt));
    }
    if (warmup < 0) warmup = 0;

    if (backend == BackendMode::Both) {
        const RunResult custom = run_backend(BackendMode::Custom, D, T, dt, nsteps, warmup);
        const RunResult cublas = run_backend(BackendMode::Cublas, D, T, dt, nsteps, warmup);

        printf("[custom] kernel=%.3f ms, end-to-end=%.3f ms, L2 error=%e\n",
               custom.kernel_ms, custom.total_ms, custom.l2err);
        printf("[cublas] kernel=%.3f ms, end-to-end=%.3f ms, L2 error=%e\n",
               cublas.kernel_ms, cublas.total_ms, cublas.l2err);
        printf("[speedup] kernel custom/cublas = %.3fx, end-to-end custom/cublas = %.3fx\n",
               custom.kernel_ms / cublas.kernel_ms,
               custom.total_ms / cublas.total_ms);

        if (timing_out) {
            printf("Warning: --timing-out is ignored for --backend both.\n");
        }
        if (rho_out) {
            printf("Warning: --rho-out is ignored for --backend both.\n");
        }
        return 0;
    }

    const RunResult result = run_backend(backend, D, T, dt, nsteps, warmup);
    printf("[%s] Kernel-only time for %d steps: %.3f ms\n",
           backend_name(backend), nsteps, result.kernel_ms);
    printf("[%s] End-to-end time (copies + kernels): %.3f ms\n",
           backend_name(backend), result.total_ms);
    printf("L2 error = %e\n", result.l2err);

    if (timing_out) {
        write_timing_file(timing_out, backend, D, T, dt, nsteps, result);
    }
    if (rho_out) {
        write_state_file(rho_out, result.state);
    }
    return 0;
}
