// rk4_cuda.cu
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
__global__ void matvec_dense_cuda(const double* L, const double* x, double* y, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        const double* row = L + (size_t)i * D;
        double sum = 0.0;
        for (int j = 0; j < D; ++j) sum += row[j] * x[j];
        y[i] = sum;
    }
}

__global__ void axpy(double alpha, const double* x, double* y, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) y[i] += alpha * x[i];
}

__global__ void copy(const double* x, double* y, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) y[i] = x[i];
}

__global__ void rk4_update(double* r, const double* k1, const double* k2,
                           const double* k3, const double* k4, double dt, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) r[i] += (dt/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
}

void rk4_time_loop(const double* dL, double* dr,
                   double* dk1, double* dk2, double* dk3, double* dk4,
                   double* dtmp, int D, double dt, int nsteps,
                   dim3 grid, dim3 block) {
    for (int n = 0; n < nsteps; ++n) {
        matvec_dense_cuda<<<grid, block>>>(dL, dr, dk1, D);

        copy<<<grid, block>>>(dr, dtmp, D);
        axpy<<<grid, block>>>(0.5 * dt, dk1, dtmp, D);
        matvec_dense_cuda<<<grid, block>>>(dL, dtmp, dk2, D);

        copy<<<grid, block>>>(dr, dtmp, D);
        axpy<<<grid, block>>>(0.5 * dt, dk2, dtmp, D);
        matvec_dense_cuda<<<grid, block>>>(dL, dtmp, dk3, D);

        copy<<<grid, block>>>(dr, dtmp, D);
        axpy<<<grid, block>>>(dt, dk3, dtmp, D);
        matvec_dense_cuda<<<grid, block>>>(dL, dtmp, dk4, D);

        rk4_update<<<grid, block>>>(dr, dk1, dk2, dk3, dk4, dt, D);
    }
}


int main(int argc, char **argv) {
    int D = 1<<12;
    double T = 1.0, dt = 1e-3;
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
        nsteps = int(round(T/dt));
    }
    size_t vbytes = (size_t)D * sizeof(double);
    size_t Lbytes = (size_t)D * D * sizeof(double);

    // host init of L and r
    std::vector<double> hL(D * (size_t)D);
    std::vector<double> hr(D, 1.0);
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            hL[(size_t)i * D + j] = (i == j) ? (-1.0 - 0.1 * i) : 0.0;
        }
    }

    // device alloc
    double *dL, *dr, *dk1, *dk2, *dk3, *dk4, *dtmp;
    cudaMalloc(&dL, Lbytes);
    cudaMalloc(&dr, vbytes);
    cudaMalloc(&dk1, vbytes);
    cudaMalloc(&dk2, vbytes);
    cudaMalloc(&dk3, vbytes);
    cudaMalloc(&dk4, vbytes);
    cudaMalloc(&dtmp, vbytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // copy to device
    cudaMemcpy(dL, hL.data(), Lbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, hr.data(), vbytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((D + block.x - 1) / block.x);

    rk4_time_loop(dL, dr, dk1, dk2, dk3, dk4, dtmp, D, dt, nsteps, grid, block);

    // copy result back and compute error on host
    cudaMemcpy(hr.data(), dr, vbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("End-to-end time (copies + kernels) for %d steps: %.3f ms\n", nsteps, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double err2 = 0.0;
    for (int i = 0; i < D; ++i) {
        double lambda_i = -1.0 - 0.1 * i;
        double exact = exp(lambda_i * T);
        double diff = hr[i] - exact;
        err2 += diff * diff;
    }
    double l2err = sqrt(err2);
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
                fprintf(fr, "%.17g\n", hr[i]);
            }
            fclose(fr);
        } else {
            printf("Warning: could not open rho file %s\n", rho_out);
        }
    }

    cudaFree(dL); cudaFree(dr); cudaFree(dk1); cudaFree(dk2); cudaFree(dk3); cudaFree(dk4); cudaFree(dtmp);
}
