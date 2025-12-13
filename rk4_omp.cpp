#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <omp.h>


void matvec_omp(const double *L, const double *x, double *y, int D) {
    #pragma omp for
    for (int i = 0; i < D; ++i) {
        double sum = 0.0;
        const double *row = L + (size_t)i * D;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < D; ++j) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

void vec_axpy_omp(double alpha, const double *x, double *y, int D) {
    #pragma omp for simd
    for (int i = 0; i < D; ++i) {
        y[i] += alpha * x[i];
    }
}

void vec_copy_omp(const double *x, double *y, int D) {
    #pragma omp for simd
    for (int i = 0; i < D; ++i) {
        y[i] = x[i];
    }
}

void rk4_step_omp(const double *L,
                     double *r,
                     double *k1,
                     double *k2,
                     double *k3,
                     double *k4,
                     double *tmp,
                     int D,
                     double dt) {
    /* k1 = L r */
    #pragma omp parallel
    {
        matvec_omp(L, r, k1, D);

        /* tmp = r + dt/2 * k1 */
        vec_copy_omp(r, tmp, D);
        vec_axpy_omp(0.5 * dt, k1, tmp, D);
        matvec_omp(L, tmp, k2, D);

        /* tmp = r + dt/2 * k2 */
        vec_copy_omp(r, tmp, D);
        vec_axpy_omp(0.5 * dt, k2, tmp, D);
        matvec_omp(L, tmp, k3, D);

        /* tmp = r + dt * k3 */
        vec_copy_omp(r, tmp, D);
        vec_axpy_omp(dt, k3, tmp, D);
        matvec_omp(L, tmp, k4, D);
        
        /* r = r + dt/6 (k1 + 2 k2 + 2 k3 + k4) */
        #pragma omp for simd
        for (int i = 0; i < D; ++i) {
            r[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
}

int main(int argc, char **argv) {
    int D = 1<<12;          /* dimension */
    double T = 1.0;     /* final time */
    double dt = 1e-3;
    int nsteps = -1;
    const char *rho_out = NULL;
    const char *timing_out = NULL;

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
    if (nsteps > 0) {
        T = nsteps * dt;
    } else {
        nsteps = (int)round(T / dt);
    }

    /* allocate L and vectors */
    double *L   = static_cast<double *>(malloc((size_t)D * D * sizeof(double)));
    double *r   = static_cast<double *>(malloc((size_t)D * sizeof(double)));
    double *k1  = static_cast<double *>(malloc((size_t)D * sizeof(double)));
    double *k2  = static_cast<double *>(malloc((size_t)D * sizeof(double)));
    double *k3  = static_cast<double *>(malloc((size_t)D * sizeof(double)));
    double *k4  = static_cast<double *>(malloc((size_t)D * sizeof(double)));
    double *tmp = static_cast<double *>(malloc((size_t)D * sizeof(double)));

    if (!L || !r || !k1 || !k2 || !k3 || !k4 || !tmp) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* example: diagonal L, L_ii = lambda_i */
    for (int i = 0; i < D; ++i) {
        double *row = L + (size_t)i * D;
        #pragma omp simd
        for (int j = 0; j < D; ++j) {
            row[j] = (i == j) ? (-1.0 - 0.1 * i) : 0.0;
        }
    }

    /* initial condition r(0) */
    for (int i = 0; i < D; ++i) {
        r[i] = 1.0;
    }

  

   
    double t = 0.0;
    auto t_start = std::chrono::steady_clock::now();
    
    for (int n = 0; n < nsteps; ++n) {
        rk4_step_omp(L, r, k1, k2, k3, k4, tmp, D, dt);
        t += dt;
    }
    auto t_end = std::chrono::steady_clock::now();
    double wall_seconds = std::chrono::duration<double>(t_end - t_start).count();
    printf("Wall time for %d steps (dt = %g): %.6f s\n", nsteps, dt, wall_seconds);

    /* optional: compare to exact solution for diagonal L */
    double err2 = 0.0;
    for (int i = 0; i < D; ++i) {
        double lambda_i = -1.0 - 0.1 * i;
        double exact = exp(lambda_i * T) * 1.0;
        double diff = r[i] - exact;
        err2 += diff * diff;
    }
    double l2err = sqrt(err2);
    printf("L2 error at T = %g is %e\n", T, l2err);
    if (timing_out) {
        FILE *ft = fopen(timing_out, "w");
        if (ft) {
            fprintf(ft, "D,%d\nT,%.17g\ndt,%.17g\nnsteps,%d\nwall_time_s,%.9f\n",
                    D, T, dt, nsteps, wall_seconds);
            fprintf(ft, "l2_error,%e\n", l2err);
            fclose(ft);
        } else {
            fprintf(stderr, "Warning: could not open timing file %s\n", timing_out);
        }
    }
    if (rho_out) {
        FILE *fr = fopen(rho_out, "w");
        if (fr) {
            for (int i = 0; i < D; ++i) {
                fprintf(fr, "%.17g\n", r[i]);
            }
            fclose(fr);
        } else {
            fprintf(stderr, "Warning: could not open rho file %s\n", rho_out);
        }
    }

    free(L);
    free(r);
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(tmp);
    return 0;
}
