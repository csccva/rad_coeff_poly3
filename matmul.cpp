#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cblas.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

// Helper function to initialize a matrix with random values
void random_init(std::vector<double> &matrix, int size, double factor) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        matrix[i] = (2.0*dis(gen)-1.0)*factor;
    }
}

// Helper function to calculate max absolute difference
double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

// Kernel for single-threaded matrix multiplication
__global__ void matmul_single_thread(const double *A, const double *B, double *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // Matrix dimensions
    const int M = 7; // Rows in A and C
    const int N = 50; // Columns in B and C
    const int K = 7; // Columns in A and rows in B

    // Host matrices
    std::vector<double> h_A(M * K), h_B(K * N), h_C1(M * N, 0.0), h_C2(M * N, 0.0), h_C3(M * N, 0.0);

    // Initialize matrices with random values
    random_init(h_A, M * K, 75000);
    random_init(h_B, K * N, 0.7);

    std::cout << "Sum of A: " << std::accumulate(h_A.begin(), h_A.end(), 0.0) << "\n";
    std::cout << "Sum of B: " << std::accumulate(h_B.begin(), h_B.end(), 0.0) << "\n";

    // Perform CPU matrix multiplication (C1 = A * B)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                h_C1[i * N + j] += h_A[i * K + k] * h_B[k * N + j];
            }
        }
    }

    // Device matrices
    double *d_A, *d_B, *d_C2;
    hipMalloc(&d_A, M * K * sizeof(double));
    hipMalloc(&d_B, K * N * sizeof(double));
    hipMalloc(&d_C2, M * N * sizeof(double));

    // Copy data to device
    hipMemcpy(d_A, h_A.data(), M * K * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), K * N * sizeof(double), hipMemcpyHostToDevice);

    // HIPBLAS handle
    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Perform DGEMM using HIPBLAS (C2 = alpha * A * B + beta * C2)
    const double alpha = 1.0;
    const double beta = 0.0;
    hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C2, N);

    // Copy result back to host
    hipMemcpy(h_C2.data(), d_C2, M * N * sizeof(double), hipMemcpyDeviceToHost);

    // Perform DGEMM using OpenBLAS (C3 = alpha * A * B + beta * C3)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, h_A.data(), K, h_B.data(), N, beta, h_C3.data(), N);

    // Launch single-thread kernel for matrix multiplication
    double *d_C1;
    hipMalloc(&d_C1, M * N * sizeof(double));
    matmul_single_thread<<<1, 1>>>(d_A, d_B, d_C1, M, N, K);

    // Copy result from kernel back to host
    std::vector<double> h_C_kernel(M * N);
    hipMemcpy(h_C_kernel.data(), d_C1, M * N * sizeof(double), hipMemcpyDeviceToHost);

    // Compare results
    double max_diff_hipblas = max_abs_diff(h_C1, h_C2);
    double max_diff_kernel = max_abs_diff(h_C1, h_C_kernel);
    double max_diff_openblas = max_abs_diff(h_C1, h_C3);

    std::cout << "Max absolute difference (HIPBLAS): " << max_diff_hipblas << "\n";
    std::cout << "Max absolute difference (Kernel): " << max_diff_kernel << "\n";
    std::cout << "Max absolute difference (OpenBLAS): " << max_diff_openblas << "\n";

    // Clean up
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C2);
    hipFree(d_C1);
    hipblasDestroy(handle);

    return 0;
}
