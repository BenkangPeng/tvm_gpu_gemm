#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

/// copy from https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/sgemm/sgemm_cublas.cu
void cublas_sgemm(float* A, float* B, float* C, int M, int N, int K) {
  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  static float alpha = 1.0;
  static float beta = 0.0;

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F,
               K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

int main(int argc, char** argv) {
  const int M = atoi(argv[1]);  // row of A
  const int N = atoi(argv[2]);  // column of B
  const int K = atoi(argv[3]);  // column of A or row of B

  float* h_A = (float*)malloc(M * K * sizeof(float));
  float* h_B = (float*)malloc(K * N * sizeof(float));
  float* h_C = (float*)malloc(M * N * sizeof(float));

  for (int i = 0; i < M * K; i++) h_A[i] = 1.0;
  for (int i = 0; i < K * N; i++) h_B[i] = 1.0;

  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, M * K * sizeof(float));
  cudaMalloc((void**)&d_B, K * N * sizeof(float));
  cudaMalloc((void**)&d_C, M * N * sizeof(float));

  // get cublas handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSetMatrix(M, K, sizeof(float), h_A, M, d_A, M);
  cublasSetMatrix(K, N, sizeof(float), h_B, K, d_B, K);

  // warmup
  cublas_sgemm(d_A, d_B, d_C, M, N, K);

  auto start = std::chrono::high_resolution_clock::now();
  const int run_times = 10;
  for (int i = 0; i < run_times; ++i) {
    cublas_sgemm(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
  }
  auto end = std::chrono::high_resolution_clock::now();

  cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C, M);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << (double)2 * M * K * N * run_times / (duration.count() * 1e9) << " TFLOPS"
            << std::endl;

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
