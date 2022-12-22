#include <stdio.h>
#include <omp.h>
#include "GPU.cuh"

__global__ void kernel(float a, float* x, int incx, float* y, int incy) {
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_index >= N || global_index * incx >= N || global_index * incy >= N) return;

	y[global_index * incy] += a * x[global_index * incx];

}

void saxpy_GPU(int blockSize, float a, const float* x, int incx, float* y, int incy, std::vector<double>* time) {
	int gridSize = N / blockSize;
	if (N % blockSize != 0) gridSize++;

	float* GPU_X;
	float* GPU_Y;
	if (cudaMalloc((void**)&GPU_X, N * sizeof(float))) printf("cudaMalloc GPU_X error\n");
	if (cudaMalloc((void**)&GPU_Y, N * sizeof(float))) printf("cudaMalloc GPU_Y error\n");

	if (cudaMemcpy(GPU_X, x, N * sizeof(float), cudaMemcpyHostToDevice)) printf("cudaMemcpy x to GPU_X error\n");
	if (cudaMemcpy(GPU_Y, y, N * sizeof(float), cudaMemcpyHostToDevice)) printf("cudaMemcpy y to GPU_Y error\n");

	double t1 = omp_get_wtime();

	kernel <<<gridSize, blockSize>>> (a, GPU_X, incx, GPU_Y, incy);
	if (cudaDeviceSynchronize()) printf("cudaDeviceSynchronize error!\n");

	double t2 = omp_get_wtime();
	(*time).push_back(t2 - t1);

	if (cudaGetLastError()) printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
	if (cudaMemcpy(y, GPU_Y, N * sizeof(float), cudaMemcpyDeviceToHost)) printf("cudaMemcpy GPU_Y to y error\n");

	cudaFree(GPU_X);
	cudaFree(GPU_Y);
}

__global__ void kernel_double(double a, double* x, int incx, double* y, int incy) {
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_index >= N || global_index * incx >= N || global_index * incy >= N) return;

	y[global_index * incy] += a * x[global_index * incx];
}

void daxpy_GPU(int blockSize, double a, const double* x, int incx, double* y, int incy, std::vector<double>* time) {
	int gridSize = N / blockSize;
	if (N % blockSize != 0) gridSize++;

	double* GPU_X;
	double* GPU_Y;

	if (cudaMalloc((void**)&GPU_X, N * sizeof(double))) printf("cudaMalloc GPU_X error\n");
	if (cudaMalloc((void**)&GPU_Y, N * sizeof(double))) printf("cudaMalloc GPU_Y error\n");

	if (cudaMemcpy(GPU_X, x, N * sizeof(double), cudaMemcpyHostToDevice)) printf("cudaMemcpy x to GPU_X error\n");
	if (cudaMemcpy(GPU_Y, y, N * sizeof(double), cudaMemcpyHostToDevice)) printf("cudaMemcpy y to GPU_Y error\n");

	double t1 = omp_get_wtime();

	kernel_double <<<gridSize, blockSize>>> (a, GPU_X, incx, GPU_Y, incy);
	if (cudaDeviceSynchronize()) printf("cudaDeviceSynchronize error\n");

	double t2 = omp_get_wtime();
	(*time).push_back(t2 - t1);

	if (cudaGetLastError()) printf("kernel_double launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
	if (cudaMemcpy(y, GPU_Y, N * sizeof(double), cudaMemcpyDeviceToHost)) printf("cudaMemcpy GPU_Y to y error\n");

	cudaFree(GPU_X);
	cudaFree(GPU_Y);
}