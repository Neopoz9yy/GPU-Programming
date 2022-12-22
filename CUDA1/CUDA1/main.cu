#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"


__global__ void kernel(float* array) {

	int block_index = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int global_index = block_index * blockDim.x * blockDim.y * blockDim.z + (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);
	array[global_index] += global_index;

	printf("I am from  (%d; %d; %d) block, (%d; %d; %d) thread (global index: %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, global_index);
}

int main() {
	int deviceCount;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&deviceCount);
	printf("Device count: %d\n\n", deviceCount);

	for (int i = 0; i < deviceCount; i++) {
		cudaGetDeviceProperties(&deviceProp, i);
		printf("Device name: %s\n\n", deviceProp.name);
	}

	int blockX = 2;
	int blockY = 2;
	int blockZ = 2;

	int threadX = 2;
	int threadY = 2;
	int threadZ = 2;

	int blockCount = blockX * blockY * blockZ;
	int threadCount = threadX * threadY * threadZ;

	const int size =  blockCount * threadCount;

	float* CPU_array = new float[size];

	for (size_t ind = 0; ind < size; ind++){
		CPU_array[ind] = ind;
		//printf("CPU_array[%d] = %.0f\n", ind, CPU_array[ind]);
	}

	float* GPU_array;
	if (cudaMalloc((void**)&GPU_array, size * sizeof(float))) printf("cudaMalloc GPU_array error\n");
	if (cudaMemcpy(GPU_array, CPU_array, size * sizeof(float), cudaMemcpyHostToDevice)) printf("cudaMemcpy CPU_array to GPU_array error\n");

	dim3 gridSize(blockX, blockY, blockZ);
	dim3 blockSize(threadX, threadY, threadZ);
	kernel <<<gridSize, blockSize>>> (GPU_array);
	if (cudaDeviceSynchronize()) printf("cudaDeviceSynchronize error!\n");

	float* CPU_array_new = new float[size];
	if (cudaMemcpy(CPU_array_new, GPU_array, size * sizeof(float), cudaMemcpyDeviceToHost)) printf("cudaMemcpy GPU_array to CPU_array_new error\n");

	//for (size_t ind = 0; ind < size; ind++) printf("CPU_array[%d] = %.0f\n", ind, CPU_array_new[ind]);

	bool flag = true;
	for (size_t i = 0; i < size; i++) {
		if (CPU_array[i] + i != CPU_array_new[i]) {
			printf("NO DONE\n\n");
			flag = false;
			break;
		}
	}
	if (flag) printf("DONE\n\n");

	delete[] CPU_array;
	cudaFree(GPU_array);
	return 0;
}