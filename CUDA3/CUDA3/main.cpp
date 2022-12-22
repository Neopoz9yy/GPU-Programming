#include <stdio.h>
#include "GPU.cuh"
#include "CPU.h"
#include <iostream>
#include <random>

#define OPTIMIZED true

float* GetRandomMatrix(int rows, int cols) {
	if (rows * cols <= 0) throw "ERROR!!!!!!";
	
	std::random_device dev;
	std::mt19937 gen(dev());
	std::uniform_real_distribution<float> urd(-1, 1);
	float* matr = new float[rows * cols];
	for (size_t i = 0; i < rows * cols; i++) matr[i] = urd(gen) * 0.1;
	
	return matr;
}


int main(){
	int deviceCount;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&deviceCount);
	printf("Device count: %d\n\n", deviceCount);

	for (int i = 0; i < deviceCount; i++) {
		cudaGetDeviceProperties(&deviceProp, i);
		printf("Device name: %s\n\n", deviceProp.name);
	}

	int size = BLOCK_SIZE * 100;
	printf("%ix%i\n\n", size, size);

	int M = size;
	int N = size;
	int K = size;
	const float* A = GetRandomMatrix(M, N);
	const float* B = GetRandomMatrix(N, K);

	double SEQ_time;
	float* SEQ = SeqMultiplication(A, B, M, N, K, &SEQ_time);
	printf("SEQ : %f(%f)\n", SEQ_time, SEQ_time / SEQ_time);

	double OMP_time;
	float* OMP = OMPMultiplication(A, B, M, N, K, &OMP_time);
	printf("OMP : %f(%f)\n", OMP_time, SEQ_time / OMP_time);

	double GPU_time;
	float* GPU = GPUMultiplication(A, B, M, N, K, !OPTIMIZED, &GPU_time);
	printf("GPU : %f(%f)\n", GPU_time, SEQ_time / GPU_time);

	double GPU_OPT_time;
	float* GPU_OPT = GPUMultiplication(A, B, M, N, K, OPTIMIZED, &GPU_OPT_time);
	printf("GPU_OPT : %f(%f)\n", GPU_OPT_time, SEQ_time / GPU_OPT_time);

	bool flag = true;
	for (size_t i = 0; i < M * K; i++){
		//printf("%f %f %f %f\n", SEQ[i], OMP[i], GPU[i], GPU_OPT[i]);
		if (std::fabs(SEQ[i] - OMP[i]) > 0.000001 || std::fabs(SEQ[i] - GPU[i]) > 0.000001 || std::fabs(SEQ[i] - GPU_OPT[i]) > 0.000001) {
			flag = false;
			printf("%d\n", std::fabs(SEQ[i] - OMP[i]) > 0.000001);
			printf("%d\n", std::fabs(SEQ[i] - GPU[i]) > 0.000001);
			printf("%d\n", std::fabs(SEQ[i] - GPU_OPT[i]) > 0.000001);
			printf("NO DONE\n");
			break;
		}
	}
	if (flag) printf("DONE\n");

	return 0;
}