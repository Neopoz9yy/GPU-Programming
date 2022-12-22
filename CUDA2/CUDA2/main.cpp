#include <stdio.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <float.h>
#include "CPU.h"
#include "GPU.cuh"
#include <random>

double average(std::vector<double> arr){
	double res = 0;
	for (size_t i = 0; i < arr.size(); i++) res += arr[i];

	return arr.size() == 0 ? 0 : res / arr.size();
}

template<typename T>
T* GetRandomMatrix(int n) {
	if (n <= 0) throw "ERROR!!!!!!";

	std::random_device dev;
	std::mt19937 gen(dev());
	std::uniform_real_distribution<T> urd(-1, 1);
	T* matr = new T[n];
	for (size_t i = 0; i < n; i++) matr[i] = urd(gen) * 0.1;

	return matr;
}

void Axpy() {
	const int iterations = 1;

	const int incx = 2, incy = 1;
	const float AF = 0.7454577;

	const std::vector<int> blockSizes{ 8, 16, 32, 64, 128, 256};
	std::vector<double> time, avgTimes;

	printf("float: %d\n", N);
	const float* XF = GetRandomMatrix<float>(N);
	const float* CNSTYF = GetRandomMatrix<float>(N);
	float* TEST_Y = new float[N];
	memcpy(TEST_Y, CNSTYF, N * sizeof(float));

	printf("Blocks\tAverage\n");
	for (size_t i = 0; i < blockSizes.size(); i++) {

		for (size_t j = 0; j < iterations; j++) saxpy_GPU(blockSizes[i], AF, XF, incx, TEST_Y, incy, &time);

		avgTimes.push_back(average(time));
		printf("%d\t%f\n", blockSizes[i], avgTimes.back());
		time.clear();
	}

	double min = avgTimes[0];
	int bestBlockSizeInd = 0;

	for (size_t i = 1; i < avgTimes.size(); i++) {
		if (avgTimes[i] < min) {
			min = avgTimes[i];
			bestBlockSizeInd = i;
		}
	}
	printf("Best size: %d\n\n", blockSizes[bestBlockSizeInd]);

	delete[] TEST_Y;

	printf("float: %d\n", N);
	double t1, t2;

	float* SEQF = new float[N];
	memcpy(SEQF, CNSTYF, N * sizeof(float));
	for (size_t i = 0; i < iterations; i++) {
		t1 = omp_get_wtime();
		saxpy(AF, XF, incx, SEQF, incy);
		t2 = omp_get_wtime();
		time.push_back(t2 - t1);
	}
	double SEQFTIME = average(time);
	printf("SEQ\t%f\t%f\n", SEQFTIME, SEQFTIME / SEQFTIME);

	time.clear();

	float* OMPF = new float[N];
	memcpy(OMPF, CNSTYF, N * sizeof(float));
	for (size_t i = 0; i < iterations; i++) {
		t1 = omp_get_wtime();
		saxpy_OMP(AF, XF, incx, OMPF, incy);
		t2 = omp_get_wtime();
		time.push_back(t2 - t1);
	}
	double OMPFTIME = average(time);
	printf("OMP\t%f\t%f\n", OMPFTIME, SEQFTIME / OMPFTIME);

	time.clear();

	float* GPUF = new float[N];
	memcpy(GPUF, CNSTYF, N * sizeof(float));
	for (size_t i = 0; i < iterations; i++) saxpy_GPU(blockSizes[bestBlockSizeInd], AF, XF, incx, GPUF, incy, &time);
	double GPUFTIME = average(time);
	printf("GPU\t%f\t%f\t\n", GPUFTIME, SEQFTIME / GPUFTIME);

	time.clear();

	bool flag = true;
	for (size_t i = 0; i < N; i++) {
		//printf("%d - %f %f %f %f\n", i, CNSTYF[i], SEQF[i], OMPF[i], GPUF[i]);
		if ( std::fabs(SEQF[i] - OMPF[i]) > 0.000001 || std::fabs(SEQF[i] - GPUF[i]) > 0.000001) {
			printf("NO DONE\n\n");
			flag = false;
			break;
		}
	}
	if (flag) printf("DONE\n\n");

	delete[] XF, CNSTYF, SEQF, OMPF, GPUF;

	/// <summary>
	/// -------------------------------------double-------------------------------------
	/// </summary>

	printf("double: %d\n", N);
	const double* XD = GetRandomMatrix<double>(N);
	const double* CNSTYD = GetRandomMatrix<double>(N);
	const double AD = 0.235265;

	double* SEQD = new double[N];
	memcpy(SEQD, CNSTYD, N * sizeof(double));
	for (size_t i = 0; i < iterations; i++) {
		t1 = omp_get_wtime();
		daxpy(AD, XD, incx, SEQD, incy);
		t2 = omp_get_wtime();
		time.push_back(t2 - t1);
	}
	double SEQDTIME = average(time);
	printf("SEQ\t%f\t%f\n", SEQDTIME, SEQDTIME / SEQDTIME);

	time.clear();

	double* OMPD = new double[N];
	memcpy(OMPD, CNSTYD, N * sizeof(double));
	for (size_t i = 0; i < iterations; i++) {
		t1 = omp_get_wtime();
		daxpy_OMP(AD, XD, incx, OMPD, incy);
		t2 = omp_get_wtime();
		time.push_back(t2 - t1);
	}
	double OMPDTIME = average(time);
	printf("OMP\t%f\t%f\n", OMPDTIME, SEQDTIME / OMPDTIME);

	time.clear();

	double* GPUD = new double[N];
	memcpy(GPUD, CNSTYD, N * sizeof(double));
	for (size_t i = 0; i < iterations; i++) daxpy_GPU(blockSizes[bestBlockSizeInd], AD, XD, incx, GPUD, incy, &time);
	double GPUDTIME = average(time);
	printf("GPU\t%f\t%f\n", GPUDTIME, SEQDTIME / GPUDTIME);

	time.clear();

	for (size_t i = 0; i < N; i++) {
		if (std::fabs(SEQD[i] - OMPD[i]) > 0.000001 || std::fabs(SEQD[i] - GPUD[i]) > 0.000001) {
			printf("NO DONE\n\n");
			flag = false;
			break;
		}
	}
	if (flag) printf("DONE\n\n");

	delete[] XD, CNSTYD, SEQD, OMPD, GPUD;
}

int main() {
	int deviceCount;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&deviceCount);
	printf("Device count: %d\n\n", deviceCount);

	for (int i = 0; i < deviceCount; i++) {
		cudaGetDeviceProperties(&deviceProp, i);

		printf("Device name: %s\n", deviceProp.name);
		/*printf("Total global memory: %d\n", deviceProp.totalGlobalMem);
		printf("Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
		printf("Registers per block: %d\n", deviceProp.regsPerBlock);
		printf("Warp size: %d\n", deviceProp.warpSize);
		printf("Memory pitch: %d\n", deviceProp.memPitch);
		printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

		printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);

		printf("Max grid size: x = %d, y = %d, z = %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);

		printf("Clock rate: %d\n", deviceProp.clockRate);
		printf("Total constant memory: %d\n", deviceProp.totalConstMem);
		printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Texture alignment: %d\n", deviceProp.textureAlignment);
		printf("Device overlap: %d\n", deviceProp.deviceOverlap);
		printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);

		printf("Kernel execution timeout enabled: %s\n\n",
			deviceProp.kernelExecTimeoutEnabled ? "true" : "false");*/
		printf("\n");
	}

	Axpy();

	return 0;
}