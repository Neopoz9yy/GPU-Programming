#include <omp.h>
#include "CPU.h"

float* SeqMultiplication(const float* A, const float* B, int M, int N, int K, double* time) {
	float* C = new float[M * K]{ 0 };
	double t1 = omp_get_wtime();

	for (size_t i = 0; i < M; i++) {
		for (size_t j = 0; j < K; j++) {
			for (size_t k = 0; k < N; k++) {
				C[i * K + j] += A[i * N + k] * B[k * K + j];
			}
		}
	}

	double t2 = omp_get_wtime();
	*time = t2 - t1;

	return C;
}

float* OMPMultiplication(const float* A, const float* B, int M, int N, int K, double* time) {
	float* C = new float[M * K]{ 0 };
	double t1 = omp_get_wtime();

#pragma omp parallel for num_threads(THREADS)
	for (int i = 0; i < M; i++) {
		for (size_t j = 0; j < K; j++) {
			for (size_t k = 0; k < N; k++) {
				C[i * K + j] += A[i * N + k] * B[k * K + j];
			}
		}
	}
#pragma omp barrier

	double t2 = omp_get_wtime();
	*time = t2 - t1;

	return C;
}