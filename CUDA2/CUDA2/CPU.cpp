#include <algorithm>
#include <omp.h>

#include "CPU.h"

void saxpy(float a, const float* x, int incx, float* y, int incy){
	for (size_t i = 0; i < N; i++){
		if (i * incx >= N || i * incy >= N) break;
		y[i * incy] += a * x[i * incx];
	}
}


void daxpy(double a, const double* x, int incx, double* y, int incy){
	for (int i = 0; i < N; i++){
		if (i * incx >= N || i * incy >= N) break;
		y[i * incy] += a * x[i * incx];
	}
}

#define THREADS 12
void saxpy_OMP(float a, const float* x, int incx, float* y, int incy){
#pragma omp parallel for num_threads(THREADS)
	for (int i = 0; i < N; i++){
		if (i * incx >= N || i * incy >= N) break;
		y[i * incy] += a * x[i * incx];
	}
#pragma omp barrier
}

void daxpy_OMP(double a, const double* x, int incx, double* y, int incy){
#pragma omp parallel for num_threads(THREADS)
	for (int i = 0; i < N; i++){
		if (i * incx >= N || i * incy >= N) break;
		y[i * incy] += a * x[i * incx];
	}
#pragma omp barrier
}