#include <stdio.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE 16

float* GPUMultiplication(const float* A, const float* B, int M, int N, int K, bool optimized, double* time);