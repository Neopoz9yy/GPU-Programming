#include <vector>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#define N 60000000

void saxpy_GPU(int blockSize, float a, const float* x, int incx, float* y, int incy, std::vector<double>* time);
void daxpy_GPU(int blockSize, double a, const double* x, int incx, double* y, int incy, std::vector<double>* time);