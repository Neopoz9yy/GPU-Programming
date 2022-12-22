#pragma once
#include <stdio.h>
#include "GPU.cuh"

void saxpy(float a, const float* x, int incx, float* y, int incy);
void daxpy(double a, const double* x, int incx, double* y, int incy);

void saxpy_OMP(float a, const float* x, int incx, float* y, int incy);
void daxpy_OMP(double a, const double* x, int incx, double* y, int incy);