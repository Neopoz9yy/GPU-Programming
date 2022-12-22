#pragma once
#define THREADS 12

float* SeqMultiplication(const float* A, const float* B, int M, int N, int K, double* time);
float* OMPMultiplication(const float* A, const float* B, int M, int N, int K, double* time);