#include "GPU.cuh"

__global__ void kernel(const float* A, const float* B, float* C, int M, int N, int K) {
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	float summ = 0.0f;

	for (size_t i = 0; i < N; i++) summ += A[Y * N + i] * B[i * K + X];

	C[Y * K + X] = summ;
}

__global__ void kernel_optimized(const float* A, const float* B, float* C, int M, int N, int K) {

	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	float summ = 0.0f;
	__shared__ float A_[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float B_[BLOCK_SIZE * BLOCK_SIZE];
	//__shared__ float A_[BLOCK_SIZE][BLOCK_SIZE];
	//__shared__ float B_[BLOCK_SIZE][BLOCK_SIZE];

	for (size_t i = 0; i < N; i += BLOCK_SIZE){
		A_[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[Y * N + (i + threadIdx.x)];
		B_[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(i + threadIdx.y) * K + X];
		//A_[threadIdx.x][threadIdx.y] = A[Y * N + (i + threadIdx.x)];
		//B_[threadIdx.x][threadIdx.y] = B[(i + threadIdx.y) * K + X];
		__syncthreads();
		for (size_t j = 0; j < BLOCK_SIZE; j++) summ += A_[threadIdx.y * BLOCK_SIZE + j] * B_[j * BLOCK_SIZE + threadIdx.x];
		//for (size_t j = 0; j < BLOCK_SIZE; j++) summ += A_[j][threadIdx.y] * B_[threadIdx.x][j];
		__syncthreads();
	}

	C[Y * K + X] = summ;
}

float* GPUMultiplication(const float* A, const float* B, int M, int N, int K, bool optimized, double* time){

	float* C = new float[M * K]{ 0 };

	float* GPU_A;
	float* GPU_B;
	float* GPU_C;

	if (cudaMalloc((void**)&GPU_A, M * N * sizeof(float))) printf("cudaMalloc GPU_A error\n");
	if (cudaMalloc((void**)&GPU_B, N * K * sizeof(float))) printf("cudaMalloc GPU_B error\n");
	if (cudaMalloc((void**)&GPU_C, M * K * sizeof(float))) printf("cudaMalloc GPU_C error\n");

	if (cudaMemcpy(GPU_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice)) printf("cudaMemcpy A to GPU_A error\n");
	if (cudaMemcpy(GPU_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice)) printf("cudaMemcpy B to GPU_B error\n");
	if (cudaMemcpy(GPU_C, C, M * K * sizeof(float), cudaMemcpyHostToDevice)) printf("cudaMemcpy C to GPU_C error\n");

	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize(M / BLOCK_SIZE, K / BLOCK_SIZE);

	double t1 = omp_get_wtime();

	if(optimized) kernel_optimized<<<gridSize, blockSize>>> (GPU_A, GPU_B, GPU_C, M, N, K);
	else kernel<<<gridSize, blockSize>>> (GPU_A, GPU_B, GPU_C, M, N, K);
	
	if (cudaDeviceSynchronize()) printf("cudaDeviceSynchronize error\n");

	double t2 = omp_get_wtime();
	*time = t2 - t1;

	if (cudaGetLastError()) printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
	if (cudaMemcpy(C, GPU_C, M * K * sizeof(float), cudaMemcpyDeviceToHost)) printf("cudaMemcpy GPU_C to C error\n");

	cudaFree(GPU_A);
	cudaFree(GPU_B);
	cudaFree(GPU_C);

	return C;
}