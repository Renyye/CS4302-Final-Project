#include <stdio.h>
#include <stdlib.h>

// matrix addition kernel
__global__ void matAdd(float *d_A, float *d_B, float *d_C, int M, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// add matrix elements
	if (row < M && col < N) {
		d_C[row * N + col] = d_A[row * N + col] + d_B[row * N + col];
	}
}
