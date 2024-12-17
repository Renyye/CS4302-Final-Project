#include <stdio.h>
#include <stdlib.h>

// dot prod kernel
__global__ void dotProd(float *d_a, float *d_b, float *d_dotprod, float *prodVec, int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	// element-wise products
	if (col < N) {
		prodVec[tid] = d_a[col] * d_b[col];
	}
	
	// determine amount of padding for parallel reduction
	int padding = 0;
	for (int e = 0; (float)N/(float)(1 << e)>= 1; ++e) {
	      	padding = e + 1;
	}

	for (int i = 0; i < (1 << padding) - N; ++i) {
		prodVec[N + i] = 0;
	}

	__syncthreads();

	// sum using parallel reduction
	for (int stride = 1 << padding; stride >= 1; stride /= 2) {
		if (col < stride) {
			prodVec[col] += prodVec[col + stride];
		}
	}	
	
	__syncthreads();

	// set dotprod
	d_dotprod[0] = prodVec[0];
}
