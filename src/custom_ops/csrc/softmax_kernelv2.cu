#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void custom_softmax_kernel_v2(float* out, const float* inp, int N, int C) { 
    // inp is (N, C)
    // in each row of C elements, first calculates maxval, then returns expf(val - maxval)
    extern __shared__ float shared[];
    int idx = blockIdx.x; // ranges [0, N)
    int tid = threadIdx.x; // ranges [0, block_size)
    int block_size = blockDim.x;
    const float* x = inp + idx * C; // idx-th row of inp
    // thread coarsening
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();
    float offset = shared[0];
    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = expf(x[i] - offset);
    }
    __syncthreads();
    // thread coarsening again, for the sum
    x = out + idx * C; // idx-th row of out
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }
    shared[tid] = sumval;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    __syncthreads();
    float sum = shared[0];
    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = x[i] / sum;
    }
}

at::Tensor custom_softmax_cuda_v2(at::Tensor input, int dim) {
    const auto N = input.size(0);
    const auto C = input.size(1);
    auto output = torch::empty_like(input);
    const int block_size = 256;  // 可根据实际情况调整
    dim3 grid(N);
    dim3 block(block_size);
    // 动态共享内存大小：block_size * sizeof(float)
    size_t sharedMemSize = block_size * sizeof(float);

    custom_softmax_kernel_v2<<<grid, block, sharedMemSize>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C
    );
    return output;
}