#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void custom_softmax_kernel_v1(float* out, const float* inp, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;
        }
    }
}

at::Tensor custom_softmax_cuda_v1(at::Tensor input, int dim) {
    // Ensure the input tensor is contiguous
    input = input.contiguous();

    // Get the number of dimensions
    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Dimension out of range");

    // Compute N (number of batches) and C (number of classes)
    int N = 1;
    int C = input.size(dim);
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            N *= input.size(i);
        }
    }

    // Prepare the output tensor
    auto output = at::empty_like(input);

    // Define CUDA kernel launch parameters
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    // Launch the CUDA kernel
    custom_softmax_kernel_v1<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N,
        C
    );

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    }

    return output;
}