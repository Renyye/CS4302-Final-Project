#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

at::Tensor custom_relu_cuda(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int N = input.numel();
    auto output = torch::zeros_like(input);

    float* d_input = input.data_ptr<float>();
    float* d_output = output.data_ptr<float>();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}
