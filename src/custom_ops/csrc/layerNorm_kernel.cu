#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void layerNorm_kernel(float* input, float* output, float* gamma, float* beta, int N, int normalized_shape) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float mean = 0.0f;
        float var = 0.0f;
        for(int i = 0; i < normalized_shape; ++i){
            mean += input[idx * normalized_shape + i];
        }
        mean /= normalized_shape;
        for(int i = 0; i < normalized_shape; ++i){
            float diff = input[idx * normalized_shape + i] - mean;
            var += diff * diff;
        }
        var /= normalized_shape;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for(int i = 0; i < normalized_shape; ++i){
            output[idx * normalized_shape + i] = gamma[i] * (input[idx * normalized_shape + i] - mean) * inv_std + beta[i];
        }
    }
}

at::Tensor custom_layerNorm_cuda(at::Tensor input, at::Tensor gamma, at::Tensor beta, int normalized_shape) {
    // 输入维度检查
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D tensor");

    int batch_size = input.size(0);
    int dim = normalized_shape;

    // 创建输出张量
    auto output = torch::zeros_like(input);

    // 获取指针
    float* d_input = input.data_ptr<float>();
    float* d_output = output.data_ptr<float>();
    float* d_gamma = gamma.data_ptr<float>();
    float* d_beta = beta.data_ptr<float>();

    // 配置block和grid
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    // 调用kernel
    layerNorm_kernel<<<blocks, threads>>>(d_input, d_output, d_gamma, d_beta, batch_size, dim);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}
