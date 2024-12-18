#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(float* input, float* output, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        float max_val = input[idx * D];
        for(int i = 1; i < D; ++i){
            if(input[idx * D + i] > max_val){
                max_val = input[idx * D + i];
            }
        }
        float sum = 0.0f;
        for(int i = 0; i < D; ++i){
            output[idx * D + i] = expf(input[idx * D + i] - max_val);
            sum += output[idx * D + i];
        }
        for(int i = 0; i < D; ++i){
            output[idx * D + i] /= sum;
        }
    }
}

at::Tensor custom_softmax_cuda(at::Tensor input, int dim) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::zeros_like(input);

    int N = 1;
    int D = 1;
    for(int i=0; i < input.dim(); ++i){
        if(i != dim){
            N *= input.size(i);
        }
        else{
            D = input.size(i);
        }
    }

    float* d_input = input.data_ptr<float>();
    float* d_output = output.data_ptr<float>();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    softmax_kernel<<<blocks, threads>>>(d_input, d_output, N, D);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}
