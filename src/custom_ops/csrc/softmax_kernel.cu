#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void custom_softmax_kernel(const float *d_input, float *d_output, int B, int S, int E, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 修正线程的计算方式
    if (idx < B * S) {
        int batch = idx / S;
        int seq = idx % S;
        int offset = batch * S * E + seq * E;

        float max_val = d_input[offset];
        for (int i = 1; i < E; ++i) {
            float val = d_input[offset + i];
            if (val > max_val) {
                max_val = val;
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < E; ++i) {
            d_output[offset + i] = expf(d_input[offset + i] - max_val);
            sum += d_output[offset + i];
        }

        for (int i = 0; i < E; ++i) {
            d_output[offset + i] /= sum;
        }
    }
}

at::Tensor custom_softmax_cuda(at::Tensor input, int dim) {
    if (dim < 0) {
        dim += input.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dim must be in range [0, ", input.dim()-1, "]");

    TORCH_CHECK(input.dim() == 3, "input must be 3D tensor");

    int B = input.size(0);
    int S = input.size(1);
    int E = input.size(2);

    auto output = torch::empty_like(input);

    const float *d_input_ptr = input.data_ptr<float>();
    float *d_output_ptr = output.data_ptr<float>();

    int total = B * S;
    int threads = 1024;  // 增加线程数
    int blocks = (total + threads - 1) / threads;

    custom_softmax_kernel<<<blocks, threads>>>(d_input_ptr, d_output_ptr, B, S, E, dim);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}
