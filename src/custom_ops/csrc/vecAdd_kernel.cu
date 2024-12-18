#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_vecAdd_kernel(float *d_a, float *d_b, float *d_c, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        d_c[col] = d_a[col] + d_b[col];
    }
}


// 封装的 C++ 函数
at::Tensor custom_vecAdd(at::Tensor A, at::Tensor B) {
    // 检查输入张量是否为 CUDA 张量
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input tensor A must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input tensor B must be of type float32");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input tensors must have the same size");

    int N = A.numel();

    // 创建输出张量
    auto C = torch::empty_like(A);

    // 获取指向数据的指针
    float *d_a = A.data_ptr<float>();
    float *d_b = B.data_ptr<float>();
    float *d_c = C.data_ptr<float>();

    // 定义 CUDA 网格和块的大小
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 启动 CUDA 内核
    custom_vecAdd_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // 同步设备，检查内核是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}