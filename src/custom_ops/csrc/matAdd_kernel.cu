#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 核函数：矩阵相加
__global__ void custom_matAdd_kernel(float *d_A, float *d_B, float *d_C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 执行矩阵加法
    if (row < M && col < N) {
        d_C[row * N + col] = d_A[row * N + col] + d_B[row * N + col];
    }
}

// PyTorch 接口封装
torch::Tensor custom_matAdd(torch::Tensor A, torch::Tensor B) {
    // 检查输入是否为 GPU 张量
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("输入张量必须在 GPU 上");
    }

    // 检查输入矩阵的形状是否一致
    if (A.sizes() != B.sizes()) {
        throw std::invalid_argument("输入张量的形状必须相同");
    }

    // 获取输入矩阵的形状
    int M = A.size(0); // 行数
    int N = A.size(1); // 列数

    // 创建输出张量
    auto C = torch::empty_like(A);

    // 设置线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用 CUDA 核函数
    custom_matAdd_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),      // 输入张量 A
        B.data_ptr<float>(),      // 输入张量 B
        C.data_ptr<float>(),      // 输出张量 C
        M, N                      // 行数和列数
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return C;
}
