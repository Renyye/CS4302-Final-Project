#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 核函数
__global__ void custom_transpose_kernel(float *d_A, float *d_T, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 执行转置操作
    if (row < M && col < N) {
        d_T[col * M + row] = d_A[row * N + col];
    }
}

// PyTorch 接口封装
torch::Tensor custom_transpose(torch::Tensor A) {
    // 检查输入是否为 GPU 张量
    if (!A.is_cuda()) {
        throw std::invalid_argument("输入张量必须在 GPU 上");
    }

    // 获取输入矩阵的形状
    int M = A.size(0); // 行数
    int N = A.size(1); // 列数

    // 创建输出张量（转置后的形状）
    auto T = torch::empty({N, M}, A.options());

    // 设置线程块和网格大小
    dim3 threadsPerBlock(16, 16); // 每个线程块的大小为 16x16
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y); // 网格大小

    // 调用 CUDA 核函数
    custom_transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),      // 输入张量
        T.data_ptr<float>(),      // 输出张量
        M, N                      // 行数和列数
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return T;
}