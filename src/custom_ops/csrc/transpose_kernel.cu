#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 核函数：批量转置
__global__ void custom_transpose_kernel(const float *d_A, float *d_T, int B, int M, int N) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < B && row < M && col < N) {
        d_T[batch * N * M + col * M + row] = d_A[batch * M * N + row * N + col];
    }
}

// PyTorch 接口封装
at::Tensor custom_transpose(at::Tensor A) {
    // 检查输入是否为 GPU 张量
    TORCH_CHECK(A.is_cuda(), "输入张量必须在 GPU 上");
    TORCH_CHECK(A.dim() == 3, "输入张量必须是 3D 张量");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "输入张量必须是 float32");

    int B = A.size(0); // 批次大小
    int M = A.size(1); // 行数
    int N = A.size(2); // 列数

    // 创建输出张量（转置后的形状）
    auto T = torch::empty({B, N, M}, A.options());

    // 设置线程块和网格大小
    dim3 threadsPerBlock(16, 16, 1); // 每个线程块的大小为 16x16
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   B); // 网格大小

    // 调用 CUDA 核函数
    custom_transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),      // 输入张量
        T.data_ptr<float>(),      // 输出张量
        B, M, N                    // 批次大小, 行数, 列数
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return T;
}
