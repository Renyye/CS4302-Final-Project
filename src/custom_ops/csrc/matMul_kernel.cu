#include <stdlib.h>
#include <torch/extension.h>

__global__ void custom_matMul_kernel(float *d_A, float *d_B, float *d_C, int M, int N, int P) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if(row < M && col < P) {
                float sum = 0.0f;

                // compute the dot product for each row of A and col of B
                for(int i = 0; i < N; ++i) {
                        sum += d_A[row * N + i] * d_B[i * P + col];
                }
                d_C[row * P + col] = sum;
        }
}


// 定义 PyTorch 接口
at::Tensor custom_matMul(at::Tensor A, at::Tensor B) {
    // 检查输入是否为 GPU 张量
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("A 和 B 必须在 GPU 上");
    }

    // 获取输入矩阵的维度
    int M = A.size(0);
    int N = A.size(1);
    int P = B.size(1);

    // 确保输入矩阵的尺寸匹配
    if (A.size(1) != B.size(0)) {
        throw std::invalid_argument("A 的列数必须等于 B 的行数");
    }

    // 分配输出张量
    auto C = torch::zeros({M, P}, A.options());

    // 设置网格和线程块大小
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用 CUDA 核函数
    custom_matMul_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), 
                                                 B.data_ptr<float>(), 
                                                 C.data_ptr<float>(), 
                                                 M, N, P);

    // 检查 CUDA 内核启动的错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return C;
}