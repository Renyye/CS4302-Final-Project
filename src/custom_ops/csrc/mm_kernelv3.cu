#include <stdlib.h>
#include <float.h>
#include <torch/extension.h>

#define TILE_WIDTH 32

__global__ void custom_matMul_kernel_v3(float *d_A, float *d_B, float *d_C, int M, int N, int P) {
    // 定义共享内存用于存储块内子矩阵
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // 以块为单位遍历 A 和 B
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // 加载 A 和 B 的块到共享内存
        if (row < M && t * TILE_WIDTH + threadIdx.x < N) {
            tile_A[threadIdx.y][threadIdx.x] = d_A[row * N + t * TILE_WIDTH + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < P && t * TILE_WIDTH + threadIdx.y < N) {
            tile_B[threadIdx.y][threadIdx.x] = d_B[(t * TILE_WIDTH + threadIdx.y) * P + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // 确保共享内存加载完成

        // 计算块内的乘积
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads(); // 确保计算完成后再加载下一块
    }

    // 写回结果到全局内存
    if (row < M && col < P) {
        d_C[row * P + col] = sum;
    }
}



// 定义 PyTorch 接口
at::Tensor custom_matrix_mul_v3(at::Tensor A, at::Tensor B) {
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
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((P + TILE_WIDTH - 1) / TILE_WIDTH,
                   (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // 调用共享内存优化的 CUDA 核函数
    custom_matMul_kernel_v3<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), 
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