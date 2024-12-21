#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// 核心参数
#define TILE_SIZE 128
#define THREAD_SIZE 16
#define K_TILE 32

// 核函数声明
__global__ void matrixMul_kernel(
    const float *A, const float *B, float *C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][K_TILE];
    __shared__ float Bs[K_TILE][TILE_SIZE];

    // row, col 是该块处理的子矩阵 C 的起始坐标
    int row = blockIdx.y * TILE_SIZE + THREAD_SIZE * threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + THREAD_SIZE * threadIdx.x;

    // 计算需要的 tile 数 (沿 K 方向)
    int tile_num = (K + K_TILE - 1) / K_TILE;

    // 用二维数组 value[THREAD_SIZE][THREAD_SIZE] 存储每个线程需计算的输出块
    float value[THREAD_SIZE][THREAD_SIZE];
    for (int rr = 0; rr < THREAD_SIZE; rr++) {
        for (int cc = 0; cc < THREAD_SIZE; cc++) {
            value[rr][cc] = 0.0f;
        }
    }

    // 主循环：分块加载并计算
    for (int tile = 0; tile < tile_num; tile++) {
        // 1) 加载 A 到共享内存 As
        for (int ind_row = 0; ind_row < THREAD_SIZE; ind_row++) {
            for (int ind_col = 0; ind_col < K_TILE; ind_col++) {
                int Arow = row + ind_row;       // 全局行
                int Acol = tile * K_TILE + ind_col; // 全局列
                if (Arow < M && Acol < K) {
                    As[threadIdx.y * THREAD_SIZE + ind_row][ind_col]
                        = A[Arow * K + Acol];
                } else {
                    As[threadIdx.y * THREAD_SIZE + ind_row][ind_col] = 0.0f;
                }
            }
        }
        __syncthreads();

        // 2) 加载 B 到共享内存 Bs
        for (int ind_row = 0; ind_row < K_TILE; ind_row++) {
            for (int ind_col = 0; ind_col < THREAD_SIZE; ind_col++) {
                int Brow = tile * K_TILE + ind_row; // 全局行
                int Bcol = col + ind_col;           // 全局列
                if (Brow < K && Bcol < N) {
                    Bs[ind_row][threadIdx.x * THREAD_SIZE + ind_col]
                        = B[Brow * N + Bcol];
                } else {
                    Bs[ind_row][threadIdx.x * THREAD_SIZE + ind_col] = 0.0f;
                }
            }
        }
        __syncthreads();

        // 3) 计算线程私有 value[][] 累加
        for (int i = 0; i < K_TILE; i++) {
            for (int ind_row = 0; ind_row < THREAD_SIZE; ind_row++) {
                for (int ind_col = 0; ind_col < THREAD_SIZE; ind_col++) {
                    value[ind_row][ind_col] +=
                        As[threadIdx.y * THREAD_SIZE + ind_row][i] *
                        Bs[i][threadIdx.x * THREAD_SIZE + ind_col];
                }
            }
        }
        __syncthreads();
    }

    // 4) 将结果写回全局内存
    for (int ind_row = 0; ind_row < THREAD_SIZE; ind_row++) {
        for (int ind_col = 0; ind_col < THREAD_SIZE; ind_col++) {
            int Crow = row + ind_row;
            int Ccol = col + ind_col;
            if (Crow < M && Ccol < N) {
                C[Crow * N + Ccol] = value[ind_row][ind_col];
            }
        }
    }
}

// 封装成一个函数供 PyTorch 调用
at::Tensor custom_matrix_mul(at::Tensor A, at::Tensor B) {
    // A: [M, K]
    // B: [K, N]
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
        "A and B must be 2D, got A.dim()=", A.dim(), ", B.dim()=", B.dim());
    TORCH_CHECK(A.size(1) == B.size(0),
        "Inner dimensions must match, A.size(1)=", A.size(1),
        ", B.size(0)=", B.size(0));

    // 确定 M, N, K
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    // 创建输出张量 [M, N]
    auto C = torch::zeros({M, N}, A.options());

    // 获取原始指针
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr       = C.data_ptr<float>();

    // 配置 grid, block
    dim3 blockDim(TILE_SIZE / THREAD_SIZE, TILE_SIZE / THREAD_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // 调用 kernel
    matrixMul_kernel<<<gridDim, blockDim>>>( (float*)A_ptr, (float*)B_ptr,
                                            (float*)C_ptr, M, N, K );

    // 同步并检查错误
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Kernel failed : ", cudaGetErrorString(err));

    return C;
}
