#include <stdlib.h>
#include <torch/extension.h>

// 定义分块大小和线程大小
#define TILE_SIZE_M 16
#define TILE_SIZE_P 16
#define TILE_SIZE_K 16


__global__ void bmm_kernel(const float *d_A, const float *d_B, float *d_C,
                                      int batch_size, int M, int N, int P) {
    // 计算当前批次、行和列
    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE_M + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_P + threadIdx.x;

    // 初始化累加器
    float sum = 0.0f;

    // 分块遍历K维度
    for (int tile = 0; tile < (N + TILE_SIZE_K - 1) / TILE_SIZE_K; ++tile) {
        // 声明共享内存
        __shared__ float As[TILE_SIZE_M][TILE_SIZE_K];
        __shared__ float Bs[TILE_SIZE_K][TILE_SIZE_P];

        // 计算A和B的全局索引
        int A_row = row;
        int A_col = tile * TILE_SIZE_K + threadIdx.x;
        int B_row = tile * TILE_SIZE_K + threadIdx.y;
        int B_col = col;

        // 加载A和B到共享内存
        if (A_row < M && A_col < N) {
            As[threadIdx.y][threadIdx.x] = d_A[batch * M * N + A_row * N + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (B_row < N && B_col < P) {
            Bs[threadIdx.y][threadIdx.x] = d_B[batch * N * P + B_row * P + B_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 同步线程，确保共享内存中的数据已经加载完成
        __syncthreads();

        // 计算部分和
        for (int k = 0; k < TILE_SIZE_K; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // 同步线程，确保所有线程都完成了部分和的计算
        __syncthreads();
    }

    // 写回结果
    if (row < M && col < P && batch < batch_size) {
        d_C[batch * M * P + row * P + col] = sum;
    }
}

__global__ void shared_weight_bmm_kernel(const float *d_A, const float *d_B, float *d_C,
                                      int batch_size, int M, int N, int P) {
    // 计算当前批次、行和列
    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE_M + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_P + threadIdx.x;

    // 初始化累加器
    float sum = 0.0f;

    // 分块遍历K维度
    for (int tile = 0; tile < (N + TILE_SIZE_K - 1) / TILE_SIZE_K; ++tile) {
        __shared__ float As[TILE_SIZE_M][TILE_SIZE_K];
        __shared__ float Bs[TILE_SIZE_K][TILE_SIZE_P];

        // 加载A和B到共享内存
        int A_row = row;
        int A_col = tile * TILE_SIZE_K + threadIdx.x;
        int B_row = tile * TILE_SIZE_K + threadIdx.y;
        int B_col = col;

        As[threadIdx.y][threadIdx.x] = (A_row < M && A_col < N) ? d_A[batch * M * N + A_row * N + A_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (B_row < N && B_col < P) ? d_B[B_row * P + B_col] : 0.0f;

        __syncthreads();

        // 计算部分和
        for (int k = 0; k < TILE_SIZE_K; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 写回结果
    if (row < M && col < P && batch < batch_size) {
        d_C[batch * M * P + row * P + col] = sum;
    }
}


// 封装函数：接收PyTorch张量并调用kernel
at::Tensor custom_bmm(at::Tensor A, at::Tensor B) {
    // 假设输入为 (batch_size, M, N), (batch_size, N, P)
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    int batch_size = A.size(0);
    int M = A.size(1);
    int N = A.size(2);
    int P = B.size(2);

    // 创建输出tensor (batch_size, M, P)
    auto C = torch::zeros({batch_size, M, P}, torch::dtype(torch::kFloat32).device(A.device()));

    // 获取raw pointers
    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((P + blockDim.x - 1)/blockDim.x,
                (M + blockDim.y - 1)/blockDim.y,
                batch_size);

    // 调用kernel
    bmm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, batch_size, M, N, P);

    // 同步和错误检查
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return C;
}

at::Tensor shared_weight_bmm(at::Tensor A, at::Tensor B) {
    // 检查输入张量
    TORCH_CHECK(A.is_cuda(), "A 必须是 CUDA 张量");
    TORCH_CHECK(B.is_cuda(), "B 必须是 CUDA 张量");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A 必须是 float32 类型");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B 必须是 float32 类型");
    TORCH_CHECK(A.dim() == 3, "A 必须是 3 维张量 [B, M, N]");
    TORCH_CHECK(B.dim() == 2, "B 必须是 2 维张量 [N, P]");
    // 确保权重 B 的形状与 A 的最后一个维度匹配
    TORCH_CHECK(A.size(2) == B.size(0), "A 的 N 维度必须等于 B 的 N 维度");

    int batch_size = A.size(0);
    int M = A.size(1);
    int N = A.size(2);
    int P = B.size(1);

    // 创建输出张量 [B, M, P]
    auto C = torch::zeros({batch_size, M, P}, A.options());

    // 获取原始指针
    const float *d_A_ptr = A.data_ptr<float>();
    const float *d_B_ptr = B.data_ptr<float>();
    float *d_C_ptr = C.data_ptr<float>();

    // 定义 block 和 grid 维度
    dim3 blockDim(TILE_SIZE_P, TILE_SIZE_M, 1);
    dim3 gridDim(
        (P + TILE_SIZE_P - 1) / TILE_SIZE_P,
        (M + TILE_SIZE_M - 1) / TILE_SIZE_M,
        batch_size
    );

    // 调用 CUDA 内核
    shared_weight_bmm_kernel<<<gridDim, blockDim>>>(d_A_ptr, d_B_ptr, d_C_ptr, batch_size, M, N, P);

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        // 抛出异常以通知 PyTorch
        TORCH_CHECK(false, "CUDA kernel failed");
    }

    return C;
}
