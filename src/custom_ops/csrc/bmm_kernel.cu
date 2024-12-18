#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 4  // 定义子块大小

template <typename scalar_t>
__global__ void bmm_kernel(const scalar_t *d_A, const scalar_t *d_B, scalar_t *d_C, int batch_size, int M, int N, int P) {
    // 计算当前线程所属的批次、行和列
    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 如果是小矩阵 (1x1)，直接执行简单的乘法
    if (M == 1 && N == 1 && P == 1) {
        if (batch < batch_size && row == 0 && col == 0) {
            d_C[batch] = d_A[batch] * d_B[batch];  // 直接进行乘法
        }
        return;
    }

    // 定义共享内存子块
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE];

    scalar_t value = 0.0;

    // 计算需要的子块数量
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; ++tile) {
        // 加载A的子块到共享内存
        if (row < M && (tile * TILE_SIZE + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = d_A[batch * M * N + row * N + tile * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        // 加载B的子块到共享内存
        if (col < P && (tile * TILE_SIZE + threadIdx.y) < N) {
            Bs[threadIdx.y][threadIdx.x] = d_B[batch * N * P + (tile * TILE_SIZE + threadIdx.y) * P + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();  // 等待所有线程加载完成

        // 计算子块的乘积
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // 等待所有线程完成计算
    }

    // 写入结果到全局内存
    if (batch < batch_size && row < M && col < P) {
        d_C[batch * M * P + row * P + col] = value;
    }
}


// Host function that wraps the CUDA kernel
at::Tensor custom_bmm(at::Tensor A, at::Tensor B) {
    // 检查输入张量属性
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // 获取数据类型
    auto dtype = A.dtype();

    // 根据数据类型选择不同的模板
    if (dtype == torch::kFloat32) {
        // 对应float32
        const float* d_A = A.data_ptr<float>();
        const float* d_B = B.data_ptr<float>();
        float* d_C = A.new_zeros({A.size(0), A.size(1), B.size(2)}, torch::dtype(torch::kFloat32)).data_ptr<float>();
        
        // 定义 block 和 grid 的维度
        dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
        dim3 gridDim((B.size(2) + TILE_SIZE - 1) / TILE_SIZE,
                     (A.size(1) + TILE_SIZE - 1) / TILE_SIZE,
                     A.size(0));

        // 调用 kernel
        bmm_kernel<float><<<gridDim, blockDim>>>(d_A, d_B, d_C, A.size(0), A.size(1), A.size(2), B.size(2));

        // 同步和错误检查
        cudaError_t err = cudaDeviceSynchronize();
        TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

        return torch::from_blob(d_C, {A.size(0), A.size(1), B.size(2)}, torch::dtype(torch::kFloat32).device(A.device()));  // 正确
    } else if (dtype == torch::kFloat64) {
        // 对应float64
        const double* d_A = A.data_ptr<double>();
        const double* d_B = B.data_ptr<double>();
        double* d_C = A.new_zeros({A.size(0), A.size(1), B.size(2)}, torch::dtype(torch::kFloat64)).data_ptr<double>();

        // 定义 block 和 grid 的维度
        dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
        dim3 gridDim((B.size(2) + TILE_SIZE - 1) / TILE_SIZE,
                     (A.size(1) + TILE_SIZE - 1) / TILE_SIZE,
                     A.size(0));

        // 调用 kernel
        bmm_kernel<double><<<gridDim, blockDim>>>(d_A, d_B, d_C, A.size(0), A.size(1), A.size(2), B.size(2));

        // 同步和错误检查
        cudaError_t err = cudaDeviceSynchronize();
        TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

        return torch::from_blob(d_C, {A.size(0), A.size(1), B.size(2)}, torch::dtype(torch::kFloat32).device(A.device()));  // 正确
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}
