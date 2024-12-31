#include <stdlib.h>
#include <float.h>
#include <torch/extension.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void custom_matMul_kernel_v4(
    float * __restrict__ a, 
    float * __restrict__ b, 
    float * __restrict__ c,
    const int M, const int N, const int K)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    // block、thread 索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 计算线程一维 ID (tid)
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];  // 128 x 8
    __shared__ float s_b[BK][BN];  //   8 x 128

    // 每线程的寄存器块 8x8
    float r_c[TM][TN] = {0.0f};


    //  - load_a_smem_m: s_a 的行 (0~127)
    //  - load_a_smem_k: s_a 的列 (0,4) (因为一次加载 float4, 跨度为4)
    int load_a_smem_m = tid >> 1;         // tid / 2
    int load_a_smem_k = (tid & 1) << 2;   // (tid % 2 == 0) ? 0 : 4

    //  - load_b_smem_k: s_b 的行 (0~7)   (tid / 32)
    //  - load_b_smem_n: s_b 的列 (0~127) (因为一次加载 float4, 跨度为4)
    int load_b_smem_k = tid >> 5;         // tid / 32
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32)*4

    // 对应全局内存中的初始地址
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // 循环遍历 K 方向 tile
    int tiles = (K + BK - 1) / BK; // 向上取整
    for (int bk = 0; bk < tiles; bk++)
    {
        // 计算全局地址
        int load_a_gmem_k = bk * BK + load_a_smem_k;  // 对应 A 矩阵列
        int load_b_gmem_k = bk * BK + load_b_smem_k;  // 对应 B 矩阵行

        // 加载 A 的一个 float4 到 s_a
        if ( (load_a_gmem_m < M) && (load_a_gmem_k + 3 < K) ) {
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = 
                 FLOAT4(a[load_a_gmem_addr]);
        }
        else {
            // 越界时，填充 0
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = make_float4(0.f,0.f,0.f,0.f);
        }

        // 加载 B 的一个 float4 到 s_b
        if ( (load_b_gmem_k < K) && (load_b_gmem_n + 3 < N) ) {
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = 
                 FLOAT4(b[load_b_gmem_addr]);
        }
        else {
            // 越界时，填充 0
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = make_float4(0.f,0.f,0.f,0.f);
        }

        __syncthreads();

        // 进行本 tile 的乘加
        #pragma unroll
        for (int k = 0; k < BK; k++)
        {
            #pragma unroll
            for (int m = 0; m < TM; m++)
            {
                // 线程在 s_a 上读取行: (ty*TM + m)
                float a_val = s_a[ty*TM + m][k];
                #pragma unroll
                for (int n = 0; n < TN; n++)
                {
                    // 线程在 s_b 上读取列: (tx*TN + n)
                    r_c[m][n] += a_val * s_b[k][tx*TN + n];
                }
            }
        }

        __syncthreads();
    }

    // 将结果写回全局内存
    // 每次写回一个 float4
    #pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int store_c_gmem_m = by * BM + ty * TM + i;
        // 每次写4个 float，故 j += 4
        #pragma unroll
        for (int j = 0; j < TN; j += 4)
        {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            if (store_c_gmem_m < M && (store_c_gmem_n + 3) < N)
            {
                int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
                FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
            }
        }
    }
}


at::Tensor custom_matrix_mul_v4(at::Tensor A, at::Tensor B) {
    // 检查输入张量在CUDA上且是float类型
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input tensor A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input tensor B must be float32");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match for matMul");

    // 获取张量维度
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // 创建输出张量
    auto C = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // 获取设备数据指针
    float* d_A = A.data_ptr<float>();
    float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 设置 CUDA 核函数的线程块和网格大小
    const int BM = 128; // 每个 block 处理的行数
    const int BN = 128; // 每个 block 处理的列数
    dim3 blockDim(16, 16); // 每个线程块的线程数
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM); // 网格大小

    // 调用 CUDA 核函数
    custom_matMul_kernel_v4<<<gridDim, blockDim>>>(
        d_A, d_B, d_C, M, N, K
    );

    // 检查 CUDA 内核是否正确执行
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // 返回结果张量
    return C;
}