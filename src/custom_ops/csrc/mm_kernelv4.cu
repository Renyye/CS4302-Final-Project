#include <stdlib.h>
#include <float.h>
#include <torch/extension.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void custom_matMul_kernel_v4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    // 规定块大小（覆盖C中的子块大小）
    const int BM = 128;   // 每个 block 负责 C 的 128 行
    const int BN = 128;   // 每个 block 负责 C 的 128 列
    const int BK = 8;     // 分块（tile）宽度（K 方向一次处理 8）
    
    // 规定线程在 block 内负责 8x8 大小的输出子块
    const int TM = 8;     // 每个线程在行方向上处理 8 个输出元素
    const int TN = 8;     // 每个线程在列方向上处理 8 个输出元素

    // 计算该 block 在网格 (grid) 中的坐标
    int bx = blockIdx.x;  // block 在 x 方向的索引, 对应输出 C 的"行块"
    int by = blockIdx.y;  // block 在 y 方向的索引, 对应输出 C 的"列块"

    // 计算该线程在 block 内的坐标
    int tx = threadIdx.x; // 线程 x 方向索引
    int ty = threadIdx.y; // 线程 y 方向索引

    // 该 block 在 C 中左上角的全局行/列索引
    int block_start_row = bx * BM;
    int block_start_col = by * BN;

    // 计算要分多少次 tile 才能覆盖完 K
    int num_tiles = (K + BK - 1) / BK;

    // -----------------------------
    // 为 A、B 分配共享内存 (更合理的排布)
    //   s_a:  128 x 8
    //   s_b:    8 x 128
    // -----------------------------
    __shared__ float s_a[128][BK];  // = [128][8]
    __shared__ float s_b[BK][128];  // = [8][128]

    // -----------------------------
    // 每个线程维护一个 8x8 的累加寄存器块
    // -----------------------------
    float r_c[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
    {
        #pragma unroll
        for (int j = 0; j < TN; j++)
        {
            r_c[i][j] = 0.0f;
        }
    }

    // -----------------------------
    // 遍历所有的 tile (在K方向)
    // -----------------------------
    for (int tile = 0; tile < num_tiles; tile++)
    {
        // 1) 把 A 的一块 [128 x BK] 加载到 s_a
        //    这里让 (threadIdx.x, threadIdx.y) 通过小循环分担所有元素
        {
            // A 的全局起始: (block_start_row, tile*BK)
            // A 的大小: M x K
            // s_a 的大小: 128 x 8
            // 我们让每个线程通过 2D 循环把需要的元素搬到共享内存

            for (int i = ty; i < 128; i += blockDim.y)    // blockDim.y=16
            {
                for (int j = tx; j < BK;   j += blockDim.x) // blockDim.x=16, BK=8
                {
                    int global_row = block_start_row + i;      // 行
                    int global_col = tile * BK + j;            // 列
                    if (global_row < M && global_col < K)
                        s_a[i][j] = A[ global_row * K + global_col ];
                    else
                        s_a[i][j] = 0.0f;
                }
            }
        }

        // 2) 把 B 的一块 [BK x 128] 加载到 s_b
        {
            // B 的全局起始: (tile*BK, block_start_col)
            // B 的大小: K x N
            // s_b 的大小: 8 x 128
            // 同理，每个线程通过 2D 循环把 B 分块加载进来

            for (int i = ty; i < BK;   i += blockDim.y)    // BK=8
            {
                for (int j = tx; j < 128; j += blockDim.x) // 128
                {
                    int global_row = tile * BK + i;         // 行
                    int global_col = block_start_col + j;    // 列
                    if (global_row < K && global_col < N)
                        s_b[i][j] = B[ global_row * N + global_col ];
                    else
                        s_b[i][j] = 0.0f;
                }
            }
        }

        // 同步，确保共享内存已填充完毕
        __syncthreads();

        // 3) 做本 tile 的乘加运算
        //    对 s_a [128 x 8] 和 s_b [8 x 128]
        //    每个线程要完成 8x8 的结果累加
        {
            // 先确定线程在 s_a、s_b 中读取的行/列范围
            // 每个线程在 C 中对 (ty*TM + i, tx*TN + j) 那些行列负责
            // 逐列 (k) 做乘加
            #pragma unroll
            for (int k = 0; k < BK; k++)
            {
                #pragma unroll
                for (int i = 0; i < TM; i++)
                {
                    float a_val = s_a[ ty*TM + i ][k];
                    #pragma unroll
                    for (int j = 0; j < TN; j++)
                    {
                        r_c[i][j] += a_val * s_b[k][ tx*TN + j ];
                    }
                }
            }
        }

        // 同步，为下一次 tile 加载做准备
        __syncthreads();
    }

    // -----------------------------
    // 写回结果到全局内存
    // -----------------------------
    for (int i = 0; i < TM; i++)
    {
        int c_row = block_start_row + ty*TM + i;
        for (int j = 0; j < TN; j++)
        {
            int c_col = block_start_col + tx*TN + j;
            if (c_row < M && c_col < N)
            {
                C[c_row * N + c_col] = r_c[i][j];
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