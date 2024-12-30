#include <stdlib.h>
#include <float.h>
#include <torch/extension.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

/**
 * sgemm_V3 Kernel:
 * 实现分块 (Block Tiling) + 向量化 (float4) 访问 + 双缓冲共享内存(double buffering) 的矩阵乘法。
 *
 * 矩阵大小:
 *    C(MxN) = A(MxK) * B(KxN)
 *
 * 需要配置 gridDim, blockDim:
 *    - 每个 block 处理 128x128 大小的输出 (BM=128, BN=128)
 *    - 在 K 方向上以 8 为步长进行分块 (BK=8)
 *    - 每个线程内再做 8x8 (TM=8, TN=8) 的寄存器块计算
 *
 * 输入参数:
 *    a, b, c : 分别为 A, B, C 矩阵在全局显存中的指针 (row-major)
 *    M, N, K : 矩阵尺寸
 */
__global__ void custom_matMul_kernel_v4(
    float * __restrict__ a, 
    float * __restrict__ b, 
    float * __restrict__ c,
    const int M, 
    const int N, 
    const int K) 
{
    // -----------------------------
    // 设定分块/线程级别参数
    // -----------------------------
    const int BM = 128;  // block 在行方向上覆盖 128 行
    const int BN = 128;  // block 在列方向上覆盖 128 列
    const int BK = 8;    // K方向一次处理 8
    const int TM = 8;    // 每个线程计算 8 行结果
    const int TN = 8;    // 每个线程计算 8 列结果

    // -----------------------------
    // block, thread 索引
    // -----------------------------
    const int bx = blockIdx.x;   // block 在列方向上的编号
    const int by = blockIdx.y;   // block 在行方向上的编号
    const int tx = threadIdx.x;  
    const int ty = threadIdx.y;

    // 每个线程的线性 ID, tid = ty * blockDim.x + tx
    const int tid = ty * blockDim.x + tx;

    // -----------------------------
    // 分配共享内存 
    //   s_a[2][8][128], s_b[2][8][128]
    //   双缓冲, 每个 buffer 大小为 BK x BM / BK x BN
    // -----------------------------
    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    // -----------------------------
    // 寄存器中需要的临时数组
    //   r_load_a, r_load_b 用于加载 float4
    //   r_comp_a, r_comp_b 用于单次计算的 A/B 行或列
    //   r_c_reg[TM][TN] 用于累加结果
    // -----------------------------
    float r_load_a[4];         
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c_reg[TM][TN] = {0.0};  // 初始化为 0

    // -----------------------------
    // 预先计算部分共享内存加载索引
    //  (参考代码中使用的位运算技巧)
    // -----------------------------
    // load_a_smem_m: 在 s_a[?][?][BM] 第三维里的“行”索引
    // load_a_smem_k: 在 s_a[?][BK][?]  第二维里的“列”索引
    int load_a_smem_m = tid >> 1;         // = tid / 2
    int load_a_smem_k = (tid & 1) << 2;   // = (tid % 2)*4

    // load_b_smem_k: 在 s_b[?][BK][?]   第二维里的“行”索引
    // load_b_smem_n: 在 s_b[?][?][BN]   第三维里的“列”索引
    int load_b_smem_k = tid >> 5;         // = tid / 32
    int load_b_smem_n = (tid & 31) << 2;  // = (tid % 32)*4

    // -----------------------------
    // 计算全局内存首地址 (行块,列块)
    // -----------------------------
    int load_a_gmem_m = by * BM + load_a_smem_m; // A 的行坐标
    int load_b_gmem_n = bx * BN + load_b_smem_n; // B 的列坐标

    //==================================================
    // 第一次加载 (bk=0 对应的分块)
    //==================================================
    {
        int load_a_gmem_k = load_a_smem_k; // K方向从 0 开始
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);

        int load_b_gmem_k = load_b_smem_k; // K方向从 0 开始
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        // 1) 从全局内存一次性加载 float4
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        // 2) 写到共享内存 buffer 0
        s_a[0][ load_a_smem_k    ][ load_a_smem_m ] = r_load_a[0];
        s_a[0][ load_a_smem_k + 1][ load_a_smem_m ] = r_load_a[1];
        s_a[0][ load_a_smem_k + 2][ load_a_smem_m ] = r_load_a[2];
        s_a[0][ load_a_smem_k + 3][ load_a_smem_m ] = r_load_a[3];

        FLOAT4(s_b[0][ load_b_smem_k ][ load_b_smem_n ]) = FLOAT4(r_load_b[0]);
    }

    __syncthreads();

    //==================================================
    // 主循环: 遍历所有 K 分块 (从 1 开始)
    //==================================================
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++)
    {
        // smem_sel 与 smem_sel_next 用于双缓冲切换
        int smem_sel       = (bk - 1) & 1; 
        int smem_sel_next  =  bk      & 1; 

        // 1) 读取下一块 A, B 到寄存器
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        // 2) 使用上一次 buffer(smem_sel) 中的数据做计算
        #pragma unroll
        for (int tk = 0; tk < BK; tk++)
        {
            // 从共享内存取 A, B
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ ty * TM / 2 ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ ty * TM / 2 + BM / 2 ]);

            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][ tx * TN / 2 ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][ tx * TN / 2 + BN / 2 ]);

            // 对应 r_c_reg 里的 8x8 计算
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    r_c_reg[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        // 3) 将下一块 (bk) 的数据写入 smem_sel_next 缓冲区
        s_a[smem_sel_next][ load_a_smem_k     ][ load_a_smem_m ] = r_load_a[0];
        s_a[smem_sel_next][ load_a_smem_k + 1 ][ load_a_smem_m ] = r_load_a[1];
        s_a[smem_sel_next][ load_a_smem_k + 2 ][ load_a_smem_m ] = r_load_a[2];
        s_a[smem_sel_next][ load_a_smem_k + 3 ][ load_a_smem_m ] = r_load_a[3];

        FLOAT4(s_b[smem_sel_next][ load_b_smem_k ][ load_b_smem_n ]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

    //==================================================
    // 处理最后一块 (也即 bk = tiles-1 对应 buffer 1)
    //==================================================
    #pragma unroll
    for (int tk = 0; tk < BK; tk++)
    {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ ty * TM / 2 ]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ ty * TM / 2 + BM / 2 ]);

        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][ tx * TN / 2 ]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][ tx * TN / 2 + BN / 2 ]);

        #pragma unroll
        for (int tm = 0; tm < TM; tm++)
        {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++)
            {
                r_c_reg[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }

    //==================================================
    // 将 8x8 的寄存器结果写回全局内存
    //==================================================
    // 分 4 次 (每次 float4) 写回 C
    #pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        // 上半部分
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        FLOAT4(c[store_c_gmem_addr])           = FLOAT4(r_c_reg[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2])  = FLOAT4(r_c_reg[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        // 下半部分
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        FLOAT4(c[store_c_gmem_addr])           = FLOAT4(r_c_reg[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2])  = FLOAT4(r_c_reg[i + TM / 2][4]);
    }
}

// ------------------------
// 3. 封装函数
// ------------------------
at::Tensor custom_matrix_mul_v4(
    at::Tensor A,  // [M, K]
    at::Tensor B   // [K, N]
) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, 
                "A, B must be 2D tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "A, B must be float32");
    TORCH_CHECK(A.size(1) == B.size(0),
                "A.size(1) must match B.size(0), shapes: [M, K], [K, N]");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // 创建输出 [M, N]
    auto C = torch::zeros({M, N}, A.options());

    // 输出A,B, 用printf
    // std::cout << "A: " << A << std::endl;
    // std::cout << "B: " << B << std::endl;


    // 获取原始指针
    float* a_ptr = (float*)A.data_ptr<float>();
    float* b_ptr = (float*)B.data_ptr<float>();
    float* c_ptr = (float*)C.data_ptr<float>();

    // 配置 block / grid
    // 与内核的 BM=128, BN=128, TM=8, TN=8 对应
    // blockDim = (BN/TN=16, BM/TM=16)
    dim3 blockDim(16, 16);

    // gridDim = ( (N+BN-1)/BN, (M+BM-1)/BM )
    dim3 gridDim((N + 128 - 1) / 128,
                 (M + 128 - 1) / 128);

    // 调用 kernel
    custom_matMul_kernel_v4<<<gridDim, blockDim>>>(
        a_ptr, b_ptr, c_ptr,
        M, N, K
    );

    // 错误检查
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}