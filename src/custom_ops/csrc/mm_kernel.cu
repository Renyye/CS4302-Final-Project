#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// ------------------------
// 1. 宏定义
// ------------------------
// Row-major 偏移计算: (row, col, ld=leading_dim)
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 用于将 float 引用 转换成 float4（需要保证地址是 16 字节对齐）
// 在此示例中保持与原 kernel 风格一致
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// ------------------------
// 2. 核函数声明 (与您给出的相同)
// ------------------------
__global__ void sgemm_V3(
    float * __restrict__ a, 
    float * __restrict__ b, 
    float * __restrict__ c,
    const int M, 
    const int N, 
    const int K) 
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    // 预先计算一些索引
    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // --- 第一次加载 ---
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        // 从全局内存一次性加载 float4
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        // 写到共享内存
        s_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }

    // --- 主循环 ---
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        // 读取下一块 A, B
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        // 对上一次加载的共享内存做计算
        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        // 把下一块加载到共享内存
        s_a[smem_sel_next][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

    // --- 处理最后一块 ---
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2         ]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2         ]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }

    // --- 写回到全局 C ---
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}


// ------------------------
// 3. 封装函数
// ------------------------
at::Tensor custom_matrix_mul(
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
    sgemm_V3<<<gridDim, blockDim>>>(
        a_ptr, b_ptr, c_ptr,
        M, N, K
    );

    // 错误检查
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}
