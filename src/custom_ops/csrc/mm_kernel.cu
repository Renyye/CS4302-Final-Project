#include <stdlib.h>
#include <float.h>
#include <torch/extension.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// // version 1 未使用 shared memory
// __global__ void custom_matMul_kernel(float *d_A, float *d_B, float *d_C, int M, int N, int P) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if(row < M && col < P) {
//         float sum = 0.0f;

//         // compute the dot product for each row of A and col of B
//         for(int i = 0; i < N; ++i) {
//             sum += d_A[row * N + i] * d_B[i * P + col];
//         }
//         d_C[row * P + col] = sum;
//     }
// }


// at::Tensor custom_matrix_mul(at::Tensor A, at::Tensor B) {
//     // 检查输入是否为 GPU 张量
//     if (!A.is_cuda() || !B.is_cuda()) {
//         throw std::invalid_argument("A 和 B 必须在 GPU 上");
//     }

//     // 获取输入矩阵的维度
//     int M = A.size(0);
//     int N = A.size(1);
//     int P = B.size(1);

//     // 确保输入矩阵的尺寸匹配
//     if (A.size(1) != B.size(0)) {
//         throw std::invalid_argument("A 的列数必须等于 B 的行数");
//     }

//     // 分配输出张量
//     auto C = torch::zeros({M, P}, A.options());

//     // 设置网格和线程块大小
//     dim3 threadsPerBlock(16, 16);
//     dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     // 调用 CUDA 核函数
//     custom_matMul_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), 
//                                                  B.data_ptr<float>(), 
//                                                  C.data_ptr<float>(), 
//                                                  M, N, P);

//     // 检查 CUDA 内核启动的错误
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
//     }

//     return C;
// }


// // version 2 使用 shared memory

// __global__ void custom_matMul_kernel(float *d_A, float *d_B, float *d_C, int M, int N, int P) {
//     // 定义共享内存用于存储块内子矩阵
//     __shared__ float tile_A[32][32];
//     __shared__ float tile_B[32][32];

//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     float sum = 0.0f;

//     // 以块为单位遍历 A 和 B
//     for (int t = 0; t < (N + 31) / 32; ++t) {
//         // 加载 A 和 B 的块到共享内存
//         if (row < M && t * 32 + threadIdx.x < N) {
//             tile_A[threadIdx.y][threadIdx.x] = d_A[row * N + t * 32 + threadIdx.x];
//         } else {
//             tile_A[threadIdx.y][threadIdx.x] = 0.0f;
//         }

//         if (col < P && t * 32 + threadIdx.y < N) {
//             tile_B[threadIdx.y][threadIdx.x] = d_B[(t * 32 + threadIdx.y) * P + col];
//         } else {
//             tile_B[threadIdx.y][threadIdx.x] = 0.0f;
//         }

//         __syncthreads(); // 确保共享内存加载完成

//         // 计算块内的乘积
//         for (int i = 0; i < 32; ++i) {
//             sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
//         }

//         __syncthreads(); // 确保计算完成后再加载下一块
//     }

//     // 写回结果到全局内存
//     if (row < M && col < P) {
//         d_C[row * P + col] = sum;
//     }
// }



// // 定义 PyTorch 接口
// at::Tensor custom_matrix_mul(at::Tensor A, at::Tensor B) {
//     // 检查输入是否为 GPU 张量
//     if (!A.is_cuda() || !B.is_cuda()) {
//         throw std::invalid_argument("A 和 B 必须在 GPU 上");
//     }

//     // 获取输入矩阵的维度
//     int M = A.size(0);
//     int N = A.size(1);
//     int P = B.size(1);

//     // 确保输入矩阵的尺寸匹配
//     if (A.size(1) != B.size(0)) {
//         throw std::invalid_argument("A 的列数必须等于 B 的行数");
//     }

//     // 分配输出张量
//     auto C = torch::zeros({M, P}, A.options());

//     // 设置网格和线程块大小
//     dim3 threadsPerBlock(32, 32);
//     dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     // 调用共享内存优化的 CUDA 核函数
//     custom_matMul_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), 
//                                                                 B.data_ptr<float>(), 
//                                                                 C.data_ptr<float>(), 
//                                                                 M, N, P);

//     // 检查 CUDA 内核启动的错误
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
//     }

//     return C;
// }


// version 3
// __global__ void custom_matMul_kernel(
//     const float* __restrict__ A,
//     const float* __restrict__ B,
//     float*       __restrict__ C,
//     int M, int N, int K)
// {
//     // 规定块大小（覆盖C中的子块大小）
//     const int BM = 128;   // 每个 block 负责 C 的 128 行
//     const int BN = 128;   // 每个 block 负责 C 的 128 列
//     const int BK = 8;     // 分块（tile）宽度（K 方向一次处理 8）
    
//     // 规定线程在 block 内负责 8x8 大小的输出子块
//     const int TM = 8;     // 每个线程在行方向上处理 8 个输出元素
//     const int TN = 8;     // 每个线程在列方向上处理 8 个输出元素

//     // 计算该 block 在网格 (grid) 中的坐标
//     int bx = blockIdx.x;  // block 在 x 方向的索引, 对应输出 C 的"行块"
//     int by = blockIdx.y;  // block 在 y 方向的索引, 对应输出 C 的"列块"

//     // 计算该线程在 block 内的坐标
//     int tx = threadIdx.x; // 线程 x 方向索引
//     int ty = threadIdx.y; // 线程 y 方向索引

//     // 该 block 在 C 中左上角的全局行/列索引
//     int block_start_row = bx * BM;
//     int block_start_col = by * BN;

//     // 计算要分多少次 tile 才能覆盖完 K
//     int num_tiles = (K + BK - 1) / BK;

//     // -----------------------------
//     // 为 A、B 分配共享内存 (更合理的排布)
//     //   s_a:  128 x 8
//     //   s_b:    8 x 128
//     // -----------------------------
//     __shared__ float s_a[128][BK];  // = [128][8]
//     __shared__ float s_b[BK][128];  // = [8][128]

//     // -----------------------------
//     // 每个线程维护一个 8x8 的累加寄存器块
//     // -----------------------------
//     float r_c[TM][TN];
//     #pragma unroll
//     for (int i = 0; i < TM; i++)
//     {
//         #pragma unroll
//         for (int j = 0; j < TN; j++)
//         {
//             r_c[i][j] = 0.0f;
//         }
//     }

//     // -----------------------------
//     // 遍历所有的 tile (在K方向)
//     // -----------------------------
//     for (int tile = 0; tile < num_tiles; tile++)
//     {
//         // 1) 把 A 的一块 [128 x BK] 加载到 s_a
//         //    这里让 (threadIdx.x, threadIdx.y) 通过小循环分担所有元素
//         {
//             // A 的全局起始: (block_start_row, tile*BK)
//             // A 的大小: M x K
//             // s_a 的大小: 128 x 8
//             // 我们让每个线程通过 2D 循环把需要的元素搬到共享内存

//             for (int i = ty; i < 128; i += blockDim.y)    // blockDim.y=16
//             {
//                 for (int j = tx; j < BK;   j += blockDim.x) // blockDim.x=16, BK=8
//                 {
//                     int global_row = block_start_row + i;      // 行
//                     int global_col = tile * BK + j;            // 列
//                     if (global_row < M && global_col < K)
//                         s_a[i][j] = A[ global_row * K + global_col ];
//                     else
//                         s_a[i][j] = 0.0f;
//                 }
//             }
//         }

//         // 2) 把 B 的一块 [BK x 128] 加载到 s_b
//         {
//             // B 的全局起始: (tile*BK, block_start_col)
//             // B 的大小: K x N
//             // s_b 的大小: 8 x 128
//             // 同理，每个线程通过 2D 循环把 B 分块加载进来

//             for (int i = ty; i < BK;   i += blockDim.y)    // BK=8
//             {
//                 for (int j = tx; j < 128; j += blockDim.x) // 128
//                 {
//                     int global_row = tile * BK + i;         // 行
//                     int global_col = block_start_col + j;    // 列
//                     if (global_row < K && global_col < N)
//                         s_b[i][j] = B[ global_row * N + global_col ];
//                     else
//                         s_b[i][j] = 0.0f;
//                 }
//             }
//         }

//         // 同步，确保共享内存已填充完毕
//         __syncthreads();

//         // 3) 做本 tile 的乘加运算
//         //    对 s_a [128 x 8] 和 s_b [8 x 128]
//         //    每个线程要完成 8x8 的结果累加
//         {
//             // 先确定线程在 s_a、s_b 中读取的行/列范围
//             // 每个线程在 C 中对 (ty*TM + i, tx*TN + j) 那些行列负责
//             // 逐列 (k) 做乘加
//             #pragma unroll
//             for (int k = 0; k < BK; k++)
//             {
//                 #pragma unroll
//                 for (int i = 0; i < TM; i++)
//                 {
//                     float a_val = s_a[ ty*TM + i ][k];
//                     #pragma unroll
//                     for (int j = 0; j < TN; j++)
//                     {
//                         r_c[i][j] += a_val * s_b[k][ tx*TN + j ];
//                     }
//                 }
//             }
//         }

//         // 同步，为下一次 tile 加载做准备
//         __syncthreads();
//     }

//     // -----------------------------
//     // 写回结果到全局内存
//     // -----------------------------
//     for (int i = 0; i < TM; i++)
//     {
//         int c_row = block_start_row + ty*TM + i;
//         for (int j = 0; j < TN; j++)
//         {
//             int c_col = block_start_col + tx*TN + j;
//             if (c_row < M && c_col < N)
//             {
//                 C[c_row * N + c_col] = r_c[i][j];
//             }
//         }
//     }
// }





// __global__ void custom_matMul_kernel(
//     float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
//     const int M, const int N, const int K) {

//     const int BM = 128;
//     const int BN = 128;
//     const int BK = 8;
//     const int TM = 8;
//     const int TN = 8;

//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int tid = ty * blockDim.x + tx;

//     __shared__ float s_a[BM][BK];
//     __shared__ float s_b[BK][BN];

//     float r_c[TM][TN] = {0.0};

//     int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
//     int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
//     int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
//     int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

//     int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
//     int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

//     for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
//         int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
//         int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
//         FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
//         int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
//         int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
//         FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

//         __syncthreads();

//         #pragma unroll
//         for (int k = 0; k < BK; k++) {
//             #pragma unroll
//             for (int m = 0; m < TM; m++) {
//                 #pragma unroll
//                 for (int n = 0; n < TN; n++) {
//                     int comp_a_smem_m = ty * TM + m;
//                     int comp_b_smem_n = tx * TN + n;
//                     r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
//                 }
//             }
//         }

//         __syncthreads();
//     }

//     #pragma unroll
//     for (int i = 0; i < TM; i++) {
//         int store_c_gmem_m = by * BM + ty * TM + i;
//         #pragma unroll
//         for (int j = 0; j < TN; j += 4) {
//             int store_c_gmem_n = bx * BN + tx * TN + j;
//             int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
//             FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
//         }
//     }
// }

// at::Tensor custom_matrix_mul(at::Tensor A, at::Tensor B) {
//     // 检查输入张量在CUDA上且是float类型
//     TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
//     TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");
//     TORCH_CHECK(A.dtype() == torch::kFloat32, "Input tensor A must be float32");
//     TORCH_CHECK(B.dtype() == torch::kFloat32, "Input tensor B must be float32");
//     TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match for matMul");

//     // 获取张量维度
//     const int M = A.size(0);
//     const int K = A.size(1);
//     const int N = B.size(1);

//     // 创建输出张量
//     auto C = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

//     // 获取设备数据指针
//     float* d_A = A.data_ptr<float>();
//     float* d_B = B.data_ptr<float>();
//     float* d_C = C.data_ptr<float>();

//     // 设置 CUDA 核函数的线程块和网格大小
//     const int BM = 128; // 每个 block 处理的行数
//     const int BN = 128; // 每个 block 处理的列数
//     dim3 blockDim(16, 16); // 每个线程块的线程数
//     dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM); // 网格大小

//     // 调用 CUDA 核函数
//     custom_matMul_kernel<<<gridDim, blockDim>>>(
//         d_A, d_B, d_C, M, N, K
//     );

//     // 检查 CUDA 内核是否正确执行
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         throw std::runtime_error(cudaGetErrorString(err));
//     }

//     // 返回结果张量
//     return C;
// }



// // Row-major 偏移计算: (row, col, ld=leading_dim)
// #define OFFSET(row, col, ld) ((row) * (ld) + (col))

// // 将 float 引用转换成 float4（需要保证地址是 16 字节对齐）
// #define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// -----------------------------------------------------------------------------
// 2. 核函数声明 (带注释)
// -----------------------------------------------------------------------------

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
__global__ void custom_matMul_kernel(
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
    custom_matMul_kernel<<<gridDim, blockDim>>>(
        a_ptr, b_ptr, c_ptr,
        M, N, K
    );

    // 错误检查
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}