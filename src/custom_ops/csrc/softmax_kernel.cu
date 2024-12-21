#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------------
// 1. 定义 warpReduceMax / warpReduceSum 辅助函数
// -------------------------------------------
__inline__ __device__ float warpReduceMax(float val) {
    // 这里使用 32 线程的 warp 进行归约
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    // 同理使用 32 线程的 warp 进行归约
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -------------------------------------------
// 2. 优化后的 Softmax Kernel
// -------------------------------------------
__global__ void custom_softmax_kernel_v2(
    const float *d_input, 
    float *d_output, 
    int B,  // batch size
    int S,  // sequence length
    int E   // embedding dimension (或 feature size)
) {
    // 让 blockIdx.x 标识第几行，即第几个 (batch, seq)
    int rowIdx = blockIdx.x;       // 取值范围: [0, B*S - 1]
    if (rowIdx >= B * S) return;

    // 计算这一行在输入、输出中的首地址
    int offset = rowIdx * E;
    const float* x = d_input + offset; 
    float* y       = d_output + offset;

    // 线程 ID
    int tid = threadIdx.x;   
    // warp 内 ID
    int laneId = tid % warpSize;
    // warp 的索引
    int warpId = tid / warpSize;
    // 每个 block 的 warp 数
    int warpsPerBlock = blockDim.x / warpSize;

    // 使用共享内存：前 half 存储各 warp 的 max 值，后 half 存储各 warp 的 sum 值
    extern __shared__ float shared[];
    float* maxvals = shared;
    float* sumvals = shared + warpsPerBlock;

    // -------------------------------------------
    // 1) 先求最大值，用于稳定数值计算
    // -------------------------------------------
    float thread_max = -INFINITY;
    // 线程循环读取该行 (E) 中的数据，做局部最大值
    for (int i = tid; i < E; i += blockDim.x) {
        thread_max = fmaxf(thread_max, x[i]);
    }

    // warp 内做最大值归约
    thread_max = warpReduceMax(thread_max);

    // 每个 warp 的 0 号线程，把这个 warp 的最大值写进共享内存
    if (laneId == 0) {
        maxvals[warpId] = thread_max;
    }
    __syncthreads();

    // block 内做 warp 间的最大值归约
    if (warpId == 0) {
        // 让前 warpsPerBlock 个线程进行再次归约，这里只需要前 warpsPerBlock 个值
        float block_max = (tid < warpsPerBlock) ? maxvals[tid] : -INFINITY;
        block_max = warpReduceMax(block_max);
        // 再次归约完毕后，把最终最大值放到 maxvals[0]
        if (tid == 0) {
            maxvals[0] = block_max;
        }
    }
    __syncthreads();

    // 取出该行的全局最大值
    float row_max = maxvals[0];

    // -------------------------------------------
    // 2) 计算 e^(x - max)
    // -------------------------------------------
    for (int i = tid; i < E; i += blockDim.x) {
        y[i] = expf(x[i] - row_max);
    }
    __syncthreads();

    // -------------------------------------------
    // 3) 归约求和
    // -------------------------------------------
    float thread_sum = 0.f;
    for (int i = tid; i < E; i += blockDim.x) {
        thread_sum += y[i];
    }
    // warp 内归约
    thread_sum = warpReduceSum(thread_sum);

    // 每个 warp 的 0 号线程写到共享内存
    if (laneId == 0) {
        sumvals[warpId] = thread_sum;
    }
    __syncthreads();

    // block 内 warp 间再归约
    if (warpId == 0) {
        float block_sum = (tid < warpsPerBlock) ? sumvals[tid] : 0.f;
        block_sum = warpReduceSum(block_sum);
        if (tid == 0) {
            sumvals[0] = block_sum;
        }
    }
    __syncthreads();

    // 取出该行的全局和
    float row_sum = sumvals[0];

    // -------------------------------------------
    // 4) 每个元素除以总和
    // -------------------------------------------
    for (int i = tid; i < E; i += blockDim.x) {
        y[i] = y[i] / row_sum;
    }
}

at::Tensor custom_softmax_cuda(at::Tensor input, int dim) {
    if (dim < 0) {
        dim += input.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dim must be in range [0, ", input.dim()-1, "]");

    TORCH_CHECK(input.dim() == 3, "input must be 3D tensor");

    int B = input.size(0);
    int S = input.size(1);
    int E = input.size(2);

    auto output = torch::empty_like(input);

    const float *d_input_ptr = input.data_ptr<float>();
    float *d_output_ptr = output.data_ptr<float>();

    int blockSize = 128;
    int gridSize  = B * S;
    int warpsPerBlock = blockSize / 32;
    size_t sharedMemBytes = 2 * warpsPerBlock * sizeof(float);

    custom_softmax_kernel_v2<<<gridSize, blockSize, sharedMemBytes>>>(d_input_ptr, d_output_ptr, B, S, E);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}
