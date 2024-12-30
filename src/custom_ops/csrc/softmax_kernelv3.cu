// With reference to https://zhuanlan.zhihu.com/p/695307283

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void custom_softmax_kernel_v3(float* out, const float* inp, int N, int C) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    int warpsPerBlock = blockDim.x / 32;

    // shared memory前半部分用于存储每个warp的max值，后半部分用于存储每个warp的sum值
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // 访问当前行
    const float* x = inp + idx * C;

    // 1. 并行求最大值
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    maxval = warpReduceMax(maxval);
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // 2. 不同warp间的最大值规约
    if (tid == 0) {
        float val = maxvals[0];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];

    // 3. 计算 exp(x - max)
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // 4. 并行求和
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // 5. 不同warp间的加和
    if (tid == 0) {
        float val = sumvals[0];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float totalSum = sumvals[0];

    // 6. 归一化
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / totalSum;
    }
}

// 定义 PyTorch 接口
at::Tensor custom_softmax_cuda_v3(at::Tensor input, int dim) {
    // 确保输入是在 GPU 上
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor.");

    // 获取输入张量的形状
    int N = input.size(0);  // 批次大小
    int C = input.size(1);  // 类别数

    // 创建输出张量
    auto output = torch::empty_like(input);

    // 设置线程块和网格的大小
    const int threadsPerBlock = 256;  // 每个线程块的线程数
    const int blocks = N;  // 网格中的块数，N为批次大小

    // 调用 CUDA 核函数
    custom_softmax_kernel_v3<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        output.data_ptr<float>(), input.data_ptr<float>(), N, C
    );

    // 检查 CUDA 内核的启动错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // 返回结果
    return output;
}
