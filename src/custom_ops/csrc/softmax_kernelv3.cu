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
    // 确保输入在 GPU 上
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor.");

    // 对于 2D 情况: (N, C)，默认对 dim=1 做 softmax
    // 对于 3D 情况: (B, S, E)，若 dim=2，则合并前两个维度，使用同一个 kernel
    auto shape = input.sizes();
    TORCH_CHECK(
        shape.size() == 2 || shape.size() == 3,
        "Only 2D or 3D tensors are supported by this custom softmax."
    );

    // block / grid 配置
    const int threadsPerBlock = 256;

    if (shape.size() == 2) {
        // 2D 输入 [N, C]
        int N = shape[0];
        int C = shape[1];
        // 若仅指定 dim=1，则直接调用
        TORCH_CHECK(
            dim == 1,
            "For a 2D tensor, only dim=1 is supported in this custom softmax."
        );

        auto output = torch::empty_like(input);
        custom_softmax_kernel_v3<<<N, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            N, C
        );
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed!");
        return output;
    } else {
        // 3D 输入 [B, S, E]
        // 如果 dim=2，则对最后一个维度做 softmax
        TORCH_CHECK(dim == 2, "For a 3D tensor, this custom softmax only supports dim=2.");
        int B = shape[0];
        int S = shape[1];
        int E = shape[2];

        // 将前面两个维度合并成 N=B*S，并按 C=E，使用 kernel
        auto input_2d = input.view({B * S, E});
        auto output_2d = torch::empty_like(input_2d);

        const int N = B * S;
        const int C = E;
        custom_softmax_kernel_v3<<<N, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            output_2d.data_ptr<float>(),
            input_2d.data_ptr<float>(),
            N, C
        );
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed!");

        // reshape 回原来的形状
        return output_2d.view({B, S, E});
    }
}