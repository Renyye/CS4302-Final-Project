#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void custom_transpose_kernel_v1(const float *d_A, float *d_T, int width, int height) {
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_A = xIndex + width * yIndex;
    int index_T = yIndex + height * xIndex;

    for (int i = 0; i < TILE_DIM; i += blockDim.x) {
        d_T[index_T + i] = d_A[index_A + i * width];
    }
}


at::Tensor custom_transpose_v1(at::Tensor A, int dim_x, int dim_y) {
    int dims = A.dim();
    
    // 处理二维矩阵
    if (dims == 2) {
        int height = A.size(0);
        int width = A.size(1);
        
        if (dim_x == 0 && dim_y == 1) {
            auto output = torch::empty({width, height}, A.options());
            dim3 block(TILE_DIM, TILE_DIM);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

            custom_transpose_kernel_v1<<<grid, block>>>(
                A.data_ptr<float>(),
                output.data_ptr<float>(),
                width,
                height
            );
            return output;
        }
        else if (dim_x == 1 && dim_y == 0) {
            auto output = torch::empty({height, width}, A.options());
            dim3 block(TILE_DIM, TILE_DIM);
            dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y);

            custom_transpose_kernel_v1<<<grid, block>>>(
                A.data_ptr<float>(),
                output.data_ptr<float>(),
                height,
                width
            );
            return output;
        }
    }
    
    // 处理三维张量
    if (dims == 3) {
        int B = A.size(0);
        int E = A.size(1);
        int S = A.size(2);

        // 这里的转置逻辑可以扩展为任意维度的交换
        if (dim_x == 0 && dim_y == 1) {
            auto output = torch::empty({E, B, S}, A.options());
            dim3 block(TILE_DIM, TILE_DIM);
            dim3 grid((E + block.x - 1) / block.x, (B + block.y - 1) / block.y);

            custom_transpose_kernel_v1<<<grid, block>>>(
                A.data_ptr<float>(),
                output.data_ptr<float>(),
                E,
                B
            );
            return output;
        }
        else if (dim_x == 0 && dim_y == 2) {
            auto output = torch::empty({S, B, E}, A.options());
            dim3 block(TILE_DIM, TILE_DIM);
            dim3 grid((S + block.x - 1) / block.x, (B + block.y - 1) / block.y);

            custom_transpose_kernel_v1<<<grid, block>>>(
                A.data_ptr<float>(),
                output.data_ptr<float>(),
                S,
                B
            );
            return output;
        }
        else if (dim_x == 1 && dim_y == 2) {
            auto output = torch::empty({B, S, E}, A.options());
            dim3 block(TILE_DIM, TILE_DIM);
            dim3 grid((B + block.x - 1) / block.x, (S + block.y - 1) / block.y);

            custom_transpose_kernel_v1<<<grid, block>>>(
                A.data_ptr<float>(),
                output.data_ptr<float>(),
                B,
                S
            );
            return output;
        }
    }
    else {
        AT_ERROR("Not supported transpose dims");
    }
}
