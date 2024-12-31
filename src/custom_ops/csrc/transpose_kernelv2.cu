#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void custom_transpose_kernel_v2(const float *d_A, float *d_T, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // 计算输入矩阵的索引
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    // 输入矩阵中的线性索引
    int index_A = yIndex * width + xIndex;
    
    // 计算转置后的索引
    int index_T = xIndex * height + yIndex;

    // 读取数据到共享内存
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = d_A[index_A];
    }
    __syncthreads();

    // 写回转置后的数据到输出矩阵
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    index_T = yIndex * height + xIndex;

    if (xIndex < height && yIndex < width) {
        d_T[index_T] = tile[threadIdx.x][threadIdx.y];
    }
}



at::Tensor custom_transpose_v2(at::Tensor A, int dim_x, int dim_y) {
    int dims = A.dim();
    
    // 处理二维矩阵
    if (dims == 2) {
        int height = A.size(0);
        int width = A.size(1);
        
        if (dim_x == 0 && dim_y == 1) {
            auto output = torch::empty({width, height}, A.options());
            dim3 block(TILE_DIM, TILE_DIM);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

            custom_transpose_kernel_v2<<<grid, block>>>(
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

            custom_transpose_kernel_v2<<<grid, block>>>(
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

            custom_transpose_kernel_v2<<<grid, block>>>(
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

            custom_transpose_kernel_v2<<<grid, block>>>(
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

            custom_transpose_kernel_v2<<<grid, block>>>(
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
