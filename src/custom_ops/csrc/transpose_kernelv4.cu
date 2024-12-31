#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 16

__global__ void custom_transpose_kernel_v4(float *odata, float *idata, int width,
                                  int height) {

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    // do diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    // 以下代码与 V3 一致

    int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

at::Tensor custom_transpose_v4(at::Tensor A, int dim_x, int dim_y) {
    int dims = A.dim();

    // Handle 2D matrices
    if (dims == 2) {
        int height = A.size(0);
        int width = A.size(1);

        if ((dim_x == 0 && dim_y == 1) || (dim_x == 1 && dim_y == 0)) {
            auto output = torch::empty({A.size(dim_y), A.size(dim_x)}, A.options());
            dim3 block(TILE_DIM, BLOCK_ROWS);  // 32 x 16 = 512 threads
            dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

            // Launch the kernel
            custom_transpose_kernel_v4<<<grid, block>>>(
                output.data_ptr<float>(), // odata
                A.data_ptr<float>(),      // idata
                width,
                height
            );
            
            // Check for errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
            }

            return output;
        }
    }

    // Handle 3D tensors
    if (dims == 3) {
        int B = A.size(0);
        int E = A.size(1);
        int S = A.size(2);

        // Determine the new shape based on dimensions to transpose
        std::vector<int64_t> new_shape;
        if (dim_x == 0 && dim_y == 1) {
            new_shape = {E, B, S};
        }
        else if (dim_x == 0 && dim_y == 2) {
            new_shape = {S, B, E};
        }
        else if (dim_x == 1 && dim_y == 2) {
            new_shape = {B, S, E};
        }
        else {
            AT_ERROR("Unsupported transpose dimensions");
        }

        auto output = torch::empty(new_shape, A.options());

        // Determine width and height based on transposed dimensions
        int width, height;
        if ((dim_x == 0 && dim_y == 1) || (dim_x == 1 && dim_y == 0)) {
            width = E;
            height = B;
        }
        else if ((dim_x == 0 && dim_y == 2) || (dim_x == 2 && dim_y == 0)) {
            width = S;
            height = B;
        }
        else if ((dim_x == 1 && dim_y == 2) || (dim_x == 2 && dim_y == 1)) {
            width = S;
            height = E;
        }

        dim3 block(TILE_DIM, BLOCK_ROWS);  // 32 x 16 = 512 threads
        dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

        // Launch the kernel
        custom_transpose_kernel_v4<<<grid, block>>>(
            output.data_ptr<float>(), // odata
            A.data_ptr<float>(),      // idata
            width,
            height
        );
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        return output;
    }

    AT_ERROR("Not supported transpose dimensions");
}

