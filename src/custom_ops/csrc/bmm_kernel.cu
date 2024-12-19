#include <stdlib.h>
#include <torch/extension.h>

__global__ void bmm_kernel(float *d_A, float *d_B, float *d_C, int batch_size, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    // printf d_B
    if (batch < batch_size && row < M && col < P) {
        printf("Batch: %d, Row: %d, Col: %d, A[Batch=%d][Row=%d][i=%d]=%f, B[Batch=%d][i=%d][Col=%d]=%f\n",
                batch, row, col,
                batch, row, 0, d_A[batch * M * N + row * N + 0],
                batch, 0, col, d_B[batch * N * P + 0 * P + col]);
    }
    if (batch < batch_size && row < M && col < P) {
        float sum = 0.0f;
        
                for (int i = 0; i < N; ++i) {
                        sum += d_A[batch * M * N + row * N + i] * d_B[batch * N * P + i * P + col];
                        printf("Batch: %d, Row: %d, Col: %d, i: %d, A[Batch=%d][Row=%d][i=%d]=%f, B[Batch=%d][i=%d][Col=%d]=%f, Partial Sum: %f\n",
                                batch, row, col, i,
                                batch, row, i, d_A[batch * M * N + row * N + i],
                                batch, i, col, d_B[batch * N * P + i * P + col],
                                sum);
                }
        d_C[batch * M * P + row * P + col] = sum;
                printf("Batch: %d, Row: %d, Col: %d, Sum: %f\n", batch, row, col, sum);
    }
}

// 封装函数：接收PyTorch张量并调用kernel
at::Tensor custom_bmm(at::Tensor A, at::Tensor B) {
    // 假设输入为 (batch_size, M, N), (batch_size, N, P)
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    int batch_size = A.size(0);
    int M = A.size(1);
    int N = A.size(2);
    int P = B.size(2);

    // 创建输出tensor (batch_size, M, P)
    auto C = torch::zeros({batch_size, M, P}, torch::dtype(torch::kFloat32).device(A.device()));

    // 获取raw pointers
    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((P + blockDim.x - 1)/blockDim.x,
                (M + blockDim.y - 1)/blockDim.y,
                batch_size);


    // 调用kernel
    bmm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, batch_size, M, N, P);

    // 同步和错误检查
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return C;
}
