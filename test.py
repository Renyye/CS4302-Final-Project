import torch
import custom_ops

# 调用bmm算子
A = torch.randn(2, 3, 4, device='cuda')
B = torch.randn(2, 4, 5, device='cuda')
C = custom_ops.custom_bmm_cuda(A, B)
assert (torch.allclose(C, torch.bmm(A, B))) # 查看是否与torch.bmm结果一致

# 调用vecAdd算子
A = torch.randn(3, 4, device='cuda')
B = torch.randn(3, 4, device='cuda')
C = custom_ops.custom_vecAdd_cuda(A, B)
assert (torch.allclose(C, A + B)) # 查看是否与A + B结果一致

# 调用matmul算子
A = torch.randn(3, 4, device='cuda')
B = torch.randn(4, 5, device='cuda')
C = custom_ops.custom_matMul_cuda(A, B)
assert (torch.allclose(C, torch.matmul(A, B))) # 查看是否与torch.matmul结果一致

# 调用transpose算子
A = torch.randn(3, 4, device='cuda')
C = custom_ops.custom_transpose_cuda(A)
assert (torch.allclose(C, A.t())) # 查看是否与A.t()结果一致

# 调用matAdd算子
A = torch.randn(3, 4, device='cuda')
B = torch.randn(3, 4, device='cuda')
C = custom_ops.custom_matAdd_cuda(A, B)
assert (torch.allclose(C, A + B)) # 查看是否与A + B结果一致

# 调用layerNormat::Tensor custom_layerNorm_cuda(at::Tensor input, at::Tensor gamma, at::Tensor beta, int normalized_shape);
A = torch.randn(3, 4, device='cuda')
gamma = torch.randn(4, device='cuda')
beta = torch.randn(4, device='cuda')
C = custom_ops.custom_layerNorm_cuda(A, gamma, beta, 4)
assert (torch.allclose(C, torch.layer_norm(A, [4], gamma, beta))) # 查看是否与torch.layer_norm结果一致

# 调用at::Tensor custom_relu_cuda(at::Tensor input);
A = torch.randn(3, 4, device='cuda')
C = custom_ops.custom_relu_cuda(A)
assert (torch.allclose(C, torch.relu(A))) # 查看是否与torch.relu结果一致

# at::Tensor custom_softmax_cuda(at::Tensor input, int dim);
A = torch.randn(3, 4, device='cuda')
C = custom_ops.custom_softmax_cuda(A, 1)
assert (torch.allclose(C, torch.softmax(A, 1))) # 查看是否与torch.softmax结果一致