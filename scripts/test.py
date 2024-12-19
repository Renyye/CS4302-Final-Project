import torch
import custom_ops

def save_tensor_to_file(tensor, filename, name):
    # 保存张量的形状和数据到文件
    with open(filename, 'a') as f:
        f.write(f"Shape: {tensor.shape}\n")
        f.write(f"Name: {name} Type: {tensor.dtype}\n")
        f.write(f"Data: {tensor.cpu().detach().numpy()}\n")
        f.write("\n" + "="*50 + "\n")\

for _ in range(1):
    A = torch.randn(2,4,1, device='cuda')
    B = torch.randn(2,1,1, device='cuda')
    C = custom_ops.custom_bmm_cuda(A, B)
    expected_C = torch.bmm(A, B)
    save_tensor_to_file(A, "test.txt", "A")
    save_tensor_to_file(B, "test.txt", "B")
    save_tensor_to_file(C, "test.txt", "C")
    try:
        assert torch.allclose(C, expected_C)
    except:
        save_tensor_to_file(A, "test.txt", "A")
        save_tensor_to_file(B, "test.txt", "B")
        save_tensor_to_file(C, "test.txt", "C")
        save_tensor_to_file(expected_C, "test.txt", "expected_C")
        raise Exception
raise Exception


for _ in range(10):
    for batch_size in range(50,60):
        for m in range(50,60):  # 维度m
            for n in range(50,60):  # 维度n
                for k in range(50,60):  # 维度k
                    A = torch.randn(batch_size, m, k, device='cuda')
                    B = torch.randn(batch_size, k, n, device='cuda')
                    C = custom_ops.custom_bmm_cuda(A, B)
                    
                    # 用torch的bmm计算参考值
                    expected_C = torch.bmm(A, B)
                    
                    # 打印测试信息并断言是否一致
                    try:
                        assert torch.allclose(C, expected_C), f"Test failed with shapes A: {A.shape}, B: {B.shape}, C: {C.shape}"
                    except:
                        save_tensor_to_file(A, "test.txt", "A")
                        save_tensor_to_file(B, "test.txt", "B")
                        save_tensor_to_file(C, "test.txt", "C")
                        save_tensor_to_file(expected_C, "test.txt", "expected_C")
                        raise Exception("Fuck you")

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
A = torch.randn(2, 3, 4, device='cuda')
C = custom_ops.custom_transpose_cuda(A)
assert (torch.allclose(C, A.transpose(1, 2))) # 查看是否与A.transpose(1, 2)结果一致

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
A = torch.randn(2, 3, 4, device='cuda')
C = custom_ops.custom_softmax_cuda(A, 2)
assert (torch.allclose(C, torch.softmax(A, 2))) # 查看是否与torch.softmax结果一致