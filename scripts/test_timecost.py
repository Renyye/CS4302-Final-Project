import torch
import custom_ops
import time

# warm up
for _ in range(1000):
    A = torch.randn(1024, 1024, device='cuda')
    B = torch.randn(1024, 1024, device='cuda')
    C = torch.matmul(A, B)

# test matmul time cost
start_n = time.time()
for _ in range(1000):
    A = torch.randn(2048, 1024, device='cuda')
    B = torch.randn(1024, 2048, device='cuda')
    C = torch.matmul(A, B)
torch.cuda.synchronize()
end_n = time.time()
print(f"Native matmul Time: {end_n - start_n:.4f} seconds")

start_c = time.time()
for _ in range(1000):
    A = torch.randn(2048, 1024, device='cuda')
    B = torch.randn(1024, 2048, device='cuda')
    C = custom_ops.custom_matMul_cuda(A, B)
torch.cuda.synchronize()
end_c = time.time()
print(f"Custom matmul Time: {end_c - start_c:.4f} seconds")
#性能比
print(f"Performance ratio: {(end_n - start_n) / (end_c - start_c):.4f}")

C_n = torch.matmul(A, B)
C_c = custom_ops.custom_matMul_cuda(A, B)
try:
    assert torch.allclose(C_n, C_c, atol=1e-4)
except:
    difference = C_n - C_c
    max_diff = difference.abs().max()
    mean_diff = difference.abs().mean()
    print(f"Max difference: {max_diff.item()}")
    print(f"Mean difference: {mean_diff.item()}")