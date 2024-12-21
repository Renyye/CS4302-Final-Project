import torch
import custom_ops
import time

# warm up
for _ in range(1000):
    A = torch.randn(30, 40, device='cuda')
    B = torch.randn(30, 40, device='cuda')
    C = custom_ops.custom_vecAdd_cuda(A, B)

# test softmax time cost
start = time.time()
for _ in range(1000):
    A = torch.randn(10, 128, 256, device='cuda')
    C = custom_ops.custom_softmax_cuda(A, 2)
torch.cuda.synchronize()
end = time.time()
print(f"Custom softmax Time: {end - start:.4f} seconds")

start = time.time()
for _ in range(1000):
    A = torch.randn(10, 128, 256, device='cuda')
    C = torch.softmax(A, 2)
torch.cuda.synchronize()
end = time.time()
print(f"Native softmax Time: {end - start:.4f} seconds")