import torch
import custom_ops
import time

def run_benchmark(transpose_func, label, iterations=1000, size=(4096, 4096)):
    A = torch.randn(size[0], size[1], device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        C = transpose_func(A)
    torch.cuda.synchronize()
    end = time.time()
    elapsed = end - start
    print(f"{label} Time: {elapsed:.4f} seconds")
    return elapsed

def run_correctness(reference_func, test_func, size=(4096, 4096), atol=1e-4):
    A = torch.randn(size[0], size[1], device='cuda')
    C_ref = reference_func(A)
    C_test = test_func(A)
    if torch.allclose(C_ref, C_test, atol=atol):
        print(f"Correctness Test Passed for {test_func.__name__}")
    else:
        diff = C_ref - C_test
        print(f"Correctness Test Failed for {test_func.__name__}")
        print(f"Max difference: {diff.abs().max().item()}")
        print(f"Mean difference: {diff.abs().mean().item()}")

# 预热
for _ in range(1000):
    A = torch.randn(1024, 1024, device='cuda')
    _ = A.transpose(0, 1)

native_time = run_benchmark(lambda x: x.transpose(0, 1), "Native transpose")

for func, name in [(custom_ops.custom_transpose_cuda_v1, "Custom transpose_kernel_v1"),
                   (custom_ops.custom_transpose_cuda_v2, "Custom transpose_kernel_v2"),
                   (custom_ops.custom_transpose_cuda_v3, "Custom transpose_kernel_v3"),
                   (custom_ops.custom_transpose_cuda, "Custom transpose_kernel_v4")]:
    custom_time = run_benchmark(func, name)
    print(f"Performance ratio: {100 * native_time / custom_time:.2f}%")
    run_correctness(lambda x: x.transpose(0, 1), func)