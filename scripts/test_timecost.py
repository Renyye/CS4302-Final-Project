import torch
import custom_ops
import time

def run_benchmark(matmul_func, label, iterations=1000, size=(2048, 1024, 1024, 2048)):
    start = time.time()
    for _ in range(iterations):
        A = torch.randn(size[0], size[1], device='cuda')
        B = torch.randn(size[2], size[3], device='cuda')
        C = matmul_func(A, B)
    torch.cuda.synchronize()
    end = time.time()
    elapsed = end - start
    print(f"{label} Time: {elapsed:.4f} seconds")
    return elapsed

def run_correctness(reference_func, test_func, size=(2048, 1024, 1024, 2048), atol=1e-4):
    A = torch.randn(size[0], size[1], device='cuda')
    B = torch.randn(size[2], size[3], device='cuda')
    C_ref = reference_func(A, B)
    C_test = test_func(A, B)
    if torch.allclose(C_ref, C_test, atol=atol):
        print(f"Correctness Test Passed for {test_func.__name__}")
    else:
        difference = C_ref - C_test
        max_diff = difference.abs().max()
        mean_diff = difference.abs().mean()
        print(f"Correctness Test Failed for {test_func.__name__}")
        print(f"Max difference: {max_diff.item()}")
        print(f"Mean difference: {mean_diff.item()}")

# Warm up
for _ in range(1000):
    A = torch.randn(1024, 1024, device='cuda')
    B = torch.randn(1024, 1024, device='cuda')
    C = torch.matmul(A, B)

# Benchmark Native matmul
native_time = run_benchmark(torch.matmul, "Native matmul")


for func,name in [(custom_ops.custom_matMul_cuda_v1, "Custom matMul_cuda_v1"),
                  (custom_ops.custom_matMul_cuda_v2, "Custom matMul_cuda_v2"),
                  (custom_ops.custom_matMul_cuda_v3, "Custom matMul_cuda_v3"),
                  (custom_ops.custom_matMul_cuda_v4, "Custom matMul_cuda_v4"),
                  (custom_ops.custom_matMul_cuda_v5, "Custom matMul_cuda_v5")]:
    custom_time = run_benchmark(func, name)
    print(f"Performance ratio: {100 * native_time / custom_time:.2f}%")
    run_correctness(torch.matmul, func)


# # Benchmark Custom matMul_cuda_v1
# custom_v1_time = run_benchmark(custom_ops.custom_matMul_cuda_v1, "Custom matMul_cuda_v1")
# print(f"Performance ratio: {100 * native_time / custom_v1_time:.2f}%")

# # Correctness Test for Custom matMul_cuda_v1
# run_correctness(torch.matmul, custom_ops.custom_matMul_cuda_v1)

# # Benchmark Custom matMul_cuda_v1
# custom_v1_time = run_benchmark(custom_ops.custom_matMul_cuda_v, "Custom matMul_cuda_vv")
# print(f"Performance ratio: {100 * native_time / custom_v1_time:.2f}%")

# # Correctness Test for Custom matMul_cuda_v1
# run_correctness(torch.matmul, custom_ops.custom_matMul_cuda_v)

# # Benchmark Custom matMul_cuda_v2
# custom_v2_time = run_benchmark(custom_ops.custom_matMul_cuda_v2, "Custom matMul_cuda_v2")
# print(f"Performance ratio: {100 * native_time / custom_v2_time:.2f}%")

# # Correctness Test for Custom matMul_cuda_v2
# run_correctness(torch.matmul, custom_ops.custom_matMul_cuda_v2)

# # Benchmark Custom matMul_cuda_v3
# custom_v3_time = run_benchmark(custom_ops.custom_matMul_cuda_v3, "Custom matMul_cuda_v3")
# print(f"Performance ratio: {100 * native_time / custom_v3_time:.2f}%")

# # Correctness Test for Custom matMul_cuda_v3
# run_correctness(torch.matmul, custom_ops.custom_matMul_cuda_v3)

# # Benchmark Custom matMul_cuda
# custom_time = run_benchmark(custom_ops.custom_matMul_cuda, "Custom matMul_cuda_v4")
# print(f"Performance ratio: {100 * native_time / custom_time:.2f}%")

# # Correctness Test for Custom matMul_cuda
# run_correctness(torch.matmul, custom_ops.custom_matMul_cuda)
