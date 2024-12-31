import torch
import custom_ops
import time

def run_benchmark(softmax_func, label, iterations=1000, size=(2048, 1024)):
    # 创建一个固定大小的输入张量
    A = torch.randn(size[0], size[1], device='cuda')
    torch.cuda.synchronize()  # 确保同步

    # 运行多次计算，获取平均时间
    start = time.time()
    for _ in range(iterations):
        C = softmax_func(A, 1)
    torch.cuda.synchronize()  # 同步GPU计算
    end = time.time()
    elapsed = end - start

    # 打印性能结果
    print(f"{label} Time: {elapsed:.4f} seconds")
    return elapsed

def run_correctness(reference_func, test_func, size=(2048, 1024), atol=1e-4):
    # 创建一个固定大小的输入张量
    A = torch.randn(size[0], size[1], device='cuda')
    # 计算参考值和测试值
    C_ref = reference_func(A, 1)
    C_test = test_func(A, 1)

    # 检查两者的相对误差
    if torch.allclose(C_ref, C_test, atol=atol):
        print(f"Correctness Test Passed for {test_func.__name__}")
    else:
        difference = C_ref - C_test
        max_diff = difference.abs().max()
        mean_diff = difference.abs().mean()
        print(f"Correctness Test Failed for {test_func.__name__}")
        print(f"Max difference: {max_diff.item()}")
        print(f"Mean difference: {mean_diff.item()}")

# 预热：调用一次softmax，避免冷启动带来的影响
for _ in range(1000):
    A = torch.randn(1024, 1024, device='cuda')
    C = torch.softmax(A, dim=1)

# 基准测试：原生softmax
native_time = run_benchmark(torch.softmax, "Native softmax")

# 基准测试：自定义CUDA softmax
for func, name in [(custom_ops.custom_softmax_cuda_v1, "Custom softmax_cuda_v1"),
                   (custom_ops.custom_softmax_cuda_v2, "Custom softmax_cuda_v2"),
                   (custom_ops.custom_softmax_cuda, "Custom softmax_cuda_v3")]:
    custom_time = run_benchmark(func, name)
    print(f"Performance ratio: {100 * native_time / custom_time:.2f}%")
    # 校验正确性
    run_correctness(torch.softmax, func)
