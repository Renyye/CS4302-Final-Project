import torch
import time
from custom_transformer import CustomTransformer
from native_ref import NativeTransformer

# 设置参数
num_layers = 2
embed_dim = 64
num_heads = 4
dim_feedforward = 128
dropout = 0.1
batch_size = 4
seq_length = 10
src = torch.randn(seq_length, batch_size, embed_dim, device='cuda')
src1 = src.clone()
src2 = src.clone()

# 设置随机种子
seed = 1234
torch.manual_seed(seed)  # CPU 上设置随机种子
torch.cuda.manual_seed(seed)  # 单个 GPU 上设置随机种子
torch.cuda.manual_seed_all(seed)  # 所有 GPU 上设置随机种子
torch.backends.cudnn.deterministic = True  # 确保结果是确定性的
torch.backends.cudnn.benchmark = False  # 禁用 cudnn 加速，这对于小规模模型和调试很有用
custom_transformer = CustomTransformer(num_layers, embed_dim, num_heads, dim_feedforward, dropout).cuda()

# 预热：确保模型已经加载并且GPU已被占用
for _ in range(10):
    _ = custom_transformer(src)

# 测量 Custom Transformer
start = time.time()
for _ in range(5000):
    output_custom = custom_transformer(src1)
torch.cuda.synchronize()
end = time.time()
custom_time = end - start

# # 重新设置随机种子
# seed = 1234
# torch.manual_seed(seed)  # CPU 上设置随机种子
# torch.cuda.manual_seed(seed)  # 单个 GPU 上设置随机种子
# torch.cuda.manual_seed_all(seed)  # 所有 GPU 上设置随机种子
# torch.backends.cudnn.deterministic = True  # 确保结果是确定性的
# torch.backends.cudnn.benchmark = False 
# native_transformer = NativeTransformer(num_layers, embed_dim, num_heads, dim_feedforward, dropout).cuda()

# for _ in range(10):
#     _ = native_transformer(src)

# # 测量 Native Transformer
# start = time.time()
# for _ in range(5000):
#     output_native = native_transformer(src2)
# torch.cuda.synchronize()
# end = time.time()
# native_time = end - start

# # 打印时间比较
# print(f"Custom Transformer Time: {custom_time:.4f} seconds")
# print(f"Native Transformer Time: {native_time:.4f} seconds")

# # 验证输出相似性
# similarity = torch.allclose(output_custom, output_native, atol=1e-4)
# percentage = (torch.abs(output_custom - output_native) / output_custom).mean().item()

# print(f"Similarity: {similarity}")
# print(f"Percentage Difference: {percentage:.4f}")