import torch
import time
from custom_transformer import CustomTransformer
from native_ref import CustomTransformer as NativeTransformer


# 设置参数
num_layers = 6
embed_dim = 512
num_heads = 8
dim_feedforward = 2048
dropout = 0.1

# 创建模型
custom_transformer = CustomTransformer(num_layers, embed_dim, num_heads, dim_feedforward, dropout).cuda()
native_transformer = NativeTransformer(num_layers, embed_dim, num_heads, dim_feedforward, dropout).cuda()

# 随机输入
batch_size = 32
seq_length = 128
src = torch.randn(seq_length, batch_size, embed_dim, device='cuda')

# 预热
for _ in range(10):
    _ = custom_transformer(src)
    _ = native_transformer(src)

# 测量Custom Transformer
start = time.time()
for _ in range(100):
    output_custom = custom_transformer(src)
torch.cuda.synchronize()
end = time.time()
custom_time = end - start

# 测量Native Transformer
start = time.time()
for _ in range(100):
    output_native = native_transformer(src)
torch.cuda.synchronize()
end = time.time()
native_time = end - start

print(f"Custom Transformer Time: {custom_time:.4f} seconds")
print(f"Native Transformer Time: {native_time:.4f} seconds")

# 验证输出相似性（可选）
# 注意：由于使用不同的算子，可能存在细微的数值差异
similarity = torch.allclose(output_custom, output_native, atol=1e-4)
percentage = (torch.abs(output_custom - output_native) / output_custom).mean().item()
print(f"Similarity: {similarity}")
print(f"Percentage Difference: {percentage:.4f}")