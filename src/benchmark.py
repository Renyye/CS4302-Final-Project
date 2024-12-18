import torch
import torch.nn as nn
import time
import custom_ops
from custom_transformer import CustomTransformer

# 定义PyTorch原生的Transformer层
class NativeTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super(NativeTransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-Attention
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward
        ff_output = self.linear2(self.dropout(nn.ReLU()(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src

class NativeTransformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super(NativeTransformer, self).__init__()
        self.layers = nn.ModuleList([
            NativeTransformerLayer(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

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
print("Output similarity:", torch.allclose(output_custom, output_native, atol=1e-4))
