import torch.nn as nn

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