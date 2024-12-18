import torch
import torch.nn as nn
import custom_ops  # 你的自定义算子模块

class CustomTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super(CustomTransformerLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.feedforward1 = nn.Linear(embed_dim, dim_feedforward)
        self.feedforward2 = nn.Linear(dim_feedforward, embed_dim)

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        # Self-Attention部分
        Q = self.q_linear(src)  # [B, S, E]
        K = self.k_linear(src)
        V = self.v_linear(src)

        # 使用自定义bmm算子计算注意力分数
        # 需要将Q和K转置以匹配矩阵乘法的维度
        K = custom_ops.custom_transpose_cuda(K)  # [B, E, S]
        attn_scores = custom_ops.custom_bmm_cuda(Q, K)  # [B, S, S]

        # Softmax
        attn_weights = custom_ops.custom_softmax_cuda(attn_scores, dim=-1)  # [B, S, S]

        # Attention输出
        attn_output = custom_ops.custom_bmm_cuda(attn_weights, V)  # [B, S, E]
        attn_output = self.out_linear(attn_output)  # [B, S, E]
        attn_output = self.dropout(attn_output)
        src = custom_ops.custom_vecAdd_cuda(src, attn_output)  # 残差连接
        src = self.layernorm1(src)

        # 前馈网络部分
        ff_output = self.feedforward1(src)
        ff_output = self.activation(ff_output)
        ff_output = self.feedforward2(ff_output)
        ff_output = self.dropout(ff_output)
        src = custom_ops.custom_vecAdd_cuda(src, ff_output)  # 残差连接
        src = self.layernorm2(src)

        return src


class CustomTransformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super(CustomTransformer, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerLayer(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.embed_dim = embed_dim

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src
