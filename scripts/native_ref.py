import torch
import torch.nn as nn

class CustomTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super(CustomTransformerLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        # 线性变换层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        # 前馈网络
        self.feedforward1 = nn.Linear(embed_dim, dim_feedforward)
        self.feedforward2 = nn.Linear(dim_feedforward, embed_dim)

        # 层归一化
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 或者使用 nn.GELU()

    def forward(self, src):
        # Self-Attention 部分
        Q = self.q_linear(src)  # [B, S, E]
        K = self.k_linear(src)
        V = self.v_linear(src)

        # 转置 K 以匹配矩阵乘法的维度
        K_transposed = K.transpose(1, 2)  # [B, E, S]

        # 计算注意力分数
        attn_scores = torch.bmm(Q, K_transposed)  # [B, S, S]

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=2)  # [B, S, S]

        # Attention 输出
        attn_output = torch.bmm(attn_weights, V)  # [B, S, E]
        attn_output = self.out_linear(attn_output)  # [B, S, E]
        attn_output = self.dropout(attn_output)

        # 残差连接和层归一化
        src = src + attn_output  # [B, S, E]
        src = self.layernorm1(src)

        # 前馈网络部分
        ff_output = self.feedforward1(src)  # [B, S, dim_feedforward]
        ff_output = self.activation(ff_output)  # [B, S, dim_feedforward]
        ff_output = self.feedforward2(ff_output)  # [B, S, E]
        ff_output = self.dropout(ff_output)

        # 残差连接和层归一化
        src = src + ff_output  # [B, S, E]
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
