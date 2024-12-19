import math
import torch
import torch.nn as nn
import torch.nn.init as init
import custom_ops  # 确保包含 custom_baddmm_cuda

def save_tensor_to_file(tensor, filename, name):
    # 保存张量的形状和数据到文件
    with open(filename, 'a') as f:
        f.write(f"Shape: {tensor.shape}, type: {type(tensor)} \n")
        f.write(f"Name: {name}, dtype: {tensor.dtype}\n")
        f.write(f"Data: {tensor.cpu().detach().numpy()}\n")
        f.write("\n" + "="*50 + "\n")


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 初始化权重和偏置
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        """
        input: [B, S, in_features]
        weight: [in_features, out_features]
        bias: [out_features] 或者 None
        """
        
        # 获取批次大小和序列长度
        B, S, E_in = input.size()
        assert E_in == self.in_features, f"输入特征维度不匹配: expected {self.in_features}, got {E_in}"

        # 将 weight 从 [in_features, out_features] 转换为 [B, in_features, out_features]
        # 以适应 custom_bmm_bias_cuda 的输入要求
        weight_batched = self.weight.unsqueeze(0).expand(B, -1, -1)  # [B, in_features, out_features]
        save_tensor_to_file(self.weight, "LinearLog.txt", "weight")
        # 处理 bias
        if self.bias is not None:
            bias = self.bias
        else:
            # 如果没有偏置，使用零偏置
            bias = torch.zeros(self.out_features, device=input.device, dtype=input.dtype)


        save_tensor_to_file(input, "LinearLog.txt", "input")
        save_tensor_to_file(weight_batched, "LinearLog.txt", "weight_batched")
        # save_tensor_to_file(bias, "LinearLog.txt", "bias")

        # 调用 custom_bmm_bias_cuda
        # assert torch.allclose(custom_ops.custom_bmm_cuda(input, weight_batched),
        #                       torch.bmm(input, weight_batched))
        weight_batched = weight_batched.clone().cuda()
        bmm_result = custom_ops.custom_bmm_cuda(input, weight_batched)  # [B, S, out_features]
        output = bmm_result + bias # [B, S, out_features]
        save_tensor_to_file(input, "LinearLog.txt", "inputafter")
        save_tensor_to_file(weight_batched, "LinearLog.txt", "weight_batchedafter")
        save_tensor_to_file(bmm_result, "LinearLog.txt", "bmm_result")
        save_tensor_to_file(output, "LinearLog.txt", "output")
        save_tensor_to_file(torch.bmm(input, weight_batched), "LinearLog.txt", "torch_bmm")
        assert torch.allclose(output,torch.bmm(input, weight_batched)+bias)
        return output


class CustomTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super(CustomTransformerLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.q_linear = CustomLinear(embed_dim, embed_dim)
        self.k_linear = CustomLinear(embed_dim, embed_dim)
        self.v_linear = CustomLinear(embed_dim, embed_dim)
        self.out_linear = CustomLinear(embed_dim, embed_dim)

        self.feedforward1 = CustomLinear(embed_dim, dim_feedforward)
        self.feedforward2 = CustomLinear(dim_feedforward, embed_dim)

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            init.constant_(param, 0.01)

    def forward(self, src):
        Q = self.q_linear(src)  # [B, S, E]
        K = self.k_linear(src)
        V = self.v_linear(src)
        save_tensor_to_file(src, "output1.txt", "src")
        save_tensor_to_file(Q, "output1.txt", "Q")
        save_tensor_to_file(K, "output1.txt", "K")
        save_tensor_to_file(V, "output1.txt", "V")

        K = custom_ops.custom_transpose_cuda(K)  # [B, E, S]
        attn_scores = custom_ops.custom_bmm_cuda(Q, K)  # [B, S, S]

        attn_weights = custom_ops.custom_softmax_cuda(attn_scores, 2)

        attn_output = custom_ops.custom_bmm_cuda(attn_weights, V)  # [B, S, E]
        attn_output = self.out_linear(attn_output)  # [B, S, E]
        attn_output = self.dropout(attn_output)

        src = custom_ops.custom_vecAdd_cuda(src, attn_output)  # 残差连接
        src = self.layernorm1(src)

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
