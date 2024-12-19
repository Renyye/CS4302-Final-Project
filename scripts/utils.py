import torch
weight = torch.Tensor([[0.01]])
weight_batched = weight.unsqueeze(0).expand(2, -1, -1)

print(weight_batched)