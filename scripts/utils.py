import torch
import custom_ops

a = torch.Tensor([[[-0.281461  ],
  [-0.3533816 ],
  [-0.11401952],
  [-0.6644559 ]],

 [[ 0.44598415],
  [ 0.74836266],
  [ 1.60429   ],
  [-0.7862584 ]]]).cuda()
b = torch.Tensor([[[0.02]],[[0.02]]]).cuda()
c = torch.Tensor([0.02]).cuda()
d = custom_ops.custom_bmm_cuda(a, b)+c
print(a.dtype, b.dtype, c.dtype, d.dtype)
print(d, d.shape)