```
"index_select_cuda" : 327
"index_select_out_cuda_impl" : 327
"ufunc_add_CUDA" : 327
"addmm_cuda" : 327
"div_true_cuda" : 327
"baddbmm_cuda" : 327
"clamp_min_scalar_cuda" : 327
"mean_cuda" : 327
"mse_cuda" : 327
"fill_cuda" : 327
"mse_backward_cuda" : 327
"sum_cuda" : 327
"threshold_cuda" : 327
"mul_cuda" : 327
"cat_cuda" : 327
"embedding_dense_backward_cuda" : 327
"addcmul_cuda" : 327
"sqrt_cuda" : 327
"addcdiv_cuda" : 327
"_local_scalar_dense_cuda" : 327

"epsilon" : 1
"check_uniform_bounds" : 1
"uniform_kernel_cpu" : 1
"add_stub" : 1
"ufunc_add_CUDA" : 1
"mul_cpu" : 1
"mul_cuda" : 1
"copy_" : 1
"slow_conv2d_cuda" : 1
```

```
index_select_cuda / index_select_out_cuda_impl
用于从输入张量中选择指定索引的元素。这个操作在 GPU 上执行时，用于高效地根据指定的索引提取张量的子集。例如，用于在模型中选择特定的嵌入向量或者其他相关的操作。

ufunc_add_CUDA
该算子执行两个张量的逐元素加法。在 GPU 上执行时，所有元素都加在一起，并且由于采用 CUDA 加速，计算会更快。

addmm_cuda
这是矩阵乘法加法的操作，即计算 A*B + C，其中 A 和 B 是矩阵，C 是一个加法项。这个算子通常用于深度学习中的前向和反向传播阶段，比如在 Transformer 中进行矩阵运算时。

div_true_cuda
用于执行张量的逐元素除法操作。与普通除法不同，它确保在除法操作中不会发生除以零的错误，并根据具体的广播规则自动扩展张量的尺寸。

baddbmm_cuda
执行批量矩阵加法的操作，即 out = beta * out + alpha * (batch1 @ batch2)，其中 batch1 和 batch2 是批量矩阵，out 是输出结果，alpha 和 beta 是缩放因子。通常用于对多批次的矩阵进行计算。

clamp_min_scalar_cuda
对输入张量的每个元素进行 "min-clamping"（最小值裁剪），即将小于某个指定值的元素设为该指定值。例如，clamp_min(5) 将把所有小于 5 的元素设为 5。

mean_cuda
计算张量的均值。它会对张量中的所有元素求平均值，在 GPU 上执行时速度较快，常用于计算损失函数或其他统计量。

mse_cuda
计算均方误差（MSE, Mean Squared Error）。这是回归问题中常用的损失函数之一，计算的是预测值与真实值之间差的平方的平均值。

fill_cuda
用于将输入张量的所有元素填充为指定的标量值。例如，tensor.fill_(5) 会将 tensor 中的所有元素设为 5。

mse_backward_cuda
计算均方误差损失函数的反向传播梯度。这是基于 MSE 计算损失时，用于计算梯度的操作。

sum_cuda
计算张量的所有元素之和。这个算子可以用来计算总损失或总和，支持对张量进行按维度求和。

threshold_cuda
用于执行阈值操作，将张量中小于某个值的元素替换为该阈值，通常用于某些非线性激活函数或在特定阈值下进行稀疏化处理。

mul_cuda
执行张量的逐元素乘法。这通常用于在模型计算中按元素的缩放或加权操作。

cat_cuda
用于将多个张量在指定维度上进行拼接（concat）。这是深度学习中经常使用的操作，例如拼接不同层的输出。

embedding_dense_backward_cuda
用于计算嵌入层（embedding layer）的反向传播梯度。在训练过程中，当嵌入层的参数需要更新时，这个算子会被调用。

addcmul_cuda
计算 tensor1 + value * (tensor2 * tensor3)。它将 tensor2 和 tensor3 逐元素相乘后，与 tensor1 相加，并乘以一个标量值。这个操作通常用于梯度更新过程中。

sqrt_cuda
对张量的每个元素计算平方根。用于许多标准化或归一化操作，或者当某些损失函数需要进行平方根变换时。

addcdiv_cuda
计算 tensor1 + value * (tensor2 / tensor3)，它是张量 tensor2 与 tensor3 逐元素相除的结果，然后与 tensor1 相加并乘以一个标量值。这个操作可以用于许多模型的更新规则。

_local_scalar_dense_cuda
这个算子用于将单个标量值从 GPU 张量提取为普通的 CPU 标量。这通常在执行一些聚合操作或在 GPU 上计算标量时使用。
```

