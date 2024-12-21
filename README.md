# Pytorch_Kernel
 CS4302 Final Project

## setup
需要 pytorch 环境  
在根目录下运行 `python setup.py install` 可以安装为 module，在此 python 环境下 import。（详见test.py）  
然后就可以运行 `test.py` 了。

## Add kernel function
1. 在 `.cu` 文件中封装函数
2. 在 `main_ops.cpp` 中注册算子
3. 在 `setup.py` 中添加源文件

## Todo
- [ ] 优化算子
    - [x] bmm 现在使用了tiling
    - [ ] 类似矩阵乘法，还可以有更多优化
    - [ ] custom_transpose_cuda
    - [ ] custom_softmax_cuda
    - [ ] custom_vecAdd_cuda?
- [ ] 扩展算子
    - [x] Self-Attention 部分使用 bmm
    - [x] linear 层不需要 bmm，实现一个更好的算子
    - [ ] LayerNorm 可以换成自己实现
