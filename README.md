# CS4302 Final Project：PyTorch CUDA算子改写

## 功能特性

实现了一些 CUDA 核函数及其逐步优化。

- 矩阵乘法
- SoftMax
- 矩阵转置
- 批量矩阵乘法（基础优化版本）

以及绑定至 Pytorch 自定义算子的方法，和一些配套的正确性及性能测试脚本。

## 文件结构

```
.
├── README.md            # 本文件
├── setup.py             # 编译构建
├── Document/Report.md   # 报告文件
├── src                  # 源文件
│   └── custom_ops
│   │   └── csrc
│   │       ├── main_ops.cpp          # 头文件,定义与绑定
│   │       ├── bmm_kernel.cu         # 核函数定义与封装
│   │       ├── <more operators>.cu   # 更多核函数
│   │       ......
│   └── transformer
│       ├── baseline_transformer.py # 使用 torch 原版算子的基准 transformer
│       ├── custom_transformer.py   # 替换使用自实现算子的 transformer
│       └── benchmark.py            # transformer 测试脚本
└── scripts
    ├── benchmark_mm.py   # 一些测试脚本
    ├── benchmark_softmax.py
    └── benchmark_transpose.py
```

## 依赖项

- `Python: 3.8.20`
- `pytorch: 1.12.1 cuda` 从官网通过 conda 安装或者从源码编译（只需要能 `#include <torch/extension.h>` 和 `import torch` 即可。不需要在 Pytorch 相关目录下。）
- `nvcc: cuda_11.3`
- GPU 环境测试于 `Tesla T4:  Driver Version: 470.182.03  CUDA Version: 11.4`
  以及 `3070Ti Laptop:  Driver Version: 566.03  CUDA Version: 12.7` (nvidia-smi 输出)

## Setup

> 使用 `pip install ninja` 并设定 `export MAX_JOBS=N` 可以加速编译。

在根目录下运行 `make build` 即可编译源文件为 Python 第三方库（在.py文件中`import custom_ops`）。

## 运行与测试

在根目录下运行 `make benchmark_all` 即可运行所有测试脚本，接着会输出原版算子与自实现算子的运行速度对比与正确性验证。

```bash
$ make benchmark_all
Testing MatMul Kernel. May take 30 seconds...
---------------------------------------------
Native matmul Time: 2.2067 seconds
Custom matMul_cuda_v1 Time: 20.7274 seconds
Performance ratio: 10.65%
```

测试脚本具体内容参见 `./scripts` 文件夹，自定义 Transformer 模块参见 `./src/transformer` 文件夹。

**完整报告**与一些结果的相应性能分析参见 `./Document/Report.md` 各算子对应模块。

## 如何自行添加算子？

1. 在 `.cu` 文件中封装函数，使用 at::Tensor 进行输入输出
2. 在 `main_ops.cpp` 中通过 `PYBIND11_MODULE` 注册算子
3. 在 `setup.py` 中添加相应 `.cu` 源文件

## TODO
- [x] 优化算子
    - [x] bmm 现在使用了tiling
    - [ ] ~~类似矩阵乘法，还可以有更多优化:  shared_weight_bmm aborted. Use mm instead.~~
    - [x] custom_transpose_cuda
    - [x] custom_softmax_cuda
    - [x] ~~custom_vecAdd_cuda Aborted~~
- [ ] ~~扩展算子 Aborted~~
    - [x] ~~Self-Attention 部分使用 bmm~~
    - [x] ~~linear 层不需要 bmm，实现一个更好的算子~~
    - [ ] ~~LayerNorm 可以换成自己实现~~
