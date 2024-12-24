from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_ops', 
            sources=[
                'src/custom_ops/csrc/main_ops.cpp',
                'src/custom_ops/csrc/mm_kernel.cu',
                'src/custom_ops/csrc/bmm_kernel.cu',
                'src/custom_ops/csrc/vecAdd_kernel.cu',
                'src/custom_ops/csrc/transpose_kernel.cu',
                'src/custom_ops/csrc/matAdd_kernel.cu',
                'src/custom_ops/csrc/layerNorm_kernel.cu',
                'src/custom_ops/csrc/relu_kernel.cu',
                'src/custom_ops/csrc/softmax_kernel.cu',
                # 其他文件...
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
