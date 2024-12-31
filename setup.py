from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_ops', 
            sources=[
                'src/custom_ops/csrc/main_ops.cpp',
                'src/custom_ops/csrc/mm_kernelv1.cu',
                'src/custom_ops/csrc/mm_kernelv2.cu',
                'src/custom_ops/csrc/mm_kernelv3.cu',
                'src/custom_ops/csrc/mm_kernelv4.cu',
                'src/custom_ops/csrc/mm_kernelv5.cu',
                'src/custom_ops/csrc/softmax_kernelv1.cu',
                'src/custom_ops/csrc/softmax_kernelv2.cu',
                'src/custom_ops/csrc/softmax_kernelv3.cu',
                'src/custom_ops/csrc/transpose_kernelv1.cu',
                'src/custom_ops/csrc/transpose_kernelv2.cu',
                'src/custom_ops/csrc/transpose_kernelv3.cu',
                'src/custom_ops/csrc/transpose_kernelv4.cu',
                'src/custom_ops/csrc/bmm_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
