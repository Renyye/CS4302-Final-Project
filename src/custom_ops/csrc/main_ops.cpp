// 主入口，声明其它cpp文件中的函数

#include <torch/extension.h>

// 声明在其他cpp文件中定义的函数
at::Tensor custom_bmm(at::Tensor A, at::Tensor B);
at::Tensor custom_vecAdd(at::Tensor A, at::Tensor B);
at::Tensor custom_matMul(at::Tensor A, at::Tensor B);
at::Tensor custom_transpose(at::Tensor A);
at::Tensor custom_matAdd(at::Tensor A, at::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_bmm_cuda", &custom_bmm, "Batched Matrix Multiplication");
    m.def("custom_vecAdd_cuda", &custom_vecAdd, "Custom Vector Addition");
    m.def("custom_matMul_cuda", &custom_matMul, "Custom Matrix Multiplication");
    m.def("custom_transpose_cuda", &custom_transpose, "Custom Transpouse");
    m.def("custom_matAdd_cuda", &custom_matAdd, "Custom Matrix Addition");
    // 此处还可以绑定更多在其他文件中实现的函数
}
