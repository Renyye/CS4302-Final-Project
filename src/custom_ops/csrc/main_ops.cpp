// 主入口，声明其它cpp文件中的函数

#include <torch/extension.h>

// 声明在其他cpp文件中定义的函数
torch::Tensor custom_bmm(torch::Tensor A, torch::Tensor B);
torch::Tensor custom_vecAdd(torch::Tensor A, torch::Tensor B);
torch::Tensor custom_matMul(torch::Tensor A, torch::Tensor B);
torch::Tensor custom_transpose(torch::Tensor A);
torch::Tensor custom_matAdd(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_bmm_cuda", &custom_bmm, "Batched Matrix Multiplication");
    m.def("custom_vecAdd_cuda", &custom_vecAdd, "Custom Vector Addition");
    m.def("custom_matMul_cuda", &custom_matMul, "Custom Matrix Multiplication");
    m.def("custom_transpose_cuda", &custom_transpose, "Custom Transpouse");
    m.def("custom_matAdd_cuda", &custom_matAdd, "Custom Matrix Addition");
    // 此处还可以绑定更多在其他文件中实现的函数
}
