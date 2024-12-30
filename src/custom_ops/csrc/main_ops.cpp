// 主入口，声明其它cpp文件中的函数

#include <torch/extension.h>

// 声明在其他cpp文件中定义的函数
at::Tensor custom_matrix_mul(at::Tensor A, at::Tensor B);
// at::Tensor custom_bmm(at::Tensor A, at::Tensor B);
// at::Tensor custom_vecAdd(at::Tensor A, at::Tensor B);
// at::Tensor custom_transpose(at::Tensor A);
// at::Tensor custom_matAdd(at::Tensor A, at::Tensor B);
// at::Tensor custom_layerNorm_cuda(at::Tensor input, at::Tensor gamma, at::Tensor beta, int normalized_shape);
// at::Tensor custom_relu_cuda(at::Tensor input);
// at::Tensor custom_softmax_cuda(at::Tensor input, int dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_matMul_cuda", &custom_matrix_mul, "Matrix Multiplication");
    // m.def("custom_bmm_cuda", &custom_bmm, "Batched Matrix Multiplication");
    // m.def("custom_vecAdd_cuda", &custom_vecAdd, "Custom Vector Addition");
    // m.def("custom_transpose_cuda", &custom_transpose, "Custom Transpouse");
    // m.def("custom_matAdd_cuda", &custom_matAdd, "Custom Matrix Addition");
    // m.def("custom_layerNorm_cuda", &custom_layerNorm_cuda, "Custom Layer Normalization");
    // m.def("custom_relu_cuda", &custom_relu_cuda, "Custom ReLU Activation");
    // m.def("custom_softmax_cuda", &custom_softmax_cuda, "Custom Softmax Activation");
    // 此处还可以绑定更多在其他文件中实现的函数
}
