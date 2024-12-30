// 主入口，声明其它cpp文件中的函数

#include <torch/extension.h>

// 声明在其他cpp文件中定义的函数
at::Tensor custom_matrix_mul_v1(at::Tensor A, at::Tensor B);
at::Tensor custom_matrix_mul_v2(at::Tensor A, at::Tensor B);
at::Tensor custom_matrix_mul_v3(at::Tensor A, at::Tensor B);
at::Tensor custom_matrix_mul_v4(at::Tensor A, at::Tensor B);
at::Tensor custom_matrix_mul_v5(at::Tensor A, at::Tensor B);
at::Tensor custom_softmax_cuda_v1(at::Tensor input, int dim);
at::Tensor custom_softmax_cuda_v2(at::Tensor input, int dim);
at::Tensor custom_softmax_cuda_v3(at::Tensor input, int dim);
at::Tensor custom_transpose_v1(at::Tensor A, int dim_x=0, int dim_y=1);
at::Tensor custom_transpose_v2(at::Tensor A, int dim_x=0, int dim_y=1);
at::Tensor custom_transpose_v3(at::Tensor A, int dim_x=0, int dim_y=1);
at::Tensor custom_transpose_v4(at::Tensor A, int dim_x=0, int dim_y=1);
// at::Tensor custom_softmax_cuda(at::Tensor input, int dim);
// at::Tensor custom_bmm(at::Tensor A, at::Tensor B);
// at::Tensor custom_vecAdd(at::Tensor A, at::Tensor B);
// at::Tensor custom_transpose(at::Tensor A);
// at::Tensor custom_matAdd(at::Tensor A, at::Tensor B);
// at::Tensor custom_layerNorm_cuda(at::Tensor input, at::Tensor gamma, at::Tensor beta, int normalized_shape);
// at::Tensor custom_relu_cuda(at::Tensor input);
// at::Tensor custom_softmax_cuda(at::Tensor input, int dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_matMul_cuda_v1", &custom_matrix_mul_v1, "Matrix Multiplication");
    m.def("custom_matMul_cuda_v2", &custom_matrix_mul_v2, "Matrix Multiplication");
    m.def("custom_matMul_cuda_v3", &custom_matrix_mul_v3, "Matrix Multiplication");
    m.def("custom_matMul_cuda_v4", &custom_matrix_mul_v4, "Matrix Multiplication");
    m.def("custom_matMul_cuda", &custom_matrix_mul_v5, "Matrix Multiplication");
    m.def("custom_softmax_cuda_v1", &custom_softmax_cuda_v1, "Custom Softmax");
    m.def("custom_softmax_cuda_v2", &custom_softmax_cuda_v2, "Custom Softmax");
    m.def("custom_softmax_cuda", &custom_softmax_cuda_v3, "Custom Softmax");
    m.def("custom_transpose_cuda_v1", &custom_transpose_v1, "Custom Transpose",
        py::arg("A"),
        py::arg("dim_x") = 0,
        py::arg("dim_y") = 1);
    m.def("custom_transpose_cuda_v2", &custom_transpose_v2, "Custom Transpose",
        py::arg("A"),
        py::arg("dim_x") = 0,
        py::arg("dim_y") = 1);
    m.def("custom_transpose_cuda_v3", &custom_transpose_v3, "Custom Transpose",
        py::arg("A"),
        py::arg("dim_x") = 0,
        py::arg("dim_y") = 1);
    m.def("custom_transpose_cuda_v4", &custom_transpose_v4, "Custom Transpose",
        py::arg("A"),
        py::arg("dim_x") = 0,
        py::arg("dim_y") = 1);
    m.def("custom_transpose_cuda", &custom_transpose_v4, "Custom Transpose",
        py::arg("A"),
        py::arg("dim_x") = 0,
        py::arg("dim_y") = 1);

}
