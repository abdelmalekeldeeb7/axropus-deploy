/*
 * bindings.cpp — PyTorch C++ bindings for AMF decode kernels.
 *
 * Compiled with the three ``.cu`` files via torch.utils.cpp_extension.
 * When any of the kernel entry points are missing (e.g. when targeting
 * hardware that does not support the format) the corresponding binding
 * raises at call time. The Python dispatch table never calls a binding
 * that is not supported by the current SM version.
 */

#include <torch/extension.h>

namespace axropus {
namespace fp8 {
torch::Tensor fp8_decode_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double k_scale,
    double v_scale,
    double softmax_scale,
    torch::optional<torch::Tensor> mask
);
}  // namespace fp8

namespace int4 {
torch::Tensor int4_decode_attention(
    torch::Tensor q,
    torch::Tensor k_packed,
    torch::Tensor v_packed,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    int64_t num_kv_tokens,
    double softmax_scale,
    torch::optional<torch::Tensor> mask
);
}  // namespace int4

namespace nvfp4 {
torch::Tensor nvfp4_decode_attention(
    torch::Tensor q,
    torch::Tensor k_packed,
    torch::Tensor v_packed,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    int64_t num_kv_tokens,
    double softmax_scale
);
}  // namespace nvfp4
}  // namespace axropus

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fp8_decode_attention",
        &axropus::fp8::fp8_decode_attention,
        "Hopper FP8 E4M3 decode attention kernel",
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("v"),
        pybind11::arg("k_scale"),
        pybind11::arg("v_scale"),
        pybind11::arg("softmax_scale"),
        pybind11::arg("mask") = torch::optional<torch::Tensor>()
    );

    m.def(
        "int4_decode_attention",
        &axropus::int4::int4_decode_attention,
        "INT4 per-block decode attention kernel",
        pybind11::arg("q"),
        pybind11::arg("k_packed"),
        pybind11::arg("v_packed"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("num_kv_tokens"),
        pybind11::arg("softmax_scale"),
        pybind11::arg("mask") = torch::optional<torch::Tensor>()
    );

    m.def(
        "nvfp4_decode_attention",
        &axropus::nvfp4::nvfp4_decode_attention,
        "NVFP4 Blackwell decode attention kernel (stub)",
        pybind11::arg("q"),
        pybind11::arg("k_packed"),
        pybind11::arg("v_packed"),
        pybind11::arg("k_scales"),
        pybind11::arg("v_scales"),
        pybind11::arg("num_kv_tokens"),
        pybind11::arg("softmax_scale")
    );
}
