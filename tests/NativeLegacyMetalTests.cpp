#include <core/compute_workspace.h>
#include <core/ggml_extend.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>
#include <ggml/src/ggml-backend-impl.h>
#include <ggml/src/ggml-metal/ggml-metal-device.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

static int verifyAttention(ggml_backend_t metal) {
    for (const int length : {64, 7424}) {
        const int channels = length == 64 ? 64 : 2048;
        const int heads = length == 64 ? 2 : 16;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true}), ggml_free);
        auto *q = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, channels, length, 1);
        auto *k = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, channels, length, 1);
        auto *v = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, channels, length, 1);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> inputBuffer(nullptr, ggml_backend_buffer_free);
        if (length == 64) {
            inputBuffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), metal));
            std::vector<float> values(channels * length, 0.0f);
            ggml_backend_tensor_set(q, values.data(), 0, values.size() * sizeof(float));
            ggml_backend_tensor_set(k, values.data(), 0, values.size() * sizeof(float));
            for (int row = 0; row < length; ++row)
                for (int channel = 0; channel < channels; ++channel)
                    values[row * channels + channel] = channel / 64.0f + row / 128.0f;
            ggml_backend_tensor_set(v, values.data(), 0, values.size() * sizeof(float));
        }
        auto *result = ggml_ext_attention_ext(context.get(), metal, q, k, v, heads, nullptr, false, true, 1.0f);
        auto *graph = ggml_new_graph_custom(context.get(), 128, false);
        ggml_build_forward_expand(graph, result);
        bool flash = false;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto *node = ggml_graph_node(graph, i);
            flash |= node->op == GGML_OP_FLASH_ATTN_EXT;
            if (ggml_nbytes(node) > 2 * channels * size_t(length) * sizeof(float)) {
                std::cerr << "Attention allocated a quadratic intermediate\n";
                return 7;
            }
        }
        if (!flash) {
            std::cerr << "CPU flash attention was discarded on a legacy Metal device\n";
            return 8;
        }
        if (length != 64) continue; // Validate production graph sizes without allocating the large inputs.
        sd::ComputeWorkspace workspace(metal);
        auto assign = [](ggml_backend_sched_t, ggml_cgraph *) {};
        auto measurement = workspace.measure(graph, 0, [&](const ggml_tensor *t) {
            return t == q || t == k || t == v ? metal : nullptr;
        }, assign);
        if (!workspace.prepare(measurement) || !workspace.allocate(graph, assign)) return 9;
        if (ggml_backend_sched_graph_compute(workspace.scheduler(), graph) != GGML_STATUS_SUCCESS) return 10;
        std::vector<float> output(channels * length);
        ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
        workspace.segment_end();
        for (int row = 0; row < length; ++row) {
            for (int channel = 0; channel < channels; ++channel) {
                const auto actual = output[row * channels + channel];
                const auto expected = channel / 64.0f + (length - 1) / 256.0f;
                if (!std::isfinite(actual) || std::abs(actual - expected) > 0.0001f) return 11;
            }
        }
    }
    return 0;
}

int main() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> metal(ggml_backend_metal_init(), ggml_backend_free);
    if (!metal) return 1;
    // Restrict the real backend to the observed A12Z capabilities. No production
    // feature overrides or unsupported GPU kernels are enabled by this test.
    auto device = static_cast<ggml_metal_device_t>(ggml_backend_get_device(metal.get())->context);
    auto *props = const_cast<ggml_metal_device_props *>(ggml_metal_device_get_props(device));
    struct Restore {
        ggml_metal_device_props *props;
        ggml_metal_device_props saved;
        ~Restore() { *props = saved; }
    } restore{props, *props};
    props->has_bfloat = false;
    props->has_simdgroup_reduction = false;
    props->has_simdgroup_mm = false;

    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({24 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true}), ggml_free);
    auto *weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_BF16, 32, 32);
    auto *input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 32);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> weights(
        ggml_backend_alloc_ctx_tensors(context.get(), metal.get()), ggml_backend_buffer_free);
    ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> values(1024);
    for (int row = 0; row < 32; ++row)
        for (int col = 0; col < 32; ++col) values[row * 32 + col] = (row + 1) / 32.0f;
    std::vector<ggml_bf16_t> bf16(values.size());
    ggml_fp32_to_bf16_row(values.data(), bf16.data(), static_cast<int64_t>(values.size()));
    ggml_backend_tensor_set(weight, bf16.data(), 0, bf16.size() * sizeof(ggml_bf16_t));
    std::vector<float> ones(32, 1.0f);
    ggml_backend_tensor_set(input, ones.data(), 0, ones.size() * sizeof(float));
    auto *view = ggml_view_2d(context.get(), weight, 32, 32, weight->nb[1], 0);
    auto *reshape = ggml_reshape_2d(context.get(), weight, 32, 32);
    auto *transpose = ggml_transpose(context.get(), weight);
    auto *permute = ggml_permute(context.get(), weight, 1, 0, 2, 3);
    for (auto *storage : {weight, view, reshape, transpose, permute}) {
        if (!ggml_backend_supports_op(metal.get(), storage)) {
            std::cerr << "BF16 storage/view incorrectly requires BF16 arithmetic support\n";
            return 2;
        }
    }
    auto *product = ggml_mul_mat(context.get(), view, input);
    if (ggml_backend_supports_op(metal.get(), product)) return 3;
    auto *result = ggml_scale(context.get(), product, 2.0f);
    auto *graph = ggml_new_graph_custom(context.get(), 32, false);
    ggml_build_forward_expand(graph, result);
    sd::ComputeWorkspace workspace(metal.get());
    auto assign = [](ggml_backend_sched_t, ggml_cgraph *) {};
    auto measurement = workspace.measure(graph, 0, [&](const ggml_tensor *t) {
        return t == weight || t == input ? metal.get() : nullptr;
    }, assign);
    if (!workspace.prepare(measurement) || !workspace.allocate(graph, assign)) return 4;
    if (ggml_backend_sched_graph_compute(workspace.scheduler(), graph) != GGML_STATUS_SUCCESS) return 5;
    std::vector<float> output(32);
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    workspace.segment_end();
    for (int row = 0; row < 32; ++row)
        if (!std::isfinite(output[row]) || std::abs(output[row] - 2.0f * (row + 1)) > 0.001f) return 6;
    return verifyAttention(metal.get());
}
