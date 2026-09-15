#include <core/compute_workspace.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

static void cpuOperation(ggml_tensor *dst, const ggml_tensor *src, int ith, int, void *) {
    if (ith != 0) return;
    auto *output = static_cast<float *>(dst->data);
    const auto *input = static_cast<const float *>(src->data);
    for (int64_t i = 0; i < ggml_nelements(src); ++i) output[i] = input[i] + 1.0f;
}

int main() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> metal(ggml_backend_metal_init(), ggml_backend_free);
    if (!metal) return 1;
    sd::ComputeWorkspace workspace(metal.get());
    for (const bool cpuInplace : {false, true}) {
        for (const int count : {128, 32, 128}) {
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
                ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true}), ggml_free);
            auto *input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, count);
            ggml_set_name(input, "input");
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> inputBuffer(
                ggml_backend_alloc_ctx_tensors(context.get(), metal.get()), ggml_backend_buffer_free);
            ggml_backend_buffer_set_usage(inputBuffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<float> values(count);
            for (int i = 0; i < count; ++i) values[i] = i + 1.0f;
            ggml_backend_tensor_set(input, values.data(), 0, count * sizeof(float));
            auto *squared = ggml_sqr(context.get(), input);
            auto *cpu = cpuInplace
                ? ggml_map_custom1_inplace(context.get(), squared, cpuOperation, 1, nullptr)
                : ggml_map_custom1(context.get(), squared, cpuOperation, 1, nullptr);
            ggml_set_name(cpu, "CPU fallback");
            auto *view = ggml_view_1d(context.get(), cpu, count, 0);
            auto *root = ggml_sqrt_inplace(context.get(), view);
            ggml_set_name(root, "sqrt after CPU fallback");
            auto *scaled = ggml_scale_inplace(context.get(), root, 2.0f);
            auto *result = ggml_mul(context.get(), scaled, input);
            auto *graph = ggml_new_graph_custom(context.get(), 32, false);
            ggml_build_forward_expand(graph, result);
            auto assign = [](ggml_backend_sched_t, ggml_cgraph *) {};
            auto measurement = workspace.measure(graph, 0, [&](const ggml_tensor *t) {
                return t == input ? metal.get() : nullptr;
            }, assign);
            if (!workspace.prepare(measurement) || !workspace.allocate(graph, assign)) return 2;
            const auto scheduler = workspace.scheduler();
            for (int n = 0; n < ggml_graph_n_nodes(graph); ++n) {
                auto *node = ggml_graph_node(graph, n);
                const auto backend = ggml_backend_sched_get_tensor_backend(scheduler, node);
                const auto buffer = node->view_src ? node->view_src->buffer : node->buffer;
                if (!backend || !buffer || !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(buffer))) {
                    std::cerr << ggml_get_name(node) << " uses an incompatible compute buffer\n";
                    workspace.segment_end();
                    return 3;
                }
            }
            if (ggml_backend_sched_get_tensor_backend(scheduler, result) != metal.get()) return 6;
            if (ggml_backend_sched_graph_compute(scheduler, graph) != GGML_STATUS_SUCCESS) return 4;
            std::vector<float> output(count);
            ggml_backend_tensor_get(result, output.data(), 0, count * sizeof(float));
            for (int i = 0; i < count; ++i) {
                const auto expected = 2.0f * values[i] * std::sqrt(values[i] * values[i] + 1.0f);
                if (!std::isfinite(output[i]) || std::abs(output[i] - expected) > 0.0001f * std::max(1.0f, expected)) return 5;
            }
            workspace.segment_end();
        }
    }
}
