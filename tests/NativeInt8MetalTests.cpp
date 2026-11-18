#include <core/compute_workspace.h>
#include <ggml-metal.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

std::vector<float> evaluate(ggml_backend_t backend, bool convrot) {
    Context ctx(ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true}), ggml_free);
    auto *weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, 64, 64);
    auto *scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
    auto *bias = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
    auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, 16);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buffer) throw std::runtime_error("fixture allocation failed");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<int8_t> weights(64 * 64);
    std::vector<float> scales(64), biases(64), inputs(64 * 16);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = int(i % 15) - 7;
    for (size_t i = 0; i < scales.size(); ++i) { scales[i] = .01f + i * .0001f; biases[i] = i * .003f; }
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = std::sin(float(i) * .1f);
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(scale, scales.data(), 0, scales.size() * sizeof(float));
    ggml_backend_tensor_set(bias, biases.data(), 0, biases.size() * sizeof(float));
    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
    auto *activation = convrot ? ggml_quantize_i8_convrot(ctx.get(), input, 64) : input;
    auto *output = ggml_mul_mat_i8_tensorwise(ctx.get(), weight, activation, scale, bias, convrot ? 64 : 0);
    if (ggml_backend_is_metal(backend) && ggml_backend_supports_op(backend, output))
        throw std::runtime_error("Metal advertised an unavailable INT8 matrix kernel");
    auto *graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    sd::ComputeWorkspace workspace(backend);
    auto assign = [](ggml_backend_sched_t, ggml_cgraph*) {};
    auto measured = workspace.measure(graph, 0, [&](const ggml_tensor *tensor) {
        return tensor == weight || tensor == scale || tensor == bias || tensor == input ? backend : nullptr;
    }, assign);
    if (!workspace.prepare(measured) || !workspace.allocate(graph, assign))
        throw std::runtime_error("mixed-backend workspace allocation failed");
    if (ggml_backend_is_metal(backend) && !workspace.scheduler())
        throw std::runtime_error("INT8 operation did not select the CPU fallback scheduler");
    const auto status = workspace.scheduler() ? ggml_backend_sched_graph_compute(workspace.scheduler(), graph)
                                               : ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("matrix computation failed");
    std::vector<float> result(64 * 16);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    workspace.segment_end();
    return result;
}
}

int main() {
    try {
        Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
        Backend metal(ggml_backend_metal_init(), ggml_backend_free);
        if (!cpu || !metal) throw std::runtime_error("CPU or Metal backend unavailable");
        for (bool convrot : {false, true}) {
            auto expected = evaluate(cpu.get(), convrot);
            auto actual = evaluate(metal.get(), convrot);
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > .0001f)
                    throw std::runtime_error("Metal/CPU fallback changed INT8 matrix results");
            }
        }
        std::cout << "INT8 tensorwise and ConvRot matrices use numerically equivalent CPU fallback\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
