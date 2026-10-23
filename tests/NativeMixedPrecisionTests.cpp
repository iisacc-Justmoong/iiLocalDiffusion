#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
void compute(ggml_backend_t cpu, int operation, int layout) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true}), ggml_free);
    require(bool(context), "Cannot allocate graph metadata");
    constexpr int columns = 17, rows = 2;
    auto *input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, columns, rows);
    auto *weight = layout == 2
        ? ggml_new_tensor_2d(context.get(), GGML_TYPE_BF16, rows, columns)
        : ggml_new_tensor_2d(context.get(), GGML_TYPE_BF16, columns, layout == 1 ? 1 : rows);
    auto *operand = layout == 2 ? ggml_transpose(context.get(), weight) : weight;
    auto *result = operation == 0 ? ggml_add(context.get(), input, operand)
        : operation == 1 ? ggml_sub(context.get(), input, operand)
        : operation == 2 ? ggml_mul(context.get(), input, operand)
        : ggml_div(context.get(), input, operand);
    require(result->type == GGML_TYPE_F32, "Mixed precision output must retain FP32");
    require(ggml_backend_supports_op(cpu, result), "CPU rejected the mixed precision graph");
    auto *graph = ggml_new_graph_custom(context.get(), 32, false);
    ggml_build_forward_expand(graph, result);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), cpu), ggml_backend_buffer_free);
    require(bool(buffer), "Cannot allocate CPU graph");
    std::vector<float> inputs(columns * rows);
    std::vector<ggml_bf16_t> weights(ggml_nelements(weight));
    for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = float(i + 1) / 4;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = ggml_fp32_to_bf16(float(i % 7 + 1) / 2);
    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(ggml_bf16_t));
    require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "CPU graph failed");
    std::vector<float> outputs(inputs.size());
    ggml_backend_tensor_get(result, outputs.data(), 0, outputs.size() * sizeof(float));
    for (int row = 0; row < rows; ++row) for (int column = 0; column < columns; ++column) {
        const auto i = row * columns + column;
        const auto weightIndex = layout == 2 ? column * rows + row : (layout == 1 ? column : i);
        const float a = inputs[i], b = ggml_bf16_to_fp32(weights[weightIndex]);
        const auto expected = operation == 0 ? a + b : operation == 1 ? a - b : operation == 2 ? a * b : a / b;
        require(std::isfinite(outputs[i]) && std::abs(outputs[i] - expected) < 0.00001f,
            "Mixed precision CPU result differs from FP32 arithmetic");
    }
}
}

int main() {
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        require(bool(cpu), "CPU backend is unavailable");
        ggml_backend_cpu_set_n_threads(cpu.get(), 2);
        for (int operation = 0; operation < 4; ++operation)
            for (int layout = 0; layout < 3; ++layout) compute(cpu.get(), operation, layout);
        std::cout << "FP32/BF16 CPU add, subtract, multiply and divide passed for contiguous, broadcast and strided inputs\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
