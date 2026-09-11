#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>
#include <algorithm>
#include <memory>
#include <thread>
#include <vector>

int main(int argc, char **) {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_metal_init(), ggml_backend_free);
    if (!backend) return 1;
    constexpr int count = 24, elements = 65536;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({count * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    if (!context) return 2;
    std::vector<ggml_tensor *> tensors;
    for (int i = 0; i < count; ++i) tensors.push_back(ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, elements));
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer || ggml_backend_metal_buffer_is_shared(nullptr)) return 3;
    const bool shared = ggml_backend_metal_buffer_is_shared(buffer.get());
    if (argc > 1) return shared ? 4 : 0; // Private storage must keep serialized uploads.
    if (!shared) return 5;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cpu(
        ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 1024), ggml_backend_buffer_free);
    if (ggml_backend_metal_buffer_is_shared(cpu.get())) return 6;
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 6; ++worker) workers.emplace_back([&, worker] {
        for (int i = worker; i < count; i += 6) {
            std::vector<float> source(elements, static_cast<float>(i + 1));
            ggml_backend_tensor_set(tensors[i], source.data(), 0, source.size() * sizeof(float));
        }
    });
    for (auto &worker : workers) worker.join();
    std::vector<float> result(elements);
    for (int i = 0; i < count; ++i) {
        ggml_backend_tensor_get(tensors[i], result.data(), 0, result.size() * sizeof(float));
        if (!std::all_of(result.begin(), result.end(), [i](float value) { return value == i + 1; })) return 7;
    }
}
