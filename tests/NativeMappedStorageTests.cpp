#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-metal.h>
#include <gguf.h>
#include <model_manager.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr auto tensorName = "model.diffusion_model.net.test.weight";
constexpr int elements = 128;

bool check(const std::filesystem::path &path, ggml_backend_t compute, ggml_backend_t storage) {
    Context context(ggml_init({8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
                              nullptr, true}), ggml_free);
    auto *weight = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, elements);
    ggml_set_name(weight, tensorName);
    ModelManager manager;
    manager.set_n_threads(1);
    manager.set_enable_mmap(true);
    manager.set_writable_mmap(false);
    if (!manager.loader().init_from_file(path.string())
        || !manager.register_param_tensors("fixture", {{tensorName, weight}},
            ModelManager::ResidencyMode::ParamBackend, compute, storage)
        || !manager.prepare_params({weight})) return false;
    if (!weight->buffer || !ggml_backend_dev_supports_buft(ggml_backend_get_device(compute),
                                                          ggml_backend_buffer_get_type(weight->buffer))) {
        std::cerr << "Mapped weight buffer is incompatible with " << ggml_backend_name(compute) << '\n';
        return false;
    }
    auto *output = ggml_scale(context.get(), weight, 2.0f);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), compute), ggml_backend_buffer_free);
    if (!buffer) return false;
    auto *graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(compute, graph) != GGML_STATUS_SUCCESS) return false;
    std::vector<float> values(elements);
    ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
    for (int i = 0; i < elements; ++i)
        if (values[i] != 2.0f * (i + 1)) return false;
    manager.release_compute_backend_params({weight});
    return true;
}
}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    const auto directory = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(directory);
    const auto path = directory / "mapped-weights.gguf";
    {
        Context context(ggml_init({ggml_tensor_overhead(), nullptr, true}), ggml_free);
        auto *weight = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, elements);
        ggml_set_name(weight, tensorName);
        std::vector<float> values(elements);
        for (int i = 0; i < elements; ++i) values[i] = i + 1;
        weight->data = values.data();
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
        gguf_add_tensor(file.get(), weight);
        if (!gguf_write_to_file(file.get(), path.string().c_str(), false)) return 2;
    }
    Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
    Backend metal(ggml_backend_metal_init(), ggml_backend_free);
    if (!cpu || !metal) return 3;
    if (!check(path, cpu.get(), cpu.get())) return 4;
    if (!check(path, metal.get(), metal.get())) return 5;
    if (!check(path, metal.get(), cpu.get())) return 6;
    std::filesystem::remove(path);
}
