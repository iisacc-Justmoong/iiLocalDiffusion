#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-metal.h>
#include <gguf.h>
#include <model_manager.h>
#include <stable-diffusion.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr auto tensorName = "model.diffusion_model.net.test.weight";
constexpr auto secondName = "model.diffusion_model.net.test2.weight";
constexpr int elements = 17 * 1024 * 1024 / sizeof(float) + 128;

bool check(const std::filesystem::path &path, ggml_backend_t compute, ggml_backend_t storage,
           ModelManager::ResidencyMode mode = ModelManager::ResidencyMode::ParamBackend, bool resident = false) {
    Context context(ggml_init({8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
                              nullptr, true}), ggml_free);
    auto *weight = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, elements);
    ggml_set_name(weight, tensorName);
    ModelManager manager;
    manager.set_n_threads(1);
    manager.set_enable_mmap(true);
    manager.set_writable_mmap(false);
    manager.loader().set_resident_memory(resident);
    if (!manager.loader().init_from_file(path.string())
        || !manager.register_param_tensors("fixture", {{tensorName, weight}},
            mode, compute, storage)
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
    if (resident) {
        // This file is a generated fixture. Remove the backing source to prove
        // that every later segment gets its exact weights from anonymous memory.
        std::filesystem::remove(path);
        for (int pass = 0; pass < 2; ++pass) {
            if (!manager.prepare_params({weight})
                || ggml_backend_graph_compute(compute, graph) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
            for (int i = 0; i < elements; ++i)
                if (values[i] != 2.0f * (i + 1)) return false;
            manager.release_compute_backend_params({weight});
        }
    }
    return true;
}
bool checkCancelledStaging(const std::filesystem::path& path, ggml_backend_t compute, ggml_backend_t storage) {
    Context context(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    auto* first = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, elements);
    auto* second = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, elements);
    ggml_set_name(first, tensorName); ggml_set_name(second, secondName);
    ModelManager manager;
    manager.set_n_threads(1);
    manager.set_enable_mmap(true);
    manager.set_writable_mmap(false);
    if (!manager.loader().init_from_file(path.string())
        || !manager.register_param_tensors("cancel fixture", {{tensorName, first}, {secondName, second}},
            ModelManager::ResidencyMode::ParamBackend, compute, storage)
        || !manager.load_all_params_eagerly()) return false;
    // Each tensor spans three read chunks. Abort after the first tensor has
    // swapped into Metal storage; retry must recover both mapped originals.
    int calls = 0;
    sd_set_abort_callback([](void* data) { return ++*static_cast<int*>(data) > 3; }, &calls);
    const bool unexpectedlyPrepared = manager.prepare_params({first, second});
    sd_set_abort_callback(nullptr, nullptr);
    if (unexpectedlyPrepared || calls != 4 || !manager.prepare_params({first, second})) return false;
    std::vector<float> values(elements);
    for (auto* tensor : {first, second}) {
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        for (int i = 0; i < elements; ++i) if (values[i] != float(i + 1)) return false;
    }
    manager.release_compute_backend_params({first, second});
    return true;
}

}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    const auto directory = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(directory);
    const auto path = directory / "mapped-weights.gguf";
    {
        Context context(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
        auto *weight = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, elements);
        ggml_set_name(weight, tensorName);
        std::vector<float> values(elements);
        for (int i = 0; i < elements; ++i) values[i] = i + 1;
        weight->data = values.data();
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
        gguf_add_tensor(file.get(), weight);
        auto* second = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, elements);
        ggml_set_name(second, secondName); second->data = values.data();
        gguf_add_tensor(file.get(), second);
        if (!gguf_write_to_file(file.get(), path.string().c_str(), false)) return 2;
    }
    Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
    Backend metal(ggml_backend_metal_init(), ggml_backend_free);
    if (!cpu || !metal) return 3;
    if (!check(path, cpu.get(), cpu.get())) return 4;
    if (!check(path, metal.get(), metal.get())) return 5;
    if (!check(path, metal.get(), cpu.get())) return 6;
    if (!check(path, metal.get(), metal.get(), ModelManager::ResidencyMode::Disk)) return 7;
    Buffer probe(ggml_backend_alloc_buffer(metal.get(), 256), ggml_backend_buffer_free);
    if (probe && ggml_backend_metal_buffer_is_shared(probe.get())
        && !checkCancelledStaging(path, metal.get(), cpu.get())) return 8;
    if (!check(path, metal.get(), metal.get(), ModelManager::ResidencyMode::Disk, true)) return 9;
    std::filesystem::remove(path);
}
