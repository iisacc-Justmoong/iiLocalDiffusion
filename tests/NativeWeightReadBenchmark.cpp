// Opt-in real checkpoint I/O probe; never run as a unit test.
#include <core/util.h>
#include <ggml-metal.h>
#include <ggml-backend.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <sys/resource.h>

int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "usage: NativeWeightReadBenchmark file offset bytes bulk|mapped\n"; return 1; }
    const auto offset = std::stoull(argv[2]);
    const auto bytes = std::stoull(argv[3]);
    const std::string mode = argv[4];
    if (mode != "bulk" && mode != "mapped") return 2;
    auto file = MmapWrapper::create(argv[1], false);
    if (!file || offset > file->size() || bytes > file->size() - offset) return 3;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_metal_init(), ggml_backend_free);
    if (!backend) return 4;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_buffer(backend.get(), bytes), ggml_backend_buffer_free);
    if (!buffer || !ggml_backend_metal_buffer_is_shared(buffer.get())) return 5;
    auto* destination = static_cast<unsigned char*>(ggml_backend_buffer_get_base(buffer.get()));
    std::memset(destination, 0, bytes);
    rusage before{}, after{}; getrusage(RUSAGE_SELF, &before);
    const auto start = std::chrono::steady_clock::now();
    if (mode == "bulk") { if (!file->copy_data(destination, bytes, offset)) return 6; }
    else std::memcpy(destination, file->data() + offset, bytes);
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    getrusage(RUSAGE_SELF, &after);
    uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < bytes; ++i) { hash ^= destination[i]; hash *= 1099511628211ull; }
    std::cout << "{\"mode\":\"" << mode << "\",\"offset\":" << offset << ",\"bytes\":" << bytes
              << ",\"seconds\":" << seconds << ",\"mib_per_second\":" << bytes / (1024.0 * 1024 * seconds)
              << ",\"major_faults\":" << after.ru_majflt - before.ru_majflt
              << ",\"fnv1a\":\"" << std::hex << hash << "\"}\n";
}
