#include <core/util.h>
#include <stable-diffusion.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <ggml-backend.h>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    const std::filesystem::path path = argv[1];
    const auto alignmentPath = path.string() + ".alignment";
    const auto alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    for (const std::size_t bytes : {1, 17, 31, 32, 33, 127, 768, 3137}) {
        const std::vector<char> data(bytes, 'x');
        { std::ofstream file(alignmentPath, std::ios::binary); file.write(data.data(), bytes); }
        auto source = MmapWrapper::create(alignmentPath, false, true);
        auto edited = MmapWrapper::create(alignmentPath, true, true);
        if (!source || !edited || reinterpret_cast<std::uintptr_t>(source->data()) % alignment
            || reinterpret_cast<std::uintptr_t>(edited->data()) % alignment) return 25;
        if (source->size() != bytes || edited->size() != bytes || source->data()[bytes - 1] != 'x') return 26;
        auto buffer = ggml_backend_cpu_buffer_from_ptr(source->writable_data(), source->size());
        if (!buffer) return 27;
        ggml_backend_buffer_free(buffer);
    }
    std::filesystem::remove(alignmentPath);
    sd_release_resident_model_memory();
    std::vector<unsigned char> expected(33 * 1024 * 1024 + 37);
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = (i * 31 + i / 251) % 256;
    { std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char*>(expected.data()), expected.size()); }
    auto mapped = MmapWrapper::create(path.string(), false);
    if (!mapped) return 2;
    std::vector<unsigned char> actual(expected.size());
    if (!mapped->copy_data(actual.data(), actual.size(), 0) || actual != expected) return 3;
    if (!mapped->copy_data(actual.data(), 1023, 19) || std::memcmp(actual.data(), expected.data() + 19, 1023)) return 4;
    if (mapped->copy_data(actual.data(), 2, expected.size() - 1)
        || mapped->copy_data(actual.data(), 0, expected.size() + 1)
        || !mapped->copy_data(actual.data(), 0, expected.size())) return 5;
    auto writable = MmapWrapper::create(path.string(), true);
    if (!writable) return 6;
    int loadProgress = 0;
    sd_set_progress_stage_callback([](sd_progress_stage_t stage, int step, int total, float, void* data) {
        if (stage == SD_PROGRESS_LOAD && step > 0 && step <= total) ++*static_cast<int*>(data);
    }, &loadProgress);
    auto resident = MmapWrapper::create(path.string(), false, true);
    sd_set_progress_stage_callback(nullptr, nullptr);
    if (!resident) return 11;
    if (loadProgress != 2) return 15;
    if (!resident->copy_data(actual.data(), actual.size(), 0) || actual != expected) return 12;
    if (resident->copy_data(actual.data(), 2, expected.size() - 1)
        || !resident->copy_data(actual.data(), 0, expected.size())) return 16;
    const auto address = resident->data();
    resident.reset(); // Execution context destruction is not runtime release.
    loadProgress = 0;
    sd_set_progress_stage_callback([](sd_progress_stage_t, int, int, float, void* data) {
        ++*static_cast<int*>(data);
    }, &loadProgress);
    resident = MmapWrapper::create(path.string(), false, true);
    if (!resident || resident->data() != address || loadProgress) return 19;
    auto edited = MmapWrapper::create(path.string(), true, true);
    if (!edited || edited->data() == address || loadProgress) return 20;
    edited->writable_data()[19] ^= 0xff;
    if (resident->data()[19] != expected[19]) return 21;
    edited.reset();
    // Explicit release evicts runtime ownership, but cannot invalidate active readers.
    sd_release_resident_model_memory();
    if (!resident->copy_data(actual.data(), actual.size(), 0) || actual != expected) return 22;
    // Cancel after one actual preload chunk, not just before opening a file.
    loadProgress = 0;
    sd_set_progress_stage_callback([](sd_progress_stage_t, int, int, float, void* data) {
        ++*static_cast<int*>(data);
    }, &loadProgress);
    sd_set_abort_callback([](void* data) { return *static_cast<int*>(data) > 0; }, &loadProgress);
    const auto interrupted = MmapWrapper::create(path.string(), false, true);
    sd_set_abort_callback(nullptr, nullptr);
    sd_set_progress_stage_callback(nullptr, nullptr);
    if (interrupted || loadProgress != 1) return 17;
    loadProgress = 0;
    sd_set_progress_stage_callback([](sd_progress_stage_t, int, int, float, void* data) {
        ++*static_cast<int*>(data);
    }, &loadProgress);
    auto reloaded = MmapWrapper::create(path.string(), false, true);
    sd_set_progress_stage_callback(nullptr, nullptr);
    if (!reloaded || loadProgress != 2) return 23; // Partial loads never enter the pool.
    writable->writable_data()[19] ^= 0xff;
    if (!writable->copy_data(actual.data(), 1023, 19) || actual[0] != (expected[19] ^ 0xff)) return 7;
    if (!mapped->copy_data(actual.data(), 1023, 19) || actual[0] != expected[19]) return 8;
#if !defined(_WIN32)
    sd_set_abort_callback([](void*) { return true; }, nullptr);
    if (mapped->copy_data(actual.data(), actual.size(), 0)) return 9;
    if (resident->copy_data(actual.data(), actual.size(), 0)) return 18;
    sd_set_abort_callback(nullptr, nullptr);
    writable.reset();
    std::filesystem::resize_file(path, 1024);
    if (mapped->copy_data(actual.data(), 2048, 0)) return 10; // Short read must fail, not SIGBUS.
    if (!resident->copy_data(actual.data(), actual.size(), 0) || actual != expected) return 13;
    auto changed = MmapWrapper::create(path.string(), false, true);
    if (!changed || changed->size() != 1024 || changed->data() == reloaded->data()) return 24;
#endif
    mapped.reset(); writable.reset();
    std::filesystem::remove(path);
    if (!resident->copy_data(actual.data(), actual.size(), 0) || actual != expected) return 14;
    resident.reset();
    reloaded.reset();
    sd_release_resident_model_memory();
    std::cout << "bulk read: mapped and anonymous-resident data, offsets, bounds, private edits, cancellation and truncation passed\n";
}
