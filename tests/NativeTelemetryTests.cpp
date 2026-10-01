#include "Generation/NativeTelemetry.hpp"
#include <stdexcept>
#include <vector>
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::filesystem::create_directories(argv[1]);
    std::string currentTrace;
    {
        iiLocalDiffusion::native_detail::NativeTelemetry trace(argv[1]);
        currentTrace = iiLocalDiffusion::native_detail::lastNativeTrace().path;
        trace.log("copied model file into anonymous memory: fixture (1024 bytes)");
        trace.log("IILD_BACKEND diffusion=Metal\n");
        trace.stage("text-encode");
        trace.stage("denoise", 0, 10);
        trace.log("IILD_WEIGHT_LOAD begin");
        trace.loading(1, 400);
        // Real stalled work must remain observable without claiming progress.
        bool heartbeat = false;
        for (int attempt = 0; attempt < 150 && !heartbeat; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::ifstream input(currentTrace); std::string line;
            while (std::getline(input, line)) {
                auto *event = json_tokener_parse(line.c_str());
                if (!event) continue; // The writer may still be flushing its last line.
                const auto kind = std::string(json_object_get_string(json_object_object_get(event, "event")));
                if (kind == "heartbeat") {
                    heartbeat = true;
                    if (json_object_get_double(json_object_object_get(event, "no_progress_ms")) < 4000)
                        throw std::runtime_error("heartbeat invented forward progress");
                }
                json_object_put(event);
            }
        }
        if (!heartbeat) throw std::runtime_error("stalled work did not produce a heartbeat");
        trace.log("IILD_WEIGHT_LOAD end elapsed_ms=5000 bytes=1024 success=1");
        trace.log("IILD_WEIGHT_TRANSFER begin backend=Metal tensors=1");
        trace.log("IILD_WEIGHT_TRANSFER end elapsed_ms=20");
        trace.stage("denoise", 1, 10);
        trace.log("loading tensors completed, taking 22.00s");
        trace.stage("vae-decode");
        trace.log("reused runtime anonymous model source: fixture (1024 bytes)");
        trace.stage("postprocess");
        trace.finish(false, "test failure");
    }
    bool denoiseMetal = false, weightLoad = false, failed = false, memory = false, transfer = false;
    bool resident = false, loadProgress = false, reused = false;
    {
        std::ifstream file(currentTrace); std::string line;
        while (std::getline(file, line)) {
            auto *event = json_tokener_parse(line.c_str());
            if (!event) throw std::runtime_error("invalid telemetry JSONL");
            const auto text = [&](const char *key) { return std::string(json_object_get_string(json_object_object_get(event, key))); };
            denoiseMetal |= text("phase") == "denoise" && text("backend") == "Metal";
            weightLoad |= text("phase") == "denoise" && text("operation") == "weight-load";
            failed |= text("event") == "failed";
            resident |= text("event") == "resident-file-loaded" && text("phase") == "model-load";
            reused |= text("event") == "resident-source-reused";
            loadProgress |= json_object_get_int(json_object_object_get(event, "load_step")) == 1
                && json_object_get_int(json_object_object_get(event, "load_total")) == 400;
            transfer |= text("phase") == "denoise" && text("operation") == "weight-transfer";
            memory |= json_object_object_get(event, "process_peak_rss_bytes") != nullptr;
            if (text("schema") != "iild-native-telemetry-v1") throw std::runtime_error("wrong schema");
            json_object_put(event);
        }
    }
    if (!denoiseMetal || !weightLoad || !failed || !transfer) throw std::runtime_error("missing timing/placement/failure evidence");
    if (!resident || !loadProgress || !reused) throw std::runtime_error("missing resident preload/reuse evidence");
#if defined(__unix__) || defined(__APPLE__)
    if (!memory) throw std::runtime_error("missing process memory evidence");
#endif
}
