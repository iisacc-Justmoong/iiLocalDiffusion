#pragma once
#include <json-c/json.h>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace iiLocalDiffusion::native_detail {
struct NativeTraceRecord { std::string path; double elapsedMs = 0; };
inline NativeTraceRecord &lastNativeTrace() {
    static thread_local NativeTraceRecord value;
    return value;
}
// Opt-in JSONL survives worker termination. No prompts, weights, or image pixels.
// Memory is process/system scoped, never mislabelled as per-model GPU allocation.
class NativeTelemetry {
    using Clock = std::chrono::steady_clock;
    std::mutex mutex;
    std::condition_variable wake;
    std::thread sampler;
    std::ofstream file;
    std::string path, phase = "model-load", operation = "initializing";
    std::map<std::string, std::string> backends;
    Clock::time_point started = Clock::now(), phaseStarted = started, stepStarted = started, lastProgress = started;
    Clock::time_point lastEmission = started;
    int step = 0, total = 0, loadStep = -1, loadTotal = -1;
    bool done = false;
    std::uint64_t peakFootprint = 0;
    static double milliseconds(Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(a - b).count();
    }
    void emit(const char *kind, std::string_view detail = {}) {
        if (!file.is_open()) return;
        const auto now = Clock::now();
        auto *json = json_object_new_object();
        const auto string = [&](const char *key, std::string_view value) {
            json_object_object_add(json, key, json_object_new_string_len(value.data(), static_cast<int>(value.size())));
        };
        const auto number = [&](const char *key, double value) { json_object_object_add(json, key, json_object_new_double(value)); };
        const auto bytes = [&](const char *key, std::uint64_t value) { json_object_object_add(json, key, json_object_new_uint64(value)); };
        string("schema", "iild-native-telemetry-v1"); string("event", kind); string("trace_path", path);
        string("phase", phase); string("operation", operation);
        const auto module = phase == "text-encode" ? "te" : phase == "denoise" ? "diffusion" : phase == "vae-decode" ? "vae" : "";
        string("backend", backends.contains(module) ? backends.at(module) : "unobserved");
        string("backend_scope", "module-runtime-placement");
        if (!detail.empty()) string("detail", detail.substr(0, 1500));
        number("elapsed_ms", milliseconds(now, started)); number("phase_elapsed_ms", milliseconds(now, phaseStarted));
        number("step_elapsed_ms", milliseconds(now, stepStarted)); number("no_progress_ms", milliseconds(now, lastProgress));
        number("step", step); number("total", total);
        number("load_step", loadStep); number("load_total", loadTotal);
        number("unix_ms", std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count());
#if defined(__unix__) || defined(__APPLE__)
        number("pid", getpid());
        rusage usage{};
        if (!getrusage(RUSAGE_SELF, &usage)) {
            bytes("process_peak_rss_bytes", static_cast<std::uint64_t>(usage.ru_maxrss)
#if !defined(__APPLE__)
                * 1024
#endif
            );
            number("process_cpu_ms", (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000.0
                + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000.0);
            number("process_major_faults", usage.ru_majflt);
        }
#endif
#if defined(__APPLE__)
        task_vm_info_data_t info{};
        mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
        if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
            bytes("process_rss_bytes", info.resident_size);
            bytes("process_footprint_bytes", info.phys_footprint);
            peakFootprint = std::max(peakFootprint, static_cast<std::uint64_t>(info.phys_footprint));
            bytes("sampled_peak_footprint_bytes", peakFootprint);
        }
#if !TARGET_OS_IPHONE
        xsw_usage swap{};
        size_t size = sizeof(swap);
        if (!sysctlbyname("vm.swapusage", &swap, &size, nullptr, 0)) bytes("system_swap_used_bytes", swap.xsu_used);
#endif
#endif
        const std::string line = json_object_to_json_string_ext(json, JSON_C_TO_STRING_PLAIN);
        file << line << '\n'; file.flush();
        // One stdio operation avoids fragmenting the worker's merged protocol.
        const auto message = "IILD_NATIVE_TELEMETRY " + line + "\n";
        std::fwrite(message.data(), 1, message.size(), stderr);
        json_object_put(json);
        lastEmission = now;
    }
public:
    explicit NativeTelemetry(const char *directory = std::getenv("IILD_NATIVE_TELEMETRY_DIR")) {
        lastNativeTrace().path.clear();
        lastNativeTrace().elapsedMs = 0;
        if (!directory || !*directory) return;
        try {
            std::filesystem::create_directories(directory);
            const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
            path = (std::filesystem::path(directory) / ("inference-" + std::to_string(stamp) + ".jsonl")).string();
            file.open(path, std::ios::out | std::ios::app);
            if (!file) return;
            lastNativeTrace().path = path;
            emit("started");
            sampler = std::thread([this] {
                std::unique_lock lock(mutex);
                while (!wake.wait_for(lock, std::chrono::seconds(5), [this] { return done; })) emit("heartbeat");
            });
        } catch (...) { file.close(); }
    }
    ~NativeTelemetry() {
        { std::lock_guard lock(mutex); done = true; }
        wake.notify_one();
        if (sampler.joinable()) sampler.join();
    }
    void stage(std::string_view next, int completed = 0, int expected = 0) {
        std::lock_guard lock(mutex);
        if (!file.is_open()) return;
        const auto now = Clock::now();
        if (phase != next) {
            emit("phase-end"); phase = next; phaseStarted = stepStarted = now;
            step = 0; total = expected; lastProgress = now;
        }
        if (step != completed || total != expected) {
            emit("step-end"); stepStarted = now; lastProgress = now;
        }
        step = completed; total = expected; operation = "compute";
        emit("progress");
    }
    void loading(int completed, int expected) {
        std::lock_guard lock(mutex);
        if (!file.is_open()) return;
        operation = "weight-load";
        if (completed != loadStep || expected != loadTotal) lastProgress = Clock::now();
        loadStep = completed; loadTotal = expected;
        if (milliseconds(Clock::now(), lastEmission) >= 1000) emit("weight-progress");
    }
    void log(std::string_view message) {
        std::lock_guard lock(mutex);
        if (!file.is_open()) return;
        if (message.find("IILD_WEIGHT_LOAD begin") != std::string_view::npos) {
            operation = "weight-load";
            emit("weight-load-start");
            return;
        }
        if (message.find("IILD_WEIGHT_LOAD end") != std::string_view::npos) {
            emit("weight-load-end", message);
            operation = "compute";
            return;
        }
        if (message.find("IILD_WEIGHT_TRANSFER begin") != std::string_view::npos) {
            operation = "weight-transfer";
            emit("weight-transfer-start", message);
            return;
        }
        if (message.find("IILD_WEIGHT_TRANSFER end") != std::string_view::npos) {
            emit("weight-transfer-end", message);
            operation = "compute";
            return;
        }
        if (message.find("reused runtime anonymous model source:") != std::string_view::npos) {
            emit("resident-source-reused", message);
            return;
        }
        if (message.find("copied model file into anonymous memory:") != std::string_view::npos) {
            emit("resident-file-loaded", message);
            return;
        }
        const auto marker = message.find("IILD_BACKEND ");
        if (marker != std::string_view::npos) {
            const auto assignment = message.substr(marker + 13);
            const auto equals = assignment.find('=');
            if (equals != std::string_view::npos) {
                auto backend = assignment.substr(equals + 1);
                backend = backend.substr(0, backend.find_first_of("\r\n "));
                backends[std::string(assignment.substr(0, equals))] = backend;
            }
            emit("backend", message);
        } else if (message.find("auto-fit:") != std::string_view::npos
            || message.find("fallback") != std::string_view::npos
            || message.find("failed") != std::string_view::npos
            || message.find("loading tensors completed") != std::string_view::npos) {
            emit("engine", message);
        }
    }
    void computing() {
        std::lock_guard lock(mutex);
        lastProgress = Clock::now();
        operation = "compute";
    }
    void note(std::string_view detail) {
        std::lock_guard lock(mutex);
        emit("configuration", detail);
    }
    void finish(bool success, std::string_view error = {}) {
        std::lock_guard lock(mutex);
        lastNativeTrace().elapsedMs = milliseconds(Clock::now(), started);
        emit(success ? "native-completed" : "failed", error);
        done = true;
        wake.notify_one();
    }
};
}
