#include "ModelPackaging/ModelPackaging.hpp"
#include <chrono>
#include <csignal>
#include <iostream>
#include <stop_token>
#include <string>
#include <vector>

namespace {
volatile std::sig_atomic_t cancelled = 0;
void cancelSignal(int) { cancelled = 1; }
std::string quote(const std::string &value) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (const auto ch : value) {
        const auto c = static_cast<unsigned char>(ch);
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else result += static_cast<char>(c);
    }
    return result + "\"";
}
}

int main(int argc, char **argv) {
    std::signal(SIGINT, cancelSignal);
    std::signal(SIGTERM, cancelSignal);
    try {
        if (argc < 2) throw std::runtime_error("Use scan|create|verify|extract --input PATH [--output PATH] [--exclude RELATIVE_FILE]");
        const std::string command = argv[1];
        std::filesystem::path input,output;
        std::vector<std::string> excluded;
        for (int i = 2; i < argc; ++i) {
            const std::string key = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("Missing argument for " + key);
            const std::string value = argv[++i];
            if (key == "--input") input = value;
            else if (key == "--output") output = value;
            else if (key == "--exclude") excluded.push_back(value);
            else throw std::runtime_error("Unknown argument: " + key);
        }
        if (input.empty()) throw std::runtime_error("Choose an input folder or model package");
        std::stop_source stop;
        auto last = std::chrono::steady_clock::time_point::min();
        std::string lastPhase;
        iild::ModelPackagingObserver observer = [&](const iild::ModelPackagingProgress &progress) {
            if (cancelled) stop.request_stop();
            const auto now = std::chrono::steady_clock::now();
            if (lastPhase == progress.phase && now - last < std::chrono::milliseconds(150)) return;
            last = now; lastPhase = progress.phase;
            std::cout << "{\"schema\":\"iild-model-package-event-v1\",\"phase\":" << quote(progress.phase)
                << ",\"detail\":" << quote(progress.detail) << ",\"completed_bytes\":" << progress.completedBytes
                << ",\"total_bytes\":" << progress.totalBytes << "}\n" << std::flush;
        };
        std::string report;
        if (command == "scan") report = iild::scanModelFolder(input,excluded,stop.get_token(),observer);
        else if (command == "create") {
            if (output.empty() || output.extension() != ".safetensors") throw std::runtime_error("Choose an output .safetensors file");
            report = iild::createModelPackage(input,output,excluded,stop.get_token(),observer);
        } else if (command == "verify") report = iild::verifyModelPackage(input,stop.get_token(),observer);
        else if (command == "extract") {
            if (output.empty()) throw std::runtime_error("Choose a new extraction folder");
            report = iild::extractModelPackage(input,output,stop.get_token(),observer);
        } else throw std::runtime_error("Unknown operation: " + command);
        std::cout << report << '\n' << std::flush;
        return 0;
    } catch (const std::exception &error) {
        std::cout << "{\"schema\":\"iild-model-package-report-v1\",\"operation\":\"error\",\"verified\":false,\"error\":"
                  << quote(error.what()) << ",\"cancelled\":" << (cancelled ? "true" : "false") << "}\n" << std::flush;
        return cancelled ? 130 : 2;
    }
}
