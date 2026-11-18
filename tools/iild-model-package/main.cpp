#include "ModelPackaging/ModelPackaging.hpp"
#include "ModelPackaging/ModelInspection.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <map>
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
        if (argc < 2) throw std::runtime_error("Use scan|create|verify|extract|calibrate|calibrate-file|weight-health|probe|inspect|inspect-index|inspect-pytorch-index|inspect-wan-directory|contains-tensors|classify|materialize --input PATH [--tensor NAME] [--output PATH] [--repair-nonfinite true|false] [--select RELATIVE_MODEL] [--exclude RELATIVE_FILE] [--assign-role RELATIVE=ROLE] [--assign-position RELATIVE=main|component] [--modalities image,video,audio,language|auto] [--role ROLE | --asset-path RELATIVE_SOURCE]");
        const std::string command = argv[1];
        std::filesystem::path input,output;
        std::vector<std::string> excluded;
        std::vector<std::filesystem::path> selected;
        std::vector<std::string> tensorNames;
        std::map<std::string,iild::ModelPackagingAssignment> choices;
        std::string modalitySelection;
        std::string assetRole = "text_encoder", assetPath;
        bool explicitRole = false;
        bool repairNonfinite = false;
        for (int i = 2; i < argc; ++i) {
            const std::string key = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("Missing argument for " + key);
            const std::string value = argv[++i];
            if (key == "--input") input = value;
            else if (key == "--output") output = value;
            else if (key == "--exclude") excluded.push_back(value);
            else if (key == "--select") selected.emplace_back(value);
            else if (key == "--tensor") tensorNames.push_back(value);
            else if (key == "--assign-role" || key == "--assign-position") {
                const auto separator=value.rfind('=');
                if (separator==std::string::npos || separator==0 || separator+1==value.size())
                    throw std::runtime_error("Use a relative path and value separated by = for " + key);
                const auto path=value.substr(0,separator);
                auto &assignment=choices[path]; assignment.relativePath=path;
                auto &field=key=="--assign-role" ? assignment.role : assignment.position;
                if (!field.empty()) throw std::runtime_error("Repeated assignment: " + path);
                field=value.substr(separator+1);
            }
            else if (key == "--modalities") modalitySelection = value;
            else if (key == "--role") { assetRole = value; explicitRole = true; }
            else if (key == "--asset-path") assetPath = value;
            else if (key == "--repair-nonfinite") {
                if (value != "true" && value != "false") throw std::runtime_error("Use true or false for --repair-nonfinite");
                repairNonfinite = value == "true";
            }
            else throw std::runtime_error("Unknown argument: " + key);
        }
        if (input.empty()) throw std::runtime_error("Choose an input folder or model package");
        if (!selected.empty() && command != "scan" && command != "create")
            throw std::runtime_error("--select is supported only for scan and create");
        if (!tensorNames.empty() && command != "contains-tensors")
            throw std::runtime_error("--tensor is supported only for contains-tensors");
        if (!choices.empty() && command != "scan" && command != "create")
            throw std::runtime_error("Assignments are supported only for scan and create");
        std::vector<iild::ModelPackagingAssignment> assignments;
        for (const auto &[path,choice] : choices) assignments.push_back(choice);
        if (repairNonfinite && command != "calibrate" && command != "calibrate-file") throw std::runtime_error("Numeric repair is available only during calibration");
        if (command == "probe") {
            std::cout << "{\"schema\":\"iild-model-container-probe-v1\",\"format\":"
                      << quote(iild::detectModelContainer(input)) << "}\n" << std::flush;
            return 0;
        }
        if (command == "contains-tensors") {
            std::vector<std::string_view> names;
            names.reserve(tensorNames.size());
            for (const auto &name:tensorNames) names.emplace_back(name);
            const auto probe=iild::probeModelTensorNames(input,names);
            const auto &found=probe.matches;
            const auto present=std::all_of(names.begin(),names.end(),[&](const auto name) {
                return std::binary_search(found.begin(),found.end(),name);
            });
            std::cout << "{\"schema\":\"iild-model-tensor-presence-v1\",\"present\":"
                      << (present ? "true" : "false") << ",\"embedded_package\":"
                      << (probe.embeddedPackage ? "true" : "false") << ",\"matches\":[";
            for (std::size_t i=0;i<found.size();++i) {
                if (i) std::cout << ',';
                std::cout << quote(found[i]);
            }
            std::cout << "]}\n" << std::flush;
            return 0;
        }
        if (command == "inspect" || command == "classify") {
            if (command == "classify") {
                if (modalitySelection.empty()) throw std::runtime_error("Choose --modalities image,video,audio,language or auto");
                std::vector<std::string> modalities;
                if (modalitySelection != "auto") {
                    std::size_t begin = 0;
                    for (;;) {
                        const auto end = modalitySelection.find(',', begin);
                        modalities.push_back(modalitySelection.substr(begin, end - begin));
                        if (end == std::string::npos) break;
                        begin = end + 1;
                    }
                }
                iild::setModelModalities(input, modalities);
            }
            auto report = iild::inspectModel(input);
            std::cout << iild::modelInspectionJson(report) << '\n' << std::flush;
            return 0;
        }
        if (command == "materialize") {
            if (output.empty()) throw std::runtime_error("Choose an output cache directory");
            if (explicitRole && !assetPath.empty()) throw std::runtime_error("Choose either --role or --asset-path");
            std::stop_source assetStop;
            const auto observer = [&](const auto &) { if (cancelled) assetStop.request_stop(); };
            const auto path = assetPath.empty()
                ? iild::materializeModelPackageAsset(input, output, assetRole, assetStop.get_token(), observer)
                : iild::materializeModelPackageAssetByPath(input, output, assetPath, assetStop.get_token(), observer);
            std::cout << "{\"schema\":\"iild-model-package-report-v1\",\"operation\":\"materialize\",\"verified\":true,\"output\":"
                      << quote(path.string()) << "}\n" << std::flush;
            return 0;
        }
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
        if (command == "scan") report = selected.empty()
            ? iild::scanModelFolderWithAssignments(input,assignments,excluded,stop.get_token(),observer)
            : iild::scanModelSelectionWithAssignments(input,selected,assignments,excluded,stop.get_token(),observer);
        else if (command == "create") {
            if (output.empty() || output.extension() != ".safetensors") throw std::runtime_error("Choose an output .safetensors file");
            report = selected.empty()
                ? iild::createModelPackageWithAssignments(input,output,assignments,excluded,stop.get_token(),observer)
                : iild::createModelPackageFromSelectionWithAssignments(input,selected,output,assignments,excluded,stop.get_token(),observer);
        } else if (command == "inspect-index") report = iild::safetensorsIndexJson(iild::inspectSafetensorsIndex(input));
        else if (command == "inspect-pytorch-index") report = iild::pyTorchIndexJson(iild::inspectPyTorchIndex(input));
        else if (command == "inspect-wan-directory") report = iild::wanPipelineDirectoryJson(iild::inspectWanPipelineDirectory(input));
        else if (command == "verify") report = iild::verifyModelPackage(input,stop.get_token(),observer);
        else if (command == "extract") {
            if (output.empty()) throw std::runtime_error("Choose a new extraction folder");
            report = iild::extractModelPackage(input,output,stop.get_token(),observer);
        } else if (command == "calibrate") {
            if (output.empty()) throw std::runtime_error("Choose a new corrected .safetensors output");
            report = iild::calibrateModelPackage(input,output,stop.get_token(),observer,
                {.replaceNonfiniteWithZero=repairNonfinite});
        } else if (command == "weight-health") {
            report = iild::inspectModelWeights(input,stop.get_token(),observer);
        } else if (command == "calibrate-file") {
            if (output.empty() || !repairNonfinite) throw std::runtime_error("Choose a new output and explicitly enable numeric repair");
            report = iild::calibrateModelFile(input,output,stop.get_token(),observer);
        } else throw std::runtime_error("Unknown operation: " + command);
        std::cout << report << '\n' << std::flush;
        return 0;
    } catch (const std::exception &error) {
        std::cout << "{\"schema\":\"iild-model-package-report-v1\",\"operation\":\"error\",\"verified\":false,\"error\":"
                  << quote(error.what()) << ",\"cancelled\":" << (cancelled ? "true" : "false") << "}\n" << std::flush;
        return cancelled ? 130 : 2;
    }
}
