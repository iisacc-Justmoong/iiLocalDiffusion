#pragma once

// UI-independent editable image-generation contract. No engine, filesystem or
// GUI state is owned here; execution capabilities are checked separately.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>
#include <utility>

namespace iiLocalDiffusion {
using ImageParameterValue = std::variant<bool, std::int64_t, double, std::string>;
struct ImageParameterSpec {
    std::string key, section;
    ImageParameterValue initial;
    double minimum = 0, maximum = 0;
    std::vector<std::string> choices;
};
inline const std::vector<ImageParameterSpec> &imageParameterSpecs()
{
    static const std::vector<ImageParameterSpec> specs{
        {"prompt", "essentials", std::string{}, 0, 32000},
        {"negativePrompt", "essentials", std::string{}, 0, 32000},
        {"model", "essentials", std::string{}, 0, 4096},
        {"width", "essentials", std::int64_t(1024), 64, 4096},
        {"height", "essentials", std::int64_t(1024), 64, 4096},
        {"outputCount", "essentials", std::int64_t(4), 1, 1000},
        {"seed", "composition", std::int64_t(-1), -1, 4294967295.0},
        {"seamlessTiling", "composition", false},
        {"transparentBackground", "composition", false},
        {"imageStrength", "references", 0.65, 0, 1},
        {"sampler", "sampling", std::string("auto"), 0, 0,
            {"auto", "euler", "heun", "euler_a", "dpmpp_2m", "dpmpp_sde", "ddim"}},
        {"scheduler", "sampling", std::string("auto"), 0, 0,
            {"auto", "normal", "karras", "exponential", "sgm_uniform"}},
        {"steps", "sampling", std::int64_t(30), 1, 1000},
        {"cfgScale", "sampling", 7.0, 0, 100},
        {"clipSkip", "sampling", std::int64_t(0), 0, 12}, // 0 retains the model's native default.
        {"eta", "sampling", 0.0, 0, 1},
        {"textualEmbeddings", "fineTuning", std::string{}, 0, 4096},
        {"vae", "fineTuning", std::string{}, 0, 4096},
        {"promptWeighting", "fineTuning", true},
        {"freeU", "fineTuning", false},
        {"refiner", "enhancement", false},
        {"refinerModel", "enhancement", std::string{}, 0, 4096},
        {"refinerSwitch", "enhancement", 0.8, 0, 1},
        {"denoiseStrength", "enhancement", 0.25, 0, 1},
        {"hiresFix", "enhancement", false},
        {"upscaler", "enhancement", std::string("lanczos"), 0, 0,
            {"nearest", "bilinear", "bicubic", "lanczos", "4x-ultra"}},
        {"upscalerModel", "enhancement", std::string{}, 0, 4096},
        {"faceRestore", "enhancement", false},
        {"detailer", "enhancement", false},
        {"detailerModel", "enhancement", std::string{}, 0, 4096},
        {"colorProfile", "output", std::string("sRGB"), 0, 0, {"sRGB", "Display P3"}},
        {"preserveMetadata", "output", true},
        {"watermark", "output", false},
        {"safetyFilter", "output", std::string("off"), 0, 0, {"off", "standard", "strict"}}
    };
    return specs;
}
struct ImageControlParameters {
    std::string id, process = "None", imageSource, model, maskSource;
    double weight = 1;
    bool ipAdapter = false, regionalMask = false, applied = false;
    std::string ipAdapterModel, ipAdapterVision;
    std::string poseDetector, poseModel;
    bool operator==(const ImageControlParameters &) const = default;
};
struct ImageLoraParameters {
    std::string id, source, name;
    double weight = 1;
    bool operator==(const ImageLoraParameters &) const = default;
};
struct ImageParameters {
    std::map<std::string, ImageParameterValue> values;
    std::vector<std::string> references;
    std::vector<ImageControlParameters> controls;
    std::vector<ImageLoraParameters> loras;
    static ImageParameters defaults()
    {
        ImageParameters p;
        for (const auto &spec : imageParameterSpecs()) p.values.emplace(spec.key, spec.initial);
        return p;
    }
    bool operator==(const ImageParameters &) const = default;
};
struct ImageParameterIssue {
    std::string field, message;
};
inline std::vector<ImageParameterIssue> validateImageParameters(const ImageParameters &p, bool submission = false)
{
    std::vector<ImageParameterIssue> issues;
    auto reject = [&](std::string key, std::string message) { issues.push_back({std::move(key), std::move(message)}); };
    const auto &specs = imageParameterSpecs();
    for (const auto &[key, value] : p.values) {
        if (std::none_of(specs.begin(), specs.end(), [&](const auto &s) { return s.key == key; }))
            reject(key, "Unknown parameter.");
    }
    for (const auto &s : specs) {
        const auto it = p.values.find(s.key);
        if (it == p.values.end() || it->second.index() != s.initial.index()) {
            reject(s.key, "Missing parameter or incorrect value type."); continue;
        }
        const auto &v = it->second;
        if (const auto *number = std::get_if<double>(&v)) {
            if (!std::isfinite(*number) || *number < s.minimum || *number > s.maximum)
                reject(s.key, "Value must be finite and within the parameter range.");
        } else if (const auto *integer = std::get_if<std::int64_t>(&v)) {
            if (*integer < s.minimum || *integer > s.maximum) reject(s.key, "Integer is outside the parameter range.");
            if ((s.key == "width" || s.key == "height") && *integer % 8) reject(s.key, "Dimensions must be multiples of 8.");
        } else if (const auto *text = std::get_if<std::string>(&v)) {
            if (text->find('\0') != std::string::npos) reject(s.key, "Embedded NUL is not allowed.");
            if (!s.choices.empty()) {
                if (std::find(s.choices.begin(), s.choices.end(), *text) == s.choices.end()) reject(s.key, "Unknown selection.");
            } else if (text->size() > s.maximum) reject(s.key, "Text is too long.");
            if (submission && s.key == "prompt" && text->find_first_not_of(" \t\r\n") == std::string::npos)
                reject(s.key, "Enter a prompt before submitting.");
        }
    }
    if (!issues.empty()) return issues; // Safe typed access only after schema validation.
    const auto seed = std::get<std::int64_t>(p.values.at("seed"));
    const auto count = std::get<std::int64_t>(p.values.at("outputCount"));
    if (seed >= 0 && seed > 4294967295LL - (count - 1)) reject("seed", "Seed sequence exceeds the unsigned 32-bit range.");
    auto validText = [](const std::string &v) { return v.size() <= 4096 && v.find('\0') == std::string::npos; };
    if (p.references.size() > 20) reject("referenceImages", "At most 20 reference images are allowed.");
    for (const auto &v : p.references) if (v.empty() || !validText(v)) reject("referenceImages", "Invalid image source.");
    if (p.controls.size() > 64) reject("controlNets", "At most 64 controls are allowed.");
    if (p.loras.size() > 64) reject("loras", "At most 64 LoRAs are allowed.");
    std::set<std::string> ids;
    const std::set<std::string> processes{"None", "Pose", "Canny", "Depth", "Line Art", "Scribble", "MLSD",
        "Normal Map", "Semantic Segment", "Shuffle", "Tile", "Reference", "IP-Adapter"};
    for (const auto &c : p.controls) {
        const auto key = "controlNets." + c.id;
        if (c.id.empty() || !ids.insert(c.id).second || !validText(c.id)) reject(key, "Control IDs must be nonempty and unique.");
        if (!processes.contains(c.process)) reject(key + ".process", "Unknown control process.");
        if (!std::isfinite(c.weight) || c.weight < 0 || c.weight > 2) reject(key + ".weight", "Control weight must be in [0, 2].");
        if (!validText(c.imageSource) || !validText(c.model) || !validText(c.maskSource)
            || !validText(c.ipAdapterModel) || !validText(c.ipAdapterVision)
            || !validText(c.poseDetector) || !validText(c.poseModel)) reject(key, "Invalid control source.");
        const bool controlNet = c.process != "None" && c.process != "IP-Adapter";
        if (c.applied && (c.imageSource.empty() || (controlNet ? c.model.empty() : !c.ipAdapter)))
            reject(key, "Applying requires an image and a configured ControlNet process or IP-Adapter.");
        if (c.applied && c.regionalMask && c.maskSource.empty()) reject(key + ".maskSource", "A regional mask requires a mask image.");
    }
    ids.clear();
    for (const auto &l : p.loras) {
        const auto key = "loras." + l.id;
        if (l.id.empty() || !ids.insert(l.id).second || !validText(l.id)) reject(key, "LoRA IDs must be nonempty and unique.");
        if (l.source.empty() || !validText(l.source) || !validText(l.name)) reject(key, "A LoRA requires a valid source.");
        if (!std::isfinite(l.weight) || l.weight < -4 || l.weight > 4) reject(key + ".weight", "LoRA weight must be in [-4, 4].");
    }
    return issues;
}

// Conservative capability contract for the current native image routes. An
// editable field is not a claim that the selected inference engine supports it.
inline std::vector<ImageParameterIssue> nativeParameterIssues(const ImageParameters &p, bool desktopWorker, bool poseAvailable = false)
{
    auto issues = validateImageParameters(p);
    if (!issues.empty()) return issues;
    auto reject = [&](const std::string &key) { issues.push_back({key, "Not supported by the current native image route."}); };
    auto flag = [&](const char *key) { return std::get<bool>(p.values.at(key)); };
    auto text = [&](const char *key) { return std::get<std::string>(p.values.at(key)); };
    // Color profile, metadata and watermark are final-publication operations.
    for (const auto *key : {"transparentBackground", "faceRestore"})
        if (flag(key)) reject(key);
    if (flag("refiner")) {
        if (desktopWorker) reject("refiner");
        if (text("refinerModel").empty()) issues.push_back({"refinerModel", "Choose a local SDXL Refiner checkpoint."});
    }
    if (flag("detailer")) {
        if (desktopWorker) reject("detailer");
        if (text("detailerModel").empty()) issues.push_back({"detailerModel", "Choose a local converted YOLOv8 detector."});
        if (std::get<double>(p.values.at("denoiseStrength")) == 0) reject("denoiseStrength");
    }
    if (desktopWorker && flag("freeU")) reject("freeU");
    if (desktopWorker && !flag("promptWeighting")) reject("promptWeighting");
    if (text("safetyFilter") != "off") reject("safetyFilter");
    if (desktopWorker && !text("textualEmbeddings").empty()) reject("textualEmbeddings");
    if (desktopWorker) {
        if (flag("seamlessTiling")) reject("seamlessTiling");
        if (text("sampler") != "auto" && text("sampler") != "euler" && text("sampler") != "heun") reject("sampler");
        if (text("scheduler") != "auto") reject("scheduler");
        if (std::get<std::int64_t>(p.values.at("clipSkip")) != 0) reject("clipSkip");
        if (std::get<double>(p.values.at("eta")) != 0) reject("eta");
        if (flag("hiresFix")) reject("hiresFix");
    } else if (flag("hiresFix")) {
        if (text("upscaler") == "4x-ultra" && text("upscalerModel").empty())
            issues.push_back({"upscalerModel", "Choose a local 4x ESRGAN model for learned upscaling."});
        if (std::get<double>(p.values.at("denoiseStrength")) == 0) reject("denoiseStrength");
    }
    if (desktopWorker && !p.references.empty()) reject("referenceImages");
    for (const auto &c : p.controls) if (c.applied) {
        const auto key = "controlNets." + c.id;
        if (desktopWorker || (c.process != "Canny" && c.process != "Tile" && c.process != "None" && c.process != "IP-Adapter"
            && !(c.process == "Pose" && poseAvailable)))
            reject(key + ".process");
        if (c.process == "Pose") {
            if (c.poseDetector.empty()) issues.push_back({key + ".poseDetector", "Choose a local YOLOX-L ONNX detector."});
            if (c.poseModel.empty()) issues.push_back({key + ".poseModel", "Choose a local DWPose SimCC ONNX model."});
        }
        if (c.ipAdapter) {
            if (desktopWorker) reject(key + ".ipAdapter");
            if (c.ipAdapterModel.empty()) issues.push_back({key + ".ipAdapterModel", "Choose a local IP-Adapter checkpoint."});
            if (c.ipAdapterVision.empty()) issues.push_back({key + ".ipAdapterVision", "Choose a local CLIP vision checkpoint."});
        }
        if (desktopWorker && c.regionalMask) reject(key + ".regionalMask");
    }
    if (desktopWorker && p.loras.size() > 1) reject("loras");
    // Both native ABI and checkpoint worker currently cap the actual RGB canvas.
    for (const auto *key : {"width", "height"})
        if (std::get<std::int64_t>(p.values.at(key)) > 2048) reject(key);
    return issues;
}
} // namespace iiLocalDiffusion
