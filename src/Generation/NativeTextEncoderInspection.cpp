#include "NativeTextEncoderInspection.hpp"

#include <stdexcept>

#if IILD_HAS_NATIVE_DIFFUSION
#include <ggml-cpu.h>
#include <model_loader.h>
#include <model/te/clip.hpp>
#include <model/te/t5.hpp>
#include <algorithm>
#include <charconv>
#include <map>
#include <memory>
#include <optional>
#include <string>
#endif

namespace iiLocalDiffusion {
namespace {
void requireFile(const std::filesystem::path &path) {
    if (path.empty() || !std::filesystem::is_regular_file(path))
        throw std::invalid_argument("Text encoder inspection requires existing model files.");
}
void requireSlot(std::string_view slot) {
    if (slot != "clip_l" && slot != "clip_g" && slot != "t5xxl")
        throw std::invalid_argument("Unknown native text encoder slot.");
}

#if IILD_HAS_NATIVE_DIFFUSION
struct Contract {
    std::string prefix;
    CLIPVersion clipVersion = OPENAI_CLIP_VIT_L_14;
    bool withFinalNorm = true;
    bool t5 = false;
};

std::optional<Contract> contract(SDVersion version, std::string_view slot) {
    if (sd_version_is_sd1(version) && slot == "clip_l")
        return Contract{"cond_stage_model.transformer.text_model", OPENAI_CLIP_VIT_L_14, true};
    if (sd_version_is_sd2(version) && slot == "clip_l")
        return Contract{"cond_stage_model.transformer.text_model", OPEN_CLIP_VIT_H_14, true};
    if (sd_version_is_sdxl(version)) {
        if (version == VERSION_SDXL_REFINER)
            return slot == "clip_g" ? std::optional(Contract{"cond_stage_model.transformer.text_model", OPEN_CLIP_VIT_BIGG_14, false}) : std::nullopt;
        if (slot == "clip_l") return Contract{"cond_stage_model.transformer.text_model", OPENAI_CLIP_VIT_L_14, false};
        if (slot == "clip_g") return Contract{"cond_stage_model.1.transformer.text_model", OPEN_CLIP_VIT_BIGG_14, false};
        return std::nullopt;
    }
    const bool flux = version == VERSION_FLUX || version == VERSION_FLUX_FILL || version == VERSION_FLUX_CONTROLS;
    if (sd_version_is_sd3(version) || flux) {
        if (slot == "clip_l") return Contract{"text_encoders.clip_l.transformer.text_model", OPENAI_CLIP_VIT_L_14, flux};
        if (slot == "clip_g" && !flux) return Contract{"text_encoders.clip_g.transformer.text_model", OPEN_CLIP_VIT_BIGG_14, false};
        if (slot == "t5xxl") return Contract{"text_encoders.t5xxl.transformer", OPENAI_CLIP_VIT_L_14, true, true};
    }
    return std::nullopt;
}

bool hasPrefix(const String2TensorStorage &storage, const std::string &prefix) {
    return std::any_of(storage.begin(), storage.end(), [&](const auto &entry) {
        return entry.first.starts_with(prefix + '.');
    });
}

// The pinned engine detects T5 dimensions from the file. Require the FLUX/SD3
// XXL architecture before constructing its metadata graph, so a malformed
// index cannot cause an unbounded number of layer objects to be allocated.
bool isT5Xxl(const String2TensorStorage &storage, const std::string &prefix) {
    const auto start = prefix + ".encoder.block.";
    for (const auto &[name, ignored] : storage) {
        (void)ignored;
        if (!name.starts_with(start)) continue;
        const auto end = name.find('.', start.size());
        if (end == std::string::npos) return false;
        int layer = -1;
        const auto *begin = name.data() + start.size();
        const auto *last = name.data() + end;
        const auto parsed = std::from_chars(begin, last, layer);
        if (parsed.ec != std::errc{} || parsed.ptr != last || layer < 0 || layer >= 24) return false;
    }
    const auto config = T5Config::detect_from_weights(storage, prefix, false);
    return config.num_layers == 24 && config.model_dim == 4096 && config.inner_dim == 4096
        && config.ff_dim == 10240 && config.num_heads == 64 && config.vocab_size == 32128;
}

bool matches(const Contract &spec, const String2TensorStorage &storage) {
    if (!hasPrefix(storage, spec.prefix) || (spec.t5 && !isT5Xxl(storage, spec.prefix))) return false;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    if (!cpu) throw std::runtime_error("The native metadata backend could not be created.");
    std::map<std::string, ggml_tensor *> required;
    std::unique_ptr<CLIPTextModelRunner> clip;
    std::unique_ptr<T5Runner> t5;
    if (spec.t5) {
        t5 = std::make_unique<T5Runner>(cpu.get(), storage, spec.prefix, false);
        t5->get_param_tensors(required, spec.prefix);
    } else {
        clip = std::make_unique<CLIPTextModelRunner>(cpu.get(), storage, spec.prefix, spec.clipVersion, spec.withFinalNorm);
        clip->get_param_tensors(required, spec.prefix);
    }
    if (required.empty()) return false;
    for (const auto &[name, tensor] : required) {
        // Match ModelManager's optional SD2 last-layer rule.
        if (name.find("cond_stage_model.transformer.text_model.encoder.layers.23") != std::string::npos) continue;
        const auto found = storage.find(name);
        if (found == storage.end() || !std::equal(tensor->ne, tensor->ne + GGML_MAX_DIMS, found->second.ne)) return false;
    }
    return true;
}

SDVersion loadModel(const std::filesystem::path &path, ModelLoader &loader) {
    if (!loader.init_from_file(path.string()))
        throw std::runtime_error("The native engine cannot inspect this model's text encoder contract.");
    loader.convert_tensors_name();
    return loader.get_sd_version();
}
#endif
}

NativeTextEncoderState inspectNativeTextEncoder(const std::filesystem::path &model, std::string_view slot) {
    requireFile(model);
    requireSlot(slot);
#if IILD_HAS_NATIVE_DIFFUSION
    ModelLoader loader;
    const auto spec = contract(loadModel(model, loader), slot);
    if (!spec) return NativeTextEncoderState::Unsupported;
    const auto &storage = loader.get_tensor_storage_map();
    if (!hasPrefix(storage, spec->prefix)) return NativeTextEncoderState::Missing;
    return matches(*spec, storage) ? NativeTextEncoderState::Embedded : NativeTextEncoderState::Incompatible;
#else
    return NativeTextEncoderState::Unsupported;
#endif
}

bool validateNativeTextEncoder(const std::filesystem::path &model, std::string_view slot,
                               const std::filesystem::path &candidate) {
    requireFile(model);
    requireFile(candidate);
    requireSlot(slot);
#if IILD_HAS_NATIVE_DIFFUSION
    ModelLoader modelLoader;
    const auto version = loadModel(model, modelLoader);
    const auto spec = contract(version, slot);
    if (!spec) throw std::runtime_error("The native engine has no text encoder contract for this model slot.");
    ModelLoader external;
    const auto prefix = slot == "t5xxl" ? "text_encoders.t5xxl.transformer." : std::string(slot) + '.';
    if (!external.init_from_file_and_convert_name(candidate.string(), prefix, version)) return false;
    return matches(*spec, external.get_tensor_storage_map());
#else
    throw std::runtime_error("This build does not include the native image engine.");
#endif
}

} // namespace iiLocalDiffusion
