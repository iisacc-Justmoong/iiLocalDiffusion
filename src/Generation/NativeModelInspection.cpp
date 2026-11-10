#include "NativeModelInspection.hpp"
#include "ImageModelRouting.hpp"
#include <json-c/json.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace iiLocalDiffusion::native_detail {
namespace model_inspection {
using Json = std::unique_ptr<json_object, decltype(&json_object_put)>;
constexpr auto invalidHeaderMessage = "The safetensors header is invalid, incomplete, or unreadable. Check the file format, access and sync status.";

inline Json parse(std::string_view bytes) {
    std::unique_ptr<json_tokener, decltype(&json_tokener_free)> parser(json_tokener_new_ex(64), json_tokener_free);
    if (!parser) throw std::runtime_error("Unable to inspect the selected model header.");
    json_tokener_set_flags(parser.get(), JSON_TOKENER_STRICT);
    Json result(json_tokener_parse_ex(parser.get(), bytes.data(), static_cast<int>(bytes.size())), json_object_put);
    const auto end = json_tokener_get_parse_end(parser.get());
    if (json_tokener_get_error(parser.get()) != json_tokener_success || !result
        || !json_object_is_type(result.get(), json_type_object) || end > bytes.size()
        || !std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(end), bytes.end(),
                        [](unsigned char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }))
        throw std::runtime_error(invalidHeaderMessage);
    return result;
}

inline json_object *member(json_object *object, const char *name, json_type type) {
    json_object *value = nullptr;
    return object && json_object_is_type(object, json_type_object)
        && json_object_object_get_ex(object, name, &value)
        && json_object_is_type(value, type) ? value : nullptr;
}

inline std::size_t syntheticMappings(json_object *root) {
    auto *metadata = member(root, "__metadata__", json_type_object);
    auto *encoded = member(metadata, "iild_merge", json_type_string);
    if (!encoded) return 0;
    try {
        const auto textLength = json_object_get_string_len(encoded);
        if (textLength < 0) return 0;
        const auto bytes = std::string_view(json_object_get_string(encoded), static_cast<std::size_t>(textLength));
        const auto merge = parse(bytes);
        auto *sources = member(merge.get(), "sources", json_type_array);
        if (!sources || json_object_array_length(sources) > 1024) return 0;
        std::size_t count = 0, visited = 0;
        for (std::size_t i = 0; i < json_object_array_length(sources); ++i) {
            auto *targets = member(json_object_array_get_idx(sources, i), "targets", json_type_array);
            if (!targets || json_object_array_length(targets) > 200000 - visited) continue;
            visited += json_object_array_length(targets);
            for (std::size_t j = 0; j < json_object_array_length(targets); ++j) {
                auto *adaptation = member(json_object_array_get_idx(targets, j), "adaptation", json_type_object);
                auto *synthetic = member(adaptation, "synthetic", json_type_boolean);
                if (synthetic && json_object_get_boolean(synthetic)) ++count;
            }
        }
        return count;
    } catch (const std::exception &) { return 0; } // Provenance is optional and untrusted.
}

inline bool tensor(json_object *root, const std::string &name) {
    return member(root, name.c_str(), json_type_object) != nullptr;
}

inline bool dimensions(json_object *root, const std::string &name, std::int64_t a, std::int64_t b) {
    auto *shape = member(member(root, name.c_str(), json_type_object), "shape", json_type_array);
    if (!shape || json_object_array_length(shape) != 2) return false;
    auto *first = json_object_array_get_idx(shape, 0), *second = json_object_array_get_idx(shape, 1);
    return json_object_is_type(first, json_type_int) && json_object_is_type(second, json_type_int)
        && json_object_get_int64(first) == a && json_object_get_int64(second) == b;
}

inline bool safeSource(const std::string &value) {
    if (value.empty() || value.find('\\') != value.npos || value.find(':') != value.npos || value.find('\0') != value.npos) return false;
    const std::filesystem::path path(value);
    if (path.is_absolute() || path.has_root_name()) return false;
    for (const auto &part : path) if (part.empty() || part == "." || part == "..") return false;
    return true;
}

// Manifest roles in older packages are sometimes wrong. Match each candidate's
// original tensor names to its mapped descriptors, never across files.
inline ImageModelInspection package(json_object *root, EmbeddedImageComponents *components, bool *extractTransformer) {
    auto *meta = member(root, "__metadata__", json_type_object);
    auto *schema = member(meta, "iild_package_schema", json_type_string);
    auto *encoded = member(meta, "iild_package_manifest", json_type_string);
    if (!schema || std::string_view(json_object_get_string(schema)) != "iild-safetensors-package-v1" || !encoded) return {};
    const auto manifest = parse(std::string_view(json_object_get_string(encoded), static_cast<std::size_t>(json_object_get_string_len(encoded))));
    auto *manifestSchema = member(manifest.get(), "schema", json_type_string);
    auto *files = member(manifest.get(), "files", json_type_array);
    if (!manifestSchema || std::string_view(json_object_get_string(manifestSchema)) != "iild-safetensors-package-v1"
        || !files || json_object_array_length(files) > 10000) throw std::runtime_error("Invalid embedded model package manifest.");
    if (components) components->packaged = true;
    std::set<std::string> paths;
    std::string transformer, encoder, vae, animaTransformer;
    std::vector<std::string> animaEncoders, animaVaes;
    bool animaNeedsExtraction = false;
    ImageWeightEncoding encoding = ImageWeightEncoding::Unknown;
    std::size_t visited = 0;
    for (std::size_t i = 0; i < json_object_array_length(files); ++i) {
        auto *file = json_object_array_get_idx(files, i);
        auto *pathValue = member(file, "path", json_type_string);
        auto *status = member(file, "status", json_type_string);
        auto *format = member(file, "format", json_type_string);
        if (!pathValue || !status || !format) throw std::runtime_error("Invalid embedded model package source.");
        const std::string path(json_object_get_string(pathValue), static_cast<std::size_t>(json_object_get_string_len(pathValue)));
        if (!safeSource(path) || !paths.insert(path).second) throw std::runtime_error("Unsafe or duplicate embedded model package source.");
        if (std::string_view(json_object_get_string(status)) != "included" || std::string_view(json_object_get_string(format)) != "safetensors") continue;
        auto *records = member(file, "tensors", json_type_array);
        if (!records || json_object_array_length(records) > 200000 - visited) throw std::runtime_error("Invalid embedded model tensor mapping.");
        visited += json_object_array_length(records);
        std::map<std::string, std::string> mapping;
        for (std::size_t j = 0; j < json_object_array_length(records); ++j) {
            auto *record = json_object_array_get_idx(records, j);
            auto *key = member(record, "key", json_type_string), *out = member(record, "output_key", json_type_string);
            if (!key || !out) throw std::runtime_error("Invalid embedded model tensor mapping.");
            const std::string original(json_object_get_string(key), static_cast<std::size_t>(json_object_get_string_len(key)));
            const std::string output(json_object_get_string(out), static_cast<std::size_t>(json_object_get_string_len(out)));
            if (!tensor(root, output) || !mapping.emplace(original, output).second) throw std::runtime_error("Invalid embedded model tensor mapping.");
        }
        const auto get = [&](const std::string &key) -> std::string { auto found = mapping.find(key); return found == mapping.end() ? std::string{} : found->second; };
        bool moss = false;
        for (const std::string prefix : {"", "model.diffusion_model.", "diffusion_model.", "transformer."}) {
            const auto video = get(prefix + "video_embedder.weight"), final = get(prefix + "final_linear_video.weight"), qkv = get(prefix + "layers.0.proj.qkv.weight");
            auto *shape = member(member(root, video.c_str(), json_type_object), "shape", json_type_array);
            if (!shape || json_object_array_length(shape) != 2 || final.empty() || qkv.empty()) continue;
            const auto width = json_object_get_int64(json_object_array_get_idx(shape, 0)), latent = json_object_get_int64(json_object_array_get_idx(shape, 1));
            if (width <= 0 || width > INT64_MAX / 3 || latent <= 0 || !dimensions(root, final, latent, width) || !dimensions(root, qkv, 3 * width, width)) continue;
            auto *dtype = member(member(root, qkv.c_str(), json_type_object), "dtype", json_type_string);
            const auto type = dtype ? std::string_view(json_object_get_string(dtype)) : std::string_view{};
            encoding = type == "I8" && !get(prefix + "layers.0.proj.qkv.comfy_quant").empty() ? ImageWeightEncoding::ComfyInt8
                : type == "F16" || type == "BF16" || type == "F32" ? ImageWeightEncoding::Floating : ImageWeightEncoding::Other;
            moss = true; break;
        }
        const bool text = !get("model.language_model.embed_tokens.weight").empty() && std::any_of(mapping.begin(), mapping.end(), [](const auto &entry) { return entry.first.starts_with("model.language_model.layers."); });
        const bool decoder = !get("encoder.conv_in.weight").empty() && !get("decoder.conv_out.weight").empty();
        const auto assign = [&](std::string &target, bool matches) { if (!matches) return; if (!target.empty()) throw std::runtime_error("Multiple matching components in the model package; select or calibrate the components explicitly."); target = path; };
        assign(transformer, moss); assign(encoder, text); assign(vae, decoder);
        bool anima = false;
        for (const std::string prefix : {"model.diffusion_model.", "diffusion_model.", ""})
            if (!get(prefix + "llm_adapter.blocks.0.cross_attn.q_proj.weight").empty()
                && !get(prefix + "blocks.0.self_attn.q_proj.weight").empty()) { anima = true; break; }
        if (anima) {
            assign(animaTransformer, true);
            animaNeedsExtraction = std::any_of(mapping.begin(), mapping.end(),
                [](const auto &entry) { return entry.first != entry.second; });
        }
        if (!get("model.embed_tokens.weight").empty() && !get("model.norm.weight").empty()
            && std::any_of(mapping.begin(), mapping.end(), [](const auto &entry) { return entry.first.starts_with("model.layers."); }))
            animaEncoders.push_back(path);
        if (!get("encoder.conv1.weight").empty() && !get("encoder.head.2.weight").empty()
            && !get("decoder.conv1.weight").empty() && !get("decoder.head.2.weight").empty())
            animaVaes.push_back(path);
    }
    if (!animaTransformer.empty()) {
        if (!transformer.empty() || animaEncoders.size() > 1 || animaVaes.size() > 1)
            throw std::runtime_error("The Anima package has multiple matching components; create or calibrate a package with an unambiguous component for each role.");
        if (components) {
            components->transformer = animaTransformer;
            if (!animaEncoders.empty()) components->textEncoder = animaEncoders.front();
            if (!animaVaes.empty()) components->vae = animaVaes.front();
        }
        if (extractTransformer) *extractTransformer = animaNeedsExtraction;
        ImageModelInspection result;
        result.architecture = ImageModelArchitecture::Anima; result.role = ImageModelRole::Checkpoint;
        result.hasTextEncoder = !animaEncoders.empty(); result.hasVae = !animaVaes.empty();
        return result;
    }
    if (transformer.empty()) return {};
    if (components) { components->transformer = transformer; components->textEncoder = encoder; components->vae = vae; }
    ImageModelInspection result;
    result.architecture = ImageModelArchitecture::MossImage; result.role = ImageModelRole::Checkpoint;
    result.encoding = encoding; result.hasTextEncoder = !encoder.empty(); result.hasVae = !vae.empty();
    result.syntheticAdapterMappings = syntheticMappings(root);
    return result;
}
}

// Checks bounded descriptors only; never loads tensor payloads or decides that
// a recognized model can actually produce an image. Unknown families continue
// to the existing engine so newly supported architectures are not blocked.
ImageModelInspection inspectImageModelHeader(const std::filesystem::path &path, EmbeddedImageComponents *components = nullptr,
                                             bool *extractTransformer = nullptr) {
    auto suffix = path.extension().string();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (suffix != ".safetensors" && suffix != ".safetensor") return {};
    const auto invalid = [] { return std::runtime_error(model_inspection::invalidHeaderMessage); };
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size < 10) throw invalid();
    const auto modified = std::filesystem::last_write_time(path, error);
    if (error) throw invalid();
    std::ifstream file(path, std::ios::binary);
    std::array<unsigned char, 8> headerPrefix{};
    if (!file.read(reinterpret_cast<char *>(headerPrefix.data()), headerPrefix.size())) throw invalid();
    std::uint64_t length = 0;
    for (unsigned i = 0; i < headerPrefix.size(); ++i) length |= std::uint64_t(headerPrefix[i]) << (i * 8);
    if (length < 2 || length > size - 8) throw invalid();
    // A header beyond this bounded inspector's budget is not necessarily
    // invalid. Leave it to the engine instead of rejecting a valid model.
    if (length > 64ull * 1024 * 1024) return {};
    std::string bytes(static_cast<std::size_t>(length), '\0');
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) throw invalid();
    const auto sizeAfter = std::filesystem::file_size(path, error);
    if (error) throw invalid();
    const auto modifiedAfter = std::filesystem::last_write_time(path, error);
    if (error || sizeAfter != size || modifiedAfter != modified)
        throw std::runtime_error("The selected model changed during header inspection. Wait for synchronization to finish and try again.");
    const auto root = model_inspection::parse(bytes);
    const auto payload = size - 8 - length;
    std::size_t tensors = 0, adapterTensors = 0, unrelatedAdapterFields = 0, unrelatedControlFields = 0;
    bool denoiser = false, vaeEncoder = false, vaeDecoder = false, textEncoder = false, control = false;
    json_object_object_foreach(root.get(), rawName, descriptor) {
        const std::string_view name(rawName);
        if (name == "__metadata__") continue;
        if (++tensors > 200000 || !json_object_is_type(descriptor, json_type_object)) throw invalid();
        auto *dtype = model_inspection::member(descriptor, "dtype", json_type_string);
        auto *shape = model_inspection::member(descriptor, "shape", json_type_array);
        auto *offsets = model_inspection::member(descriptor, "data_offsets", json_type_array);
        if (!dtype || !shape || !offsets || json_object_array_length(offsets) != 2) throw invalid();
        auto *begin = json_object_array_get_idx(offsets, 0), *end = json_object_array_get_idx(offsets, 1);
        if (!json_object_is_type(begin, json_type_int) || !json_object_is_type(end, json_type_int)
            || json_object_get_int64(begin) < 0 || json_object_get_int64(end) < json_object_get_int64(begin)
            || static_cast<std::uint64_t>(json_object_get_int64(end)) > payload) throw invalid();
        for (std::size_t i = 0; i < json_object_array_length(shape); ++i) {
            auto *dimension = json_object_array_get_idx(shape, i);
            if (!json_object_is_type(dimension, json_type_int) || json_object_get_int64(dimension) < 0) throw invalid();
        }
        const bool adapter = name.find("lora_down.") != name.npos || name.find("lora_up.") != name.npos
            || name.find("lora_A.") != name.npos || name.find("lora_B.") != name.npos;
        if (adapter) ++adapterTensors;
        else if (!name.ends_with(".alpha") && name.find("dora_scale") == name.npos) ++unrelatedAdapterFields;
        denoiser |= name.starts_with("model.diffusion_model.") || name.starts_with("diffusion_model.")
            || name.starts_with("unet.") || name.starts_with("transformer.")
            || name.starts_with("input_blocks.") || name.starts_with("double_blocks.")
            || name.starts_with("joint_blocks.") || name.starts_with("transformer_blocks.")
            || name.starts_with("layers.") || name.starts_with("blocks.") || name.starts_with("txtfusion.")
            || name.starts_with("text_fusion.") || name.starts_with("x_embedder.");
        vaeEncoder |= name == "encoder.conv_in.weight" || name == "vae.encoder.conv_in.weight";
        vaeDecoder |= name == "decoder.conv_out.weight" || name == "vae.decoder.conv_out.weight";
        textEncoder |= name.starts_with("text_encoders.") || name.starts_with("text_encoder.")
            || name.starts_with("model.language_model.");
        const bool controlField = name.starts_with("control_model.") || name.starts_with("controlnet_cond_embedding.")
            || name.starts_with("controlnet_down_blocks.") || name.starts_with("controlnet_mid_block.");
        control |= controlField;
        if (!controlField) ++unrelatedControlFields;
    }
    if (!tensors) throw invalid();
    const auto embedded = model_inspection::package(root.get(), components, extractTransformer);
    if (embedded.architecture != ImageModelArchitecture::Unknown) return embedded;
    for (const std::string prefix : {"", "model.diffusion_model.", "diffusion_model.", "transformer."}) {
        auto *video = model_inspection::member(model_inspection::member(root.get(), (prefix + "video_embedder.weight").c_str(), json_type_object), "shape", json_type_array);
        if (!video || json_object_array_length(video) != 2) continue;
        const auto width = json_object_get_int64(json_object_array_get_idx(video, 0));
        const auto latent = json_object_get_int64(json_object_array_get_idx(video, 1));
        if (width <= 0 || width > INT64_MAX / 3 || latent <= 0
            || !model_inspection::dimensions(root.get(), prefix + "final_linear_video.weight", latent, width)
            || !model_inspection::dimensions(root.get(), prefix + "layers.0.proj.qkv.weight", 3 * width, width)) continue;
        ImageModelInspection result;
        result.architecture = ImageModelArchitecture::MossImage;
        result.role = ImageModelRole::Checkpoint;
        result.hasTextEncoder = textEncoder;
        result.hasVae = vaeEncoder;
        result.syntheticAdapterMappings = model_inspection::syntheticMappings(root.get());
        auto *weight = model_inspection::member(root.get(), (prefix + "layers.0.proj.qkv.weight").c_str(), json_type_object);
        auto *dtype = model_inspection::member(weight, "dtype", json_type_string);
        const auto encoding = dtype ? std::string_view(json_object_get_string(dtype)) : std::string_view{};
        if (encoding == "I8"
            && model_inspection::tensor(root.get(), prefix + "layers.0.proj.qkv.comfy_quant"))
            result.encoding = ImageWeightEncoding::ComfyInt8;
        else if (encoding == "F16" || encoding == "BF16" || encoding == "F32")
            result.encoding = ImageWeightEncoding::Floating;
        else result.encoding = ImageWeightEncoding::Other;
        return result;
    }
    ImageModelInspection result;
    result.hasTextEncoder = textEncoder;
    result.hasVae = vaeEncoder;
    if (!denoiser && adapterTensors >= 2 && unrelatedAdapterFields == 0)
        result.role = ImageModelRole::Lora;
    else if (!denoiser && vaeEncoder && vaeDecoder)
        result.role = ImageModelRole::Vae;
    else if (!denoiser && control && unrelatedControlFields == 0)
        result.role = ImageModelRole::ControlNet;
    return result;
}

void inspectNativeImageModel(const std::filesystem::path &path) {
    const auto model = inspectImageModelHeader(path);
    if (model.architecture == ImageModelArchitecture::MossImage) {
        std::string message = "Unsupported native model: the tensor layout identifies a MOSS Image transformer, which this native image engine cannot run.";
        if (model.encoding == ImageWeightEncoding::ComfyInt8)
            message += " This file contains INT8 Comfy quantized transformer weights; a MOSS workflow must explicitly support this quantization.";
        if (!model.hasTextEncoder)
            message += " A compatible MOSS Qwen3.5 text encoder and tokenizer are needed separately.";
        if (!model.hasVae)
            message += " A compatible MOSS VAE is needed separately.";
        if (model.syntheticAdapterMappings)
            message += " Merge metadata reports " + std::to_string(model.syntheticAdapterMappings) + " synthetic adapter mappings; this does not establish output quality.";
        message += " Use a compatible MOSS workflow and verified components, or choose a checkpoint supported by this native engine. The source file was not changed.";
        throw std::runtime_error(message);
    }
    if (model.role == ImageModelRole::Lora)
        throw std::runtime_error("Unsupported native model: the selected file is a LoRA adapter, not a complete image checkpoint. Attach it to a compatible base model.");
    if (model.role == ImageModelRole::Vae)
        throw std::runtime_error("Unsupported native model: the selected file is a VAE, not an image checkpoint. Select a compatible base model and attach the VAE separately.");
    if (model.role == ImageModelRole::ControlNet)
        throw std::runtime_error("Unsupported native model: the selected file is a ControlNet, not an image checkpoint. Attach it to a compatible base model and conditioning input.");
}
bool embeddedAnimaTransformerNeedsExtraction(const std::filesystem::path &path) {
    bool extract = false;
    (void)inspectImageModelHeader(path, nullptr, &extract);
    return extract;
}
}

iiLocalDiffusion::ImageModelInspection iiLocalDiffusion::inspectImageModel(const std::filesystem::path &path) {
    return native_detail::inspectImageModelHeader(path);
}

iiLocalDiffusion::EmbeddedImageComponents iiLocalDiffusion::inspectEmbeddedImageComponents(const std::filesystem::path &path) {
    EmbeddedImageComponents components;
    (void)native_detail::inspectImageModelHeader(path, &components);
    return components;
}
