#include "NativeModelInspection.hpp"
#include "ImageModelRouting.hpp"
#include "ModelPackaging/ModelInspection.hpp"
#include "ModelPackaging/ModelPackaging.hpp"
#include <json-c/json.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <map>
#include <optional>
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

// Map only known Anima denoiser roots. Keeping this independent of filenames
// also lets us reject mixed aliases that would otherwise overwrite a tensor.
inline std::optional<std::string> animaTensorKey(std::string_view name) {
    constexpr std::array prefixes{std::string_view{"model.diffusion_model.net."},
        std::string_view{"model.diffusion_model."}, std::string_view{"diffusion_model.net."},
        std::string_view{"diffusion_model."}, std::string_view{"transformer.net."},
        std::string_view{"transformer."}, std::string_view{"net."}, std::string_view{""}};
    constexpr std::array roots{std::string_view{"llm_adapter."}, std::string_view{"blocks."},
        std::string_view{"x_embedder."}, std::string_view{"t_embedder."},
        std::string_view{"t_embedding_norm."}, std::string_view{"final_layer."}};
    for (const auto prefix : prefixes) {
        if (!name.starts_with(prefix)) continue;
        const auto suffix = name.substr(prefix.size());
        for (const auto root : roots)
            if (suffix.starts_with(root)) return std::string(suffix);
    }
    return std::nullopt;
}

class AnimaTensorInspection {
public:
    void add(const std::string &name, json_object *descriptor, bool hasQuantizationMarker) {
        const auto canonical = animaTensorKey(name);
        if (!canonical) return;
        if (!normalized.insert(*canonical).second)
            throw std::runtime_error("The Anima checkpoint has conflicting tensor aliases: " + *canonical + '.');
        if (!canonical->starts_with("blocks.") || !canonical->ends_with(".weight")) return;
        auto *dtype = member(descriptor, "dtype", json_type_string);
        const auto type = dtype ? std::string_view(json_object_get_string(dtype)) : std::string_view{};
        if (type == "I8" && hasQuantizationMarker) quantized = true;
        else if (type == "F16" || type == "BF16" || type == "F32") floating = true;
    }

    ImageWeightEncoding encoding() const {
        return quantized ? ImageWeightEncoding::ComfyInt8
            : floating ? ImageWeightEncoding::Floating : ImageWeightEncoding::Other;
    }
private:
    std::set<std::string> normalized;
    bool quantized = false;
    bool floating = false;
};

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

// Manifest roles in older packages are sometimes wrong. Match original tensor
// names to mapped descriptors and combine only explicitly indexed shard groups.
inline ImageModelInspection package(json_object *root, EmbeddedImageComponents *components, bool *extractTransformer,
                                    const std::filesystem::path &modelPath, EmbeddedImageCatalog *catalog) {
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
    if (catalog) catalog->packaged = true;
    std::set<std::string> paths, indexes, duplicatePaths;
    std::map<std::string,std::map<std::string,std::string>> shardMappings;
    std::map<std::string,std::map<std::string,std::string>> singleMappings;
    std::map<std::string,bool> sourceUserMain, groupUserMain;
    bool hasLegacyMappings = false;
    std::vector<std::string> encoders, vaes, animaEncoders, animaVaes;
    std::map<std::string,ImageWeightEncoding> mossModels, animaModels;
    std::map<std::string,bool> animaNeedsExtraction;
    const auto inspectMapping = [&](const std::string &path, const std::map<std::string,std::string> &mapping) {
        const auto get = [&](const std::string &key) -> std::string { auto found = mapping.find(key); return found == mapping.end() ? std::string{} : found->second; };
        bool moss = false;
        auto encoding = ImageWeightEncoding::Unknown;
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
        if (moss) mossModels.emplace(path, encoding);
        if (text) encoders.push_back(path);
        if (decoder) vaes.push_back(path);
        bool anima = false;
        for (const std::string prefix : {"model.diffusion_model.net.", "model.diffusion_model.",
                                          "diffusion_model.net.", "diffusion_model.",
                                          "transformer.net.", "transformer.", "net.", ""})
            if (!get(prefix + "llm_adapter.blocks.0.cross_attn.q_proj.weight").empty()
                && !get(prefix + "blocks.0.self_attn.q_proj.weight").empty()) { anima = true; break; }
        if (moss && anima)
            throw std::runtime_error("Conflicting model signatures in one embedded package component: " + path);
        if (anima) {
            AnimaTensorInspection inspected;
            for (const auto &[original, output] : mapping) {
                const bool hasMarker = original.ends_with(".weight")
                    && !get(original.substr(0, original.size() - 7) + ".comfy_quant").empty();
                inspected.add(original, member(root, output.c_str(), json_type_object), hasMarker);
            }
            animaModels.emplace(path, inspected.encoding());
            // Indexed transformers must be reconstructed even when their
            // packaged names already match: materialization verifies the index
            // and all shard checksums before handing weights to the engine.
            animaNeedsExtraction.emplace(path, indexes.contains(path) || std::any_of(mapping.begin(), mapping.end(),
                [](const auto &entry) { return entry.first != entry.second; }));
        }
        if (!get("model.embed_tokens.weight").empty() && !get("model.norm.weight").empty()
            && std::any_of(mapping.begin(), mapping.end(), [](const auto &entry) { return entry.first.starts_with("model.layers."); }))
            animaEncoders.push_back(path);
        if (!get("encoder.conv1.weight").empty() && !get("encoder.head.2.weight").empty()
            && !get("decoder.conv1.weight").empty() && !get("decoder.head.2.weight").empty())
            animaVaes.push_back(path);
    };
    std::size_t visited = 0;
    for (std::size_t i = 0; i < json_object_array_length(files); ++i) {
        auto *file = json_object_array_get_idx(files, i);
        auto *pathValue = member(file, "path", json_type_string);
        auto *status = member(file, "status", json_type_string);
        auto *format = member(file, "format", json_type_string);
        if (!pathValue || !status || !format) throw std::runtime_error("Invalid embedded model package source.");
        const std::string path(json_object_get_string(pathValue), static_cast<std::size_t>(json_object_get_string_len(pathValue)));
        if (!safeSource(path) || !paths.insert(path).second) throw std::runtime_error("Unsafe or duplicate embedded model package source.");
        const auto sourceStatus = std::string_view(json_object_get_string(status));
        if (sourceStatus != "included" && sourceStatus != "duplicate") continue;
        auto *position = member(file, "position", json_type_string);
        auto *positionSource = member(file, "position_source", json_type_string);
        const bool userMain = position && positionSource
            && std::string_view(json_object_get_string(position)) == "main"
            && std::string_view(json_object_get_string(positionSource)) == "user";
        auto *indexMarker=member(file,"shard_index",json_type_boolean);
        if (sourceStatus == "included" && std::string_view(json_object_get_string(format)) == "asset"
            && (indexMarker ? json_object_get_boolean(indexMarker) : iild::hasSafetensorsIndexSuffix(path)))
            indexes.insert(path);
        if (catalog && sourceStatus == "included" && std::string_view(json_object_get_string(format)) == "gguf")
            catalog->groups.push_back(EmbeddedImageGroup{path, {}, userMain});
        if (std::string_view(json_object_get_string(format)) != "safetensors") continue;
        sourceUserMain.emplace(path, userMain);
        if (sourceStatus == "duplicate") duplicatePaths.insert(path);
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
        if (auto *groupValue=member(file,"group",json_type_string)) {
            const std::string group(json_object_get_string(groupValue),static_cast<std::size_t>(json_object_get_string_len(groupValue)));
            if (group!=path) {
                if (!safeSource(group))
                    throw std::runtime_error("Invalid embedded model shard group.");
                auto &combined=shardMappings[group];
                auto [selected, inserted] = groupUserMain.try_emplace(group, true);
                (void)inserted;
                selected->second = selected->second && userMain;
                for (const auto &[name,output] : mapping)
                    if (!combined.emplace(name,output).second) throw std::runtime_error("Duplicate tensor in embedded model shard group.");
                continue;
            }
        } else hasLegacyMappings = true;
        singleMappings.emplace(path,std::move(mapping));
    }
    if (hasLegacyMappings && !indexes.empty()) {
        std::map<std::string,std::string> claimed;
        for (const auto &index : iild::detail::inspectPackageShardIndexes(modelPath)) {
            if (shardMappings.contains(index.path)) continue;
            std::map<std::string,std::string> expected; std::set<std::string> members;
            bool available=true;
            for (const auto &[name,path] : index.weights) {
                if (!expected.emplace(name,path).second) throw std::runtime_error("Duplicate tensor in embedded model shard index.");
                const auto file=singleMappings.find(path);
                if (file==singleMappings.end() || !file->second.contains(name)) { available=false; break; }
                members.insert(path);
            }
            if (!available || members.empty()) continue;
            std::map<std::string,std::string> combined;
            for (const auto &path : members) {
                if (!claimed.emplace(path,index.path).second) throw std::runtime_error("Multiple indexes claim the same embedded model shard.");
                for (const auto &[name,output] : singleMappings.at(path)) {
                    const auto indexed=expected.find(name);
                    if (indexed==expected.end() || indexed->second!=path || !combined.emplace(name,output).second)
                        throw std::runtime_error("Embedded model shard index does not cover its tensors.");
                }
            }
            if (combined.size()!=expected.size()) throw std::runtime_error("Embedded model shard index does not cover its tensors.");
            groupUserMain[index.path] = std::all_of(members.begin(), members.end(),
                [&](const auto &path) { return sourceUserMain.at(path); });
            shardMappings.emplace(index.path,std::move(combined));
        }
        for (const auto &[path,index] : claimed) { (void)index; singleMappings.erase(path); }
    }
    const auto process = [&](const auto &path, const auto &mapping, bool userMain) {
        if (catalog) {
            EmbeddedImageGroup group;
            group.path = path;
            group.tensorMappings.assign(mapping.begin(), mapping.end());
            group.userSelectedMain = userMain;
            catalog->groups.push_back(std::move(group));
        }
        inspectMapping(path, mapping);
    };
    for (const auto &[path,mapping] : singleMappings)
        if (!duplicatePaths.contains(path)) process(path,mapping,sourceUserMain.at(path));
    for (const auto &[path,mapping] : shardMappings) {
        if (!indexes.contains(path)) throw std::runtime_error("Embedded model shard index is missing.");
        process(path,mapping,groupUserMain.at(path));
    }
    if (mossModels.size() + animaModels.size() > 1) {
        if (catalog) return {};
        throw std::runtime_error("Multiple image models in the package; select the model explicitly.");
    }
    if (!animaModels.empty()) {
        if (!catalog && (animaEncoders.size() > 1 || animaVaes.size() > 1))
            throw std::runtime_error("The Anima package has multiple matching components; create or calibrate a package with an unambiguous component for each role.");
        const auto &[animaTransformer, animaEncoding] = *animaModels.begin();
        if (components) {
            components->transformer = animaTransformer;
            if (animaEncoders.size() == 1) components->textEncoder = animaEncoders.front();
            if (animaVaes.size() == 1) components->vae = animaVaes.front();
        }
        if (extractTransformer) *extractTransformer = animaNeedsExtraction.at(animaTransformer);
        ImageModelInspection result;
        result.architecture = ImageModelArchitecture::Anima; result.role = ImageModelRole::Checkpoint;
        result.encoding = animaEncoding;
        result.hasTextEncoder = animaEncoders.size() == 1; result.hasVae = animaVaes.size() == 1;
        return result;
    }
    if (mossModels.empty()) return {};
    if (!catalog && (encoders.size() > 1 || vaes.size() > 1))
        throw std::runtime_error("The MOSS package has multiple matching components; select or calibrate the components explicitly.");
    const auto &[transformer, encoding] = *mossModels.begin();
    if (components) {
        components->transformer = transformer;
        if (encoders.size() == 1) components->textEncoder = encoders.front();
        if (vaes.size() == 1) components->vae = vaes.front();
    }
    ImageModelInspection result;
    result.architecture = ImageModelArchitecture::MossImage; result.role = ImageModelRole::Checkpoint;
    result.encoding = encoding; result.hasTextEncoder = encoders.size() == 1; result.hasVae = vaes.size() == 1;
    result.syntheticAdapterMappings = syntheticMappings(root);
    return result;
}
}

// Checks bounded descriptors and, for older packages, verified shard indexes;
// it never loads weight payloads or assumes a model can produce an image.
// Unknown families continue to the existing engine.
ImageModelInspection inspectImageModelHeader(const std::filesystem::path &path, EmbeddedImageComponents *components = nullptr,
                                             bool *extractTransformer = nullptr, EmbeddedImageCatalog *catalog = nullptr) {
    auto suffix = path.extension().string();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool namedSafetensors = suffix == ".safetensors" || suffix == ".safetensor";
    const auto invalid = [] { return std::runtime_error(model_inspection::invalidHeaderMessage); };
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size < 10) {
        if (namedSafetensors) throw invalid();
        return {};
    }
    const auto modified = std::filesystem::last_write_time(path, error);
    if (error) {
        if (namedSafetensors) throw invalid();
        return {};
    }
    std::optional<iild::SafetensorsIndex> index;
    model_inspection::Json root(nullptr,json_object_put);
    std::uint64_t payload=0;
    if (iild::detectModelContainer(path)=="safetensors-index") {
        index.emplace(iild::inspectSafetensorsIndex(path));
        root=model_inspection::Json(json_object_new_object(),json_object_put);
        auto *metadata=json_object_new_object();
        for (const auto &[key,value] : index->sharedMetadata) {
            // A shard group is not itself an embedded iild package.
            if (key=="iild_package_schema" || key=="iild_package_manifest") continue;
            json_object_object_add(metadata,key.c_str(),json_object_new_string_len(value.data(),static_cast<int>(value.size())));
        }
        json_object_object_add(root.get(),"__metadata__",metadata);
        for (const auto &tensor : index->tensors) {
            auto *descriptor=json_object_new_object();
            json_object_object_add(descriptor,"dtype",json_object_new_string_len(tensor.dtype.data(),static_cast<int>(tensor.dtype.size())));
            auto *shape=json_object_new_array();
            for (const auto dimension : tensor.shape) json_object_array_add(shape,json_object_new_uint64(dimension));
            json_object_object_add(descriptor,"shape",shape);
            // Actual offsets and sizes were validated in each original shard.
            auto *offsets=json_object_new_array();
            json_object_array_add(offsets,json_object_new_int(0));json_object_array_add(offsets,json_object_new_int(0));
            json_object_object_add(descriptor,"data_offsets",offsets);
            json_object_object_add(root.get(),tensor.name.c_str(),descriptor);
        }
    } else {
        std::ifstream file(path, std::ios::binary);
        std::array<unsigned char, 8> headerPrefix{};
        if (!file.read(reinterpret_cast<char *>(headerPrefix.data()), headerPrefix.size())) {
            if (namedSafetensors) throw invalid();
            return {};
        }
        std::uint64_t length = 0;
        for (unsigned i = 0; i < headerPrefix.size(); ++i) length |= std::uint64_t(headerPrefix[i]) << (i * 8);
        if (length < 2 || length > size - 8) {
            if (namedSafetensors) throw invalid();
            return {};
        }
        // A header beyond this bounded inspector's budget is not necessarily invalid.
        if (length > 64ull * 1024 * 1024) return {};
        std::string bytes(static_cast<std::size_t>(length), '\0');
        if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
            if (namedSafetensors) throw invalid();
            return {};
        }
        if (!namedSafetensors) {
            const auto first = std::find_if(bytes.begin(), bytes.end(), [](unsigned char c) { return !std::isspace(c); });
            if (first == bytes.end() || *first != '{') return {};
        }
        root=model_inspection::parse(bytes);
        payload=size-8-length;
    }
    const auto sizeAfter = std::filesystem::file_size(path, error);
    if (error) throw invalid();
    const auto modifiedAfter = std::filesystem::last_write_time(path, error);
    if (error || sizeAfter != size || modifiedAfter != modified)
        throw std::runtime_error("The selected model changed during header inspection. Wait for synchronization to finish and try again.");
    std::size_t tensors = 0, adapterTensors = 0, unrelatedAdapterFields = 0, unrelatedControlFields = 0;
    bool denoiser = false, vaeEncoder = false, vaeDecoder = false, textEncoder = false, control = false;
    json_object_object_foreach(root.get(), rawName, descriptor) {
        const std::string_view name(rawName);
        if (name == "__metadata__") continue;
        if (++tensors > (index ? 1000000u : 200000u) || !json_object_is_type(descriptor, json_type_object)) throw invalid();
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
    EmbeddedImageComponents localComponents;
    auto *packageComponents = components ? components : &localComponents;
    const auto embedded = model_inspection::package(root.get(), packageComponents, extractTransformer, path, catalog);
    const auto finalSize = std::filesystem::file_size(path,error);
    if (error || finalSize!=size || std::filesystem::last_write_time(path,error)!=modified || error)
        throw std::runtime_error("The selected model changed during package inspection. Wait for synchronization to finish and try again.");
    if (index && !iild::verifySafetensorsIndex(*index))
        throw std::runtime_error("The selected model shards changed during header inspection.");
    // Package-level names can mix unrelated models, adapters and encoders.
    // Never fall back to treating that flattened namespace as one checkpoint.
    if (packageComponents->packaged) return embedded;
    for (const std::string prefix : {"model.diffusion_model.net.", "model.diffusion_model.",
                                      "diffusion_model.net.", "diffusion_model.",
                                      "transformer.net.", "transformer.", "net.", ""}) {
        if (!model_inspection::tensor(root.get(), prefix + "llm_adapter.blocks.0.cross_attn.q_proj.weight")
            || !model_inspection::tensor(root.get(), prefix + "blocks.0.self_attn.q_proj.weight")) continue;
        model_inspection::AnimaTensorInspection inspected;
        json_object_object_foreach(root.get(), rawKey, animaDescriptor) {
            const std::string key(rawKey);
            const bool hasMarker = key.ends_with(".weight")
                && model_inspection::tensor(root.get(), key.substr(0, key.size() - 7) + ".comfy_quant");
            inspected.add(key, animaDescriptor, hasMarker);
        }
        ImageModelInspection result;
        result.architecture = ImageModelArchitecture::Anima;
        result.role = ImageModelRole::Checkpoint;
        result.encoding = inspected.encoding();
        result.hasTextEncoder = model_inspection::tensor(root.get(), "text_encoders.llm.model.embed_tokens.weight")
            && model_inspection::tensor(root.get(), "text_encoders.llm.model.norm.weight");
        result.hasVae = (model_inspection::tensor(root.get(), "first_stage_model.encoder.conv1.weight")
            && model_inspection::tensor(root.get(), "first_stage_model.decoder.conv1.weight"))
            || (model_inspection::tensor(root.get(), "vae.encoder.conv1.weight")
            && model_inspection::tensor(root.get(), "vae.decoder.conv1.weight"))
            || (vaeEncoder && vaeDecoder);
        return result;
    }
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

iiLocalDiffusion::EmbeddedImageCatalog iiLocalDiffusion::inspectEmbeddedImageCatalog(const std::filesystem::path &path) {
    EmbeddedImageCatalog catalog;
    (void)native_detail::inspectImageModelHeader(path, nullptr, nullptr, &catalog);
    return catalog;
}

iiLocalDiffusion::EmbeddedImagePackageInspection iiLocalDiffusion::inspectEmbeddedImagePackage(const std::filesystem::path &path) {
    EmbeddedImagePackageInspection result;
    result.model = native_detail::inspectImageModelHeader(path, &result.components, nullptr, &result.catalog);
    return result;
}
