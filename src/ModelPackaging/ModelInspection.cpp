#include "ModelInspection.hpp"
#include <iiFileProviderNative.h>
#include <iiFileProviderPresentation.h>
#include <json-c/json.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace iild {
namespace {
namespace fs = std::filesystem;
using Json = std::unique_ptr<json_object, decltype(&json_object_put)>;
constexpr std::uint64_t headerLimit = 64 * 1024 * 1024;
constexpr std::uint64_t jsonLimit = 1024 * 1024;
json_object *field(json_object *object, const char *key) {
    json_object *value = nullptr;
    if (object && json_object_is_type(object, json_type_object))
        json_object_object_get_ex(object, key, &value);
    return value;
}
std::string string(json_object *value) {
    return value && json_object_is_type(value, json_type_string)
        ? std::string(json_object_get_string(value), static_cast<std::size_t>(json_object_get_string_len(value))) : std::string{};
}
Json parse(const std::string &text) {
    std::unique_ptr<json_tokener, decltype(&json_tokener_free)> tok(json_tokener_new_ex(64), json_tokener_free);
    json_tokener_set_flags(tok.get(), JSON_TOKENER_STRICT | JSON_TOKENER_VALIDATE_UTF8);
    auto *value = json_tokener_parse_ex(tok.get(), text.data(), static_cast<int>(text.size()));
    Json result(value, json_object_put);
    if (json_tokener_get_error(tok.get()) != json_tokener_success || !value
        || !json_object_is_type(value, json_type_object))
        throw std::runtime_error("Invalid model metadata JSON");
    return result;
}
Json objectValue(json_object *value) {
    if (value && json_object_is_type(value, json_type_object)) return Json(json_object_get(value), json_object_put);
    const auto text = string(value);
    return text.empty() ? Json(nullptr, json_object_put) : parse(text);
}
bool isFile(const fs::path &path) { std::error_code error; return fs::is_regular_file(path, error); }
bool isDirectory(const fs::path &path) { std::error_code error; return fs::is_directory(path, error); }
std::string pathKey(const fs::path &path) {
    const auto value = path.generic_u8string();
    return {value.begin(), value.end()};
}
std::vector<fs::path> sidecars(const fs::path &path) {
    if (isDirectory(path)) return {path / "society.model.json"};
    auto stem = path; stem.replace_extension(".model.json");
    auto full = path; full += ".model.json";
    return {stem, full};
}
std::string identity(const fs::path &path) {
    std::error_code error;
    auto result = pathKey(path);
    const auto status = fs::status(path, error);
    if (error || !fs::exists(status)) return result + ":missing";
    result += ':' + std::to_string(static_cast<long long>(fs::last_write_time(path, error).time_since_epoch().count()));
    if (fs::is_regular_file(status)) result += ':' + std::to_string(fs::file_size(path, error));
#ifndef _WIN32
    struct stat info{};
    if (::stat(path.c_str(), &info) == 0) {
        result += ':' + std::to_string(info.st_dev) + ':' + std::to_string(info.st_ino);
#ifdef __APPLE__
        const auto changed = info.st_ctimespec;
#else
        const auto changed = info.st_ctim;
#endif
        result += ':' + iiFileProvider::native::Presentation::normalizeChangeTime(
            static_cast<std::uint64_t>(info.st_dev), static_cast<std::uint64_t>(info.st_ino), changed.tv_sec, changed.tv_nsec);
    }
#endif
    return result;
}
std::string cacheKey(const fs::path &path) {
    auto result = identity(path);
    if (isDirectory(path)) for (const auto *name : {"model_index.json", "config.json", "transformer/config.json", "vae/config.json"})
        result += '|' + identity(path / name);
    for (const auto &sidecar : sidecars(path)) result += '|' + identity(sidecar);
    return result;
}
Json readJson(const fs::path &path, ModelInspection &result) {
    const auto text = iiFileProvider::native::File::read(path, jsonLimit);
    result.metadataBytesRead += text.size();
    return parse(text);
}
Json readHeader(const fs::path &path, ModelInspection &result) {
    auto file = iiFileProvider::native::File::openRead(path);
    const auto size = file->size();
    if (size < 10) throw std::runtime_error("Model has no complete safetensors header");
    const auto bytes = file->read(8);
    std::uint64_t length = 0;
    for (std::size_t i = 0; i < 8; ++i) length |= std::uint64_t(static_cast<unsigned char>(bytes[i])) << (8 * i);
    if (length < 2 || length > headerLimit || length > size - 8)
        throw std::runtime_error("Invalid or oversized safetensors metadata header");
    const auto text = file->read(static_cast<std::int64_t>(length));
    if (text.size() != length) throw std::runtime_error("Truncated model metadata header");
    result.metadataBytesRead += length + 8;
    return parse(text);
}
bool allowed(const std::string &value) {
    return value == "image" || value == "video" || value == "audio" || value == "language";
}
std::vector<std::string> normalized(const std::vector<std::string> &values) {
    std::vector<std::string> result;
    for (const auto &value : values) {
        if (!allowed(value)) throw std::invalid_argument("Unknown model modality: " + value);
        if (std::find(result.begin(), result.end(), value) == result.end()) result.push_back(value);
    }
    std::sort(result.begin(), result.end());
    return result;
}
void declaration(json_object *metadata, ModelInspection &result) {
    auto society = objectValue(field(metadata, "society"));
    auto *plural = field(society.get(), "modalities");
    auto *singular = field(society.get(), "modality");
    if (!plural) plural = field(metadata, "society.modalities");
    if (!singular) singular = field(metadata, "society.modality");
    if (string(singular) == "auto") {
        result.modalities = result.detectedModalities; result.userOverride = false; return;
    }
    if (!plural && !singular) return;
    std::vector<std::string> values;
    if (plural) {
        if (!json_object_is_type(plural, json_type_array) || json_object_array_length(plural) == 0)
            throw std::runtime_error("society.modalities must be a nonempty array");
        for (std::size_t i = 0; i < json_object_array_length(plural); ++i)
            values.push_back(string(json_object_array_get_idx(plural, i)));
    } else values.push_back(string(singular));
    result.modalities = normalized(values); result.userOverride = true;
}
void className(const std::string &name, ModelInspection &result) {
    if (name.empty()) return;
    static const std::set<std::string> ltx2{"LTX2Pipeline", "LTX2ConditionPipeline", "LTX2ImageToVideoPipeline",
        "LTX2VideoTransformer3DModel", "AVTransformer3DModel"};
    static const std::set<std::string> ltx1{"LTXPipeline", "LTXConditionPipeline", "LTXImageToVideoPipeline",
        "LTXVideoTransformer3DModel", "LTXModel"};
    if (ltx2.contains(name)) {
        result.architecture = name;
        if (!result.family.starts_with("LTX 2.")) result.family = "LTX 2";
        result.detectedModalities = {"audio", "video"}; result.recognized = true;
    } else if (ltx1.contains(name) && result.family != "LTX 2") {
        result.architecture = name; result.family = "LTX";
        result.detectedModalities = {"video"}; result.recognized = true;
    } else if (!result.recognized && (name.starts_with("StableDiffusion") || name.starts_with("Flux")
        || name == "SD3Transformer2DModel" || name == "UNet2DConditionModel")) {
        result.architecture = name; result.family = name.starts_with("Flux") ? "Flux" : "Stable Diffusion";
        result.detectedModalities = {"image"}; result.recognized = true;
    }
}
void config(json_object *root, ModelInspection &result) {
    className(string(field(root, "_class_name")), result);
    auto transformer = objectValue(field(root, "transformer"));
    if (transformer) className(string(field(transformer.get(), "_class_name")), result);
}
ModelInspection inspect(const fs::path &path) {
    ModelInspection result;
    Json metadata(nullptr, json_object_put);
    try {
        if (isDirectory(path)) {
            for (const auto *name : {"model_index.json", "config.json", "transformer/config.json"})
                if (isFile(path / name)) { auto root = readJson(path / name, result); config(root.get(), result); }
        } else if (isFile(path)) {
            auto header = readHeader(path, result);
            metadata = objectValue(field(header.get(), "__metadata__"));
            if (metadata) {
                result.version = string(field(metadata.get(), "model_version"));
                auto modelConfig = objectValue(field(metadata.get(), "config"));
                if (modelConfig) config(modelConfig.get(), result);
                className(string(field(metadata.get(), "_class_name")), result);
                auto manifest = objectValue(field(metadata.get(), "iild_package_manifest"));
                const auto family = manifest ? string(field(manifest.get(), "family")) : std::string{};
                if (family == "LTX 2" || family.starts_with("LTX 2.")) {
                    className("LTX2Pipeline", result); result.family = family;
                }
            }
            bool audio = false, projection = false, transformer = false;
            json_object_object_foreach(header.get(), key, value) {
                (void)value;
                const std::string_view name(key);
                audio = audio || name.starts_with("audio_vae.");
                projection = projection || name.starts_with("text_embedding_projection.audio_aggregate_embed.");
                transformer = transformer || name.starts_with("model.diffusion_model.transformer_blocks.");
            }
            if (audio && projection && transformer) className("AVTransformer3DModel", result);
        } else result.diagnostic = "Model is not locally available; only companion metadata can be inspected";
    } catch (const std::exception &error) { result.diagnostic = error.what(); }
    result.modalities = result.detectedModalities;
    try {
        if (metadata) declaration(metadata.get(), result);
        for (const auto &sidecar : sidecars(path)) if (isFile(sidecar)) {
            auto root = readJson(sidecar, result); declaration(root.get(), result);
        }
    } catch (const std::exception &error) { result.diagnostic = error.what(); }
    return result;
}
std::mutex cacheMutex;
std::map<std::string, std::pair<std::string, ModelInspection>> cache;
}

ModelInspection inspectModel(const fs::path &source) {
    const auto path = fs::absolute(source).lexically_normal();
    const auto key = cacheKey(path);
    std::lock_guard lock(cacheMutex);
    const auto found = cache.find(pathKey(path));
    if (found != cache.end() && found->second.first == key) {
        auto result = found->second.second; result.cacheHit = true; result.metadataBytesRead = 0; return result;
    }
    auto result = inspect(path);
    if (cacheKey(path) != key) { result.diagnostic = "Model metadata changed during inspection; refresh to retry"; return result; }
    if (cache.size() >= 128) cache.erase(cache.begin());
    cache[pathKey(path)] = {key, result};
    return result;
}

void setModelModalities(const fs::path &source, const std::vector<std::string> &values) {
    const auto modalities = normalized(values);
    const auto path = fs::absolute(source).lexically_normal();
    if (!isFile(path) && !isDirectory(path)) throw std::runtime_error("Model must be locally available to save its classification");
    const auto sidecar = sidecars(path).back();
    const bool present = isFile(sidecar);
    const auto previous = present ? iiFileProvider::native::File::read(sidecar, jsonLimit) : std::string{};
    auto root = present ? parse(previous) : Json(json_object_new_object(), json_object_put);
    auto society = objectValue(field(root.get(), "society"));
    if (!society) society = Json(json_object_new_object(), json_object_put);
    json_object_object_del(society.get(), "modality"); json_object_object_del(society.get(), "modalities");
    json_object_object_del(root.get(), "society.modality"); json_object_object_del(root.get(), "society.modalities");
    if (modalities.empty()) json_object_object_add(society.get(), "modality", json_object_new_string("auto"));
    else if (modalities.size() == 1) json_object_object_add(society.get(), "modality", json_object_new_string(modalities.front().c_str()));
    else {
        auto *array = json_object_new_array();
        for (const auto &value : modalities) json_object_array_add(array, json_object_new_string(value.c_str()));
        json_object_object_add(society.get(), "modalities", array);
    }
    json_object_object_add(root.get(), "society", json_object_get(society.get()));
    const std::string replacement = std::string(json_object_to_json_string_ext(root.get(), JSON_C_TO_STRING_PRETTY)) + '\n';
    if (present) iiFileProvider::native::File::update(sidecar, previous, replacement);
    else iiFileProvider::native::File::create(sidecar, replacement);
}

std::string modelInspectionJson(const ModelInspection &value) {
    Json root(json_object_new_object(), json_object_put);
    const auto add = [&](const char *key, const std::string &text) { json_object_object_add(root.get(), key, json_object_new_string(text.c_str())); };
    add("schema", "iild-model-inspection-v1"); add("architecture", value.architecture); add("family", value.family);
    add("version", value.version); add("diagnostic", value.diagnostic);
    json_object_object_add(root.get(), "recognized", json_object_new_boolean(value.recognized));
    json_object_object_add(root.get(), "user_override", json_object_new_boolean(value.userOverride));
    json_object_object_add(root.get(), "cache_hit", json_object_new_boolean(value.cacheHit));
    json_object_object_add(root.get(), "metadata_bytes_read", json_object_new_uint64(value.metadataBytesRead));
    for (const auto &[key, list] : {std::pair{"modalities", &value.modalities}, std::pair{"detected_modalities", &value.detectedModalities}}) {
        auto *array = json_object_new_array();
        for (const auto &item : *list) json_object_array_add(array, json_object_new_string(item.c_str()));
        json_object_object_add(root.get(), key, array);
    }
    return json_object_to_json_string_ext(root.get(), JSON_C_TO_STRING_PLAIN);
}
}
