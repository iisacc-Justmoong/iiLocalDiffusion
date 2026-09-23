#pragma once
#include "NativeCachePolicy.hpp"
#include <json-c/json.h>
#include <cctype>
#include <functional>
#include <cmath>
#include <memory>
#include <set>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <sstream>
#include <vector>
#if IILD_HAS_NATIVE_DIFFUSION
#include "zip.h"
#endif

namespace iiLocalDiffusion::native_detail {
struct UnifiedStage {
    std::filesystem::path model;
    float strength;
    std::string identity;
    std::filesystem::path vae;
    std::string vaeIdentity;
};
struct UnifiedModel {
    std::filesystem::path package;
    std::string packageIdentity;
    std::filesystem::path manifest;
    std::string identity;
    std::vector<UnifiedStage> stages;
    void verify() const {
        if (!package.empty() && modelIdentity(package) != packageIdentity)
            throw std::runtime_error("The unified model package changed during generation.");
        if (modelIdentity(manifest) != identity)
            throw std::runtime_error("The unified model manifest changed during generation.");
        for (const auto &stage : stages) {
            if (modelIdentity(stage.model) != stage.identity)
                throw std::runtime_error("A unified model member changed during generation.");
            if (!stage.vae.empty() && modelIdentity(stage.vae) != stage.vaeIdentity)
                throw std::runtime_error("A unified model VAE changed during generation.");
        }
    }
};
inline bool isUnifiedModelPackage(const std::filesystem::path &path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".iildmodel" && std::filesystem::is_regular_file(path);
}
inline bool safePackageMember(const std::string &name) {
    if (name.empty() || name.front() == '/' || name.find_first_of("\\:\0") != std::string::npos) return false;
    const std::filesystem::path path(name);
    if (path.is_absolute()) return false;
    for (const auto &part : path)
        if (part.empty() || part == "." || part == "..") return false;
    return true;
}
#if IILD_HAS_NATIVE_DIFFUSION
inline std::filesystem::path materializeUnifiedModel(const std::filesystem::path &package) {
    namespace fs = std::filesystem;
    if (!package.is_absolute() || !isUnifiedModelPackage(package) || fs::canonical(package) != package)
        throw std::runtime_error("Choose an available canonical .iildmodel package file.");
    const auto identity = modelIdentity(package);
    std::ostringstream encoded;
    encoded << std::hex << std::hash<std::string>{}(identity);
    const auto cache = fs::temp_directory_path() / "iiLocalDiffusion" / "iildmodel-v1";
    const auto destination = cache / encoded.str();
    const auto marker = destination / ".iildmodel-source";
    const auto ready = [&] {
        std::ifstream input(marker, std::ios::binary);
        return fs::is_directory(destination) && input
            && std::string(std::istreambuf_iterator<char>(input), {}) == identity
            && fs::is_regular_file(destination / "model_index.json");
    };
    static std::mutex mutex;
    const std::lock_guard lock(mutex);
    if (ready()) return fs::canonical(destination);
    fs::create_directories(cache);
    const auto staging = cache / ("." + encoded.str() + "-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(staging);
    struct Archive { zip_t *value = nullptr; ~Archive() { if (value) zip_close(value); } } archive;
    try {
        archive.value = zip_open(package.string().c_str(), 0, 'r');
        if (!archive.value) throw std::runtime_error("The .iildmodel ZIP64 package cannot be opened.");
        const auto count = zip_entries_total(archive.value);
        if (count < 1 || count > 1024) throw std::runtime_error("An .iildmodel package requires 1 to 1024 file entries.");
        std::set<std::string> names;
        std::uint64_t total = 0;
        for (ssize_t index = 0; index < count; ++index) {
            if (zip_entry_openbyindex(archive.value, static_cast<std::size_t>(index)) != 0)
                throw std::runtime_error("Cannot inspect an .iildmodel package member.");
            const auto close = [&] { zip_entry_close(archive.value); };
            const char *raw = zip_entry_name(archive.value);
            const std::string name = raw ? raw : "";
            const auto size = zip_entry_size(archive.value);
            const auto compressed = zip_entry_comp_size(archive.value);
            if (!safePackageMember(name) || zip_entry_isdir(archive.value) != 0 || !size
                || size != compressed || !names.insert(name).second
                || size > 16ull * 1024 * 1024 * 1024 * 1024 - total) {
                close(); throw std::runtime_error("The .iildmodel package contains an invalid, duplicate or compressed member.");
            }
            total += size;
            const auto target = staging / fs::path(name);
            fs::create_directories(target.parent_path());
            if (zip_entry_fread(archive.value, target.string().c_str()) != 0) {
                close(); throw std::runtime_error("Cannot extract an .iildmodel package member.");
            }
            close();
            if (!fs::is_regular_file(target) || fs::file_size(target) != size)
                throw std::runtime_error("An .iildmodel package member changed while extracting.");
        }
        if (!names.contains("model_index.json")) throw std::runtime_error("The .iildmodel manifest is missing.");
        if (modelIdentity(package) != identity) throw std::runtime_error("The .iildmodel package changed while extracting.");
        { std::ofstream output(staging / ".iildmodel-source", std::ios::binary); output << identity; }
        std::error_code error;
        fs::rename(staging, destination, error);
        if (error && !ready()) throw std::runtime_error("Cannot publish the .iildmodel extraction cache.");
        if (error) fs::remove_all(staging);
    } catch (...) {
        fs::remove_all(staging);
        throw;
    }
    if (!ready()) throw std::runtime_error("The .iildmodel extraction cache is incomplete.");
    return fs::canonical(destination);
}
#endif
inline UnifiedModel loadUnifiedModel(const std::filesystem::path &source) {
    namespace fs = std::filesystem;
    auto root = source;
    std::string packageIdentity;
    if (isUnifiedModelPackage(source)) {
#if IILD_HAS_NATIVE_DIFFUSION
        packageIdentity = modelIdentity(source);
        root = materializeUnifiedModel(source);
#else
        throw std::runtime_error("Packaged .iildmodel files require the native diffusion runtime.");
#endif
    }
    if (!root.is_absolute() || !fs::is_directory(root) || fs::canonical(root) != root)
        throw std::runtime_error("Choose an available canonical unified model package.");
    const auto manifest = root / "model_index.json";
    if (!fs::is_regular_file(manifest) || fs::canonical(manifest) != manifest || fs::file_size(manifest) > 1024 * 1024)
        throw std::runtime_error("The unified model manifest is missing, redirected or oversized.");
    UnifiedModel result{packageIdentity.empty() ? fs::path{} : source, packageIdentity,
                        manifest, modelIdentity(manifest), {}};
    std::unique_ptr<json_object, decltype(&json_object_put)> json(json_object_from_file(manifest.string().c_str()), json_object_put);
    const auto field = [](json_object *object, const char *key, json_type type) {
        json_object *value = nullptr;
        if (!object || !json_object_object_get_ex(object, key, &value) || !json_object_is_type(value, type))
            throw std::runtime_error(std::string("Invalid unified model field: ") + key);
        return value;
    };
    const auto string = [&](json_object *object, const char *key) {
        auto *value = field(object, key, json_type_string);
        return std::string(json_object_get_string(value),
                           static_cast<std::size_t>(json_object_get_string_len(value)));
    };
    if (string(json.get(), "schema") != "iild-unified-model-v1"
        || string(json.get(), "_class_name") != "IILDUnifiedCascade"
        || string(json.get(), "composition") != "ordered-image-refinement")
        throw std::runtime_error("This directory is not a supported unified model cascade.");
    if (!packageIdentity.empty() && string(json.get(), "container") != "zip-stored-v1")
        throw std::runtime_error("A packaged unified model must declare the zip-stored-v1 container.");
    auto *stages = field(json.get(), "stages", json_type_array);
    const auto count = json_object_array_length(stages);
    if (!count || count > 64) throw std::runtime_error("A unified model requires 1 to 64 stages.");
    std::set<fs::path> members;
    for (std::size_t i = 0; i < count; ++i) {
        auto *stage = json_object_array_get_idx(stages, i);
        const auto name = string(stage, "model");
        const fs::path relative(name);
        if (name.empty() || name.find_first_of("\\:") != std::string::npos || name.find('\0') != std::string::npos
            || relative.is_absolute()) throw std::runtime_error("Invalid unified model member path.");
        for (const auto &part : relative)
            if (part == ".." || part == "." || part.empty()) throw std::runtime_error("A unified model member escapes its package.");
        const auto model = root / relative;
        const auto size = json_object_get_int64(field(stage, "size_bytes", json_type_int));
        if (!fs::is_regular_file(model) || fs::canonical(model) != model || size <= 0
            || fs::file_size(model) != static_cast<std::uint64_t>(size) || !members.insert(model).second)
            throw std::runtime_error("A unified model member is missing, changed, duplicated or redirected.");
        json_object *strength = nullptr;
        if (!json_object_object_get_ex(stage, "strength", &strength)
            || (!json_object_is_type(strength, json_type_double) && !json_object_is_type(strength, json_type_int)))
            throw std::runtime_error("Invalid unified model strength.");
        const auto value = json_object_get_double(strength);
        if (!std::isfinite(value) || value < 0 || value > 1 || (i == 0 && value != 1))
            throw std::runtime_error("Unified model strengths must be in [0,1], with base strength 1.");
        result.stages.push_back({model, static_cast<float>(value), modelIdentity(model), {}, {}});
        json_object *vae = nullptr;
        if (json_object_object_get_ex(stage, "vae", &vae)) {
            const auto vaeName = string(stage, "vae");
            const fs::path relativeVae(vaeName);
            if (vaeName.empty() || vaeName.find_first_of("\\:") != std::string::npos
                || vaeName.find('\0') != std::string::npos || relativeVae.is_absolute())
                throw std::runtime_error("Invalid unified VAE path.");
            for (const auto &part : relativeVae)
                if (part == ".." || part == "." || part.empty())
                    throw std::runtime_error("A unified VAE escapes its package.");
            const auto path = root / relativeVae;
            if (!fs::is_regular_file(path) || fs::canonical(path) != path || fs::file_size(path) == 0)
                throw std::runtime_error("A unified VAE is missing, empty or redirected.");
            result.stages.back().vae = path;
            result.stages.back().vaeIdentity = modelIdentity(path);
        }
    }
    result.verify();
    return result;
}
}
