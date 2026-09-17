#pragma once
#include "NativeCachePolicy.hpp"
#include <json-c/json.h>
#include <cmath>
#include <memory>
#include <set>
#include <vector>

namespace iiLocalDiffusion::native_detail {
struct UnifiedStage {
    std::filesystem::path model;
    float strength;
    std::string identity;
};
struct UnifiedModel {
    std::filesystem::path manifest;
    std::string identity;
    std::vector<UnifiedStage> stages;
    void verify() const {
        if (modelIdentity(manifest) != identity)
            throw std::runtime_error("The unified model manifest changed during generation.");
        for (const auto &stage : stages)
            if (modelIdentity(stage.model) != stage.identity)
                throw std::runtime_error("A unified model member changed during generation.");
    }
};
inline UnifiedModel loadUnifiedModel(const std::filesystem::path &root) {
    namespace fs = std::filesystem;
    if (!root.is_absolute() || !fs::is_directory(root) || fs::canonical(root) != root)
        throw std::runtime_error("Choose an available canonical unified model directory.");
    const auto manifest = root / "model_index.json";
    if (!fs::is_regular_file(manifest) || fs::canonical(manifest) != manifest || fs::file_size(manifest) > 1024 * 1024)
        throw std::runtime_error("The unified model manifest is missing, redirected or oversized.");
    UnifiedModel result{manifest, modelIdentity(manifest), {}};
    std::unique_ptr<json_object, decltype(&json_object_put)> json(json_object_from_file(manifest.string().c_str()), json_object_put);
    const auto field = [](json_object *object, const char *key, json_type type) {
        json_object *value = nullptr;
        if (!object || !json_object_object_get_ex(object, key, &value) || !json_object_is_type(value, type))
            throw std::runtime_error(std::string("Invalid unified model field: ") + key);
        return value;
    };
    const auto string = [&](json_object *object, const char *key) {
        auto *value = field(object, key, json_type_string);
        return std::string(json_object_get_string(value), json_object_get_string_len(value));
    };
    if (string(json.get(), "schema") != "iild-unified-model-v1"
        || string(json.get(), "_class_name") != "IILDUnifiedCascade"
        || string(json.get(), "composition") != "ordered-image-refinement")
        throw std::runtime_error("This directory is not a supported unified model cascade.");
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
        result.stages.push_back({model, static_cast<float>(value), modelIdentity(model)});
    }
    result.verify();
    return result;
}
}
