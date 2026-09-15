#include "GenerationDefaults.hpp"
#include "NativeCachePolicy.hpp"
#include <json-c/json.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace iiLocalDiffusion::native_detail {
namespace {
std::filesystem::path libraryDirectory() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&generationResourceDirectory), &module);
    std::wstring buffer(32768, L'\0');
    const auto length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length) { buffer.resize(length); return std::filesystem::path(buffer).parent_path(); }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&generationResourceDirectory), &info) && info.dli_fname)
        return std::filesystem::canonical(info.dli_fname).parent_path();
#endif
    throw std::runtime_error("Cannot locate the iiLocalDiffusion runtime library.");
}
json_object *field(json_object *object, const char *name, json_type type) {
    json_object *value = nullptr;
    if (!json_object_object_get_ex(object, name, &value) || !json_object_is_type(value, type))
        throw std::runtime_error(std::string("Invalid generation-defaults field: ") + name);
    return value;
}
std::filesystem::path resourcePath(const std::filesystem::path &root, json_object *item) {
    const auto relative = std::filesystem::path(json_object_get_string(field(item, "file", json_type_string)));
    if (relative.empty() || relative.is_absolute()) throw std::runtime_error("Invalid default resource path.");
    for (const auto &part : relative)
        if (part == "..") throw std::runtime_error("A default resource escapes its package.");
    const auto path = std::filesystem::canonical(root / relative);
    const auto inside = path.lexically_relative(root);
    if (inside.empty() || *inside.begin() == ".." || !std::filesystem::is_regular_file(path)
        || std::filesystem::file_size(path) != static_cast<std::uint64_t>(json_object_get_int64(field(item, "size", json_type_int))))
        throw std::runtime_error("A bundled generation resource is missing or changed: " + path.string());
    return path;
}
std::filesystem::path resource(const std::filesystem::path &root, json_object *item) {
    const auto path = resourcePath(root, item);
    // Reject an unfetched Git LFS pointer before calling any tensor loader.
    std::ifstream input(path, std::ios::binary);
    std::uint64_t headerLength = 0;
    input.read(reinterpret_cast<char *>(&headerLength), sizeof(headerLength));
    if (!input || headerLength < 2 || headerLength > 16 * 1024 * 1024
        || headerLength + 8 > std::filesystem::file_size(path))
        throw std::runtime_error("Invalid bundled safetensors resource; fetch Git LFS objects: " + path.string());
    return path;
}

std::string validateVaeConfig(const std::filesystem::path &root, json_object *item, const std::string &family) {
    const auto path = resourcePath(root, field(item, "config", json_type_object));
    if (std::filesystem::file_size(path) > 1024 * 1024) throw std::runtime_error("VAE configuration is too large.");
    const auto before = modelIdentity(path);
    std::unique_ptr<json_object, decltype(&json_object_put)> config(json_object_from_file(path.string().c_str()), json_object_put);
    if (!config || std::string(json_object_get_string(field(config.get(), "_class_name", json_type_string)))
            != json_object_get_string(field(item, "class_name", json_type_string)))
        throw std::runtime_error("The VAE configuration has an incompatible architecture.");
    const auto number = [&](const char *name, double expected, bool optional = false) {
        json_object *value = nullptr;
        if (!json_object_object_get_ex(config.get(), name, &value) && optional) return;
        if (!value || (!json_object_is_type(value, json_type_int) && !json_object_is_type(value, json_type_double))
            || !std::isfinite(json_object_get_double(value)) || std::abs(json_object_get_double(value) - expected) > 1e-6)
            throw std::runtime_error("Incompatible " + family + " VAE configuration: " + name);
    };
    if (family == "qwen-image") {
        number("z_dim", 16);
        number("input_channels", 3, true);
        number("in_channels", 3, true);
        number("out_channels", 3, true);
        // Match the pinned native Wan/Qwen decoder's latent normalization.
        const double means[] = {-.7571, -.7089, -.9113, .1075, -.1745, .9653, -.1517, 1.5508,
            .4134, -.0715, .5517, -.3632, -.1922, -.9497, .2503, -.2921};
        const double stds[] = {2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916};
        for (const auto *name : {"latents_mean", "latents_std"}) {
            auto *values = field(config.get(), name, json_type_array);
            if (json_object_array_length(values) != 16) throw std::runtime_error("Invalid Qwen VAE latent normalization.");
            const auto *expected = std::string(name) == "latents_mean" ? means : stds;
            for (std::size_t i = 0; i < 16; ++i) {
                auto *value = json_object_array_get_idx(values, i);
                if ((!json_object_is_type(value, json_type_int) && !json_object_is_type(value, json_type_double))
                    || !std::isfinite(json_object_get_double(value)) || std::abs(json_object_get_double(value) - expected[i]) > 1e-6)
                    throw std::runtime_error("Incompatible Qwen VAE latent normalization.");
            }
        }
    } else {
        number("in_channels", 3);
        number("out_channels", 3);
        number("latent_channels", family == "sdxl-base" ? 4 : family == "flux1" ? 16 : 32);
        if (json_object_array_length(field(config.get(), "block_out_channels", json_type_array)) != 4)
            throw std::runtime_error("Incompatible VAE downsampling configuration.");
        if (family == "flux2") {
            auto *patch = field(config.get(), "patch_size", json_type_array);
            if (json_object_array_length(patch) != 2) throw std::runtime_error("FLUX.2 VAE requires 2x2 patches.");
            for (std::size_t i = 0; i < 2; ++i)
                if (!json_object_is_type(json_object_array_get_idx(patch, i), json_type_int)
                    || json_object_get_int(json_object_array_get_idx(patch, i)) != 2)
                    throw std::runtime_error("FLUX.2 VAE requires 2x2 patches.");
            number("batch_norm_eps", 1e-4);
        } else {
            number("scaling_factor", family == "sdxl-base" ? .13025 : .3611);
            number("shift_factor", family == "sdxl-base" ? 0 : .1159, family == "sdxl-base");
        }
    }
    if (modelIdentity(path) != before) throw std::runtime_error("VAE configuration changed while reading.");
    return before;
}
}

std::string canonicalLoraFamily(const std::string &family) {
    if (family == "sd1") return "sd15";
    if (family == "sdxl") return "sdxl-base";
    if (family == "flux1-schnell" || family == "flux1-dev") return "flux1";
    return family;
}

std::filesystem::path generationResourceDirectory() {
    if (const auto *override = std::getenv("IILD_GENERATION_RESOURCES")) {
        if (!*override) throw std::runtime_error("IILD_GENERATION_RESOURCES must not be empty.");
        return std::filesystem::canonical(override);
    }
    const auto library = libraryDirectory();
    for (const auto &candidate : {
            library / "../share/iiLocalDiffusion/resources",
            library / "../Resources/iiLocalDiffusion/resources",
            library / "../iiLocalDiffusion/resources",
            library / "../../iiLocalDiffusion/resources",
            library / "iiLocalDiffusion/resources"}) {
        if (std::filesystem::is_regular_file(candidate / "generation-defaults.json"))
            return std::filesystem::canonical(candidate);
    }
#if defined(IILD_BUILD_LIBRARY_DIRECTORY) && defined(IILD_BUILD_RESOURCE_DIRECTORY)
    if (library == std::filesystem::path(IILD_BUILD_LIBRARY_DIRECTORY))
        return std::filesystem::canonical(IILD_BUILD_RESOURCE_DIRECTORY);
#endif
    throw std::runtime_error("iiLocalDiffusion generation-defaults.json is missing. Install the SDK resources or set IILD_GENERATION_RESOURCES.");
}

GenerationDefaults loadGenerationDefaults(const std::filesystem::path &directory) {
    const auto root = directory.empty() ? generationResourceDirectory() : std::filesystem::canonical(directory);
    const auto manifestPath = root / "generation-defaults.json";
    if (std::filesystem::file_size(manifestPath) > 1024 * 1024)
        throw std::runtime_error("Generation defaults manifest is too large.");
    const auto before = modelIdentity(manifestPath);
    std::unique_ptr<json_object, decltype(&json_object_put)> manifest(json_object_from_file(manifestPath.string().c_str()), json_object_put);
    if (!manifest || json_object_get_int(field(manifest.get(), "version", json_type_int)) != 1)
        throw std::runtime_error("Unsupported generation-defaults resource manifest.");
    GenerationDefaults result;
    result.identity = before;
    std::set<std::string> seenFamilies;
    auto addLora = [&](json_object *item) {
        DefaultLora lora;
        lora.path = resource(root, item);
        json_object *scale = nullptr;
        if (!json_object_object_get_ex(item, "scale", &scale)
            || (!json_object_is_type(scale, json_type_double) && !json_object_is_type(scale, json_type_int)))
            throw std::runtime_error("Invalid fallback LoRA strength.");
        lora.scale = static_cast<float>(json_object_get_double(scale));
        if (!std::isfinite(lora.scale) || lora.scale == 0)
            throw std::runtime_error("A fallback LoRA must have a finite nonzero strength.");
        auto *families = field(item, "families", json_type_array);
        if (!json_object_array_length(families)) throw std::runtime_error("A fallback LoRA requires explicit model families.");
        for (std::size_t i = 0; i < json_object_array_length(families); ++i) {
            auto *value = json_object_array_get_idx(families, i);
            if (!json_object_is_type(value, json_type_string)) throw std::runtime_error("Invalid LoRA model family.");
            auto family = canonicalLoraFamily(json_object_get_string(value));
            if (family.empty() || family == "*" || family == "other" || family == "sd1-or-sd2")
                throw std::runtime_error("A fallback LoRA requires an unambiguous model family.");
            if (!seenFamilies.insert(family).second) throw std::runtime_error("Duplicate fallback LoRA family: " + family);
            lora.families.push_back(std::move(family));
        }
        result.identity += ':' + modelIdentity(lora.path);
        result.loras.push_back(std::move(lora));
    };
    json_object *legacyLora = nullptr;
    if (json_object_object_get_ex(manifest.get(), "fallback_lora", &legacyLora))
        addLora(field(manifest.get(), "fallback_lora", json_type_object));
    json_object *loraEntries = nullptr;
    if (json_object_object_get_ex(manifest.get(), "fallback_loras", &loraEntries)) {
        loraEntries = field(manifest.get(), "fallback_loras", json_type_array);
        if (json_object_array_length(loraEntries) > 64) throw std::runtime_error("Too many fallback LoRAs.");
        for (std::size_t i = 0; i < json_object_array_length(loraEntries); ++i)
            addLora(json_object_array_get_idx(loraEntries, i));
    }
    if (result.loras.empty() || result.loras.size() > 64)
        throw std::runtime_error("The defaults manifest requires 1..64 fallback LoRAs.");
    auto *entries = field(manifest.get(), "negative_embeddings", json_type_array);
    const auto count = json_object_array_length(entries);
    if (count > 64) throw std::runtime_error("Invalid default embedding count.");
    for (std::size_t i = 0; i < count; ++i) {
        auto *item = json_object_array_get_idx(entries, i);
        DefaultEmbedding embedding;
        embedding.path = resource(root, item);
        embedding.token = json_object_get_string(field(item, "token", json_type_string));
        if (!std::regex_match(embedding.token, std::regex("[a-z][a-z0-9_]*")))
            throw std::runtime_error("Invalid default embedding token.");
        auto *families = field(item, "families", json_type_array);
        for (std::size_t j = 0; j < json_object_array_length(families); ++j) {
            auto *family = json_object_array_get_idx(families, j);
            if (!json_object_is_type(family, json_type_string))
                throw std::runtime_error("Invalid embedding model family.");
            embedding.sd15 |= std::string(json_object_get_string(family)) == "sd15";
        }
        result.identity += ':' + modelIdentity(embedding.path);
        result.embeddings.push_back(std::move(embedding));
    }
    if (modelIdentity(manifestPath) != before) throw std::runtime_error("Generation defaults changed while reading.");
    return result;
}

DefaultVae loadFallbackVae(const std::filesystem::path &directory, const std::string &family) {
    const auto root = directory.empty() ? generationResourceDirectory() : std::filesystem::canonical(directory);
    const auto manifestPath = root / "generation-defaults.json";
    if (std::filesystem::file_size(manifestPath) > 1024 * 1024)
        throw std::runtime_error("Generation defaults manifest is too large.");
    const auto before = modelIdentity(manifestPath);
    std::unique_ptr<json_object, decltype(&json_object_put)> manifest(json_object_from_file(manifestPath.string().c_str()), json_object_put);
    if (!manifest || json_object_get_int(field(manifest.get(), "version", json_type_int)) != 1)
        throw std::runtime_error("Unsupported generation-defaults resource manifest.");
    const auto requested = canonicalLoraFamily(family);
    json_object *item = nullptr;
    std::set<std::string> seen;
    auto inspect = [&](json_object *entry) {
        auto *families = field(entry, "families", json_type_array);
        if (!json_object_array_length(families)) throw std::runtime_error("Missing fallback VAE families.");
        const std::string architecture = json_object_get_string(field(entry, "class_name", json_type_string));
        for (std::size_t i = 0; i < json_object_array_length(families); ++i) {
            auto *value = json_object_array_get_idx(families, i);
            if (!json_object_is_type(value, json_type_string)) throw std::runtime_error("Invalid VAE family.");
            const auto member = canonicalLoraFamily(json_object_get_string(value));
            const std::string expected = member == "qwen-image" ? "AutoencoderKLQwenImage"
                : member == "flux2" ? "AutoencoderKLFlux2"
                : (member == "sdxl-base" || member == "flux1") ? "AutoencoderKL" : "";
            if (expected.empty() || architecture != expected)
                throw std::runtime_error("Incompatible fallback VAE architecture or family.");
            if (!seen.insert(member).second) throw std::runtime_error("Duplicate fallback VAE family: " + member);
            if (member == requested) item = entry;
        }
    };
    json_object *legacy = nullptr;
    if (json_object_object_get_ex(manifest.get(), "fallback_vae", &legacy))
        inspect(field(manifest.get(), "fallback_vae", json_type_object));
    json_object *entries = nullptr;
    if (json_object_object_get_ex(manifest.get(), "fallback_vaes", &entries)) {
        entries = field(manifest.get(), "fallback_vaes", json_type_array);
        if (json_object_array_length(entries) > 64) throw std::runtime_error("Too many fallback VAEs.");
        for (std::size_t i = 0; i < json_object_array_length(entries); ++i)
            inspect(json_object_array_get_idx(entries, i));
    }
    if (!item) throw std::runtime_error("Missing fallback VAE for " + requested + "; install the SDK VAE resources.");
    const auto path = resource(root, item);
    const auto configIdentity = validateVaeConfig(root, item, requested);
    if (modelIdentity(manifestPath) != before)
        throw std::runtime_error("Generation defaults changed while reading.");
    return {path, before + ':' + modelIdentity(path) + ':' + configIdentity};
}

std::string appendNegativeTokens(std::string prompt, const std::vector<std::string> &tokens) {
    for (const auto &token : tokens) {
        if (std::regex_search(prompt, std::regex("(^|[^A-Za-z0-9_])" + token + "($|[^A-Za-z0-9_])"))) continue;
        if (!prompt.empty()) prompt += ", ";
        prompt += token;
    }
    return prompt;
}
}
