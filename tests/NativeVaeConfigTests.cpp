#include "Generation/GenerationDefaults.hpp"
#include <json-c/json.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
using Json = std::unique_ptr<json_object, decltype(&json_object_put)>;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
json_object *get(json_object *value, const char *name) { return json_object_object_get(value, name); }
std::string string(json_object *value) { return json_object_get_string(value); }
Json clone(json_object *value) { return {json_tokener_parse(json_object_to_json_string(value)), json_object_put}; }
void save(const std::filesystem::path &path, json_object *value) {
    require(json_object_to_file_ext(path.string().c_str(), value, JSON_C_TO_STRING_PRETTY) == 0, "Cannot write fixture JSON");
}
}

int main(int argc, char **argv) {
    try {
        if (argc != 3) return 1;
        const auto resources = std::filesystem::canonical(argv[1]);
        const auto directory = std::filesystem::absolute(argv[2]);
        Json manifest(json_object_from_file((resources / "generation-defaults.json").string().c_str()), json_object_put);
        require(bool(manifest), "Cannot read installed VAE manifest");
        std::vector<json_object *> entries{get(manifest.get(), "fallback_vae")};
        auto *fallbacks = get(manifest.get(), "fallback_vaes");
        for (std::size_t i = 0; i < json_object_array_length(fallbacks); ++i)
            entries.push_back(json_object_array_get_idx(fallbacks, i));
        for (auto *original : entries) {
            const auto family = string(json_object_array_get_idx(get(original, "families"), 0));
            const auto root = directory / family;
            std::filesystem::create_directories(root);
            const auto weights = root / "weights.safetensors";
            if (!std::filesystem::exists(weights))
                std::filesystem::create_hard_link(resources / string(get(original, "file")), weights);
            const auto originalConfig = resources / string(get(get(original, "config"), "file"));
            Json config(json_object_from_file(originalConfig.string().c_str()), json_object_put);
            require(bool(config), "Cannot read official VAE configuration");
            auto entry = clone(original);
            json_object_object_add(entry.get(), "file", json_object_new_string("weights.safetensors"));
            auto *configEntry = get(entry.get(), "config");
            json_object_object_add(configEntry, "file", json_object_new_string("config.json"));
            Json fixture(json_object_new_object(), json_object_put);
            json_object_object_add(fixture.get(), "version", json_object_new_int(1));
            json_object_object_add(fixture.get(), "fallback_vae", json_object_get(entry.get()));
            const auto write = [&](json_object *value) {
                save(root / "config.json", value);
                json_object_object_add(configEntry, "size", json_object_new_int64(std::filesystem::file_size(root / "config.json")));
                save(root / "generation-defaults.json", fixture.get());
            };
            write(config.get());
            const auto first = iiLocalDiffusion::native_detail::loadFallbackVae(root, family);
            require(first.path == weights, "Correct configured VAE was not selected");
            const auto reject = [&](const char *field, json_object *value) {
                auto wrong = clone(config.get());
                json_object_object_add(wrong.get(), field, value);
                write(wrong.get());
                bool rejected = false;
                try { iiLocalDiffusion::native_detail::loadFallbackVae(root, family); }
                catch (const std::exception &) { rejected = true; }
                require(rejected, "Incompatible VAE configuration was accepted");
            };
            reject("_class_name", json_object_new_string("AutoencoderKLWrong"));
            reject(family == "qwen-image" ? "z_dim" : "latent_channels", json_object_new_int(48));
            reject("out_channels", json_object_new_int(4));
            if (family == "qwen-image") {
                auto means = clone(get(config.get(), "latents_mean"));
                json_object_array_put_idx(means.get(), 0, json_object_new_double(.1));
                reject("latents_mean", means.release());
                reject("latents_std", json_object_new_array());
            } else if (family == "flux2") {
                auto patch = Json(json_tokener_parse("[1, 1]"), json_object_put);
                reject("patch_size", patch.release());
                reject("batch_norm_eps", json_object_new_double(.01));
            } else {
                reject("scaling_factor", json_object_new_double(.18215));
                reject("shift_factor", json_object_new_double(.5));
                reject("scaling_factor", json_object_new_string("0.13025"));
            }
            write(config.get());
            const auto restored = iiLocalDiffusion::native_detail::loadFallbackVae(root, family);
            require(!restored.identity.empty(), "Restored configuration was not accepted");
            // A valid, same-size config edit must invalidate the context identity.
            json_object_object_add(config.get(), "sample_size", json_object_new_int(1024));
            write(config.get());
            const auto before = iiLocalDiffusion::native_detail::loadFallbackVae(root, family).identity;
            const auto size = std::filesystem::file_size(root / "config.json");
            json_object_object_add(config.get(), "sample_size", json_object_new_int(2048));
            // Keep the manifest bytes and timestamps unchanged: only the config
            // may account for this identity change.
            save(root / "config.json", config.get());
            require(std::filesystem::file_size(root / "config.json") == size
                && iiLocalDiffusion::native_detail::loadFallbackVae(root, family).identity != before,
                "Same-size VAE configuration edits must invalidate the native cache");
            json_object_object_add(configEntry, "file", json_object_new_string("../config.json"));
            save(root / "generation-defaults.json", fixture.get());
            bool rejected = false;
            try { iiLocalDiffusion::native_detail::loadFallbackVae(root, family); }
            catch (const std::exception &) { rejected = true; }
            require(rejected, "VAE config must remain inside its resource package");
            std::cout << family << ": architecture, latent contract and config cache identity passed\n";
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
