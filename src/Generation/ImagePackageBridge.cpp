#include "ImagePackageBridge.hpp"
#include "ImageModelRouting.hpp"
#include "NativeVaeInspection.hpp"
#include "NativeTextEncoderInspection.hpp"
#include "ModelPackaging/ModelPackaging.hpp"
#include "ModelPackaging/ModelInspection.hpp"

#include <json-c/json.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>

struct iild_image_package_result_v1 { std::string json; };

namespace {
std::filesystem::path utf8Path(const char *value) {
    if (!value || !*value) throw std::invalid_argument("An image package path is required.");
    const std::string_view bytes(value);
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(bytes.data()),bytes.size()));
}
std::string utf8Text(const std::filesystem::path &value) {
    const auto bytes=value.generic_u8string(); return {bytes.begin(),bytes.end()};
}
const char *architecture(iiLocalDiffusion::ImageModelArchitecture value) {
    using A=iiLocalDiffusion::ImageModelArchitecture;
    switch (value) { case A::MossImage:return "moss-image"; case A::Anima:return "anima"; default:return "unknown"; }
}
const char *role(iiLocalDiffusion::ImageModelRole value) {
    using R=iiLocalDiffusion::ImageModelRole;
    switch (value) { case R::Checkpoint:return "checkpoint"; case R::Lora:return "lora";
        case R::Vae:return "vae"; case R::ControlNet:return "controlnet"; default:return "unknown"; }
}
const char *encoding(iiLocalDiffusion::ImageWeightEncoding value) {
    using E=iiLocalDiffusion::ImageWeightEncoding;
    switch (value) { case E::Floating:return "floating"; case E::ComfyInt8:return "comfy-int8";
        case E::Other:return "other"; default:return "unknown"; }
}
void text(json_object *object,const char *key,const std::string &value) {
    json_object_object_add(object,key,json_object_new_string_len(value.data(),static_cast<int>(value.size())));
}
void catalogJson(json_object *result,const iiLocalDiffusion::EmbeddedImageCatalog &catalog) {
    auto *groups=json_object_new_array();
    json_object_object_add(result,"groups",groups);
    for (const auto &group : catalog.groups) {
        auto *item=json_object_new_object();
        json_object_array_add(groups,item);
        text(item,"path",group.path);
        json_object_object_add(item,"user_selected_main",json_object_new_boolean(group.userSelectedMain));
        auto *mapping=json_object_new_array();
        json_object_object_add(item,"tensors",mapping);
        for (const auto &[original,output] : group.tensorMappings) {
            auto *entry=json_object_new_array();
            json_object_array_add(entry,json_object_new_string_len(original.data(),static_cast<int>(original.size())));
            json_object_array_add(entry,json_object_new_string_len(output.data(),static_cast<int>(output.size())));
            json_object_array_add(mapping,entry);
        }
    }
}
template <typename F> iild_image_package_result_v1 *answer(F &&operation) noexcept {
    try {
        const std::unique_ptr<json_object,decltype(&json_object_put)> object(json_object_new_object(),json_object_put);
        if (!object) return nullptr;
        text(object.get(),"schema","iild-image-package-bridge-v1");
        try {
            operation(object.get());
            json_object_object_add(object.get(),"ok",json_object_new_boolean(true));
        } catch (const std::exception &error) {
            json_object_object_add(object.get(),"ok",json_object_new_boolean(false));
            text(object.get(),"error",error.what());
        } catch (...) {
            json_object_object_add(object.get(),"ok",json_object_new_boolean(false));
            text(object.get(),"error","Unable to inspect or prepare the image package.");
        }
        const auto *json=json_object_to_json_string_ext(object.get(),JSON_C_TO_STRING_PLAIN);
        if (!json) return nullptr;
        return new iild_image_package_result_v1{json};
    } catch (...) { return nullptr; }
}
}

extern "C" {
iild_image_package_result_v1 *iild_image_package_inspect_v1(const char *model) {
    return answer([&](json_object *result) {
        const auto path=utf8Path(model);
        const auto package=iiLocalDiffusion::inspectEmbeddedImagePackage(path);
        const auto &inspection=package.model;
        const auto &components=package.components;
        text(result,"architecture",architecture(inspection.architecture));
        text(result,"role",role(inspection.role));
        text(result,"encoding",encoding(inspection.encoding));
        json_object_object_add(result,"packaged",json_object_new_boolean(components.packaged));
        text(result,"transformer",components.transformer);
        text(result,"text_encoder",components.textEncoder);
        text(result,"vae",components.vae);
        catalogJson(result,package.catalog);
    });
}
iild_image_package_result_v1 *iild_image_package_catalog_v1(const char *model) {
    return answer([&](json_object *result) {
        const auto catalog=iiLocalDiffusion::inspectEmbeddedImageCatalog(utf8Path(model));
        json_object_object_add(result,"packaged",json_object_new_boolean(catalog.packaged));
        catalogJson(result,catalog);
    });
}
iild_image_package_result_v1 *iild_image_shard_index_inspect_v1(const char *index) {
    return answer([&](json_object *result) {
        const auto report=iild::safetensorsIndexJson(iild::inspectSafetensorsIndex(utf8Path(index)));
        auto *value=json_tokener_parse(report.c_str());
        if (!value) throw std::runtime_error("Cannot encode the shard inspection report.");
        json_object_object_add(result,"index",value);
    });
}
iild_image_package_result_v1 *iild_image_shard_index_paths_v1(const char *index) {
    return answer([&](json_object *result) {
        auto *paths=json_object_new_array();
        json_object_object_add(result,"paths",paths);
        for (const auto &path : iild::safetensorsIndexShardPaths(utf8Path(index))) {
            const auto value=utf8Text(path);
            json_object_array_add(paths,json_object_new_string_len(value.data(),static_cast<int>(value.size())));
        }
    });
}
iild_image_package_result_v1 *iild_image_pytorch_index_inspect_v1(const char *index) {
    return answer([&](json_object *result) {
        const auto report=iild::pyTorchIndexJson(iild::inspectPyTorchIndex(utf8Path(index)));
        auto *value=json_tokener_parse(report.c_str());
        if (!value) throw std::runtime_error("Cannot encode the PyTorch shard inspection report.");
        json_object_object_add(result,"index",value);
    });
}
iild_image_package_result_v1 *iild_image_container_probe_v1(const char *model) {
    return answer([&](json_object *result) {
        text(result,"format",iild::detectModelContainer(utf8Path(model)));
    });
}
iild_image_package_result_v1 *iild_video_wan_directory_inspect_v1(const char *directory) {
    return answer([&](json_object *result) {
        const auto report=iild::wanPipelineDirectoryJson(iild::inspectWanPipelineDirectory(utf8Path(directory)));
        auto *value=json_tokener_parse(report.c_str());
        if (!value) throw std::runtime_error("Cannot encode the Wan directory inspection report.");
        json_object_object_add(result,"directory",value);
    });
}
iild_image_package_result_v1 *iild_image_package_materialize_v1(const char *model,const char *cache,
    const char *relative_component,iild_image_package_progress_v1 progress,void *user) {
    return answer([&](json_object *result) {
        const auto source=utf8Path(model),directory=utf8Path(cache),relative=utf8Path(relative_component);
        std::stop_source stop;
        const auto observer=[&](const iild::ModelPackagingProgress &event) {
            if (progress && progress(event.detail.c_str(),event.completedBytes,event.totalBytes,user)) stop.request_stop();
        };
        const auto path=iild::materializeModelPackageAssetByPath(source,directory,relative,stop.get_token(),observer);
        text(result,"path",utf8Text(path));
    });
}
iild_image_package_result_v1 *iild_image_vae_inspect_v1(const char *model) {
    return answer([&](json_object *result) {
        const auto info=iiLocalDiffusion::inspectNativeVae(utf8Path(model));
        using S=iiLocalDiffusion::NativeVaeState;
        const char *state = info.state == S::Missing ? "missing" : info.state == S::Embedded ? "embedded"
            : info.state == S::Incompatible ? "incompatible" : "unsupported";
        text(result,"state",state);
        text(result,"model_family",info.modelFamily);
        text(result,"vae_family",info.vaeFamily);
    });
}
iild_image_package_result_v1 *iild_image_vae_validate_v1(const char *model,const char *candidate) {
    return answer([&](json_object *result) {
        json_object_object_add(result,"valid",json_object_new_boolean(
            iiLocalDiffusion::validateNativeVae(utf8Path(model),utf8Path(candidate))));
    });
}
iild_image_package_result_v1 *iild_image_text_encoder_inspect_v1(const char *model,const char *slot) {
    return answer([&](json_object *result) {
        if (!slot) throw std::invalid_argument("A text encoder slot is required.");
        const auto state=iiLocalDiffusion::inspectNativeTextEncoder(utf8Path(model),slot);
        using S=iiLocalDiffusion::NativeTextEncoderState;
        text(result,"state",state == S::Missing ? "missing" : state == S::Embedded ? "embedded"
            : state == S::Incompatible ? "incompatible" : "unsupported");
    });
}
iild_image_package_result_v1 *iild_image_text_encoder_validate_v1(const char *model,const char *slot,const char *candidate) {
    return answer([&](json_object *result) {
        if (!slot) throw std::invalid_argument("A text encoder slot is required.");
        json_object_object_add(result,"valid",json_object_new_boolean(
            iiLocalDiffusion::validateNativeTextEncoder(utf8Path(model),slot,utf8Path(candidate))));
    });
}
const char *iild_image_package_json_v1(const iild_image_package_result_v1 *result) {
    return result?result->json.c_str():nullptr;
}
void iild_image_package_free_v1(iild_image_package_result_v1 *result) { delete result; }
}
