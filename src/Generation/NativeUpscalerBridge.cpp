#include "NativeUpscalerBridge.hpp"
#include "NativeUpscaler.hpp"
#include "NativeUpscalerBridgeInternal.hpp"
#include <json-c/json.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

struct iild_upscale_result_v1 {
    iiLocalDiffusion::UpscaleExecution execution;
    std::string error, metadata;
    bool cancelled = false;
};
namespace iiLocalDiffusion::native_detail {
std::unique_ptr<Upscaler> makeBridgeUpscaler(const iild_upscale_request_v1 &request,
    const iild_upscale_request_v2 *extended, const iild_upscale_request_v3 *swin) {
    if (request.size != sizeof(request) || request.kind < 0 || request.kind > 7
        || request.cpu < 0 || request.cpu > 4
        || (extended && (extended->size != sizeof(*extended) || extended->srvgg_activation < 0 || extended->srvgg_activation > 2))
        || (swin && swin->size != sizeof(*swin)))
        throw std::invalid_argument("Invalid upscale request or ABI version.");
    if (extended && request.kind < 5 &&
        ((extended->max_weight_bytes && extended->max_weight_bytes != 1024ULL*1024*1024)
         || (extended->max_working_bytes && extended->max_working_bytes != 512ULL*1024*1024)
         || extended->srvgg_activation != 0))
        throw std::invalid_argument("CPU model settings do not apply to the selected upscaler.");
    if (extended && request.kind < 4 && (request.cpu || request.tile_size != 128 || request.threads != 1))
        throw std::invalid_argument("Model device and tiling settings do not apply to interpolation.");
    if (extended && request.kind == 6 && extended->srvgg_activation != 0)
        throw std::invalid_argument("SRVGG activation does not apply to SwinIR.");
    if (swin && request.kind < 6 && (swin->swinir_image_range != 1.f || swin->swinir_qk_scale != 0.f))
        throw std::invalid_argument("SwinIR settings do not apply to the selected upscaler.");
    if (request.kind == 4) {
        if (!request.model) throw std::invalid_argument("ESRGAN needs a model path.");
        if (request.cpu > 1) throw std::invalid_argument("RRDB does not accept this explicit device selection.");
        return makeEsrganUpscaler({request.model, request.cpu ? UpscaleDevice::Cpu : UpscaleDevice::Automatic,
                                   request.tile_size, request.threads});
    }
    if (request.kind >= 5) {
        if (!request.model) throw std::invalid_argument("A native upscaler model path is required.");
        NativeModelUpscalerOptions model;
        std::u8string utf8Model;
        for (const char byte : std::string_view(request.model))
            utf8Model.push_back(static_cast<char8_t>(static_cast<unsigned char>(byte)));
        model.model=std::filesystem::path(utf8Model);
        model.device=static_cast<UpscaleDevice>(request.cpu);
        model.architecture=request.kind==5 ? UpscaleModelArchitecture::SRVGG :
            request.kind==6 ? UpscaleModelArchitecture::SwinIR : UpscaleModelArchitecture::Automatic;
        model.tileSize=request.tile_size; model.threads=request.threads;
        if (extended) {
            model.maxWeightBytes=extended->max_weight_bytes;
            model.maxWorkingBytes=extended->max_working_bytes;
            model.srvggActivation=static_cast<SrvggActivation>(extended->srvgg_activation);
        }
        if (swin) { model.swinIRImageRange=swin->swinir_image_range; model.swinIRQkScale=swin->swinir_qk_scale; }
        return makeNativeModelUpscaler(std::move(model));
    }
    return makeInterpolationUpscaler(static_cast<UpscaleFilter>(request.kind));
}
UpscaleOptions bridgeUpscaleOptions(const iild_upscale_request_v1 &request) {
    UpscaleOptions options;
    options.scale=request.scale; options.targetWidth=request.target_width; options.targetHeight=request.target_height;
    options.passes=request.passes; options.maxOutputBytes=request.max_output_bytes;
    return options;
}
}
extern "C" {
static iild_upscale_result_v1 *upscale_request(const iild_upscale_request_v1 *request,
                                        const iild_upscale_request_v2 *extended,
                                        const iild_upscale_request_v3 *swin,
                                        iild_upscale_progress_v1 callback, void *user) {
    try {
        auto result = std::make_unique<iild_upscale_result_v1>();
        try {
            if (!request || !request->pixels || !request->width || !request->height
                || (request->channels != 1 && request->channels != 3 && request->channels != 4))
                throw std::invalid_argument("Invalid upscale request or ABI version.");
            const auto w=std::size_t(request->width), h=std::size_t(request->height), c=std::size_t(request->channels);
            if (w > std::numeric_limits<std::size_t>::max()/h || w*h > std::numeric_limits<std::size_t>::max()/c
                || w*h*c != request->pixel_bytes)
                throw std::invalid_argument("Invalid input buffer length.");
            using namespace iiLocalDiffusion;
            auto provider = native_detail::makeBridgeUpscaler(*request, extended, swin);
            auto options = native_detail::bridgeUpscaleOptions(*request);
            UpscaleImage image{request->width,request->height,request->channels,
                               std::vector<std::uint8_t>(request->pixels,request->pixels+request->pixel_bytes)};
            result->execution=upscaleImage(std::move(image),*provider,options,{},[&](int done,int total){
                return !callback || callback(done,total,user)==0;
            });
        } catch (const iiLocalDiffusion::UpscaleCancelled &e) { result->cancelled=true; result->error=e.what(); }
          catch (const std::exception &e) { result->error=e.what(); }
        using Json = std::unique_ptr<json_object,decltype(&json_object_put)>;
        Json metadata(json_object_new_object(),json_object_put);
        if (!metadata) return nullptr;
        auto field=[&](const char *key,json_object *value){ json_object_object_add(metadata.get(),key,value); };
        field("error",json_object_new_string(result->error.c_str()));
        field("cancelled",json_object_new_boolean(result->cancelled));
        field("upscaler",json_object_new_string(result->execution.upscaler.c_str()));
        field("applied_scale",json_object_new_double(result->execution.appliedScale));
        field("width",json_object_new_int64(result->execution.image.width));
        field("height",json_object_new_int64(result->execution.image.height));
        field("channels",json_object_new_int64(result->execution.image.channels));
        auto *passes=json_object_new_array();
        for (const auto &pass : result->execution.passes) {
            auto *entry=json_object_new_object();
            json_object_object_add(entry,"width",json_object_new_int64(pass.width));
            json_object_object_add(entry,"height",json_object_new_int64(pass.height));
            json_object_array_add(passes,entry);
        }
        field("passes",passes);
        auto *details=json_object_new_object();
        for (const auto &[key,value] : result->execution.details)
            json_object_object_add(details,key.c_str(),json_object_new_string(value.c_str()));
        field("details",details);
        result->metadata=json_object_to_json_string_ext(metadata.get(),JSON_C_TO_STRING_PLAIN);
        return result.release();
    } catch (...) { return nullptr; }
}
iild_upscale_result_v1 *iild_upscale_v1(const iild_upscale_request_v1 *request,
                                        iild_upscale_progress_v1 callback, void *user) {
    return upscale_request(request,nullptr,nullptr,callback,user);
}
iild_upscale_result_v1 *iild_upscale_v2(const iild_upscale_request_v2 *request,
                                        iild_upscale_progress_v1 callback, void *user) {
    if (!request || request->size != sizeof(*request)) return upscale_request(nullptr,nullptr,nullptr,callback,user);
    return upscale_request(&request->image,request,nullptr,callback,user);
}
iild_upscale_result_v1 *iild_upscale_v3(const iild_upscale_request_v3 *request,
                                        iild_upscale_progress_v1 callback, void *user) {
    if (!request || request->size != sizeof(*request)) return upscale_request(nullptr,nullptr,nullptr,callback,user);
    return upscale_request(&request->upscaler.image,&request->upscaler,request,callback,user);
}
const char *iild_upscale_metadata_v1(const iild_upscale_result_v1 *result) { return result ? result->metadata.c_str() : nullptr; }
const uint8_t *iild_upscale_pixels_v1(const iild_upscale_result_v1 *result,size_t *size) {
    if (size) *size=result ? result->execution.image.pixels.size() : 0;
    return result && !result->execution.image.pixels.empty() ? result->execution.image.pixels.data() : nullptr;
}
void iild_upscale_free_v1(iild_upscale_result_v1 *result) { delete result; }
}
