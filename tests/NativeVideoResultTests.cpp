#include <Generation/NativeVideo.hpp>
#include <Generation/NativeVideoBridge.hpp>
#include <Generation/NativeCachePolicy.hpp>
#include <Generation/NativeEngineAccess.hpp>
#include <ModelPackaging/ModelPackaging.hpp>
#include <iiFileProviderNative.h>
#include <stable-diffusion.h>
#include <json-c/json.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <cmath>

namespace {
int loaded = 0, destroyed = 0, freed = 0, generated = 0, failure = 0;
bool supports = true, continued = false;
sd_progress_stage_cb_t stageCallback = nullptr;
void *stageUser = nullptr;
sd_graph_eval_callback_t evalCallback = nullptr;
void *evalUser = nullptr;
sd_log_cb_t logCallback = nullptr;
void *logUser = nullptr;
std::string backend, paramsBackend, encoder, t5Encoder, vae, highNoise, vision, effectiveModel, diffusionModel;
float expertBoundary = 0, expertGuidance = 0;
int expertSteps = 0;
bool prefetchDisabled = false, tiledDecode = false;
std::timed_mutex engineMutex;
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
std::string safetensors(const char *name, const std::string &payload) {
    auto *header = json_object_new_object();
    auto *tensor = json_object_new_object();
    auto *shape = json_object_new_array();
    auto *offsets = json_object_new_array();
    json_object_array_add(shape, json_object_new_int64(static_cast<int64_t>(payload.size())));
    json_object_array_add(offsets, json_object_new_int(0));
    json_object_array_add(offsets, json_object_new_int64(static_cast<int64_t>(payload.size())));
    json_object_object_add(tensor, "dtype", json_object_new_string("U8"));
    json_object_object_add(tensor, "shape", shape);
    json_object_object_add(tensor, "data_offsets", offsets);
    json_object_object_add(header, name, tensor);
    const std::string encoded = json_object_to_json_string_ext(header, JSON_C_TO_STRING_PLAIN);
    json_object_put(header);
    std::string result(8, '\0');
    for (unsigned i = 0; i < 8; ++i) result[i] = static_cast<char>((encoded.size() >> (8 * i)) & 255);
    return result + encoded + payload;
}
std::string safetensorsMany(const std::vector<std::string> &names,const char *dtype="U8") {
    auto *header=json_object_new_object();
    std::size_t offset=0;
    for (const auto &name:names) {
        auto *tensor=json_object_new_object(), *shape=json_object_new_array(), *offsets=json_object_new_array();
        json_object_array_add(shape,json_object_new_int(1));
        json_object_array_add(offsets,json_object_new_int64(static_cast<int64_t>(offset)));
        json_object_array_add(offsets,json_object_new_int64(static_cast<int64_t>(offset+1)));
        json_object_object_add(tensor,"dtype",json_object_new_string(dtype));
        json_object_object_add(tensor,"shape",shape);json_object_object_add(tensor,"data_offsets",offsets);
        json_object_object_add(header,name.c_str(),tensor);++offset;
    }
    const std::string encoded=json_object_to_json_string_ext(header,JSON_C_TO_STRING_PLAIN);
    json_object_put(header);
    std::string prefix(8,'\0');
    for (unsigned i=0;i<8;++i) prefix[i]=static_cast<char>((encoded.size()>>(8*i))&255);
    return prefix+encoded+std::string(names.size(),'x');
}
std::string ggufFixture(const std::vector<std::string> &names={"token_embd.weight"}) {
    std::string result = "GGUF";
    const auto integer = [&](std::uint64_t value, unsigned count) {
        for (unsigned i = 0; i < count; ++i) result.push_back(static_cast<char>((value >> (8 * i)) & 255));
    };
    const auto text = [&](const std::string &value) { integer(value.size(), 8); result += value; };
    integer(3, 4); integer(names.size(), 8); integer(0, 8);
    for (std::size_t i=0;i<names.size();++i) {
        text(names[i]); integer(1, 4); integer(1, 8); integer(0, 4); integer(i*32, 8);
    }
    result.append((32 - result.size() % 32) % 32, '\0');
    result.append((names.size()-1)*32,'\0');integer(0x3e800000, 4);
    return result;
}
}
struct sd_ctx_t {};
extern "C" {
void sd_ctx_params_init(sd_ctx_params_t *value) { *value = {}; }
void sd_vid_gen_params_init(sd_vid_gen_params_t *value) { *value = {}; }
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *value) {
    ++loaded; backend = value->backend ? value->backend : "";
    effectiveModel = value->model_path ? value->model_path : "";
    diffusionModel = value->diffusion_model_path ? value->diffusion_model_path : "";
    paramsBackend = value->params_backend ? value->params_backend : "";
    prefetchDisabled = value->disable_prefetch;
    encoder = value->llm_path ? value->llm_path : "";
    t5Encoder = value->t5xxl_path ? value->t5xxl_path : "";
    vae = value->vae_path ? value->vae_path : "";
    highNoise = value->high_noise_diffusion_model_path ? value->high_noise_diffusion_model_path : "";
    vision = value->clip_vision_path ? value->clip_vision_path : "";
    return new sd_ctx_t;
}
void free_sd_ctx(sd_ctx_t *value) { ++destroyed; delete value; }
bool sd_ctx_supports_video_generation(const sd_ctx_t *) { return supports; }
int32_t sd_get_num_physical_cores() { return 2; }
void sd_set_abort_callback(sd_abort_cb_t, void *) {}
void sd_set_log_callback(sd_log_cb_t callback, void *data) { logCallback = callback; logUser = data; }
void sd_set_progress_stage_callback(sd_progress_stage_cb_t callback, void *data) { stageCallback = callback; stageUser = data; }
void sd_set_backend_eval_callback(sd_graph_eval_callback_t callback, void *data) { evalCallback = callback; evalUser = data; }
sd_type_t str_to_sd_type(const char *type) {
    return std::string(type) == "f16" ? SD_TYPE_F16 : std::string(type) == "bf16" ? SD_TYPE_BF16 : SD_TYPE_F32;
}
bool generate_video(sd_ctx_t *, const sd_vid_gen_params_t *parameters, sd_image_t **output, int *count, sd_audio_t **) {
    ++generated;
    tiledDecode = parameters->vae_tiling_params.enabled;
    expertBoundary = parameters->moe_boundary;
    expertSteps = parameters->high_noise_sample_params.sample_steps;
    expertGuidance = parameters->high_noise_sample_params.guidance.txt_cfg;
    if (generated > 1 && parameters->init_image.data) continued = parameters->init_image.data[0] == 42;
    require(parameters->sample_params.sample_steps == 2, "Sampling steps must reach native engine");
    require(parameters->sample_params.guidance.txt_cfg == 3, "Guidance must reach native engine");
    if (evalCallback) evalCallback(nullptr, false, evalUser);
    if (stageCallback) stageCallback(SD_PROGRESS_SAMPLE, 1, 2, 0, stageUser);
    if (logCallback && failure == 1) {
        const std::string oversized(3000, 'w');
        logCallback(SD_LOG_WARN, oversized.c_str(), logUser);
    }
    if (logCallback && failure != 4) logCallback(SD_LOG_WARN, "model manager cannot make enough memory available on Metal", logUser);
    if (failure == 3 && backend != "cpu") return false;
    if (failure == 4 || failure == 5) return false;
    if (failure == 1) {
        if (logCallback) {
            const std::string oversized(3000, 'e');
            logCallback(SD_LOG_ERROR, oversized.c_str(), logUser);
            logCallback(SD_LOG_ERROR, "segment weight preparation failed", logUser);
        }
        return false;
    }
    *count = parameters->video_frames;
    *output = static_cast<sd_image_t *>(std::calloc(static_cast<std::size_t>(*count), sizeof(sd_image_t)));
    for (int i = 0; i < *count; ++i) {
        auto &frame = (*output)[i];
        frame.width = static_cast<unsigned>(parameters->width); frame.height = static_cast<unsigned>(parameters->height);
        frame.channel = failure == 2 ? 4 : 3;
        frame.data = static_cast<std::uint8_t *>(std::malloc(static_cast<std::size_t>(frame.width) * frame.height * 3));
        std::memset(frame.data, 42, static_cast<std::size_t>(frame.width) * frame.height * 3);
    }
    return true;
}
void free_sd_images(sd_image_t *images, int count) {
    ++freed;
    for (int i = 0; i < count; ++i) std::free(images[i].data);
    std::free(images);
}
}
namespace iiLocalDiffusion::native_detail {
std::timed_mutex &nativeEngineMutex() { return engineMutex; }
void nativeEngineOnThread(bool) {}
void releaseImageCacheForVideoLocked() {}
ResourceLimits resourceLimits() { return {8ull << 30, 4ull << 30, 6ull << 30, true}; }
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        namespace fs = std::filesystem;
        using iiFileProvider::native::File;
        const auto root = fs::absolute(argv[1]); File::createDirectories(root);
        const auto model = root / "model.safetensors", textEncoder = root / "encoder.gguf";
        File::write(model, "fixture model"); File::write(textEncoder, "fixture encoder");
        iiLocalDiffusion::NativeVideoRequest request;
        request.modelPath = model; request.textEncoderPath = textEncoder;
        request.width = request.height = 32; request.steps = 2; request.cpu = true;
        request.shots = {{"First shot", "noise", 1, 9, 24}, {"Second shot", "noise", 2, 9, 24}};
        request.shots[1].continuePrevious = true;
        std::atomic_bool cancelled{false}; int delivered = 0;
        const auto observer = [&](std::size_t, int, int width, int height, std::span<const std::uint8_t> rgb) {
            require(width == 32 && height == 32 && rgb.size() == 32 * 32 * 3 && rgb.front() == 42, "RGB frame callback contract"); ++delivered;
        };
        std::vector<int> batches;
        auto result = iiLocalDiffusion::generateNativeVideo(request, cancelled, [&](const auto &event) {
            if (event.stage == iiLocalDiffusion::NativeGenerationStage::Computing) batches.push_back(event.step);
        }, observer);
        require(result.error.empty() && result.frameCounts == std::vector<int>{9, 9}, "All shots must complete");
        require(delivered == 18 && loaded == 1 && destroyed == 1 && freed == 2, "One context per batch; every output is freed");
        require(continued && backend == "cpu" && paramsBackend == "cpu" && encoder == textEncoder.string(),
            "Continuity, CPU and explicit encoder bindings");
        require(t5Encoder.empty(), "The LTX encoder must not be sent to the T5 slot");
        require(batches == std::vector<int>{1, 2} && !evalCallback, "Only completed graph batches report progress; callback is cleared");
        const auto legacy = root / "legacy.onnx";
        File::write(legacy, "legacy model");
        auto rejected = request; rejected.modelPath = legacy;
        result = iiLocalDiffusion::generateNativeVideo(rejected, cancelled, {}, observer);
        require(result.error.find("ONNX") != std::string::npos && loaded == 1,
            "Video model format must be rejected before native execution");
        rejected = request; rejected.textEncoderPath = legacy;
        result = iiLocalDiffusion::generateNativeVideo(rejected, cancelled, {}, observer);
        require(result.error.find("ONNX") != std::string::npos && loaded == 1,
            "Text encoder format must be rejected before native execution");
        const auto upper = root / "legacy.ONNX";
        File::write(upper, "legacy model");
        rejected = request; rejected.textEncoderPath = upper;
        result = iiLocalDiffusion::generateNativeVideo(rejected, cancelled, {}, observer);
        require(result.error.find("ONNX") != std::string::npos && loaded == 1,
            "ONNX rejection must be case insensitive");
        const auto wanFile=root/"wan-checkpoint.safetensors";
        File::write(wanFile,safetensorsMany({"model.diffusion_model.patch_embedding.weight",
            "model.diffusion_model.blocks.0.cross_attn.norm_k.weight"}));
        auto wan=request;wan.modelPath=wanFile;wan.shots.resize(1);
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && encoder.empty() && t5Encoder==textEncoder.string() && diffusionModel.empty(),
            "Wan's explicit text encoder must bind to T5 rather than the LTX LLM slot");
        const std::vector<std::string> standaloneDiffusersNames{"patch_embedding.weight", "blocks.0.attn1.to_q.weight",
            "blocks.0.attn2.norm_k.weight", "condition_embedder.time_embedder.linear_1.weight"};
        const auto standaloneDiffusers=root/"standalone-diffusers.safetensors";
        for (const std::string prefix : {"", "model.diffusion_model.", "diffusion_model.", "transformer."}) {
            std::vector<std::string> names;
            for (const auto &name:standaloneDiffusersNames) names.push_back(prefix+name);
            File::write(standaloneDiffusers,safetensorsMany(names));
            wan.modelPath=standaloneDiffusers;
            result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
            require(result.error.empty() && t5Encoder==textEncoder.string() && diffusionModel==standaloneDiffusers.string(),
                "Standalone Diffusers Wan weights must reach the native diffusion namespace");
        }
        File::write(standaloneDiffusers,safetensorsMany(standaloneDiffusersNames,"FUTURE8"));
        wan.modelPath=standaloneDiffusers;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && diffusionModel==standaloneDiffusers.string(),
            "Metadata discovery must not preempt the engine's decision about an unknown dtype");
        const auto diffusersIndexDir=root/"diffusers-index";
        File::createDirectories(diffusersIndexDir);
        File::write(diffusersIndexDir/"first.safetensors",safetensorsMany({standaloneDiffusersNames[0],standaloneDiffusersNames[1]}));
        File::write(diffusersIndexDir/"second.safetensors",safetensorsMany({standaloneDiffusersNames[2],standaloneDiffusersNames[3]}));
        const auto standaloneDiffusersIndex=diffusersIndexDir/"model.safetensors.index.json";
        File::write(standaloneDiffusersIndex,R"({"weight_map":{"patch_embedding.weight":"first.safetensors","blocks.0.attn1.to_q.weight":"first.safetensors","blocks.0.attn2.norm_k.weight":"second.safetensors","condition_embedder.time_embedder.linear_1.weight":"second.safetensors"}})");
        wan.modelPath=standaloneDiffusersIndex;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && t5Encoder==textEncoder.string() && diffusionModel==standaloneDiffusersIndex.string(),
            "Sharded Diffusers Wan weights must reach the native diffusion namespace");
        const auto linkedSnapshot=root/"hf-video-snapshot",linkedBlobs=root/"hf-video-blobs";
        File::createDirectories(linkedSnapshot);File::createDirectories(linkedBlobs);
        const auto linkedIndex=linkedSnapshot/"model.safetensors.index.json",indexBlob=linkedBlobs/"index-hash";
        File::write(indexBlob,File::read(standaloneDiffusersIndex,65536));
        File::write(linkedBlobs/"first-hash",File::read(diffusersIndexDir/"first.safetensors",65536));
        File::write(linkedBlobs/"second-hash",File::read(diffusersIndexDir/"second.safetensors",65536));
        std::error_code symlinkError;
        fs::create_symlink(indexBlob,linkedIndex,symlinkError);
        if (!symlinkError) {
            fs::create_symlink(linkedBlobs/"first-hash",linkedSnapshot/"first.safetensors");
            fs::create_symlink(linkedBlobs/"second-hash",linkedSnapshot/"second.safetensors");
            wan.modelPath=linkedIndex;
            result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
            require(result.error.empty() && effectiveModel==linkedIndex.string() && diffusionModel==linkedIndex.string(),
                "Linked video index must reach the native loader with its logical shard base");
            const auto replacement=linkedBlobs/"replacement-index";
            File::write(replacement,File::read(indexBlob,65536));
            bool changedLink=false;
            result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},
                [&](std::size_t shot,int number,int width,int height,auto rgb) {
                    observer(shot,number,width,height,rgb);
                    if (!changedLink) {fs::remove(linkedIndex);fs::create_symlink(replacement,linkedIndex);changedLink=true;}
                });
            require(changedLink && result.error.find("changed")!=std::string::npos,
                "Retargeting a video index during generation must invalidate its frames");
        }
        const auto standaloneDiffusersGguf=root/"standalone-diffusers.gguf";
        const auto diffusersGgufBytes=ggufFixture(standaloneDiffusersNames);
        File::write(standaloneDiffusersGguf,diffusersGgufBytes);
        wan.modelPath=standaloneDiffusersGguf;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && t5Encoder==textEncoder.string() && diffusionModel==standaloneDiffusersGguf.string(),
            "Standalone Diffusers Wan GGUF weights must reach the native diffusion namespace");
        const auto wrappedDiffusersInput=root/"wrapped-diffusers-input";
        File::createDirectories(wrappedDiffusersInput);
        File::write(wrappedDiffusersInput/"model.gguf",diffusersGgufBytes);
        File::write(wrappedDiffusersInput/"text_encoder.gguf",ggufFixture());
        const auto wrappedDiffusers=root/"wrapped-diffusers.safetensors";
        const std::vector<iild::ModelPackagingAssignment> wrappedChoices{{"model.gguf","model","main"}};
        (void)iild::createModelPackageWithAssignments(wrappedDiffusersInput,wrappedDiffusers,wrappedChoices);
        wan.modelPath=wrappedDiffusers;wan.textEncoderPath.clear();wan.assetCacheDirectory=root/"wrapped-diffusers-cache";
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && !diffusionModel.empty() && diffusionModel==effectiveModel &&
            File::read(effectiveModel)==diffusersGgufBytes && !t5Encoder.empty(),
            "A wrapped Diffusers Wan GGUF must be restored and mounted in the native diffusion namespace");
        wan.textEncoderPath=textEncoder;
        wan.modelPath=wanFile;
        const auto vaeFile=root/"video-vae.safetensors", highFile=root/"high.safetensors", visionFile=root/"vision.safetensors";
        File::write(vaeFile,safetensors("vae.fixture","a"));
        File::write(highFile,safetensors("high.fixture","b"));
        File::write(visionFile,safetensors("vision.fixture","c"));
        iiLocalDiffusion::NativeVideoComponents components;
        components.vaePath=vaeFile;components.highNoiseModelPath=highFile;components.visionEncoderPath=visionFile;
        components.moeBoundary=0.75f;components.highNoiseSteps=7;components.highNoiseGuidance=3.5f;
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,components,cancelled,{},observer);
        require(result.error.empty() && vae==vaeFile.string() && highNoise==highFile.string() && vision==visionFile.string(),
            "Explicit Wan components must reach their native engine slots");
        require(expertBoundary==0.75f && expertSteps==7 && expertGuidance==3.5f,
            "High-noise sampling controls must reach native video generation");
        auto invalidComponents=components;invalidComponents.highNoiseModelPath.clear();
        const auto beforeBadComponents=loaded;
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,invalidComponents,cancelled,{},observer);
        require(!result.error.empty() && loaded==beforeBadComponents,"Expert controls require the expert model");
        result=iiLocalDiffusion::generateNativeVideoWithComponents(request,components,cancelled,{},observer);
        require(!result.error.empty() && loaded==beforeBadComponents,"Wan expert controls reject non-Wan checkpoints");
        invalidComponents=components;invalidComponents.vaePath=legacy;
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,invalidComponents,cancelled,{},observer);
        require(result.error.find("ONNX")!=std::string::npos && loaded==beforeBadComponents,
            "ONNX components are rejected before model load");
        for (float bad : {-2.0f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
            invalidComponents=components;invalidComponents.moeBoundary=bad;
            result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,invalidComponents,cancelled,{},observer);
            require(!result.error.empty() && loaded==beforeBadComponents,"Invalid expert controls fail before model load");
        }
        bool componentModified=false;
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,components,cancelled,{},
            [&](std::size_t shot,int number,int width,int height,auto rgb) {
                observer(shot,number,width,height,rgb);
                if(!componentModified) { File::write(highFile,safetensors("high.fixture","changed"));componentModified=true; }
            });
        require(componentModified && result.error.find("changed")!=std::string::npos && loaded==destroyed,
            "Changed component weights must invalidate the result");
        File::write(highFile,safetensors("high.fixture","b"));
        const auto componentDir=root/"component-shards";
        File::createDirectories(componentDir);
        const auto componentShard=componentDir/"part.safetensors", componentIndex=componentDir/"vae.safetensors.index.json";
        File::write(componentShard,safetensors("vae.fixture","one"));
        File::write(componentIndex,R"({"weight_map":{"vae.fixture":"part.safetensors"}})");
        auto shardedComponents=components;shardedComponents.vaePath=componentIndex;
        componentModified=false;
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,shardedComponents,cancelled,{},
            [&](std::size_t shot,int number,int width,int height,auto rgb) {
                observer(shot,number,width,height,rgb);
                if(!componentModified) {File::write(componentShard,safetensors("vae.fixture","new"));componentModified=true;}
            });
        require(componentModified && result.error.find("changed")!=std::string::npos,
            "A modified component shard invalidates the result");
        const auto componentSnapshot=root/"linked-component-snapshot";
        File::createDirectories(componentSnapshot);
        const auto linkedComponentIndex=componentSnapshot/"vae.safetensors.index.json";
        std::error_code componentLinkError;
        fs::create_symlink(componentIndex,linkedComponentIndex,componentLinkError);
        if (!componentLinkError) {
            const auto linkedComponentShard=componentSnapshot/"part.safetensors";
            fs::create_symlink(componentShard,linkedComponentShard);
            shardedComponents.vaePath=linkedComponentIndex;
            wan.textEncoderPath=linkedComponentIndex;
            result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,shardedComponents,cancelled,{},observer);
            require(result.error.empty() && vae==linkedComponentIndex.string() && t5Encoder==linkedComponentIndex.string(),
                "Linked video VAE and text-encoder indexes must retain their logical shard bases");
            const auto replacementShard=componentDir/"replacement.safetensors";
            File::write(replacementShard,File::read(componentShard,65536));
            componentModified=false;
            result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,shardedComponents,cancelled,{},
                [&](std::size_t shot,int number,int width,int height,auto rgb) {
                    observer(shot,number,width,height,rgb);
                    if (!componentModified) {
                        fs::remove(linkedComponentShard);fs::create_symlink(replacementShard,linkedComponentShard);
                        componentModified=true;
                    }
                });
            require(componentModified && result.error.find("changed")!=std::string::npos,
                "Retargeting a video component shard must invalidate generated frames");
            wan.textEncoderPath=textEncoder;
        }
        const auto embeddedWan=root/"wan-with-t5.safetensors";
        File::write(embeddedWan,safetensorsMany({"patch_embedding.weight", "blocks.0.cross_attn.norm_k.weight",
            "text_encoders.t5xxl.transformer.shared.weight",
            "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight",
            "text_encoders.t5xxl.transformer.encoder.final_layer_norm.weight"}));
        wan.modelPath=embeddedWan;wan.textEncoderPath.clear();
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && encoder.empty() && t5Encoder.empty(),
            "Wan checkpoints containing T5 tensors must not require an external encoder");
        wan.textEncoderPath=textEncoder;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && encoder.empty() && t5Encoder==textEncoder.string(),
            "An explicit Wan encoder must retain precedence over embedded tensors");
        File::write(embeddedWan,safetensorsMany({"patch_embedding.weight", "blocks.0.cross_attn.norm_k.weight",
            "text_encoders.t5xxl.transformer.token_embd.weight",
            "text_encoders.t5xxl.transformer.enc.output_norm.weight",
            "text_encoders.t5xxl.transformer.enc.blk.0.attn_q.weight"}));
        wan.textEncoderPath.clear();
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && encoder.empty() && t5Encoder.empty(),
            "Wan GGUF-style T5 tensor names must retain their native conversion path");
        File::write(embeddedWan,safetensorsMany({"patch_embedding.weight", "blocks.0.cross_attn.norm_k.weight",
            "text_encoders.t5xxl.transformer.shared.weight"}));
        const auto beforeIncompleteWan=loaded;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(!result.error.empty() && loaded==beforeIncompleteWan,
            "Partial Wan T5 metadata must not bypass the missing encoder check");
        const auto wanInput=root/"wan-package-input";
        File::createDirectories(wanInput);
        File::write(wanInput/"model.safetensors",safetensorsMany({"model.diffusion_model.patch_embedding.weight",
            "model.diffusion_model.blocks.0.cross_attn.norm_k.weight",
            "text_encoders.t5xxl.transformer.shared.weight",
            "text_encoders.t5xxl.transformer.encoder.final_layer_norm.weight",
            "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight"}));
        const auto wanEncoderBytes=ggufFixture();
        File::write(wanInput/"text_encoder.gguf",wanEncoderBytes);
        const auto wanPackage=root/("wan-package-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".safetensors");
        (void)iild::createModelPackage(wanInput,wanPackage);
        wan.modelPath=wanPackage;wan.assetCacheDirectory=root/"wan-assets";
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && encoder.empty() && !t5Encoder.empty() && File::read(t5Encoder)==wanEncoderBytes && vae.empty(),
            "A packaged Wan encoder must be verified and bound to T5 even if flat markers are present");
        const auto packageVaeBytes=safetensors("decoder.weight","wan-vae");
        File::write(wanInput/"video_vae.safetensors",packageVaeBytes);
        const auto wanVaePackage=root/("wan-vae-package-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".safetensors");
        (void)iild::createModelPackage(wanInput,wanVaePackage);
        wan.modelPath=wanVaePackage;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && !vae.empty() && File::read(vae)==packageVaeBytes,
            "An included Wan VAE must be verified, materialized and bound automatically");
        auto explicitVae=iiLocalDiffusion::NativeVideoComponents{};explicitVae.vaePath=vaeFile;
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,explicitVae,cancelled,{},observer);
        require(result.error.empty() && vae==vaeFile.string(),"An explicit VAE must take precedence over the packaged VAE");
        const auto expertInput=root/"wan-expert-input";
        File::createDirectories(expertInput);
        fs::copy_file(wanInput/"model.safetensors",expertInput/"model.safetensors",fs::copy_options::overwrite_existing);
        File::write(expertInput/"text_encoder.gguf",wanEncoderBytes);
        const auto expertBytes=safetensors("patch_embedding.weight","high");
        const auto visionBytes=safetensors("vision.weight","vision");
        File::write(expertInput/"expert.safetensors",expertBytes);
        File::write(expertInput/"vision.safetensors",visionBytes);
        const std::vector<iild::ModelPackagingAssignment> expertChoices{
            {"model.safetensors","model","main"},{"expert.safetensors","high_noise_model","component"},
            {"vision.safetensors","vision_encoder","component"}};
        const auto expertPackage=root/"wan-experts.safetensors";
        (void)iild::createModelPackageWithAssignments(expertInput,expertPackage,expertChoices);
        wan.modelPath=expertPackage;
        auto packageControls=iiLocalDiffusion::NativeVideoComponents{};packageControls.moeBoundary=0.75f;
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,packageControls,cancelled,{},observer);
        require(result.error.empty() && !highNoise.empty() && File::read(highNoise)==expertBytes &&
            !vision.empty() && File::read(vision)==visionBytes && expertBoundary==0.75f,
            "Packaged Wan experts and vision encoder must reach the native engine");
        packageControls.highNoiseModelPath=highFile;packageControls.visionEncoderPath=visionFile;
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,packageControls,cancelled,{},observer);
        require(result.error.empty() && highNoise==highFile.string() && vision==visionFile.string(),
            "Explicit high-noise and vision components override package roles");
        File::write(expertInput/"another.safetensors",safetensors("other.weight","different"));
        auto ambiguousChoices=expertChoices;
        ambiguousChoices.push_back({"another.safetensors","high_noise_model","component"});
        const auto ambiguousExpertPackage=root/"wan-experts-ambiguous.safetensors";
        (void)iild::createModelPackageWithAssignments(expertInput,ambiguousExpertPackage,ambiguousChoices);
        wan.modelPath=ambiguousExpertPackage;
        const auto beforeAmbiguousExpert=loaded;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.find("Multiple")!=std::string::npos && loaded==beforeAmbiguousExpert,
            "Multiple packaged experts must not be selected arbitrarily");
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,packageControls,cancelled,{},observer);
        require(result.error.empty() && highNoise==highFile.string(),
            "An explicit expert resolves package ambiguity");
        const auto diffusersInput=root/"diffusers-wan-input";
        File::createDirectories(diffusersInput/"transformer");
        File::createDirectories(diffusersInput/"transformer_2");
        const std::vector<std::string> diffusersNames{"patch_embedding.weight", "blocks.0.attn1.to_q.weight",
            "blocks.0.attn2.norm_k.weight", "condition_embedder.time_embedder.linear_1.weight"};
        const auto diffusersBytes=safetensorsMany(diffusersNames);
        File::write(diffusersInput/"transformer"/"weights.safetensors",diffusersBytes);
        File::write(diffusersInput/"transformer_2"/"weights.safetensors",diffusersBytes);
        File::write(diffusersInput/"text_encoder.gguf",wanEncoderBytes);
        File::write(diffusersInput/"model_index.json",R"({"_class_name":"WanPipeline","boundary_ratio":0.73,"transformer":["diffusers","WanTransformer3DModel"],"transformer_2":["diffusers","WanTransformer3DModel"]})");
        const auto diffusersPackage=root/"wan-diffusers-package.safetensors";
        (void)iild::createModelPackage(diffusersInput,diffusersPackage);
        wan.modelPath=diffusersPackage;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && effectiveModel==diffusersPackage.string() && !highNoise.empty() &&
            File::read(highNoise)==diffusersBytes && !t5Encoder.empty() && std::abs(expertBoundary-0.73f)<0.00001f,
            "A Diffusers Wan package must route its normalized primary, expert, and text encoder");
        packageControls.moeBoundary=0.5f;
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,packageControls,cancelled,{},observer);
        require(result.error.empty() && expertBoundary==0.5f,"Explicit expert boundary must override the packaged value");
        wan.modelPath=wanVaePackage;
        const auto ggufPackageInput=root/"wan-gguf-package-input";
        File::createDirectories(ggufPackageInput);
        const auto wanGgufBytes=ggufFixture({"patch_embedding.weight","blocks.0.cross_attn.norm_k.weight"});
        File::write(ggufPackageInput/"model.gguf",wanGgufBytes);
        File::write(ggufPackageInput/"text_encoder.gguf",wanEncoderBytes);
        File::write(ggufPackageInput/"video_vae.safetensors",packageVaeBytes);
        const auto ggufWanPackage=root/("wan-gguf-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".safetensors");
        const std::vector<iild::ModelPackagingAssignment> ggufAssignment{{"model.gguf","model","main"}};
        (void)iild::createModelPackageWithAssignments(ggufPackageInput,ggufWanPackage,ggufAssignment);
        wan.modelPath=ggufWanPackage;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && fs::path(effectiveModel).extension()==".gguf" && File::read(effectiveModel)==wanGgufBytes &&
            diffusionModel==effectiveModel &&
            !t5Encoder.empty() && File::read(t5Encoder)==wanEncoderBytes && !vae.empty() && File::read(vae)==packageVaeBytes,
            "A package's GGUF primary model, text encoder and VAE must reach their native slots");
        const auto bareWanPath=root/"bare-wan.gguf";File::write(bareWanPath,wanGgufBytes);
        auto bareWanRequest=request;bareWanRequest.modelPath=bareWanPath;bareWanRequest.shots.resize(1);
        result=iiLocalDiffusion::generateNativeVideo(bareWanRequest,cancelled,{},observer);
        require(result.error.empty() && effectiveModel==bareWanPath.string() && diffusionModel==effectiveModel,
            "A standalone bare Wan GGUF must be mounted in the diffusion namespace");
        auto ggufNoCache=wan;ggufNoCache.textEncoderPath=textEncoder;ggufNoCache.assetCacheDirectory.clear();
        const auto beforeGgufNoCache=loaded;
        result=iiLocalDiffusion::generateNativeVideo(ggufNoCache,cancelled,{},observer);
        require(result.error.find("cache")!=std::string::npos && loaded==beforeGgufNoCache,
            "A packaged main GGUF requires caller-owned cache before loading");
        const auto selfContainedInput=root/"wan-gguf-embedded-encoder";
        File::createDirectories(selfContainedInput);
        const auto embeddedGguf=ggufFixture({"patch_embedding.weight","blocks.0.cross_attn.norm_k.weight",
            "text_encoders.t5xxl.transformer.shared.weight",
            "text_encoders.t5xxl.transformer.encoder.final_layer_norm.weight",
            "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight"});
        File::write(selfContainedInput/"model.gguf",embeddedGguf);
        File::write(selfContainedInput/"video_vae.safetensors",packageVaeBytes);
        const auto selfContainedPackage=root/("wan-gguf-embedded-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".safetensors");
        (void)iild::createModelPackageWithAssignments(selfContainedInput,selfContainedPackage,ggufAssignment);
        wan.modelPath=selfContainedPackage;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && t5Encoder.empty() && diffusionModel==effectiveModel && File::read(effectiveModel)==embeddedGguf,
            "A complete encoder in the packaged main GGUF must not require a separate encoder");
        auto ggufExplicit=wan;ggufExplicit.textEncoderPath=textEncoder;
        result=iiLocalDiffusion::generateNativeVideo(ggufExplicit,cancelled,{},observer);
        require(result.error.empty() && t5Encoder==textEncoder.string(),"An explicit encoder takes precedence over the main GGUF encoder");
        File::write(selfContainedInput/"text_encoder.gguf",wanEncoderBytes);
        const auto separateEncoderPackage=root/("wan-gguf-separate-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".safetensors");
        (void)iild::createModelPackageWithAssignments(selfContainedInput,separateEncoderPackage,ggufAssignment);
        wan.modelPath=separateEncoderPackage;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && !t5Encoder.empty() && File::read(t5Encoder)==wanEncoderBytes,
            "A separate packaged encoder takes precedence over an encoder embedded in the GGUF main model");
        const auto corruptGgufPackage=root/"corrupt-wan-gguf.safetensors";
        auto corruptBytes=File::read(ggufWanPackage);const auto embeddedOffset=corruptBytes.find(wanGgufBytes);
        require(embeddedOffset!=std::string::npos,"The synthetic GGUF payload was not found");
        corruptBytes[embeddedOffset+wanGgufBytes.size()-1]^=1;
        File::write(corruptGgufPackage,corruptBytes);
        auto corrupted=wan;corrupted.modelPath=corruptGgufPackage;corrupted.assetCacheDirectory=root/"corrupt-assets";
        const auto beforeCorrupt=loaded;
        result=iiLocalDiffusion::generateNativeVideo(corrupted,cancelled,{},observer);
        require(result.error.find("checksum")!=std::string::npos && loaded==beforeCorrupt,
            "A corrupt packaged GGUF payload must fail checksum verification before model load");
        auto changedCache=wan;changedCache.modelPath=ggufWanPackage;changedCache.assetCacheDirectory=root/"changing-assets";
        bool changedModelCache=false;
        result=iiLocalDiffusion::generateNativeVideo(changedCache,cancelled,{},[&](std::size_t shot,int number,int width,int height,auto rgb) {
            observer(shot,number,width,height,rgb);
            if(!changedModelCache) {File::write(effectiveModel,"changed");changedModelCache=true;}
        });
        require(changedModelCache && result.error.find("changed")!=std::string::npos,
            "Changing a restored main GGUF during generation must invalidate results");
        const auto changedGgufPackage=root/"changed-wan-gguf.safetensors";
        fs::copy_file(ggufWanPackage,changedGgufPackage,fs::copy_options::overwrite_existing);
        wan.modelPath=changedGgufPackage;bool changedGguf=false;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},[&](std::size_t shot,int number,int width,int height,auto rgb) {
            observer(shot,number,width,height,rgb);
            if(!changedGguf) {File::write(changedGgufPackage,"changed");changedGguf=true;}
        });
        require(changedGguf && result.error.find("changed")!=std::string::npos,
            "Changing the source package during generation must invalidate results");
        wan.modelPath=wanVaePackage;
        auto withoutCache=wan;withoutCache.textEncoderPath=textEncoder;withoutCache.assetCacheDirectory.clear();
        const auto beforeMissingCache=loaded;
        result=iiLocalDiffusion::generateNativeVideo(withoutCache,cancelled,{},observer);
        require(result.error.find("cache")!=std::string::npos && loaded==beforeMissingCache,
            "A packaged VAE without a caller-owned cache must fail before loading");
        result=iiLocalDiffusion::generateNativeVideoWithComponents(withoutCache,explicitVae,cancelled,{},observer);
        require(result.error.empty() && vae==vaeFile.string(),"An explicit VAE does not require package extraction");
        auto cancelVae=wan;cancelVae.textEncoderPath=textEncoder;
        const auto beforeVaeCancel=loaded;
        result=iiLocalDiffusion::generateNativeVideo(cancelVae,cancelled,[&](const auto &event) {
            if (event.stage==iiLocalDiffusion::NativeGenerationStage::Preparing) cancelled=true;
        },observer);
        require(result.cancelled && result.error.empty() && loaded==beforeVaeCancel,
            "Canceling packaged VAE preparation must not start the engine");
        cancelled=false;
        File::write(wanInput/"second_vae.safetensors",safetensors("decoder.weight","different"));
        const auto ambiguousWan=root/("ambiguous-wan-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".safetensors");
        (void)iild::createModelPackage(wanInput,ambiguousWan);
        wan.modelPath=ambiguousWan;
        const auto beforeAmbiguous=loaded;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.find("Multiple")!=std::string::npos && loaded==beforeAmbiguous,
            "Multiple packaged VAEs must not be selected arbitrarily");
        result=iiLocalDiffusion::generateNativeVideoWithComponents(wan,explicitVae,cancelled,{},observer);
        require(result.error.empty() && vae==vaeFile.string(),"An explicit VAE resolves package ambiguity");
        const auto duplicateInput=root/"wan-embedded-vae";
        File::createDirectories(duplicateInput);
        File::write(duplicateInput/"model.safetensors",safetensorsMany({"model.diffusion_model.patch_embedding.weight",
            "model.diffusion_model.blocks.0.cross_attn.norm_k.weight","vae.decoder.weight"}));
        File::write(duplicateInput/"text_encoder.gguf",wanEncoderBytes);
        File::write(duplicateInput/"video_vae.safetensors",safetensors("decoder.weight","x"));
        const auto duplicatePackage=root/("duplicate-wan-vae-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".safetensors");
        (void)iild::createModelPackage(duplicateInput,duplicatePackage);
        wan.modelPath=duplicatePackage;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && vae.empty(),"A duplicate embedded VAE must not be mounted twice");
        const auto shardPackageInput=root/"wan-vae-shards";
        File::createDirectories(shardPackageInput/"video_vae");
        File::write(shardPackageInput/"model.safetensors",File::read(wanInput/"model.safetensors"));
        File::write(shardPackageInput/"text_encoder.gguf",wanEncoderBytes);
        File::write(shardPackageInput/"video_vae"/"part-a.safetensors",safetensors("vae.a","a"));
        File::write(shardPackageInput/"video_vae"/"part-b.safetensors",safetensors("vae.b","b"));
        File::write(shardPackageInput/"video_vae"/"model.safetensors.index.json",
            R"({"weight_map":{"vae.a":"part-a.safetensors","vae.b":"part-b.safetensors"}})");
        const auto shardVaePackage=root/("sharded-wan-vae-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".safetensors");
        (void)iild::createModelPackage(shardPackageInput,shardVaePackage);
        wan.modelPath=shardVaePackage;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && !vae.empty() && File::read(vae).find("vae.a")!=std::string::npos &&
            File::read(vae).find("vae.b")!=std::string::npos,"A packaged indexed VAE must restore every shard");
        const auto wanSplit=root/"wan-sharded";
        File::createDirectories(wanSplit);
        File::write(wanSplit/"part-a.safetensors",safetensors("patch_embedding.weight","a"));
        File::write(wanSplit/"part-b.safetensors",safetensors("blocks.0.cross_attn.norm_k.weight","b"));
        const auto wanIndex=wanSplit/"model.safetensors.index.json";
        File::write(wanIndex,R"({"weight_map":{"patch_embedding.weight":"part-a.safetensors","blocks.0.cross_attn.norm_k.weight":"part-b.safetensors"}})");
        wan.modelPath=wanIndex;wan.textEncoderPath=textEncoder;
        result=iiLocalDiffusion::generateNativeVideo(wan,cancelled,{},observer);
        require(result.error.empty() && encoder.empty() && t5Encoder==textEncoder.string() && diffusionModel==wanIndex.string(),
            "A sharded Wan model must retain its family and T5 encoder binding");
        const auto split = root / "standalone-video-shards";
        File::createDirectories(split);
        const auto shard1 = split / "part-1.safetensors", shard2 = split / "part-2.safetensors";
        File::write(shard1, safetensors("video.first", "one"));
        File::write(shard2, safetensors("video.second", "two"));
        const auto index = split / "model.safetensors.index.json";
        File::write(index, R"({"weight_map":{"video.first":"part-1.safetensors","video.second":"part-2.safetensors"}})");
        auto sharded = request; sharded.modelPath = index; sharded.shots.resize(1);
        const auto beforeSharded = loaded;
        result = iiLocalDiffusion::generateNativeVideo(sharded, cancelled, {}, observer);
        require(result.error.empty() && loaded == beforeSharded + 1,
            "A complete standalone video shard index with an explicit encoder must reach the engine");
        const auto renamedIndex=split/"video-index.data";
        File::write(renamedIndex,File::read(index));
        auto renamedVideo=sharded;renamedVideo.modelPath=renamedIndex;
        result=iiLocalDiffusion::generateNativeVideo(renamedVideo,cancelled,{},observer);
        if (!result.error.empty() || effectiveModel!=renamedIndex.string())
            throw std::runtime_error("A renamed video index did not reach the native engine: "+result.error);
        bool renamedChanged=false;
        result=iiLocalDiffusion::generateNativeVideo(renamedVideo,cancelled,{},[&](std::size_t shot,int number,int width,int height,auto rgb) {
            observer(shot,number,width,height,rgb);
            if (!renamedChanged) {File::write(shard2,safetensors("video.second","new"));renamedChanged=true;}
        });
        require(renamedChanged && result.error.find("changed")!=std::string::npos,
            "A changed shard of a renamed video index was not detected");
        File::write(shard2,safetensors("video.second","two"));
        auto withoutEncoder=sharded; withoutEncoder.textEncoderPath.clear();
        result = iiLocalDiffusion::generateNativeVideo(withoutEncoder, cancelled, {}, observer);
        require(result.error.find("explicit compatible text encoder") != std::string::npos,
            "A standalone indexed video model must report its missing encoder");
        const std::string embed="text_encoders.llm.model.embed_tokens.weight";
        const std::string norm="text_encoders.llm.model.norm.weight";
        const std::string query="text_encoders.llm.model.layers.0.self_attn.q_proj.weight";
        const auto completeFile=split/"self-contained.safetensors";
        File::write(completeFile,safetensorsMany({"model.diffusion_model.fixture",embed,norm,query}));
        auto complete=withoutEncoder;complete.modelPath=completeFile;
        result = iiLocalDiffusion::generateNativeVideo(complete, cancelled, {}, observer);
        require(result.error.empty() && encoder.empty(),
            "A self-contained video checkpoint must reach the engine without an external encoder");
        auto explicitEncoder=complete;explicitEncoder.textEncoderPath=textEncoder;
        result = iiLocalDiffusion::generateNativeVideo(explicitEncoder, cancelled, {}, observer);
        require(result.error.empty() && encoder==textEncoder.string(),
            "An explicitly selected video text encoder must retain precedence");
        File::write(split/"embedded-1.safetensors",safetensorsMany({"model.diffusion_model.fixture",embed}));
        File::write(split/"embedded-2.safetensors",safetensorsMany({norm,query}));
        const auto embeddedIndex=split/"embedded.SAFETENSORS.Index.JSON";
        File::write(embeddedIndex,R"({"weight_map":{"model.diffusion_model.fixture":"embedded-1.safetensors","text_encoders.llm.model.embed_tokens.weight":"embedded-1.safetensors","text_encoders.llm.model.norm.weight":"embedded-2.safetensors","text_encoders.llm.model.layers.0.self_attn.q_proj.weight":"embedded-2.safetensors"}})");
        complete.modelPath=embeddedIndex;
        result = iiLocalDiffusion::generateNativeVideo(complete, cancelled, {}, observer);
        require(result.error.empty() && encoder.empty(),
            "An indexed video checkpoint with an embedded encoder must reach the engine without an external encoder");
        const auto compactFile=split/"gguf-names.safetensors";
        File::write(compactFile,safetensorsMany({"model.diffusion_model.fixture",
            "text_encoders.llm.token_embd.weight", "text_encoders.llm.output_norm.weight",
            "text_encoders.llm.blk.0.attn_q.weight"}));
        auto compact=withoutEncoder;compact.modelPath=compactFile;
        result = iiLocalDiffusion::generateNativeVideo(compact, cancelled, {}, observer);
        require(result.error.empty() && encoder.empty(),
            "GGUF-style embedded LLM tensor names must preserve their engine conversion path");
        File::write(compactFile,safetensorsMany({"model.diffusion_model.fixture",embed,
            "text_encoders.llm.output_norm.weight",query}));
        result = iiLocalDiffusion::generateNativeVideo(compact, cancelled, {}, observer);
        require(result.error.empty() && encoder.empty(),
            "Mixed embedded LLM tensor naming conventions must not require an external encoder");
        const auto compactGguf=split/"self-contained.gguf";
        File::write(compactGguf,ggufFixture({"model.diffusion_model.fixture",
            "text_encoders.llm.token_embd.weight", "text_encoders.llm.output_norm.weight",
            "text_encoders.llm.blk.0.attn_q.weight"}));
        compact.modelPath=compactGguf;
        result = iiLocalDiffusion::generateNativeVideo(compact, cancelled, {}, observer);
        require(result.error.empty() && encoder.empty(),
            "A self-contained GGUF video checkpoint must not require an external encoder");
        File::write(split/"partial.safetensors",safetensorsMany({"model.diffusion_model.fixture",embed,norm}));
        const auto partialIndex=split/"partial.safetensors.index.json";
        File::write(partialIndex,R"({"weight_map":{"model.diffusion_model.fixture":"partial.safetensors","text_encoders.llm.model.embed_tokens.weight":"partial.safetensors","text_encoders.llm.model.norm.weight":"partial.safetensors"}})");
        auto partial=withoutEncoder;partial.modelPath=partialIndex;
        const auto beforePartial=loaded;
        result = iiLocalDiffusion::generateNativeVideo(partial, cancelled, {}, observer);
        require(result.error.find("explicit compatible text encoder")!=std::string::npos && loaded==beforePartial,
            "Partial embedded LLM metadata must not bypass the missing encoder check");
        bool modified = false;
        result = iiLocalDiffusion::generateNativeVideo(sharded, cancelled, {}, [&](std::size_t shot, int number, int width, int height, auto rgb) {
            observer(shot, number, width, height, rgb);
            if (!modified) { File::write(shard2, safetensors("video.second", "new")); modified = true; }
        });
        require(modified && result.error.find("changed") != std::string::npos,
            "A modified shard must invalidate a video request before publication");
        File::write(shard2, safetensors("video.second", "two"));
        auto shardedEncoder=request; shardedEncoder.textEncoderPath=index; shardedEncoder.shots.resize(1);
        modified = false;
        result = iiLocalDiffusion::generateNativeVideo(shardedEncoder, cancelled, {}, [&](std::size_t shot, int number, int width, int height, auto rgb) {
            observer(shot, number, width, height, rgb);
            if (!modified) { File::write(shard2, safetensors("video.second", "new")); modified = true; }
        });
        require(modified && result.error.find("changed") != std::string::npos,
            "A modified indexed text encoder must invalidate the video request");
        File::write(index, R"({"weight_map":{"video.first":"../outside.safetensors"}})");
        const auto beforeInvalid = loaded;
        result = iiLocalDiffusion::generateNativeVideo(sharded, cancelled, {}, observer);
        require(!result.error.empty() && loaded == beforeInvalid,
            "An unsafe or incomplete video index must be rejected before the engine loads");
        const auto input = root / "package-input";
        File::createDirectories(input);
        File::write(input / "main.safetensors", safetensors("model.diffusion_model.fixture", "fixture model"));
        File::write(input / "text_encoder.safetensors", safetensors("text_model.encoder.weight", "fixture encoder"));
        const auto package = root / ("package-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
        (void)iild::createModelPackage(input, package);
        auto embedded = request; embedded.modelPath = package; embedded.textEncoderPath.clear();
        embedded.assetCacheDirectory = root / "assets";
        for (int attempt = 0; attempt < 2; ++attempt) {
            std::vector<std::pair<int, int>> preparation;
            result = iiLocalDiffusion::generateNativeVideo(embedded, cancelled, [&](const auto &event) {
                if (event.stage == iiLocalDiffusion::NativeGenerationStage::Preparing)
                    preparation.emplace_back(event.step, event.total);
            }, observer);
            require(result.error.empty() && !preparation.empty(), "Embedded asset prepares and reuses its verified cache");
            require(fs::path(encoder).extension() == ".safetensors" && fs::is_regular_file(encoder),
                "The native engine receives a materialized safetensors encoder");
            require(preparation.front().second == 1 && preparation.back() == std::pair{1, 1},
                "Preparation and cached verification report a nonzero total and final completion");
        }
        // Flat package headers may contain a few LLM names from the main
        // checkpoint while the actual separate text encoder is another asset.
        // Keep the package's verified component path instead of assuming the
        // flattened names constitute a complete in-model text encoder.
        File::write(input/"main.safetensors",safetensorsMany({"model.diffusion_model.fixture",embed,norm,query}));
        const auto mixedPackage=root/("mixed-package-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".safetensors");
        (void)iild::createModelPackage(input,mixedPackage);
        embedded.modelPath=mixedPackage;
        result=iiLocalDiffusion::generateNativeVideo(embedded,cancelled,{},observer);
        require(result.error.empty() && !encoder.empty() && fs::is_regular_file(encoder),
            "A package's separate encoder must retain precedence over flattened embedded markers");
        const auto shardsInput = root / "sharded-package-input";
        File::createDirectories(shardsInput / "text_encoder");
        File::write(shardsInput / "main.safetensors", safetensors("model.diffusion_model.fixture", "fixture model"));
        File::write(shardsInput / "text_encoder" / "part-1.safetensors", safetensors("text_model.first.weight", "first"));
        File::write(shardsInput / "text_encoder" / "part-2.safetensors", safetensors("text_model.second.weight", "second"));
        File::write(shardsInput / "text_encoder" / "model.safetensors.index.json",
            R"({"weight_map":{"text_model.first.weight":"part-1.safetensors","text_model.second.weight":"part-2.safetensors"}})");
        const auto shardsPackage = root / ("sharded-package-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
        (void)iild::createModelPackage(shardsInput, shardsPackage);
        embedded.modelPath = shardsPackage;
        result = iiLocalDiffusion::generateNativeVideo(embedded, cancelled, {}, observer);
        const auto mergedEncoder = result.error.empty() ? File::read(encoder) : std::string{};
        require(result.error.empty() && fs::path(encoder).extension() == ".safetensors" &&
                mergedEncoder.find("text_model.first.weight") != std::string::npos &&
                mergedEncoder.find("text_model.second.weight") != std::string::npos,
            "The native video engine receives all indexed encoder shards in one verified file");
        const auto ggufInput = root / "gguf-package-input";
        File::createDirectories(ggufInput);
        File::write(ggufInput / "main.safetensors", safetensors("model.diffusion_model.fixture", "fixture model"));
        const auto ggufBytes = ggufFixture();
        File::write(ggufInput / "text_encoder.gguf", ggufBytes);
        const auto ggufPackage = root / ("gguf-package-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
        (void)iild::createModelPackage(ggufInput, ggufPackage);
        embedded.modelPath = ggufPackage;
        result = iiLocalDiffusion::generateNativeVideo(embedded, cancelled, {}, observer);
        require(result.error.empty() && fs::path(encoder).extension() == ".gguf" && File::read(encoder) == ggufBytes,
            "The native engine receives the original byte-exact GGUF encoder");
        failure = 2;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, observer);
        require(result.error.find("invalid RGB") != std::string::npos && loaded == destroyed, "Invalid native output cannot be published");
        failure = 1;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, observer);
        require(!result.error.empty() && loaded == destroyed, "Inference failure releases context");
        require(result.error.find("cannot make enough memory available") != std::string::npos &&
            result.error.find("segment weight preparation failed") != std::string::npos,
            "Video failures must retain bounded warning and error evidence");
        require(result.error.size() < 4000, "Warning and error evidence must remain bounded");
        failure = 0; supports = false;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, observer);
        require(result.error.find("not supported") != std::string::npos, "Manual classification cannot turn an image engine into video");
        supports = true;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, [&](const auto &event) {
            if (event.stage == iiLocalDiffusion::NativeGenerationStage::Denoising) cancelled = true;
        }, observer);
        require(result.cancelled && result.error.empty() && loaded == destroyed && !stageCallback && !logCallback,
            "Cancellation clears callbacks and owns no stale context");
        cancelled = false;
        result = iiLocalDiffusion::generateNativeVideo(request, cancelled, {}, [&](std::size_t, int index, int, int, auto) {
            if (index == 8) cancelled = true;
        });
        require(result.cancelled && result.frameCounts.empty() && loaded == destroyed, "Cancellation on the final frame prevents shot publication");
        cancelled = false;
        auto placement = request; placement.cpu = false; placement.cpuTextEncoding = true; placement.backend = "auto";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.empty() && backend == "te=cpu" && paramsBackend == "diffusion=disk",
            "CPU text encoding must leave GPU diffusion weights reloadable when auto-fit is disabled");
        placement.cpuTextEncoding = false; placement.backend = "metal";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.empty() && backend == "metal" && paramsBackend == "diffusion=disk",
            "Explicit GPU placement must leave diffusion weights reloadable");
        placement.backend = "cpu";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.empty() && backend == "cpu" && paramsBackend.empty(),
            "A CPU backend request must not use GPU residency policy");
        placement.backend = "auto";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.empty() && backend.empty() && paramsBackend.empty(),
            "Automatic placement without overrides must keep the engine auto-fit contract");
        auto lowVram = placement; lowVram.vaeTiling = false; lowVram.shots.resize(1);
        result = iiLocalDiffusion::generateNativeVideoLowVram(lowVram, cancelled, {}, observer);
        require(result.error.empty() && backend == "te=cpu,vae=cpu" &&
                    paramsBackend == "te=disk,diffusion=disk,vae=disk" && prefetchDisabled && tiledDecode,
            "Low VRAM must offload text, VAE and weights, disable prefetch and enable tiled decode");
        lowVram.backend = "metal";
        result = iiLocalDiffusion::generateNativeVideoLowVram(lowVram, cancelled, {}, observer);
        require(result.error.empty() && backend == "te=cpu,diffusion=metal,vae=cpu" && prefetchDisabled,
            "Low VRAM must preserve an explicitly selected compute device");
        lowVram.cpu = true;
        result = iiLocalDiffusion::generateNativeVideoLowVram(lowVram, cancelled, {}, observer);
        require(result.error.empty() && backend == "cpu" && paramsBackend == "te=disk,diffusion=disk,vae=disk",
            "Low VRAM must preserve an explicit CPU request");
#if !defined(_WIN32)
        // POSIX test files are sparse; do not allocate a multi-GiB fixture on
        // platforms where resize_file may reserve the entire extent.
        const auto largeModel = root / "large-sparse-fixture.safetensors";
        File::write(largeModel, "fixture model");
        fs::resize_file(largeModel, (4ull << 30) + 1);
        auto large = placement; large.modelPath = largeModel; large.shots.resize(1);
        result = iiLocalDiffusion::generateNativeVideo(large, cancelled, {}, observer);
        require(result.error.empty() && backend.empty() && paramsBackend.empty(),
            "Large fully automatic requests must preserve engine auto-fit");
        large.cpuTextEncoding = true;
        result = iiLocalDiffusion::generateNativeVideo(large, cancelled, {}, observer);
        require(result.error.empty() && backend == "te=cpu" && paramsBackend == "te=disk,diffusion=disk",
            "A large explicitly placed video checkpoint must use disk-backed weights");
        large.cpu = true;
        result = iiLocalDiffusion::generateNativeVideo(large, cancelled, {}, observer);
        require(result.error.empty() && backend == "cpu" && paramsBackend == "te=disk,diffusion=disk",
            "A large CPU video checkpoint must stream its text and diffusion weights");
        fs::remove(largeModel);
        const auto sparseShard=split / "large.safetensors";
        const auto sparseIndex=split / "large.safetensors.index.json";
        constexpr std::uint64_t sparseBytes=(4ull << 30) + 1;
        const auto header=std::string(R"({"video.large":{"dtype":"U8","shape":[)")+
            std::to_string(sparseBytes)+R"(],"data_offsets":[0,)"+std::to_string(sparseBytes)+R"(]}})";
        std::string prefix(8,'\0');
        for (unsigned i=0;i<8;++i) prefix[i]=static_cast<char>((header.size()>>(8*i))&255);
        File::write(sparseShard,prefix+header);
        fs::resize_file(sparseShard,8+header.size()+sparseBytes);
        File::write(sparseIndex,R"({"weight_map":{"video.large":"large.safetensors"}})");
        auto largeSharded=large; largeSharded.modelPath=sparseIndex;
        result = iiLocalDiffusion::generateNativeVideo(largeSharded, cancelled, {}, observer);
        require(result.error.empty() && paramsBackend == "te=disk,diffusion=disk",
            "Video streaming policy must use the total shard size rather than the small index size");
        fs::remove(sparseShard);
#endif
        failure = 3;
        const auto attemptsBefore = generated;
        placement.shots.resize(1);
        int fallbackBatches = 0;
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, [&](const auto &event) {
            if (event.stage == iiLocalDiffusion::NativeGenerationStage::Computing) ++fallbackBatches;
        }, observer);
        require(result.error.empty() && generated == attemptsBefore + 2 && backend == "cpu" &&
                    paramsBackend == "te=disk,diffusion=disk" && loaded == destroyed && fallbackBatches == 1 && !evalCallback,
            "Automatic GPU budget failure must retry with CPU compute and disk-backed diffusion weights");
        const auto otherFailure = generated;
        failure = 4;
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(!result.error.empty() && generated == otherFailure + 1,
            "Unrelated failures must not trigger a disk-backed retry");
        const auto explicitFailure = generated;
        failure = 3; placement.backend = "metal";
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(!result.error.empty() && generated == explicitFailure + 1 && backend == "metal",
            "Explicit GPU selection must not silently change to CPU");
        placement.backend = "auto"; failure = 5;
        result = iiLocalDiffusion::generateNativeVideo(placement, cancelled, {}, observer);
        require(result.error.find("Initial GPU failure") != std::string::npos &&
                    result.error.find("Fallback failure") != std::string::npos && loaded == destroyed,
            "Both attempts must be reported when the disk-backed fallback also fails");
        failure = 0;
        const auto modelPath = model.string(), encoderPath = textEncoder.string();
        iild_native_video_shot_v1 abiShot{};
        abiShot.prompt = "ABI shot"; abiShot.negative_prompt = "noise";
        abiShot.frames = 9; abiShot.fps = 24; abiShot.conditioning_strength = 1;
        iild_native_video_request_v1 abiRequest{};
        abiRequest.size = sizeof(abiRequest); abiRequest.model = modelPath.c_str();
        abiRequest.text_encoder = encoderPath.c_str(); abiRequest.backend = "auto";
        abiRequest.precision = "auto"; abiRequest.width = abiRequest.height = 32;
        abiRequest.steps = 2; abiRequest.guidance_scale = 3;
        abiRequest.shots = &abiShot; abiRequest.shot_count = 1;
        const auto abiFrame = [](size_t, int32_t, int32_t, int32_t, const uint8_t*, size_t, void*) { return 0; };
        auto *abi = iild_native_video_generate_low_vram_v1(&abiRequest, nullptr, abiFrame, nullptr);
        require(abi && std::string(iild_native_video_metadata_v1(abi)).find("low-vram") != std::string::npos &&
                    backend == "te=cpu,vae=cpu" && paramsBackend == "te=disk,diffusion=disk,vae=disk" && prefetchDisabled,
            "Low VRAM C ABI must reach the real video adapter without changing the request layout");
        iild_native_video_free_v1(abi);
        abi = iild_native_video_generate_v1(&abiRequest, nullptr, abiFrame, nullptr);
        require(abi && std::string(iild_native_video_metadata_v1(abi)).find("standard") != std::string::npos &&
                    backend.empty() && paramsBackend.empty() && !prefetchDisabled,
            "Existing C ABI must preserve standard placement");
        iild_native_video_free_v1(abi);
        const auto wanPath=wanFile.string(), vaePath=vaeFile.string(), highPath=highFile.string(), visionPath=visionFile.string();
        abiRequest.model=wanPath.c_str();
        iild_native_video_components_v1 abiComponents{sizeof(iild_native_video_components_v1),
            vaePath.c_str(),highPath.c_str(),visionPath.c_str(),0.75f,7,3.5f};
        abi=iild_native_video_generate_with_components_v1(&abiRequest,&abiComponents,nullptr,abiFrame,nullptr);
        require(abi && std::string(iild_native_video_metadata_v1(abi)).find("\"error\":\"\"")!=std::string::npos &&
            vae==vaePath && highNoise==highPath && vision==visionPath && expertSteps==7,
            "The extended C ABI carries paths and controls into the engine");
        iild_native_video_free_v1(abi);
        abi=iild_native_video_generate_low_vram_with_components_v1(&abiRequest,&abiComponents,nullptr,abiFrame,nullptr);
        require(abi && std::string(iild_native_video_metadata_v1(abi)).find("low-vram")!=std::string::npos &&
            highNoise==highPath && prefetchDisabled,"Low VRAM retains the components extension");
        iild_native_video_free_v1(abi);
        abiComponents.size=0;
        abi=iild_native_video_generate_with_components_v1(&abiRequest,&abiComponents,nullptr,abiFrame,nullptr);
        require(abi && std::string(iild_native_video_metadata_v1(abi)).find("components ABI")!=std::string::npos,
            "The extended ABI rejects an invalid extension size");
        iild_native_video_free_v1(abi);
        abiRequest.model=modelPath.c_str();
        auto *invalid = iild_native_video_generate_v1(nullptr, nullptr, nullptr, nullptr);
        require(invalid && std::string(iild_native_video_metadata_v1(invalid)).find("Invalid native video ABI") != std::string::npos, "C ABI rejects invalid request");
        iild_native_video_free_v1(invalid);
        invalid = iild_native_video_generate_low_vram_v1(nullptr, nullptr, nullptr, nullptr);
        require(invalid && std::string(iild_native_video_metadata_v1(invalid)).find("low-vram") != std::string::npos,
            "Low VRAM C ABI must report its execution mode even for invalid requests");
        iild_native_video_free_v1(invalid);
        std::cout << "Native video frames, batch, continuity, cleanup, cancellation and ABI checks passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
