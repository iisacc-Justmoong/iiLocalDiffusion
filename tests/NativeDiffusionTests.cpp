#include "Generation/NativeDiffusion.hpp"
#include "Generation/ImageModelRouting.hpp"
#include <atomic>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdlib>

static void writeSafetensors(const std::filesystem::path &path, const std::string &header, std::size_t bytes, bool pad = true) {
    std::ofstream file(path, std::ios::binary);
    auto padded = header;
    while (pad && padded.size() % 8) padded.push_back(' ');
    const auto size = static_cast<std::uint64_t>(padded.size());
    for (unsigned shift = 0; shift != 64; shift += 8) file.put(static_cast<char>((size >> shift) & 0xff));
    file.write(padded.data(), static_cast<std::streamsize>(padded.size()));
    const std::string payload(bytes, '\0');
    file.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

static std::string jsonString(const std::string &value) {
    std::string result = "\"";
    for (const char c : value) { if (c == '\\' || c == '"') result += '\\'; result += c; }
    return result + "\"";
}

int main(int argc, char **argv) {
    using namespace iiLocalDiffusion;
    if (argc != 2) return 1;
    const auto directory = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(directory);
    std::atomic_bool cancelled{true};
    NativeGenerationRequest request;
    auto result = generateNativeImage(request, cancelled);
    if (!result.cancelled || !result.rgb.empty()) return 2;
    cancelled = false;
    request.modelPath = directory / "missing.safetensors";
    request.prompt = "a forest";
    {
        auto unresolved = request;
        unresolved.width = 0; unresolved.height = 0; unresolved.steps = 0; unresolved.seed = -1;
        const auto defaults = nativeImageParameterDefaults(unresolved.modelPath);
        const auto resolved = resolveNativeImageDefaults(unresolved);
        if (resolved.width != std::get<std::int64_t>(defaults.values.at("width"))
            || resolved.height != std::get<std::int64_t>(defaults.values.at("height"))
            || resolved.steps != std::get<std::int64_t>(defaults.values.at("steps")) || resolved.seed != -1) return 51;
        unresolved.width = 768; unresolved.height = 576; unresolved.steps = 47; unresolved.seed = 2147483648LL;
        const auto explicitValues = resolveNativeImageDefaults(unresolved);
        if (explicitValues.width != 768 || explicitValues.height != 576 || explicitValues.steps != 47
            || explicitValues.seed != 2147483648LL) return 52;
    }
    result = generateNativeImage(request, cancelled);
    if (result.error.empty() || !result.rgb.empty()) return 3;
    request.modelPath = directory / "invalid.safetensors";
    { std::ofstream file(request.modelPath); file << "not a checkpoint"; }
    {
        auto gguf = request; gguf.modelPath = directory / "weights.gguf";
        if (generateNativeImage(gguf, cancelled).error.find("ONNX") != std::string::npos) return 19;
        auto onnx = request; onnx.modelPath = directory / "weights.ONNX";
        if (generateNativeImage(onnx, cancelled).error.find("ONNX") == std::string::npos) return 20;
        NativeModelComponents components; components.llm = directory / "encoder.onnx";
        if (generateNativeImageWithComponents(request, {}, components, NativeComputeBackend::Cpu, cancelled).error.find("ONNX") == std::string::npos) return 21;
        NativeGenerationOptions options; options.loras.push_back({directory / "adapter.ONNX", 1.0f});
        if (generateNativeImageWithOptions(request, options, NativeComputeBackend::Cpu, cancelled).error.find("ONNX") == std::string::npos) return 22;
        NativeAdvancedControls advanced; advanced.refinerModel = directory / "refiner.onnx";
        if (generateNativeAdvancedImage(request, {}, {}, advanced, NativeComputeBackend::Cpu, cancelled).error.find("ONNX") == std::string::npos) return 23;
    }
    request.width = 63;
    result = generateNativeImage(request, cancelled);
    if (result.error != "Invalid native image generation parameters.") return 4;
    request.width = 64;
    request.timeoutMilliseconds = 0;
    result = generateNativeImageWithProgress(request, cancelled, {});
    if (result.error != "Invalid native image generation parameters.") return 6;
    request.timeoutMilliseconds = 900000;
    // Legacy iiLocalDiffusion packages may put the language encoder at the
    // root, while the actual MOSS transformer is a named embedded component.
    {
        const auto model = directory / "embedded-moss.safetensors";
        const std::string prefix = "components/model/Other/moss.safetensors/";
        const std::string vae = "components/video_vae/VAE/flux.safetensors/";
        const auto manifest = std::string(R"({"schema":"iild-safetensors-package-v1","files":[{"path":"Other/moss.safetensors","role":"model","format":"safetensors","status":"included","tensors":[)")
            + "{\"key\":\"video_embedder.weight\",\"output_key\":" + jsonString(prefix + "video_embedder.weight") + "},"
            + "{\"key\":\"final_linear_video.weight\",\"output_key\":" + jsonString(prefix + "final_linear_video.weight") + "},"
            + "{\"key\":\"layers.0.proj.qkv.weight\",\"output_key\":" + jsonString(prefix + "layers.0.proj.qkv.weight") + "},"
            + "{\"key\":\"layers.0.proj.qkv.comfy_quant\",\"output_key\":" + jsonString(prefix + "layers.0.proj.qkv.comfy_quant") + "}]},"
            + R"({"path":"VLM/encoder.safetensors","role":"model","format":"safetensors","status":"included","tensors":[{"key":"model.language_model.embed_tokens.weight","output_key":"model.language_model.embed_tokens.weight"},{"key":"model.language_model.layers.0.weight","output_key":"model.language_model.layers.0.weight"}]},)"
            + "{\"path\":\"VAE/flux.safetensors\",\"role\":\"video_vae\",\"format\":\"safetensors\",\"status\":\"included\",\"tensors\":["
            + "{\"key\":\"encoder.conv_in.weight\",\"output_key\":" + jsonString(vae + "encoder.conv_in.weight") + "},"
            + "{\"key\":\"decoder.conv_out.weight\",\"output_key\":" + jsonString(vae + "decoder.conv_out.weight") + "}]}]}";
        const auto header = std::string("{\"__metadata__\":{\"iild_package_schema\":\"iild-safetensors-package-v1\",\"iild_package_manifest\":") + jsonString(manifest) + "},"
            + jsonString(prefix + "video_embedder.weight") + ":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[0,16]},"
            + jsonString(prefix + "final_linear_video.weight") + ":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[16,32]},"
            + jsonString(prefix + "layers.0.proj.qkv.weight") + ":{\"dtype\":\"I8\",\"shape\":[12,4],\"data_offsets\":[32,80]},"
            + jsonString(prefix + "layers.0.proj.qkv.comfy_quant") + ":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[80,82]},"
            + "\"model.language_model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[82,86]},"
            + "\"model.language_model.layers.0.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[86,90]},"
            + jsonString(vae + "encoder.conv_in.weight") + ":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[90,94]},"
            + jsonString(vae + "decoder.conv_out.weight") + ":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[94,98]}}";
        writeSafetensors(model, header, 98);
        const auto inspected = inspectImageModel(model);
        const auto embedded = inspectEmbeddedImageComponents(model);
        if (inspected.architecture != ImageModelArchitecture::MossImage || inspected.encoding != ImageWeightEncoding::ComfyInt8
            || !inspected.hasTextEncoder || !inspected.hasVae || !embedded.packaged
            || embedded.transformer != "Other/moss.safetensors" || embedded.textEncoder != "VLM/encoder.safetensors"
            || embedded.vae != "VAE/flux.safetensors") return 40;
        auto requestForPackage = request; requestForPackage.modelPath = model;
        if (generateNativeImage(requestForPackage, cancelled).error.find("MOSS Image") == std::string::npos) return 41;
    }
    // A MOSS denoiser, including the user's quantized fused-QKV layout, must
    // fail with its real compatibility boundary rather than an SD version error.
    for (const std::string prefix : {"", "model.diffusion_model.", "diffusion_model.", "transformer."}) {
        const auto model = directory / (prefix.empty() ? "moss-unrelated-filename.safetensor" : "moss-unrelated-filename.safetensors");
        const auto header = "{\"__metadata__\":{\"iild_merge\":\"{\\\"sources\\\":[5,{\\\"targets\\\":[false,{\\\"adaptation\\\":\\\"bad\\\"},{\\\"adaptation\\\":{\\\"synthetic\\\":true}}]}]}\"},\"" + prefix + "video_embedder.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[0,16]},"
            "\"" + prefix + "final_linear_video.weight\":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[16,32]},"
            "\"" + prefix + "layers.0.proj.qkv.weight\":{\"dtype\":\"I8\",\"shape\":[12,4],\"data_offsets\":[32,80]},"
            "\"" + prefix + "layers.0.proj.qkv.comfy_quant\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[80,82]}}";
        writeSafetensors(model, header, 82);
        auto moss = request; moss.modelPath = model;
        const auto inspected = inspectImageModel(model);
        if (inspected.architecture != ImageModelArchitecture::MossImage
            || inspected.role != ImageModelRole::Checkpoint
            || inspected.encoding != ImageWeightEncoding::ComfyInt8
            || inspected.hasTextEncoder || inspected.hasVae
            || inspected.syntheticAdapterMappings != 1) return 34;
        const auto failure = generateNativeImage(moss, cancelled);
        if (failure.error.find("MOSS Image") == std::string::npos
            || failure.error.find("INT8") == std::string::npos
            || failure.error.find("1 synthetic adapter mappings") == std::string::npos
            || failure.error.find("native") == std::string::npos || !failure.rgb.empty()) return 24;
        (void)nativeImageParameterDefaults(model);
        const auto residentFailure = generateNativeImageWithResidentWeights(
            moss, {}, NativeComputeBackend::Cpu, cancelled, {}, {});
        if (residentFailure.error.find("MOSS Image") == std::string::npos
            || residentFailure.error.find("INT8") == std::string::npos || !residentFailure.rgb.empty()) {
            std::cerr << "Resident MOSS preflight: " << residentFailure.error << '\n';
            return 32;
        }
    }
    {
        const auto model = directory / "app-style.safetensor";
        const auto header = R"({"video_embedder.weight":{"dtype":"BF16","shape":[4,2],"data_offsets":[0,16]},"final_linear_video.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[16,32]},"layers.0.proj.qkv.weight":{"dtype":"I8","shape":[12,4],"data_offsets":[32,80]},"layers.0.proj.qkv.comfy_quant":{"dtype":"U8","shape":[2],"data_offsets":[80,82]}})";
        writeSafetensors(model, header, 82, false);
        auto moss = request; moss.modelPath = model;
        const auto failure = generateNativeImageWithResidentWeights(moss, {}, NativeComputeBackend::Cpu, cancelled, {}, {});
        if (failure.error.find("MOSS Image") == std::string::npos || failure.error.find("INT8") == std::string::npos) {
            std::cerr << "Application-style MOSS preflight: " << failure.error << '\n';
            return 33;
        }
    }
    {
        const auto malformed = directory / "truncated.safetensors";
        writeSafetensors(malformed, "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", 4);
        auto invalid = request; invalid.modelPath = malformed;
        if (generateNativeImage(invalid, cancelled).error.find("safetensors") == std::string::npos) return 25;
    }
    {
        const auto component = directory / "component.safetensors";
        auto test = request; test.modelPath = component;
        writeSafetensors(component, "{\"layer.lora_down.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"layer.lora_up.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8);
        if (inspectImageModel(component).role != ImageModelRole::Lora) return 35;
        if (generateNativeImage(test, cancelled).error.find("LoRA adapter") == std::string::npos) return 26;
        writeSafetensors(component, "{\"encoder.conv_in.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"decoder.conv_out.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8);
        if (inspectImageModel(component).role != ImageModelRole::Vae) return 36;
        if (generateNativeImage(test, cancelled).error.find("VAE") == std::string::npos) return 27;
        writeSafetensors(component, "{\"control_model.input_blocks.0.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4);
        if (inspectImageModel(component).role != ImageModelRole::ControlNet) return 37;
        if (generateNativeImage(test, cancelled).error.find("ControlNet") == std::string::npos) return 28;
        writeSafetensors(component, "{\"layer.lora_down.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"layer.lora_up.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]},\"layers.0.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}", 12);
        if (inspectImageModel(component).role != ImageModelRole::Unknown) return 38;
        if (generateNativeImage(test, cancelled).error.find("LoRA adapter") != std::string::npos) return 30;
        writeSafetensors(component, "{\"encoder.conv_in.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"decoder.conv_out.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]},\"layers.0.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}", 12);
        if (generateNativeImage(test, cancelled).error.find("selected file is a VAE") != std::string::npos) return 31;
    }
    {
        const auto near = directory / "near-moss.safetensors";
        writeSafetensors(near, "{\"video_embedder.weight\":{\"dtype\":\"F32\",\"shape\":[4,2],\"data_offsets\":[0,32]},\"final_linear_video.weight\":{\"dtype\":\"F32\",\"shape\":[3,4],\"data_offsets\":[32,80]},\"layers.0.proj.qkv.weight\":{\"dtype\":\"F32\",\"shape\":[12,4],\"data_offsets\":[80,272]}}", 272);
        auto unknown = request; unknown.modelPath = near;
        if (generateNativeImage(unknown, cancelled).error.find("MOSS Image") != std::string::npos) return 29;
    }
    NativeGenerationOptions sharedOptions;
    sharedOptions.defaultModifiers = false;
    sharedOptions.resourceDirectory = directory / "society-resources-not-installed";
    result = generateNativeImageWithOptions(request, sharedOptions, NativeComputeBackend::Cpu, cancelled);
    if (result.error.empty() || !result.rgb.empty()
        || result.error.find("generation-defaults.json") != std::string::npos) return 11;
    cancelled = true;
    result = generateNativeImageWithOptions(request, sharedOptions, NativeComputeBackend::Cpu, cancelled);
    if (!result.cancelled || !result.rgb.empty()) return 12;
    cancelled = false;
    bool falseStep = false;
    result = generateNativeImage(request, cancelled, [&](int, int) { falseStep = true; });
    if (falseStep) return 7; // Loading a rejected file cannot emit denoising steps.
    result = generateNativeImage(request, cancelled);
    if (result.error.empty() || !result.rgb.empty()) return 5;
    std::filesystem::remove(request.modelPath);
    if (const auto *model = std::getenv("IILD_NATIVE_TEST_MODEL")) {
        request.modelPath = std::filesystem::canonical(model);
        request.width = request.height = 64;
        request.steps = 1;
        for (const auto stage : {NativeGenerationStage::Loading, NativeGenerationStage::Denoising}) {
            cancelled = false;
            bool requested = false;
            const auto started = std::chrono::steady_clock::now();
            result = generateNativeImageWithProgress(request, cancelled, [&](const auto &event) {
                if (event.stage == stage && (stage != NativeGenerationStage::Loading || event.step > 0))
                    requested = cancelled = true;
            });
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cout << "real_model_cancel_stage=" << static_cast<int>(stage) << "; seconds=" << seconds << '\n';
            if (!requested || !result.cancelled || !result.rgb.empty() || seconds > 30) return 8;
        }
        cancelled = false;
        request.timeoutMilliseconds = 100;
        result = generateNativeImageWithProgress(request, cancelled, {});
        if (result.cancelled || !result.rgb.empty() || result.error.find("time limit") == std::string::npos) return 9;
        std::cout << "real_model_deadline_passed\n";
        request.timeoutMilliseconds = 900000;
        request.q8CacheDirectory = directory / "q8-cancel";
        std::filesystem::remove_all(request.q8CacheDirectory);
        bool preparationProgress = false;
        const auto started = std::chrono::steady_clock::now();
        result = generateNativeImageWithProgress(request, cancelled, [&](const auto &event) {
            if (event.stage == NativeGenerationStage::Preparing && event.step > 0)
                preparationProgress = cancelled = true;
        });
        if (!preparationProgress || !result.cancelled || !result.rgb.empty()
            || !std::filesystem::is_empty(request.q8CacheDirectory)
            || std::chrono::steady_clock::now() - started > std::chrono::seconds(30)) return 10;
        std::cout << "real_model_q8_cancel_passed\n";
    }
    std::cout << "native_available=" << nativeDiffusionAvailable() << "; cancellation, invalid inputs, model rejection passed\n";
}
