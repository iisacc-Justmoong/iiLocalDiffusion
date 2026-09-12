#include "Generation/NativeDiffusion.hpp"
#include "Generation/NativeCachePolicy.hpp"
#include <stable-diffusion.h>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <future>
#include <thread>

// The real adapter is compiled into this test. Only the upstream C API is a
// fixture: mimic SDXL alignment, identifiable RGB pixels, and failure cases.
struct sd_ctx_t {};
namespace {
sd_log_cb_t logCallback = nullptr;
void *logData = nullptr;
sd_progress_stage_cb_t progressCallback = nullptr;
void *progressData = nullptr;
sd_abort_cb_t abortCallback = nullptr;
void *abortData = nullptr;
enum class Output { valid, failed, engineError, wrongSize, wrongChannels, missing, multiple };
Output output = Output::valid;
int allocatedImages = 0;
constexpr auto warning = "No valid VAE specified with --vae or --force-sdxl-vae-conv-scale flag set, using Conv2D scale 0.031";
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
}
#if defined(__APPLE__)
namespace iiLocalDiffusion::native_detail {
ResourceLimits resourceLimits() { return {8ull << 30, 4ull << 30, 6ull << 30, true}; }
}
#endif
extern "C" {
void sd_set_log_callback(sd_log_cb_t callback, void *data) { logCallback = callback; logData = data; }
void sd_set_progress_stage_callback(sd_progress_stage_cb_t callback, void *data) { progressCallback = callback; progressData = data; }
void sd_set_abort_callback(sd_abort_cb_t callback, void *data) { abortCallback = callback; abortData = data; }
int32_t sd_get_num_physical_cores() { return 2; }
void sd_ctx_params_init(sd_ctx_params_t *parameters) { *parameters = {}; }
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *) {
    if (logCallback) logCallback(SD_LOG_WARN, warning, logData);
    return new sd_ctx_t;
}
void free_sd_ctx(sd_ctx_t *context) { delete context; }
bool sd_ctx_supports_image_generation(const sd_ctx_t *) { return true; }
sample_method_t sd_get_default_sample_method(const sd_ctx_t *) { return static_cast<sample_method_t>(0); }
scheduler_t sd_get_default_scheduler(const sd_ctx_t *, sample_method_t) { return static_cast<scheduler_t>(0); }
void sd_img_gen_params_init(sd_img_gen_params_t *parameters) { *parameters = {}; }
void sd_cancel_generation(sd_ctx_t *, sd_cancel_mode_t) {}
bool convert_with_components(const char *, const char *, const char *, const char *, const char *,
    const char *, const char *, sd_type_t, const char *, bool, int) { return false; }
bool generate_image(sd_ctx_t *, const sd_img_gen_params_t *parameters, sd_image_t **images, int *count) {
    *images = nullptr; *count = 0;
    if (abortCallback && abortCallback(abortData)) return false;
    if (output == Output::failed) return false;
    if (output == Output::engineError) {
        if (logCallback) logCallback(SD_LOG_ERROR, "VAE decode allocation failed", logData);
        return false;
    }
    if (output == Output::missing) return true;
    *count = output == Output::multiple ? 2 : 1;
    *images = static_cast<sd_image_t *>(std::calloc(*count, sizeof(sd_image_t)));
    allocatedImages += *count;
    for (int i = 0; i < *count; ++i) {
        auto &image = (*images)[i];
        image.width = (parameters->width + 63) / 64 * 64;
        image.height = (parameters->height + 63) / 64 * 64;
        if (output == Output::wrongSize) image.height -= 64;
        image.channel = output == Output::wrongChannels ? 4 : 3;
        image.data = static_cast<uint8_t *>(std::calloc(image.width * image.height, image.channel));
        for (unsigned y = 0; y < image.height; ++y)
            for (unsigned x = 0; x < image.width; ++x) {
                const auto offset = (y * image.width + x) * image.channel;
                image.data[offset] = x % 251;
                image.data[offset + 1] = y % 251;
                image.data[offset + 2] = (x + y) % 251;
            }
    }
    if (progressCallback) progressCallback(SD_PROGRESS_DECODE, 1, 1, 0, progressData);
    return true;
}
void free_sd_images(sd_image_t *images, int count) {
    for (int i = 0; i < count; ++i) std::free(images[i].data);
    std::free(images);
    allocatedImages -= count;
}
}

int main(int argc, char **argv) {
    using namespace iiLocalDiffusion;
    try {
        if (argc != 2) return 1;
        const auto directory = std::filesystem::path(argv[1]);
        std::filesystem::create_directories(directory);
        const auto model = directory / "result-fixture.safetensors";
        { std::ofstream file(model); file << "C API fixture"; }
        NativeGenerationRequest request;
        request.modelPath = std::filesystem::canonical(model);
        request.prompt = "portrait";
        request.steps = 10;
        std::atomic_bool cancelled{false};
        // width, height, horizontal crop origin, vertical crop origin.
        const std::array<std::array<int, 4>, 6> sizes{{
            {1024, 1368, 0, 20}, {1368, 1024, 20, 0}, {1024, 1824, 0, 16},
            {1824, 1024, 16, 0}, {1024, 1024, 0, 0}, {88, 72, 20, 28}}};
        for (const auto &size : sizes) {
            releaseNativeDiffusionCache();
            request.width = size[0]; request.height = size[1];
            bool decoded = false;
            const auto result = generateNativeImageWithProgress(request, cancelled, [&](const auto &event) {
                decoded |= event.stage == NativeGenerationStage::Decoding;
            });
            require(result.error.empty(), "Completed image rejected: " + result.error);
            require(decoded && !result.cancelled, "Missing decode completion");
            require(result.width == size[0] && result.height == size[1], "Wrong public dimensions");
            require(result.rgb.size() == static_cast<size_t>(size[0] * size[1] * 3), "Wrong RGB byte count");
            for (int y = 0; y < size[1]; ++y)
                for (int x = 0; x < size[0]; ++x) {
                    const auto offset = (y * size[0] + x) * 3;
                    require(result.rgb[offset] == (x + size[2]) % 251
                        && result.rgb[offset + 1] == (y + size[3]) % 251
                        && result.rgb[offset + 2] == (x + size[2] + y + size[3]) % 251,
                        "Crop changed pixels or read the wrong row stride");
                }
            require(allocatedImages == 0, "Upstream image allocation leaked");
            std::cout << "RGB and center crop verified: " << size[0] << 'x' << size[1] << '\n';
        }
        for (const auto failure : {Output::failed, Output::engineError, Output::wrongSize,
                                  Output::wrongChannels, Output::missing, Output::multiple}) {
            releaseNativeDiffusionCache();
            output = failure;
            const auto result = generateNativeImageWithProgress(request, cancelled, {});
            require(!result.error.empty() && result.rgb.empty(), "Invalid result accepted");
            require(result.error.find(warning) == std::string::npos, "Warning hid the real failure");
            if (failure == Output::engineError)
                require(result.error.find("VAE decode allocation failed") != std::string::npos, "Lost backend error");
            if (failure == Output::wrongSize)
                require(result.error.find("128x128") != std::string::npos
                    && result.error.find("128x64") != std::string::npos, "Missing expected/actual dimensions");
            require(allocatedImages == 0, "Failed image allocation leaked");
            require(!logCallback && !progressCallback, "Dangling callback state");
        }
        releaseNativeDiffusionCache();
        output = Output::valid;
        request.width = request.height = 64;
        request.timeoutMilliseconds = 200;
        for (const bool cancelPaused : {false, true}) {
            cancelled = false;
            auto control = std::make_shared<NativeExecutionControl>();
            auto future = std::async(std::launch::async, [&] {
                return generateNativeImageWithExecutionControl(request, cancelled, [&](const auto &event) {
                    if (event.stage == NativeGenerationStage::Encoding) control->setPaused(true);
                }, control);
            });
            const auto waitDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!control->isWaiting() && std::chrono::steady_clock::now() < waitDeadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            const bool parked = control->isWaiting();
            // Stay paused longer than the entire inference timeout. The same
            // request must finish after resuming, without an expired deadline.
            const bool blocked = future.wait_for(std::chrono::milliseconds(300)) == std::future_status::timeout;
            if (cancelPaused) cancelled = true;
            else control->setPaused(false);
            if (!parked || !blocked) { cancelled = true; control->setPaused(false); }
            const auto result = future.get();
            require(parked && blocked, "Paused engine submitted further compute");
            if (cancelPaused) require(result.cancelled && result.rgb.empty(), "Paused cancellation did not unwind");
            else require(result.error.empty() && !result.rgb.empty(), "Pause consumed the inference deadline: " + result.error);
            require(allocatedImages == 0 && !abortCallback, "Paused request leaked engine state");
        }
        std::cout << "Paused native requests resume without timeout and cancel while parked\n";
        releaseNativeDiffusionCache();
        std::filesystem::remove(model);
        std::cout << "Malformed images rejected; warning/error separation and cleanup verified\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
