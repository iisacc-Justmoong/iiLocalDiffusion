#pragma once
#include "NativeImageBridge.hpp"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct iild_native_video_shot_v1 {
    const char *prompt, *negative_prompt;
    int64_t seed;
    int32_t frames, fps;
    const uint8_t *first_rgb, *last_rgb;
    size_t first_size, last_size;
    float conditioning_strength;
    int32_t continue_previous;
} iild_native_video_shot_v1;
typedef struct iild_native_video_request_v1 {
    size_t size;
    const char *model, *text_encoder, *asset_cache;
    const char *backend, *precision;
    int32_t width, height, steps, threads;
    float guidance_scale;
    int32_t cpu, vae_tiling, cpu_text_encoding;
    const iild_native_video_shot_v1 *shots;
    size_t shot_count;
} iild_native_video_request_v1;
// Optional additive request; -1 selects native defaults for expert controls.
typedef struct iild_native_video_components_v1 {
    size_t size;
    const char *video_vae, *high_noise_model, *vision_encoder;
    float moe_boundary;
    int32_t high_noise_steps;
    float high_noise_guidance;
} iild_native_video_components_v1;
typedef int (*iild_native_video_frame_v1)(size_t shot, int32_t index, int32_t width, int32_t height,
    const uint8_t *rgb, size_t size, void *user);
typedef struct iild_native_video_result_v1 iild_native_video_result_v1;
IILD_EXPORT iild_native_video_result_v1 *iild_native_video_generate_v1(const iild_native_video_request_v1 *,
    iild_native_progress_v1, iild_native_video_frame_v1, void *user);
// Same v1 request layout, with opt-in Low VRAM execution.
IILD_EXPORT iild_native_video_result_v1 *iild_native_video_generate_low_vram_v1(const iild_native_video_request_v1 *,
    iild_native_progress_v1, iild_native_video_frame_v1, void *user);
IILD_EXPORT iild_native_video_result_v1 *iild_native_video_generate_with_components_v1(const iild_native_video_request_v1 *,
    const iild_native_video_components_v1 *, iild_native_progress_v1, iild_native_video_frame_v1, void *user);
IILD_EXPORT iild_native_video_result_v1 *iild_native_video_generate_low_vram_with_components_v1(const iild_native_video_request_v1 *,
    const iild_native_video_components_v1 *, iild_native_progress_v1, iild_native_video_frame_v1, void *user);
IILD_EXPORT const char *iild_native_video_metadata_v1(const iild_native_video_result_v1 *);
IILD_EXPORT void iild_native_video_free_v1(iild_native_video_result_v1 *);
#ifdef __cplusplus
}
#endif
