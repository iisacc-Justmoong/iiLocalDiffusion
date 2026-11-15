#pragma once
#include "Export.hpp"
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Stable C boundary. Input and strings are borrowed for the call.
typedef struct iild_upscale_request_v1 {
    size_t size;
    const uint8_t *pixels;
    size_t pixel_bytes;
    uint32_t width, height, channels;
    // 0 nearest, 1 bilinear, 2 bicubic, 3 Lanczos3, 4 ESRGAN/RRDB,
    // 5 SRVGG, 6 SwinIR, 7 automatic native model architecture.
    int32_t kind;
    const char *model;
    // Device: 0 automatic, 1 CPU, 2 Metal, 3 DirectX, 4 Vulkan.
    // Explicit GPU choices apply to SRVGG/SwinIR (kinds 5-7).
    int32_t cpu, tile_size, threads;
    double scale;
    uint32_t target_width, target_height;
    int32_t passes;
    uint64_t max_output_bytes;
} iild_upscale_request_v1;
// Return nonzero to cancel after the completed pass.
typedef int (*iild_upscale_progress_v1)(int completed, int total, void *user);
typedef struct iild_upscale_result_v1 iild_upscale_result_v1;
IILD_EXPORT iild_upscale_result_v1 *iild_upscale_v1(const iild_upscale_request_v1 *, iild_upscale_progress_v1, void *user);
// V2 retains V1 and adds resource limits and SRVGG activation selection.
typedef struct iild_upscale_request_v2 {
    size_t size;
    iild_upscale_request_v1 image;
    uint64_t max_weight_bytes, max_working_bytes;
    int32_t srvgg_activation; // 0 PReLU, 1 ReLU, 2 LeakyReLU (slope 0.1).
} iild_upscale_request_v2;
IILD_EXPORT iild_upscale_result_v1 *iild_upscale_v2(const iild_upscale_request_v2 *, iild_upscale_progress_v1, void *user);
// V3 adds explicit SwinIR image range and QK scaling (0 selects model default).
typedef struct iild_upscale_request_v3 {
    size_t size;
    iild_upscale_request_v2 upscaler;
    float swinir_image_range, swinir_qk_scale;
} iild_upscale_request_v3;
IILD_EXPORT iild_upscale_result_v1 *iild_upscale_v3(const iild_upscale_request_v3 *, iild_upscale_progress_v1, void *user);
IILD_EXPORT const char *iild_upscale_metadata_v1(const iild_upscale_result_v1 *);
IILD_EXPORT const uint8_t *iild_upscale_pixels_v1(const iild_upscale_result_v1 *, size_t *size);
IILD_EXPORT void iild_upscale_free_v1(iild_upscale_result_v1 *);
#ifdef __cplusplus
}
#endif
