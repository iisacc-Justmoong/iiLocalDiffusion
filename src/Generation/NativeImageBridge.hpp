#pragma once
#include "Export.hpp"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Versioned C boundary for the SDK's Python worker. Strings are UTF-8 and
// borrowed for the duration of the call. Results must be freed by this library.
typedef struct iild_native_lora_v1 { const char *path; float strength; } iild_native_lora_v1;
typedef struct iild_native_request_v1 {
    size_t size;
    const char *model;
    const char *prompt;
    const char *negative_prompt;
    const char *resources;
    int32_t width, height, steps;
    int64_t seed;
    int32_t default_modifiers, prepare_only;
    // Zero retains the desktop worker's unrestricted wall time. Positive
    // values set a caller-owned time limit; cancellation remains available.
    int32_t timeout_milliseconds;
    const iild_native_lora_v1 *loras;
    size_t lora_count;
} iild_native_request_v1;
// Returning nonzero requests cooperative cancellation. The worker may also
// terminate its own process; there are no child inference servers or processes.
typedef int (*iild_native_progress_v1)(int stage, int step, int total, void *user);
// RGB bytes are borrowed only until the callback returns. Nonzero cancels.
typedef int (*iild_native_preview_v1)(int sequence, int step, int total, int width, int height,
    const uint8_t *rgb, size_t size, void *user);
typedef struct iild_native_result_v1 iild_native_result_v1;

IILD_EXPORT int iild_native_available_v1(void);
IILD_EXPORT iild_native_result_v1 *iild_native_generate_v1(const iild_native_request_v1 *,
    iild_native_progress_v1, void *user);
IILD_EXPORT iild_native_result_v1 *iild_native_generate_with_preview_v1(const iild_native_request_v1 *,
    iild_native_progress_v1, iild_native_preview_v1, void *user);
IILD_EXPORT const char *iild_native_metadata_v1(const iild_native_result_v1 *);
IILD_EXPORT const uint8_t *iild_native_rgb_v1(const iild_native_result_v1 *, size_t *size);
IILD_EXPORT void iild_native_free_v1(iild_native_result_v1 *);
IILD_EXPORT void iild_native_release_v1(void);

#ifdef __cplusplus
}
#endif
