#pragma once

#include "Export.hpp"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// UTF-8 inputs are borrowed for the call. Results and their JSON strings are
// owned by the library until iild_image_package_free_v1 is called.
typedef struct iild_image_package_result_v1 iild_image_package_result_v1;
typedef int (*iild_image_package_progress_v1)(const char *detail, uint64_t completed_bytes,
                                               uint64_t total_bytes, void *user);

// Inspection reads bounded model metadata and any required embedded indexes;
// it does not load weights or promise that an engine can execute the model.
IILD_EXPORT iild_image_package_result_v1 *iild_image_package_inspect_v1(const char *model);

// Original component groups and tensor-name mappings. Indexed legacy groups
// are reconstructed from their verified embedded index during inspection.
IILD_EXPORT iild_image_package_result_v1 *iild_image_package_catalog_v1(const char *model);

// Standalone index inspection and inexpensive dependency discovery. The
// complete inspection verifies the exact tensor-to-shard mapping.
IILD_EXPORT iild_image_package_result_v1 *iild_image_shard_index_inspect_v1(const char *index);
IILD_EXPORT iild_image_package_result_v1 *iild_image_shard_index_paths_v1(const char *index);
IILD_EXPORT iild_image_package_result_v1 *iild_image_pytorch_index_inspect_v1(const char *index);
IILD_EXPORT iild_image_package_result_v1 *iild_image_container_probe_v1(const char *model);

// Inspect a local Wan Diffusers directory without loading tensor data.
IILD_EXPORT iild_image_package_result_v1 *iild_video_wan_directory_inspect_v1(const char *directory);

// Reconstructs one exact original component or shard-index path. The observer
// may request cooperative cancellation by returning nonzero.
IILD_EXPORT iild_image_package_result_v1 *iild_image_package_materialize_v1(
    const char *model, const char *cache, const char *relative_component,
    iild_image_package_progress_v1 progress, void *user);

// Inspect the model's VAE contract and validate a separate candidate using
// the native engine's full metadata contract. No weights are executed.
IILD_EXPORT iild_image_package_result_v1 *iild_image_vae_inspect_v1(const char *model);
IILD_EXPORT iild_image_package_result_v1 *iild_image_vae_validate_v1(const char *model, const char *candidate);
IILD_EXPORT iild_image_package_result_v1 *iild_image_text_encoder_inspect_v1(const char *model, const char *slot);
IILD_EXPORT iild_image_package_result_v1 *iild_image_text_encoder_validate_v1(
    const char *model, const char *slot, const char *candidate);

IILD_EXPORT const char *iild_image_package_json_v1(const iild_image_package_result_v1 *result);
IILD_EXPORT void iild_image_package_free_v1(iild_image_package_result_v1 *result);

#ifdef __cplusplus
}
#endif
