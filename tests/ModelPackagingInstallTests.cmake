if(NOT DEFINED BUILD_DIRECTORY OR NOT DEFINED STAGE_DIRECTORY OR NOT DEFINED BIN_DIRECTORY)
    message(FATAL_ERROR "Build, staging, and binary directories are required")
endif()

if(EXISTS "${STAGE_DIRECTORY}" OR IS_SYMLINK "${STAGE_DIRECTORY}")
    message(FATAL_ERROR "Packaging stage already exists; refusing to replace it")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIRECTORY}"
        --prefix "${STAGE_DIRECTORY}" --component ModelPackaging
    RESULT_VARIABLE install_result OUTPUT_VARIABLE install_output ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "ModelPackaging installation failed: ${install_output}\n${install_error}")
endif()
if(NOT EXISTS "${STAGE_DIRECTORY}/share/iiLocalDiffusion/python/iild_model_packaging.py")
    message(FATAL_ERROR "The ModelPackaging Python API was not installed")
endif()

# An argument error must reach the installed CLI after the loader resolves all
# dependencies; do not supply a developer-machine dynamic-library search path.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        --unset=DYLD_LIBRARY_PATH --unset=DYLD_FALLBACK_LIBRARY_PATH
        --unset=DYLD_FRAMEWORK_PATH --unset=DYLD_FALLBACK_FRAMEWORK_PATH
        --unset=DYLD_VERSIONED_LIBRARY_PATH --unset=DYLD_VERSIONED_FRAMEWORK_PATH
        --unset=DYLD_INSERT_LIBRARIES --unset=DYLD_ROOT_PATH
        "${STAGE_DIRECTORY}/${BIN_DIRECTORY}/iild-model-package"
    RESULT_VARIABLE run_result OUTPUT_VARIABLE run_output ERROR_VARIABLE run_error)
if(NOT run_result EQUAL 2 OR NOT run_output MATCHES "iild-model-package-report-v1")
    message(FATAL_ERROR "Installed ModelPackaging CLI failed: ${run_result}\n${run_output}\n${run_error}")
endif()

if(DEFINED PYTHON_EXECUTABLE AND EXISTS "${PYTHON_EXECUTABLE}")
    # Exercise the installed Python interface and the relocated CLI together
    # using a tiny synthetic header; no pretrained model is executed.
    set(probe [=[
import json, pathlib, struct, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / 'share/iiLocalDiffusion/python'))
import iild_model_packaging as api
model = root / 'probe.safetensors'
header = json.dumps({'probe.weight': {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
model.write_bytes(struct.pack('<Q', len(header)) + header + b'x')
tool = root / sys.argv[2] / 'iild-model-package'
report = api.contains_tensors(model, ['probe.weight', 'missing.weight'], executable=tool)
assert report['present'] is False and report['matches'] == ['probe.weight']
assert api.find_tensor_names(model, ['probe.weight'], executable=tool) == ['probe.weight']
snapshot = root / 'linked-snapshot'
snapshot.mkdir()
index_blob = root / 'linked-index-blob'
index_blob.write_text(json.dumps({'weight_map': {'probe.weight': 'part.safetensors'}}))
index_link = snapshot / 'model.safetensors.index.json'
shard_link = snapshot / 'part.safetensors'
try:
    index_link.symlink_to(index_blob)
    shard_link.symlink_to(model)
except (OSError, NotImplementedError):
    pass  # Windows can deny symlink creation to an unprivileged test account.
else:
    indexed = api.inspect_sharded_index(index_link, executable=tool)
    assert pathlib.Path(indexed['path']) == index_link
    assert pathlib.Path(indexed['resolved_path']) == index_blob.resolve()
    assert pathlib.Path(indexed['shards'][0]['resolved_path']) == model.resolve()
    assert api.find_tensor_names(index_link, ['probe.weight'], executable=tool) == ['probe.weight']
wan = root / 'wan-pipeline'
wan.mkdir()
(wan / 'model_index.json').write_text(json.dumps({
    '_class_name': 'WanPipeline', 'scheduler': ['diffusers', 'UniPCMultistepScheduler'],
    'text_encoder': ['transformers', 'UMT5EncoderModel'], 'tokenizer': ['transformers', 'T5TokenizerFast'],
    'transformer': ['diffusers', 'WanTransformer3DModel'], 'vae': ['diffusers', 'AutoencoderKLWan']}))
for name in ('transformer', 'text_encoder', 'vae'):
    (wan / name).mkdir()
    (wan / name / 'model.safetensors').write_bytes(model.read_bytes())
route = api.inspect_wan_directory(wan, executable=tool)
assert pathlib.Path(route['model']) == (wan / 'transformer/model.safetensors').resolve()
assert route['scheduler'] == 'UniPCMultistepScheduler' and route['high_noise_model'] is None
manifest = json.loads((wan / 'model_index.json').read_text())
manifest.update({'_class_name': 'WanImageToVideoPipeline',
                 'image_encoder': ['transformers', 'CLIPVisionModelWithProjection'],
                 'image_processor': ['transformers', 'CLIPProcessor']})
(wan / 'model_index.json').write_text(json.dumps(manifest))
(wan / 'image_encoder').mkdir()
(wan / 'image_encoder/model.safetensors').write_bytes(model.read_bytes())
route = api.inspect_wan_directory(wan, executable=tool)
assert route['pipeline'] == 'WanImageToVideoPipeline'
assert pathlib.Path(route['vision_encoder']) == (wan / 'image_encoder/model.safetensors').resolve()
]=])
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        --unset=DYLD_LIBRARY_PATH --unset=DYLD_FALLBACK_LIBRARY_PATH
        --unset=DYLD_FRAMEWORK_PATH --unset=DYLD_FALLBACK_FRAMEWORK_PATH
        --unset=DYLD_VERSIONED_LIBRARY_PATH --unset=DYLD_VERSIONED_FRAMEWORK_PATH
        --unset=DYLD_INSERT_LIBRARIES --unset=DYLD_ROOT_PATH
        "PYTHONDONTWRITEBYTECODE=1" "${PYTHON_EXECUTABLE}" -c "${probe}" "${STAGE_DIRECTORY}" "${BIN_DIRECTORY}"
        RESULT_VARIABLE probe_result OUTPUT_VARIABLE probe_output ERROR_VARIABLE probe_error)
    if(NOT probe_result EQUAL 0)
        message(FATAL_ERROR "Installed ModelPackaging Python API failed: ${probe_result}\n${probe_output}\n${probe_error}")
    endif()
endif()
