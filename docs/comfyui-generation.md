# Local ComfyUI generation

`reference/generate.py --backend comfyui` executes an explicit API workflow
against an existing loopback ComfyUI server. Model architectures, quantized/GGUF
loaders, split components, videos, audio and meshes are handled by their installed
nodes. The client adds no package dependency and does not install nodes or weights.

Export a workflow with ComfyUI **File → Export (API)**. Visual-editor JSON with
`nodes`/`links` is rejected. API JSON maps node IDs to `class_type` and `inputs`.
Check availability without queueing:

```bash
python3 reference/generate.py --backend comfyui \
  --workflow /absolute/path/workflow-api.json --validate-only
```

The workflow determines all components, task inputs, LoRA, sampling and export.
Set inputs explicitly by node ID; prompt nodes are never guessed:

```bash
python3 reference/generate.py --backend comfyui --base-model Illustrious \
  --workflow /absolute/path/workflow-api.json \
  --workflow-inputs '{"6":{"text":"a lighthouse at sunrise"},"3":{"seed":42}}' \
  --output-dir build/reference/illustrious-workflow
```

IDs `6`/`3` are examples that must exist in the supplied workflow.
`--workflow-inputs @/path/values.json` reads the same object from a file. Only
existing inputs can be overridden. Local model names must match the server's
inventory; image/video inputs must already exist on that local server.

`--print-config` is offline. `--validate-only` checks live node availability,
required inputs, model/enum choices, graph links and cycles. `/prompt` performs
the server's final validation. A successful preflight is not a generation result.
Hosted API nodes identified by ComfyUI are rejected. Endpoints are restricted to
localhost/loopback; proxy variables and redirects cannot redirect model prompts.

`submission.json` records the prompt ID, patched workflow and its SHA-256 before
waiting for a successful completed `/history/{prompt_id}` entry. Node errors,
empty outputs, timeouts and download errors fail. A timeout does not cancel or
resubmit a potentially running job; inspect its recorded prompt ID in ComfyUI.

Image/video/audio/mesh file descriptors returned by output nodes are downloaded
through `/view`. Paths are checked, streaming has a size limit, and files publish
atomically. `generation.json` is written only after all artifacts arrive, with
hashes and sizes. A new or empty output directory preserves existing user files.

Defaults: `http://127.0.0.1:8188`, 3600-second job timeout, one-second polling,
2 GiB per-artifact limit. `--timeout`, `--poll-interval`, and
`--max-artifact-bytes` expose these limits. Requests are bounded to 30 seconds or
the shorter job timeout. Output storage defaults under `build/`.

The base-model label is declared provenance; the client does not inspect
server-side tensors or prove that a workflow loaded that named model. HTTP
integration tests use a local fixture server to verify preflight, queueing,
completion, bytes, hashes, errors and preservation of existing files. They do
not prove actual generation support for every model or plugin.

Protocol sources: [server routes](https://docs.comfy.org/development/comfyui-server/comms_routes)
and [official API examples](https://github.com/Comfy-Org/ComfyUI/tree/master/script_examples).
