# Core ML component conversion

This is a separate offline conversion/validation tool. The C++ runtime loads
its compiled artifact directly without calling Python. It neither downloads
a model nor converts a whole diffusion pipeline. Exact linear weight keys
are required; synthetic weights require the explicit `--fixture` flag.
An empty model argument is an error and never selects synthetic weights.

Use a dedicated Python 3.13 environment. coremltools 9.0 is the reviewed stable
release (BSD-3-Clause) with a matching macOS wheel; do not install it into the
Python 3.14 Diffusers environment. Native loading uses the system Core ML
framework, not the Python package. This conversion path requires macOS 14.4+
for compute-plan inspection.

```bash
UV_PYTHON_INSTALL_DIR="$PWD/build/dependencies/python" \
UV_CACHE_DIR="$PWD/build/dependencies/uv-cache" \
  uv venv --python 3.13 build/reference/coreml-venv
UV_CACHE_DIR="$PWD/build/dependencies/uv-cache" \
  uv pip install --python build/reference/coreml-venv/bin/python \
    -r reference/coreml/requirements.txt
cmake -S . -B build
cmake --build build --parallel

build/reference/coreml-venv/bin/python reference/coreml/convert_linear.py \
  --model /absolute/path/to/text-encoder.safetensors \
  --weight-key text_model.encoder.layers.0.mlp.fc1.weight \
  --bias-key text_model.encoder.layers.0.mlp.fc1.bias --batch-size 128 \
  --output-dir build/reference/neural/my-clip-component --native-test build/CoreMLTests

./build/iild-run neural-compute \
  --model build/reference/neural/my-clip-component/model.mlmodelc
```

The source may contain FP32, FP16, or BF16 tensors. Weights are converted to
FP16 internal computation; non-finite or out-of-FP16-range values are rejected.
`--io-precision fp32|fp16` controls the compiled interface (default FP32).
The conversion maps `[out,in]` weights to `[out,in,1,1]` convolution and
binds inputs to `[1,in,1,batch]`, outputs to `[1,out,1,batch]`. Batch size and
shape influence Core ML placement; a small model can prefer CPU despite ANE
being available. The default rejects such a plan; `--allow-cpu-plan` is an
explicit diagnostic override, not a way to force ANE.

Outputs are `model.mlpackage`, `model.mlmodelc`, `plan.json`, `oracle.json`,
and `provenance.json`; `--native-test` additionally records native logs.
Existing files are never overwritten. `--native-test` compares actual C++
prediction with the independent NumPy FP32 oracle; omission reports
conversion without numerical validation. The default tolerance is
`0.005 + 0.01*abs(expected)`, recorded along with any explicit `--atol`/`--rtol`
override. Source and artifact hashes identify exactly what was tested.
Models' licenses and rights remain the caller's responsibility; a component
conversion is not a full-model compatibility or image-quality certification.

Run an explicit synthetic API check, or add a verified fixture to CTest:

```bash
build/reference/coreml-venv/bin/python reference/coreml/convert_linear.py \
  --fixture --io-precision fp16 --output-dir build/reference/neural/my-fixture \
  --native-test build/CoreMLTests
cmake -S . -B build \
  -DIILD_COREML_TEST_FIXTURE="$PWD/build/reference/neural/my-clip-component"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

See [Neural Engine / Tensor Core contracts](../../docs/neural-accelerators.md)
for permitted devices, fail-closed plan checks, unsupported interfaces,
hardware-evidence limits, and the independent CUDA precision policy.
