# Model calibration

`iiLocalDiffusion::ModelPackaging` provides C++23 APIs for bounded Safetensors
weight inspection and for creating a corrected copy. `inspectModelWeights`
counts non-finite IEEE F16, BF16, F32 and F64 values; it reports unchecked
tensor types separately. Inspection does not execute a model or establish its
generation quality. Integer, FP8 and other encodings are not numerically
validated by this operation.

`calibrateModelPackage` verifies an iiLocalDiffusion package, reconstructs its
included original components, reclassifies their roles using the current
packager, and creates and verifies a new package. It leaves tensor bytes
unchanged by default. Qwen language-model tensors are recognized as a text
encoder when a main diffusion transformer is not present. A full MOSS
transformer signature takes precedence over language tensors. Recognition
does not guarantee that a downstream inference engine can execute the model.

An explicit `replaceNonfiniteWithZero` option replaces non-finite values in
supported floating-point tensors in the reconstructed copy. For standalone
Safetensors, `calibrateModelFile` provides the same explicit numeric operation;
it refuses embedded iiLocalDiffusion packages so their checksums cannot be
invalidated. Replacing values changes weights and is not a quality guarantee.
The source and any existing destination are never overwritten. Temporary
files are removed on normal completion, failure and cooperative cancellation.
The operation can require space for reconstructed components and a second
package. Power loss or forcible process termination may leave temporary files.

The `iild-model-package` CLI exposes `weight-health`, `calibrate`, and
`calibrate-file` operations. Numeric repair requires `--repair-nonfinite true`.
The Python module provides `inspect_weights`, `calibrate_package`, and
`repair_standalone_weights`. `tests/ModelPackagingTests.py` uses synthetic
models to verify classification, package reconstruction, numeric repair,
source preservation, output verification, and unchecked encodings. No
pretrained model inference or visual-quality improvement is asserted.
