# Native Pose preprocessing

The Figma `237:6488` Process=Pose control requires actual image-to-pose inference,
not passing the original RGB image through as a ControlNet hint. This SDK now
contains a native DWPose path connected to the product's resource selection,
immutable queue snapshot and ControlNet forwarding. This is a minimum operational
integration; real trained-model quality and detailed behavior remain deferred.

## Pipeline and boundaries

- `NativePoseSession` owns inline YOLOX-L and DWPose ONNX files and two CPU
  inference sessions. `process` returns an owned RGB pose hint of the same size.
- `PoseProcessing` is a private, Qt/OpenCV/Python-free C++23 numerical boundary:
  640x640 top-left YOLOX letterbox, BGR CHW detector input, person-only grid
  decoding and NMS, per-person 1.25x affine crop, normalized RGB pose input,
  133-joint SimCC decoding, synthetic neck and OpenPose body remapping, colored
  body/hand limbs and face landmarks on a black canvas.
- Detector output is `[1,8400,85]`, strides 8/16/32, person score >0.3 and inclusive
  IoU NMS 0.45. Pose input is `[1,3,256,192]` or `[1,3,384,288]`; outputs are ordered
  x/y SimCC tensors `[1,133,2*width]`, `[1,133,2*height]` with split ratio 2.
- The RGB crop convention follows MMPose's `bgr_to_rgb=True` training
  preprocessor, mean `[123.675,116.28,103.53]`, std `[58.395,57.12,57.375]`.
  This resolves the inconsistent color convention in the original BGR example
  versus RGB Gradio caller. Real-checkpoint parity remains to be established.
- No detections produces a black hint, not a fabricated full-image person.
  At most 64 people and images up to 2048 per side are accepted; exceeding these
  limits fails rather than silently dropping people. Detector/person ordering
  remains deterministic. Feet are not rendered, matching DWPose's ControlNet
  visualization; body, hands and face are rendered.
- Rasterization uses clipped analytic ellipses/circles and segment distance,
  not OpenCV's polygon routines. Color/topology contracts are tested; exact
  pixel parity with an OpenCV rendering is not asserted.

## Memory and lifetime

Model files are canonical local regular `.onnx` files, read completely into owned
anonymous byte arrays in cancellable chunks, with before/after metadata identity
checks. ONNX sessions are created **from those arrays**, not filenames. Both arrays
and sessions remain in the owner across calls and pauses. No memory-pressure/idle
eviction, mmap, model download, conversion cache, or serialized optimized model
is introduced. The OS may compress/page anonymous memory normally.

`OnnxInlineModel` walks bounded protobuf wire fields before creating a session,
rejecting external data/data-location fields in initializers, attributes, nested
graphs, functions, sparse tensors and training graphs. External sidecar tensors
are forbidden even when an absolute path exists; an array-based session alone
is not relied upon as proof of that invariant. ONNX Runtime still performs
semantic graph validation. Inline model files must be nonempty and below 2 GiB.

`processNativePose` retains sessions in a process-runtime cache keyed by absolute
model pair and thread count, until explicit `releaseNativePoseCache`,
`releaseNativeDiffusionCache`, or teardown. Cache hits do not re-stat or read the
source files; replacing files at the same paths requires explicit release.
Idle, pause and cancellation do not evict loaded models. Calls
serialize through a cancellable lock. `NativeExecutionControl`
parks between runs/person crops; an ONNX run already active finishes on pause.
Cancellation requests ORT RunOptions termination, and a cancelled call does not
destroy its session. Session initialization cannot yet be interrupted inside
ORT; cancellation is checked around it and while reading files.

## Build and execution provider

`IILD_ENABLE_POSE=ON` enables the optional CPU ONNX path; it is OFF by default.
The current macOS ARM64 build downloads ONNX Runtime
1.30.0 into `build/_deps`, pinned by SHA-256. Other targets require an explicit
target-native `IILD_ONNXRUNTIME_ROOT`; iOS and other packaged platforms remain
unverified. No system/global runtime is installed. ONNX provider selection is
explicit CPU, not a silent fallback from the diffusion engine's Metal backend.
Native CPU intra-op threads default to hardware concurrency, inter-op=1;
idle thread spinning is disabled. No CoreML compiled-model cache is used.

```
env -u CPATH -u CPLUS_INCLUDE_PATH cmake -S . -B build -DIILD_ENABLE_POSE=ON
env -u CPATH -u CPLUS_INCLUDE_PATH cmake --build build --target iiLocalDiffusion PoseProcessingTests NativePoseTests NativePoseDisabledTests -j 4
ctest --test-dir build -R '^(PoseProcessingTests|NativePoseTests|NativePoseDisabledTests)$' --output-on-failure
```

When a previously offline build needs its first dependency download, temporarily
configure `FETCHCONTENT_FULLY_DISCONNECTED=OFF`, then restore it after population.
Packaged consumers must carry the ORT shared library and license notices; build
availability is not evidence of a packaged or installed app update.

## Verification scope and remaining work

The numerical tests cover color ordering, letterboxing, all detector strides,
multi-person NMS, coordinate transforms, SimCC confidence, neck/body remapping,
hands/face, black background and malformed/nonfinite input. Native tests construct
small ONNX graphs with known tensor outputs and execute the real C++ ORT API.
They verify two independently positioned people, shape rejection, cancellation,
pause/resume and repeated inference after both fixture model files are removed.
These fixtures are **not learned models** and do not prove detector accuracy,
generation quality, real-model throughput or installed-app behavior.

On 2026-09-29, the shared SDK plus all three Pose test targets built successfully.
The 26 selected native suites (Pose, parameters, memory policy, ControlNet,
IP-Adapter, Refiner, Detailer, upscaling and conditioning) passed in 24.40 seconds.
Evidence: `build/pose-red.log`, `build/pose-build.log`, `build/pose-tests.log` and
`build/pose-native-regression.log`. The red numerical contract failed before
implementation. No SDK installation/staging or Dreamscapes reinstall was done.

The minimum integration adds `poseDetector`/`poseModel` resources, atomic product
selection, queue forwarding, runtime retention and a pose hint passed to the
existing ControlNet API. Tests cover cache reuse after source removal, explicit
release, queue resource forwarding and picker cancellation. Real YOLOX/DWPose
checkpoint accuracy, image parity and end-to-end learned-model generation remain
unverified and are deferred under the user's minimum-operation scope.

### Minimum integration verification (2026-09-29)

Shared SDK and affected test targets rebuilt successfully. Six selected suites
passed in 28.78 seconds: PoseProcessing, NativePose, disabled Pose, parameters,
native result forwarding and its mobile-contract variant. The SDK was staged
under `build/install` for the product build, including ONNX Runtime and licenses;
the global SDK installation was not changed. Logs:
`build/pose-integration-build.log`, `build/pose-integration-tests.log`,
`build/pose-integration-stage.log`.

## Provenance

Tensor/pose conventions reference [DWPose's ONNX implementation](https://github.com/IDEA-Research/DWPose/tree/3dca5db79d9f9ffdd378753ddf6ec66535aace88/ControlNet-v1-1-nightly/annotator/dwpose)
and its [MMPose configuration](https://github.com/IDEA-Research/DWPose/blob/3dca5db79d9f9ffdd378753ddf6ec66535aace88/mmpose/configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py).
Modified native implementation; original attributions/license are retained in
`docs/licenses/DWPose.txt`. ONNX Runtime's license and third-party notices are
installed alongside its shared libraries when enabled. Model weight rights are
separate from source/runtime licensing; no pretrained weight is bundled here.
