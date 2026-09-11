# Native inference third-party notices

These license texts accompany the pinned stable-diffusion.cpp backend.
The server/httplib, WebP and WebM targets are disabled.

- nlohmann-json-LICENSE.txt: https://raw.githubusercontent.com/nlohmann/json/v3.11.2/LICENSE.MIT
- zip-LICENSE.txt: https://raw.githubusercontent.com/kuba--/zip/master/LICENSE.txt
- pytorch-rng-LICENSE.txt: https://raw.githubusercontent.com/pytorch/pytorch/d01a7b0241ed1c4cded7e7ca097249feb343f072/LICENSE
- sentencepiece-LICENSE.txt: https://raw.githubusercontent.com/google/sentencepiece/v0.2.0/LICENSE
- stb-LICENSE.txt: https://raw.githubusercontent.com/nothings/stb/master/LICENSE

Darts Clone copyright and license are copied from the pinned source.
SentencePiece tokenizer code is attributed to Google Inc.; the RNG adaptation is from the referenced PyTorch revision.

The pinned native runtime also applies iiLocalDiffusion patches for staged cancellation, direct mapped uploads, and notification-based tensor-loader completion. See docs/native-image-generation.md in the SDK source for ownership, fallback, and validation contracts.

The pinned ggml Metal backend includes `native-metal-shared-upload.patch` to identify shared storage. The mapped loader copies disjoint shared-memory tensors concurrently; private GPU buffers retain serialized uploads. MIT licenses remain unchanged.

Q8 disk preparation uses the pinned MIT backend streaming converter. `native-conversion-cancellation.patch` adds cooperative abort checks before and after tensor conversion, preserving worker joining and error propagation. No new dependency is introduced.
The conversion patch also uses read-only mmap and reduces queued output reservations to 256 MiB to leave headroom for source and conversion scratch memory on mobile devices.

`native-thread-safe-logging.patch` gives concurrent logging calls independent buffers and passes preformatted ggml messages as string arguments. Conversion scratch contexts use RAII so cancellation exceptions release them.
