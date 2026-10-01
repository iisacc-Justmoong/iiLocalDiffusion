# Desktop resident-weight execution

Dreamscapes uses the component-aware C bridge (V2/V3), including Krea2's custom
sampling schedule. These calls now use `generateNativeImageWithResidentWeights`
for both preparation and generation. Previously only the separate Society host
entry point requested anonymous weights; the desktop bridge still read file-backed
weights again for each denoising segment.

The Python worker enforces `queued batch -> prepare-only -> anonymous storage
confirmation -> generation`. A failed preload or an old SDK without the anonymous
storage contract cannot start inference. Preparation is recorded separately in
`performance.preparation`; source reuse avoids a second full preload. The legacy
V1 C bridge now uses the same resident storage contract, including Anima routing.
Telemetry records `load_step/load_total`, `resident-file-loaded` after each actual
source copy and `resident_model_ready=1` only after context construction succeeds.
The final ready marker is the preload barrier; an allocation or heartbeat alone
does not mean the model is ready.

## Contract

- Read each model/component file into anonymous process memory before constructing
  the execution context. The process runtime owns these pristine source buffers,
  independently of that context. Subsequent tensor reads and segment reloads use that
  memory, not `pread` of the original external checkpoint.
- Preload is synchronous and completes before inference; it reports actual bytes
  loaded through the existing loading callback. Cancellation is checked every
  32 MiB during preload and every 8 MiB during subsequent memory copies.
- Anonymous does not mean pinned or guaranteed physical RAM: the OS may compress
  or swap pages. No application-owned disk cache, swap file or reduced-precision
  derivative is introduced. The initial source read and final image publication
  still require storage. Allocation/read failure is an error, not a silent
  file-backed fallback.
- Preparation and generation share one cached context; model/component identity,
  backend and storage policy invalidate it. Existing C request structures and
  exported symbols retain their layouts. Existing explicitly nonresident C++ APIs
  retain their compatibility behavior; all C bridge versions use resident weights.
- Preserve sampler, sigmas, seed, steps, output dimensions, source precision,
  preview and cancellation contracts. Placement stays automatic (Metal where
  supported); CPU is not run redundantly to inflate utilization.
- `performance.weight_storage` is `anonymous` for V1/V2/V3. Native traces also record
  the requested weight-storage policy. Backend labels describe module placement,
  not per-operator GPU utilization. OS swap and GPU allocations remain separately
  budgeted/observed; a 31.8 GB model on a 32 GiB machine can still exceed RAM.

## Runtime lifetime and explicit release

The native source pool has no memory-pressure eviction, idle timeout, background
release or capacity-based eviction. Successful source loads remain retained across
completed jobs, invalid execution contexts, backend changes, adapter destruction
and model selection changes. File identity (canonical path, device/inode, size,
mtime and ctime) prevents a changed source from reusing stale bytes; previously
loaded versions remain owned until explicit release. A failed/partial preload is
never published. No source checksum is recomputed merely to look up this pool.

Read-only execution contexts share the pristine anonymous allocation. Writable
in-place LoRA contexts receive private memory copies so they cannot poison later
generations. The source disk is still consulted for identity/header validation;
this policy eliminates repeated **bulk weight reads**, not every metadata access.

`releaseNativeDiffusionCache()` / `iild_native_release_v1()` releases execution and
source residency immediately when idle or after the active request. Python worker
protocol action `release` (empty arguments) and worker EOF are explicit lifecycle
boundaries. Request errors clear unsafe execution state, not the source pool.
The native library is held by the worker session, not a per-request Python adapter.
Foreground without a selected model hides readiness without unloading weights.

The application never writes a swap file to the model volume. The OS chooses
anonymous-page compression and swap placement; the SDK neither pins these pages
nor reconfigures OS swap. Allocation failures remain errors, never file-backed
fallbacks. Crash, forced process termination (including the existing desktop
hard-cancel/watchdog path), or OS kill destroys the runtime; memory cannot survive
that boundary. GPU working buffers may still have different paging restrictions.
Keeping anonymous weights does not guarantee survival under iOS memory pressure.

Dreamscapes retains its worker across idle/background transitions and no longer
kills an ongoing preparation merely because the selected model changed. It waits
for the current preparation boundary and then prepares the pending selection.
Native memory-warning callbacks log pressure without calling release. Normal app
shutdown remains the teardown owner.

## Regression and performance verification

`NativeResultTests` and `NativeMobileResultTests` check actual backend parameters
for V2/V3, prepare-to-generate cache reuse, component binding changes, custom
sigmas, output dimensions and CPU selection. `NativeBulkReadTests` checks preload
progress, cancellation after a partial preload, bounded memory-copy cancellation,
bounds and source truncation/removal. `NativeMappedMetalStorage` and
`NativeMappedPrivateStorage` load a generated fixture, remove that fixture, then
reload its segment and execute Metal twice, checking every output value.
`NativeBulkReadTests` also verifies pointer reuse after wrapper destruction,
private writable copies, explicit release with live readers, cancelled-load
nonpublication and changed-source invalidation. `NativeResultTests` verifies that
failed generation does not release runtime sources. `ForegroundInferenceTests`
checks idle/selection transitions, explicit release, request failure and EOF.

Use real traces to compare total generation time and completed images; memory-copy
benchmarks and fixtures alone do not establish an end-to-end speedup. Never start
a second full-size model while the active worker already consumes constrained
unified memory. A running worker retains its loaded dylib/context; new code cannot
retroactively change its weight storage.
