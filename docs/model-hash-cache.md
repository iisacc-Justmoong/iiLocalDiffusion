# Persistent model hash cache

`cached_model_sha256` retains verified SHA-256 values across worker restarts.
The identity includes canonical path, device, inode, size, nanosecond mtime and
metadata change time (Windows uses ChangeTime). A cache miss still reads every
byte; a changed identity always invalidates the entry. Before publishing a hash,
the signature is checked again. Publisher manifests never seed this cache.

SQLite stores at most 1024 paths. macOS defaults to
`~/Library/Caches/iiLocalDiffusion/model-hashes.sqlite3`; other platforms use
`$XDG_CACHE_HOME/iiLocalDiffusion` or `~/.cache/iiLocalDiffusion`.
`IILD_MODEL_HASH_CACHE` overrides the file; `off` disables persistent caching.
The in-process cache remains bounded. Cache I/O failure, corruption, or a lock
lasting more than 50 ms falls back to a full content hash. This cache is a local
performance aid, not protection from an attacker who can modify the same user's
cache. Output digests continue to use uncached `file_sha256`.

Full reads reuse one 8 MiB buffer rather than allocating 1 MiB blocks repeatedly.
Progress still reports actual bytes. First verification and model loading remain
real work; an unchanged cache hit reads no tensor bytes.

Build/check: compile the Python sources with `python3 -m compileall`, then run
`tests/PersistentModelHashTests.py` and `tests/InferenceCacheTests.py`. The former
is also registered with CTest. Tests cover reuse in a new process, same-size edits
with restored mtime, inode replacement, mutation during hashing, corrupt and
unavailable cache files. Tests isolate all generated files under `build/`.

## Metadata-only interactive generation

Dreamscapes sets `IILD_MODEL_VALIDATION=metadata` on its inference worker and
inherited runtime subprocesses. `model_content_sha256` then returns `None` after
checking file metadata; it never consults the checksum cache or reads model
contents. This applies on the first request, not only cache hits. Local model
provenance records `sha256: null` and `validation: metadata` rather than inventing
or trusting a checksum. Checkpoint, LoRA, VAE, ControlNet and generic directory
model inputs use this policy. Unified packages read their small manifest, check
canonical member paths and nonempty regular files, and proceed to the native
loader without matching member SHA-256 or published member sizes.

Legacy conversion still performs the necessary weights-only deserialization and
writes converted tensors. In metadata mode its cache key is explicitly prefixed
`metadata-` and derived from the source file signature. It does not perform extra
whole-file checksum passes; strict conversion caches remain separate. The
weights-only loader and path containment rules are unchanged.

Content/architecture failures from the real loader or generator propagate to
the caller. Generated output checksums remain real checksums. Strict SDK callers
and import/merge checksum APIs retain full verification unless they explicitly
select metadata mode. The persistent cache above serves those strict callers.

`MetadataValidationTests` guards the hash API with an exception and verifies
first-use resolution, generic models, safetensors conversion entry points and a
bad-checksum package reaching the loader. `CheckpointConversionTests` verifies
metadata conversion/reuse without stream checksums while retaining weights-only
loading. Dreamscapes tests the worker policy during preparation and generation.

The native C++ `modelIdentity` also honors `IILD_MODEL_VALIDATION=metadata`.
This uses file stat identity for package members and external VAEs, avoiding a
second full-model FNV pass after the Python preflight. Without this option,
offline native tools retain their content identity behavior.
