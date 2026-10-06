<a id="모델-리소스"></a>

# Model resources

The `.safetensors` and `.pt` originals in this directory are managed via Git LFS. Model files are not placed in Git's regular blobs, and the originals are downloaded with the following command after receiving the repository.

```sh
git lfs install --local
git lfs pull
git lfs fsck
```

`.gitattributes` defines tracking rules by resource extension. Each original in the LFS pointer contains the SHA-256 and byte size. The `generation-defaults.json` specified LoRA and negative embeddings are now global image generation defaults and are included in CMake installation. SDXL uses 7 embeddings and does not apply automatic LoRA. The default for `fallback_loras` is an empty array. Detailed priority and consumer deployment are in the [global defaults documentation](../docs/generation-defaults.md).

The default LoRA of other model families registers compatible files, strength, family, hash, and size in the `fallback_loras` array. While existing single `fallback_lora` formats are also supported, the bundle does not register the default LoRA. Image and Deforum, and the native runtime use the same family selection rules. Changing only the family name of SDXL weights is not a conversion; actual LoRA files matching each base model are required.

`embeddings/` contains copies of 3 original `.pt` files preserved as tensor-only safetensors. When the originals change, recreate them with `scripts/prepare_generation_defaults.py` and verify vector equality and hashes with `--check`. The original `.pt` files are also retained.

`vae/` contains the original VAEs and configurations for Qwen Image RGB, SDXL, FLUX.1, and FLUX.2. `fallback_vae` (the existing Qwen entry) and the `fallback_vaes` array record pinned revisions, SHA-256, sizes, and compatible families. Native/Diffusers attaches the corresponding family's VAE to models without one; this remains active even when the default LoRA is disabled. An embedded VAE takes precedence, and different latent spaces are not mixed. Redownload and verification commands are in the [defaults document](../docs/generation-defaults.md#vae-자동-폴백). `.cache` is excluded from Git and installation. Qwen/FLUX retain the original Apache-2.0 text, while SDXL preserves the official model card declaring MIT. Evidence that the FLUX.1 original and public copy are identical is in `vae/flux1/NOTICE.md`.
