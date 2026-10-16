<a id="모델-리소스"></a>

# Model resources

The `.safetensors` and `.pt` originals in this directory are managed via Git LFS, except for the oversized Qwen3-VL encoder described below. Model files are not placed in Git's regular blobs, and the LFS-managed originals are downloaded with the following command after receiving the repository.

```sh
git lfs install --local
git lfs pull
git lfs fsck
```

`.gitattributes` defines tracking rules by resource extension. Each original in the LFS pointer contains the SHA-256 and byte size. The `generation-defaults.json` specified LoRA and negative embeddings are now global image generation defaults and are included in CMake installation. SDXL uses 7 embeddings and does not apply automatic LoRA. The default for `fallback_loras` is an empty array. Detailed priority and consumer deployment are in the [global defaults documentation](../docs/generation-defaults.md).

The default LoRA of other model families registers compatible files, strength, family, hash, and size in the `fallback_loras` array. While existing single `fallback_lora` formats are also supported, the bundle does not register the default LoRA. Image and Deforum, and the native runtime use the same family selection rules. Changing only the family name of SDXL weights is not a conversion; actual LoRA files matching each base model are required.

`embeddings/` contains copies of 3 original `.pt` files preserved as tensor-only safetensors. When the originals change, recreate them with `scripts/prepare_generation_defaults.py` and verify vector equality and hashes with `--check`. The original `.pt` files are also retained.

`vae/` contains the original VAEs and configurations for Qwen Image RGB, SDXL, FLUX.1, FLUX.2, and Wan2.1. `fallback_vae` (the existing Qwen entry) and the `fallback_vaes` array record pinned revisions, SHA-256, sizes, and compatible families. Native/Diffusers attaches the corresponding family's VAE to models without one; this remains active even when the default LoRA is disabled. An embedded VAE takes precedence, and different latent spaces are not mixed. Redownload and verification commands are in the [defaults document](../docs/generation-defaults.md#vae-자동-폴백). `.cache` is excluded from Git and installation. Qwen/FLUX retain the original Apache-2.0 text, while SDXL preserves the official model card declaring MIT. Evidence that the FLUX.1 original and public copy are identical is in `vae/flux1/NOTICE.md`.

`text_encoders/` contains the pinned original Qwen3-VL-4B BF16 encoder for split Krea2 denoisers. `fallback_llms` records its family, class, revision, size, and SHA-256. The native loader normalizes the original Hugging Face `model.language_model` namespace in memory, validates all required tensor shapes, and mounts the encoder together with the Wan2.1 VAE. Complete embedded components and explicit selections retain precedence. These required components remain active when optional generation modifiers are disabled. `scripts/prepare_generation_defaults.py --check` verifies their hashes using streaming reads. Consumer applications provision these resources outside the app bundle, in the Society drive.

The encoder is 8,875,719,384 bytes, exceeding [GitHub's maximum Git LFS object size](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). Its payload is therefore ignored by Git; its pinned source identity, license, expected size, and SHA-256 remain versioned. Existing local copies are preserved. After cloning, provision this original from its immutable upstream revision before installing the resources or using split Krea2 models:

```sh
mkdir -p resources/text_encoders
curl --fail --location \
  'https://huggingface.co/Comfy-Org/Krea-2/resolve/eb1eddd3983a54678545a9b2c178c5853b30f7be/text_encoders/qwen3vl_4b_bf16.safetensors' \
  --output resources/text_encoders/qwen3vl_4b_bf16.safetensors
printf '%s  %s\n' \
  '36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34' \
  'resources/text_encoders/qwen3vl_4b_bf16.safetensors' | shasum -a 256 --check
```

The installed layout and resource manifest are unchanged. `git lfs pull` alone does not provision this one encoder. Run `scripts/prepare_generation_defaults.py --check` with the documented Python environment to verify the complete resource set after provisioning.
