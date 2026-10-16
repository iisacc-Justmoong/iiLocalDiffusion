# Qwen3-VL-4B source identity

The original Qwen/Qwen3-VL-4B-Instruct encoder is used unchanged under Apache-2.0.
Single-file weights: `Comfy-Org/Krea-2`, revision `eb1eddd3983a54678545a9b2c178c5853b30f7be`, `text_encoders/qwen3vl_4b_bf16.safetensors`.
The upstream model is `Qwen/Qwen3-VL-4B-Instruct`. This package includes only the encoder, not the Krea2 denoiser weights. SHA-256 and size are recorded in `generation-defaults.json`.
The SDK mounts the original `model.language_model` namespace in memory. It does not rewrite the file or change tensor precision.
