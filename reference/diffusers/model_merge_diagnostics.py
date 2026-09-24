"""Header-based, base-scoped diagnostics shared by CLI and desktop clients."""

from collections import Counter
import math

from model_merge_ecosystem import identify_ecosystem


def component_kind(component, key):
    name = f"{component}.{key}".lower()
    # Ownership takes priority over a generic transformer/encoder token.
    if any(token in name for token in ("text_encoder", "text_model", "cond_stage_model",
                                      "conditioner.embedders", "lora_te", "clip.", "t5.")):
        return "text_encoder"
    if any(token in name for token in ("vae.", "vae/", "first_stage_model", "lora_vae")):
        return "vae"
    if any(token in name for token in ("double_blocks", "single_blocks", "joint_blocks",
                                      "llm_adapter", "tproj", "txt_fusion")):
        return "dit"
    if any(token in name for token in ("unet.", "unet/", "lora_unet", "input_blocks",
                                      "output_blocks", "middle_block", "down_blocks", "up_blocks")):
        return "unet"
    if any(token in name for token in ("transformer_blocks", "transformer.", "transformer/")):
        return "dit"
    return "other"


def model_profile(readers, layout):
    identity = identify_ecosystem(readers, layout)
    groups = {name: {"tensor_count": 0, "parameter_count": 0, "dtypes": Counter(), "examples": []}
              for name in ("unet", "dit", "vae", "text_encoder", "other")}
    for address, filename in sorted(layout.items()):
        value = readers[filename].get_slice(address[1])
        kind = component_kind(*address)
        denoiser_namespace = address[1].startswith(("model.diffusion_model.", "diffusion_model."))
        if kind == "other" and denoiser_namespace and set(identity.families) & {"sd1", "sd2", "sdxl"}:
            kind = "unet"
        if kind == "other" and any(family in identity.families for family in ("flux1", "flux2", "anima", "sd3")):
            if denoiser_namespace or any(token in address[1] for token in ("blocks.", "img_in.", "txt_in.", "final_layer.")):
                kind = "dit"
        group = groups[kind]
        group["tensor_count"] += 1
        group["parameter_count"] += math.prod(value.get_shape())
        group["dtypes"][value.get_dtype()] += 1
        if len(group["examples"]) < 6:
            group["examples"].append({"component": address[0], "tensor": address[1],
                                      "shape": value.get_shape(), "dtype": value.get_dtype()})
    for group in groups.values():
        group["dtypes"] = dict(group["dtypes"])
    return {"ecosystems": list(identity.families), "evidence": list(identity.evidence),
            "label": identity.label, "components": groups, "tensor_count": len(layout),
            "classification": "header-evidence; not an inference-quality guarantee"}


def preflight_summary(request, entries, common_layers, stages):
    included = [entry for entry in entries if entry["compatible"]]
    excluded = [entry for entry in entries if not entry["compatible"]]
    unified = request.mode == "unified"
    risks = ["Header inspection cannot certify finite tensor values or generated-image quality."]
    if unified:
        risks.append("Unified is sequential image refinement with independent networks, not conversion into one network. Copied stages retain their original numeric values.")
    else:
        risks.append("Invalid values are repaired and unusable contributions are omitted. Equal output size does not mean unchanged weights.")
        if any(report["projected_tensors"] for report in common_layers.values()):
            risks.append("Coordinate fitting changes shapes or layer assignments. This approximation is not trained or semantically equivalent conversion.")
        if any(report["base_preserved_tensors"] for report in common_layers.values()):
            risks.append("Unmatched layers, empty tensors and runtime buffers retain the base values.")
    if excluded:
        risks.append(f"{len(excluded)} resource(s) will be excluded; weights are resolved using only included resources.")
    if not included or not any(request.weights or ()):
        risks.append("No effective material contribution is expected; the output may contain only the preserved or repaired base.")
    if request.weight_normalization:
        risks.append("Checkpoint coefficients exceeding one were proportionally normalized.")
    conditional = any(entry["status"] == "conditional" for entry in included)
    return {"status": "incompatible" if not included else "conditional" if conditional or excluded else "compatible",
            "output_contract": "independent-network-cascade" if unified else "base-tensor-layout",
            "method_description": ("Independent models refine an image in order; checkpoint weights are refinement strengths."
                                   if unified else "Weighted sum combines base and material coordinates; weighted difference subtracts materials. LoRAs contribute deltas."),
            "expected_stage_count": len(stages) if unified else 1,
            "included_material_count": len(included), "excluded_material_count": len(excluded),
            "base_weight": request.base_weight, "weights": list(request.weights or ()),
            "projected_tensor_count": sum(report["projected_tensors"] for report in common_layers.values()),
            "risks": risks, "inspection": "headers-and-lora-targets; no full numeric scan"}
