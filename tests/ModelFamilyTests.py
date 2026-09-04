#!/usr/bin/env python3
"""Model-family contracts preserve training configuration and reject cross-family weights."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import controlnet
import generation_scheduler
import model_loading
import presets


def component(class_name, **configuration):
    result = type(class_name, (), {})()
    result.config = SimpleNamespace(**configuration)
    return result


def sdxl_pipeline(scheduler="EulerAncestralDiscreteScheduler", prediction="v_prediction"):
    components = {
        "tokenizer": component("CLIPTokenizer", model_max_length=77),
        "tokenizer_2": component("CLIPTokenizer", model_max_length=77),
        "text_encoder": component("CLIPTextModel", hidden_size=768,
            max_position_embeddings=77, projection_dim=768),
        "text_encoder_2": component("CLIPTextModelWithProjection", hidden_size=1280,
            max_position_embeddings=77, projection_dim=1280),
        "unet": component("UNet2DConditionModel", in_channels=4, out_channels=4,
            sample_size=64, cross_attention_dim=2048, addition_embed_type="text_time",
            addition_time_embed_dim=256, projection_class_embeddings_input_dim=2816,
            use_linear_projection=True),
        "vae": component("AutoencoderKL", in_channels=3, out_channels=3,
            latent_channels=4, scaling_factor=0.13025, force_upcast=False,
            sample_size=512, block_out_channels=[128, 256, 512, 512]),
        "scheduler": component(scheduler, prediction_type=prediction,
            timestep_spacing="trailing", beta_end=0.02, rescale_betas_zero_snr=True),
    }
    result = type("StableDiffusionXLPipeline", (), {})()
    result.config = SimpleNamespace(force_zeros_for_empty_prompt=False)
    result.components = components
    for name, value in components.items():
        setattr(result, name, value)
    return result


class EulerDiscreteScheduler:
    def __init__(self, prediction_type="epsilon", rescale_betas_zero_snr=False):
        self.config = dict(prediction_type=prediction_type,
                           rescale_betas_zero_snr=rescale_betas_zero_snr)

    @property
    def compatibles(self):
        return [EulerDiscreteScheduler, OtherScheduler]

    @classmethod
    def from_config(cls, configuration, **overrides):
        return cls(**{**configuration, **overrides})


class OtherScheduler(EulerDiscreteScheduler):
    pass


class ModelFamilyTests(unittest.TestCase):
    def test_architecture_routing_is_independent_from_model_name(self):
        for name in ("sdxl", "illustrious", "noobai", "noobai-v-pred", "pony"):
            preset = presets.PRESETS[name]
            self.assertEqual(preset.name, name)
            self.assertEqual(preset.family, "sdxl-base")
            self.assertFalse(preset.strict_contract)
            self.assertIsNone(preset.runtime.weight_variant)
        for name in ("flux1-dev", "flux1-krea-dev"):
            preset = presets.PRESETS[name]
            self.assertEqual(preset.family, "flux1-schnell")
            self.assertTrue(preset.requires_guidance_embeds)
            self.assertEqual(preset.runtime.max_sequence_length, 512)
            self.assertGreater(preset.guidance_scale, 0)
        self.assertTrue(presets.PONY_PRESET.requires_model_override)

    def test_sd15_compatible_preserves_architecture_with_noncanonical_scheduler(self):
        preset = presets.SD15_COMPATIBLE_PRESET
        components = {
            "tokenizer": component("CLIPTokenizer", model_max_length=77),
            "text_encoder": component("CLIPTextModel", hidden_size=768, max_position_embeddings=77),
            "unet": component("UNet2DConditionModel", in_channels=4, out_channels=4, cross_attention_dim=768),
            "vae": component("AutoencoderKL", in_channels=3, out_channels=3,
                latent_channels=4, scaling_factor=0.18215, block_out_channels=[128, 256, 512, 512]),
            "scheduler": component("EulerAncestralDiscreteScheduler", prediction_type="v_prediction"),
        }
        pipeline = type("StableDiffusionPipeline", (), {})()
        pipeline.components = components
        for name, value in components.items():
            setattr(pipeline, name, value)
        presets.validate_pipeline_contract(pipeline, preset)
        for name, field, value in (("unet", "in_channels", 9), ("unet", "out_channels", 8),
                                    ("unet", "cross_attention_dim", 1024),
                                    ("text_encoder", "hidden_size", 1024)):
            configuration = components[name].config
            original = getattr(configuration, field)
            setattr(configuration, field, value)
            with self.subTest(name=name, field=field), self.assertRaisesRegex(RuntimeError, field):
                presets.validate_pipeline_contract(pipeline, preset)
            setattr(configuration, field, original)

    def test_compatible_defaults_retain_original_identity_and_guidance_architecture(self):
        for strict, compatible in (
            (presets.SD15_PRESET, presets.SD15_COMPATIBLE_PRESET),
            (presets.FLUX1_SCHNELL_PRESET, presets.FLUX1_SCHNELL_COMPATIBLE_PRESET),
        ):
            self.assertTrue(strict.strict_contract)
            self.assertFalse(compatible.strict_contract)
            self.assertEqual(strict.model_id, compatible.model_id)
            self.assertEqual(strict.revision, compatible.revision)
            self.assertEqual(strict.family, compatible.family)
        preset = presets.FLUX1_SCHNELL_COMPATIBLE_PRESET
        self.assertFalse(preset.requires_guidance_embeds)
        self.assertTrue(presets.compatible_scheduler_class(preset, "FlowMatchHeunDiscreteScheduler"))
        self.assertFalse(presets.compatible_scheduler_class(preset, "EulerDiscreteScheduler"))
        with self.assertRaisesRegex(ValueError, "guidance"):
            model_loading.classify_model_weights(preset, {"img_in.weight", "guidance_in.in_layer.weight"})

    def test_controlnet_augmented_presets_preserve_family_and_training_contract(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                augmented = controlnet.controlnet_preset(preset)
                self.assertEqual(augmented.family, preset.family)
                self.assertEqual(augmented.requires_guidance_embeds, preset.requires_guidance_embeds)
                self.assertEqual(augmented.scheduler_defaults, preset.scheduler_defaults)
                self.assertEqual(augmented.expected_components[-1][0], "controlnet")

    def test_sdxl_controlnet_augmented_pipeline_validates_and_rejects_wrong_components(self):
        preset = controlnet.controlnet_preset(presets.NOOBAI_V_PRED_PRESET)
        pipeline = sdxl_pipeline()
        pipeline.__class__ = type("StableDiffusionXLControlNetPipeline", (), {})
        pipeline.controlnet = component("ControlNetModel")
        pipeline.components["controlnet"] = pipeline.controlnet
        presets.validate_pipeline_contract(pipeline, preset)
        pipeline.components["controlnet"] = component("FluxControlNetModel")
        with self.assertRaisesRegex(RuntimeError, "controlnet class"):
            presets.validate_pipeline_contract(pipeline, preset)
        pipeline.components["controlnet"] = pipeline.controlnet
        pipeline.unet.config.in_channels = 9
        with self.assertRaisesRegex(RuntimeError, "in_channels"):
            presets.validate_pipeline_contract(pipeline, preset)

    def test_sdxl_derivatives_accept_training_schedule_and_vae_precision_variations(self):
        for name in ("sdxl", "illustrious", "noobai", "noobai-v-pred", "pony"):
            for prediction in ("epsilon", "v_prediction"):
                with self.subTest(name=name, prediction=prediction):
                    presets.validate_pipeline_contract(sdxl_pipeline(prediction=prediction), presets.PRESETS[name])

    def test_derivatives_still_reject_inpaint_refiner_and_wrong_text_contracts(self):
        for component_name, field, value in (
            ("unet", "in_channels", 9), ("unet", "cross_attention_dim", 1280),
            ("unet", "out_channels", 8), ("text_encoder_2", "hidden_size", 768),
            ("vae", "latent_channels", 16), ("vae", "scaling_factor", 0.18215),
        ):
            pipeline = sdxl_pipeline()
            setattr(getattr(pipeline, component_name).config, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field):
                presets.validate_pipeline_contract(pipeline, presets.ILLUSTRIOUS_PRESET)

    def test_diffusion_family_does_not_accept_flow_or_custom_scheduler(self):
        for name in ("FlowMatchEulerDiscreteScheduler", "UntrustedScheduler"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "scheduler class"):
                presets.validate_pipeline_contract(sdxl_pipeline(scheduler=name), presets.NOOBAI_PRESET)
        with self.assertRaisesRegex(RuntimeError, "prediction_type"):
            presets.validate_pipeline_contract(sdxl_pipeline(prediction="unrecognized"), presets.NOOBAI_PRESET)

    def test_single_file_configuration_accepts_known_variant_scheduler_only(self):
        (ROOT / "build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="model-family-", dir=ROOT / "build") as temporary:
            root = Path(temporary)
            preset = presets.ILLUSTRIOUS_PRESET
            index = {"_class_name": preset.pipeline_class}
            for name, cls in preset.expected_components:
                index[name] = ["transformers" if name.startswith(("text_encoder", "tokenizer")) else "diffusers", cls]
            selection = presets.ModelSelection(str(root), None, True)
            for library, cls, accepted in (
                ("diffusers", "DPMSolverMultistepScheduler", True),
                ("diffusers", "EulerAncestralDiscreteScheduler", True),
                ("remote.custom", "EulerDiscreteScheduler", False),
                ("diffusers", "FlowMatchEulerDiscreteScheduler", False),
            ):
                index["scheduler"] = [library, cls]
                (root / "model_index.json").write_text(json.dumps(index))
                with self.subTest(library=library, cls=cls):
                    if accepted:
                        self.assertEqual(model_loading.resolve_configuration_directory(selection, preset, root, True), root)
                    else:
                        with self.assertRaises(ValueError):
                            model_loading.resolve_configuration_directory(selection, preset, root, True)

    def test_noobai_v_prediction_fixes_incorrect_packaged_epsilon_scheduler(self):
        args = SimpleNamespace(preset="noobai-v-pred", scheduler="auto", scheduler_config={}, prediction_type="auto")
        pipeline = SimpleNamespace(scheduler=OtherScheduler())
        metadata = generation_scheduler.configure_scheduler(pipeline, args)
        self.assertIs(type(pipeline.scheduler), EulerDiscreteScheduler)
        self.assertEqual(pipeline.scheduler.config, dict(prediction_type="v_prediction", rescale_betas_zero_snr=True))
        self.assertEqual(metadata["preset_defaults"]["prediction_type"], "v_prediction")
        args.preset = "noobai"
        original = OtherScheduler()
        pipeline.scheduler = original
        generation_scheduler.configure_scheduler(pipeline, args)
        self.assertIs(pipeline.scheduler, original)

    def test_explicit_prediction_override_is_applied_and_conflicts_fail(self):
        args = SimpleNamespace(preset="sdxl", scheduler="auto", scheduler_config={}, prediction_type="v_prediction")
        pipeline = SimpleNamespace(scheduler=EulerDiscreteScheduler())
        generation_scheduler.configure_scheduler(pipeline, args)
        self.assertEqual(pipeline.scheduler.config["prediction_type"], "v_prediction")
        args.scheduler_config = {"prediction_type": "epsilon"}
        with self.assertRaisesRegex(ValueError, "conflicts"):
            generation_scheduler.configure_scheduler(pipeline, args)

    def test_flux_guidance_weights_cannot_be_silently_discarded_with_schnell_config(self):
        for keys in (
            {"img_in.weight", "guidance_in.in_layer.weight"},
            {"x_embedder.weight", "time_text_embed.guidance_embedder.linear_1.weight"},
            {"model.diffusion_model.img_in.weight", "model.diffusion_model.guidance_in.in_layer.weight"},
        ):
            for preset in (presets.FLUX1_DEV_PRESET, presets.FLUX1_KREA_DEV_PRESET):
                self.assertEqual(model_loading.classify_model_weights(preset, keys), "transformer")
            with self.assertRaisesRegex(ValueError, "guidance"):
                model_loading.classify_model_weights(presets.FLUX1_SCHNELL_PRESET, keys)
        with self.assertRaisesRegex(ValueError, "guidance"):
            model_loading.classify_model_weights(presets.FLUX1_DEV_PRESET, {"img_in.weight"})

    def test_remote_model_overrides_continue_to_require_immutable_revision(self):
        for preset in (presets.ILLUSTRIOUS_PRESET, presets.NOOBAI_PRESET, presets.FLUX1_KREA_DEV_PRESET):
            selection = presets.resolve_model_selection(preset, None, None)
            self.assertRegex(selection.requested_revision, "^[a-f0-9]{40}$")
            with self.assertRaisesRegex(ValueError, "commit"):
                presets.resolve_model_selection(preset, "untrusted/replacement", "main")


if __name__ == "__main__":
    unittest.main()
