#!/usr/bin/env python3
"""Architectural graph contracts; no server, downloads, or tensor inference."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
from comfyui_image_workflow import build_workflow, workflow_requirements, resolve_workflow_hires
from comfyui_runtime import validate_workflow


def node(inputs, outputs, *, optional=None, output_node=False):
    return {"input": {"required": inputs, "optional": optional or {}}, "output": outputs,
            "output_node": output_node, "python_module": "nodes"}


INT = ["INT", {"min": 1, "max": 16384}]
FLOAT = ["FLOAT", {"min": 0.0, "max": 100.0}]
TEXT = ["STRING"]
MODELS = [["checkpoint.safetensors"]]
DENOISERS = [["denoiser.safetensors", "negative.safetensors", "decoder.safetensors"]]
ENCODERS = [["clip.safetensors", "clip2.safetensors", "clip3.safetensors", "clip4.safetensors", "clip.gguf"]]
TYPES = [["stable_diffusion", "sdxl", "sd3", "flux", "flux2", "pixart", "hidream", "chroma",
          "qwen_image", "lumina2", "boogu", "krea2", "lens", "mage", "ideogram4", "stable_cascade"]]
SAMPLERS = [["euler", "euler_ancestral", "lcm", "uni_pc", "dpmpp_2m_sde", "res_multistep"]]
SCHEDULERS = [["normal", "simple", "sgm_uniform", "beta"]]


def object_info():
    objects = {
        "CheckpointLoaderSimple": node({"ckpt_name": MODELS}, ["MODEL", "CLIP", "VAE"]),
        "UNETLoader": node({"unet_name": DENOISERS, "weight_dtype": [["default"]]}, ["MODEL"]),
        "UnetLoaderGGUF": node({"unet_name": [["denoiser.gguf"]]}, ["MODEL"]),
        "VAELoader": node({"vae_name": [["vae.safetensors"]]}, ["VAE"]),
        "CLIPLoader": node({"clip_name": ENCODERS, "type": TYPES}, ["CLIP"]),
        "DualCLIPLoader": node({"clip_name1": ENCODERS, "clip_name2": ENCODERS, "type": TYPES}, ["CLIP"]),
        "TripleCLIPLoader": node({f"clip_name{i}": ENCODERS for i in range(1, 4)}, ["CLIP"]),
        "QuadrupleCLIPLoader": node({f"clip_name{i}": ENCODERS for i in range(1, 5)}, ["CLIP"]),
        "CLIPTextEncode": node({"clip": ["CLIP"], "text": TEXT}, ["CONDITIONING"]),
        "CLIPTextEncodePixArtAlpha": node({"clip": ["CLIP"], "text": TEXT, "width": INT, "height": INT}, ["CONDITIONING"]),
        "CLIPTextEncodeHunyuanDiT": node({"clip": ["CLIP"], "bert": TEXT, "mt5xl": TEXT}, ["CONDITIONING"]),
        "CLIPTextEncodeLumina2": node({"clip": ["CLIP"], "user_prompt": TEXT, "system_prompt": [["superior", "alignment"]]}, ["CONDITIONING"]),
        "TextEncodeMageFlowEdit": node({"clip": ["CLIP"], "prompt": TEXT, "negative_prompt": TEXT,
                                        "width": INT, "height": INT, "batch_size": INT},
                                       ["CONDITIONING", "CONDITIONING", "LATENT"], optional={"vae": ["VAE"]}),
        "KSampler": node({"model": ["MODEL"], "positive": ["CONDITIONING"], "negative": ["CONDITIONING"],
                          "latent_image": ["LATENT"], "seed": ["INT", {"min": 0, "max": 2**64 - 1}],
                          "steps": INT, "cfg": FLOAT, "sampler_name": SAMPLERS, "scheduler": SCHEDULERS,
                          "denoise": ["FLOAT", {"min": 0.0, "max": 1.0}]}, ["LATENT"]),
        "VAEDecode": node({"samples": ["LATENT"], "vae": ["VAE"]}, ["IMAGE"]),
        "VAEEncode": node({"pixels": ["IMAGE"], "vae": ["VAE"]}, ["LATENT"]),
        "ImageScale": node({"image": ["IMAGE"], "width": INT, "height": INT,
                            "upscale_method": [["nearest-exact", "bilinear", "bicubic", "lanczos"]],
                            "crop": [["disabled", "center"]]}, ["IMAGE"]),
        "SaveImage": node({"images": ["IMAGE"], "filename_prefix": TEXT}, [], output_node=True),
        "FluxGuidance": node({"conditioning": ["CONDITIONING"], "guidance": FLOAT}, ["CONDITIONING"]),
        "ConditioningZeroOut": node({"conditioning": ["CONDITIONING"]}, ["CONDITIONING"]),
        "T5TokenizerOptions": node({"clip": ["CLIP"], "min_padding": ["INT"], "min_length": ["INT"]}, ["CLIP"]),
        "ModelSamplingDiscrete": node({"model": ["MODEL"], "sampling": [["eps", "v_prediction", "lcm", "x0"]], "zsnr": ["BOOLEAN"]}, ["MODEL"]),
        "ModelSamplingSD3": node({"model": ["MODEL"], "shift": FLOAT}, ["MODEL"]),
        "ModelSamplingAuraFlow": node({"model": ["MODEL"], "shift": FLOAT}, ["MODEL"]),
        "ModelSamplingFlux": node({"model": ["MODEL"], "base_shift": FLOAT, "max_shift": FLOAT, "width": INT, "height": INT}, ["MODEL"]),
        "CFGNorm": node({"model": ["MODEL"], "strength": FLOAT}, ["MODEL"], optional={"pre_cfg": ["BOOLEAN"]}),
        "CLIPSetLastLayer": node({"clip": ["CLIP"], "stop_at_clip_layer": ["INT", {"min": -24, "max": -1}]}, ["CLIP"]),
        "CFGGuider": node({"model": ["MODEL"], "positive": ["CONDITIONING"], "negative": ["CONDITIONING"], "cfg": FLOAT}, ["GUIDER"]),
        "DualModelGuider": node({"model": ["MODEL"], "positive": ["CONDITIONING"], "cfg": FLOAT}, ["GUIDER"],
                                optional={"model_negative": ["MODEL"], "negative": ["CONDITIONING"]}),
        "Flux2Scheduler": node({"steps": INT, "width": INT, "height": INT}, ["SIGMAS"]),
        "BasicScheduler": node({"model": ["MODEL"], "scheduler": SCHEDULERS, "steps": INT,
                                "denoise": ["FLOAT", {"min": 0.0, "max": 1.0}]}, ["SIGMAS"]),
        "SplitSigmasDenoise": node({"sigmas": ["SIGMAS"], "denoise": ["FLOAT", {"min": 0.0, "max": 1.0}]}, ["SIGMAS", "SIGMAS"]),
        "Ideogram4Scheduler": node({"steps": INT, "width": ["INT", {"min": 256, "max": 8192}],
                                    "height": ["INT", {"min": 256, "max": 8192}], "mu": FLOAT, "std": FLOAT}, ["SIGMAS"]),
        "RandomNoise": node({"noise_seed": ["INT", {"min": 0, "max": 2**64 - 1}]}, ["NOISE"]),
        "KSamplerSelect": node({"sampler_name": SAMPLERS}, ["SAMPLER"]),
        "SamplerCustomAdvanced": node({"noise": ["NOISE"], "guider": ["GUIDER"], "sampler": ["SAMPLER"],
                                       "sigmas": ["SIGMAS"], "latent_image": ["LATENT"]}, ["LATENT", "LATENT"]),
        "StableCascade_StageB_Conditioning": node({"conditioning": ["CONDITIONING"], "stage_c": ["LATENT"]}, ["CONDITIONING"]),
    }
    for name in ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyFlux2LatentImage", "EmptyHiDreamO1LatentImage"):
        objects[name] = node({"width": INT, "height": INT, "batch_size": INT}, ["LATENT"])
    objects["StableCascade_EmptyLatentImage"] = node({"width": ["INT", {"min": 256, "max": 16384}],
                                                    "height": ["INT", {"min": 256, "max": 16384}], "batch_size": INT,
                                                    "compression": ["INT", {"default": 42, "min": 4, "max": 128}]}, ["LATENT", "LATENT"])
    return deepcopy(objects)


def classes(graph, name):
    return [(key, value["inputs"]) for key, value in graph.items() if value["class_type"] == name]


class ComfyUIImageWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.objects = object_info()

    def build(self, name="Illustrious", model="checkpoint.safetensors", components=None, **kwargs):
        return build_workflow(name, model, components or {}, "a lighthouse", self.objects, **kwargs)

    def test_sdxl_derivatives_keep_checkpoint_model_clip_and_vae_roles(self):
        for name in ("SDXL 1.0", "Illustrious", "NoobAI", "Pony"):
            with self.subTest(name=name):
                graph = self.build(name)
                loader, _ = classes(graph, "CheckpointLoaderSimple")[0]
                sampler = classes(graph, "KSampler")[0][1]
                self.assertEqual(sampler["model"], [loader, 0])
                self.assertEqual(classes(graph, "CLIPTextEncode")[0][1]["clip"], [loader, 1])
                self.assertEqual(classes(graph, "VAEDecode")[0][1]["vae"], [loader, 2])
                self.assertEqual(classes(graph, "EmptyLatentImage")[0][1]["width"], 1024)
                validate_workflow(graph, self.objects)

    def test_all_documented_family_graphs_have_valid_links_and_image_output(self):
        for name in ("SD 1.5", "SD 2.0", "SDXL 1.0", "SD 3", "Flux.1 D", "Flux.2 D", "Flux.2 Klein 4B",
                     "AuraFlow", "Chroma", "PixArt a", "PixArt E", "Hunyuan 1", "HiDream", "HiDream-O1", "Qwen",
                     "Lumina", "ZImageBase", "Anima", "Ernie", "Boogu", "Krea 2", "Lens", "MageFlow", "Ideogram 4.0", "Stable Cascade"):
            with self.subTest(name=name):
                extras = {"Ideogram 4.0": {"model_negative": "negative.safetensors"},
                          "Stable Cascade": {"decoder": "decoder.safetensors"}}.get(name, {})
                graph = self.build(name, components=extras)
                validate_workflow(graph, self.objects)
                self.assertEqual(len(classes(graph, "SaveImage")), 1)
                self.assertEqual(len(classes(graph, "VAEDecode")), 1)

    def test_explicit_parameters_reach_sampler_and_dimensions(self):
        graph = self.build(seed=123, steps=7, cfg=5.5, width=768, height=512, batch_size=2,
                           sampler_name="euler_ancestral", scheduler="simple", negative_prompt="blur")
        sampler = classes(graph, "KSampler")[0][1]
        self.assertEqual({key: sampler[key] for key in ("seed", "steps", "cfg", "sampler_name", "scheduler")},
                         {"seed": 123, "steps": 7, "cfg": 5.5, "sampler_name": "euler_ancestral", "scheduler": "simple"})
        self.assertEqual(classes(graph, "EmptyLatentImage")[0][1], {"width": 768, "height": 512, "batch_size": 2})
        self.assertIn("blur", [inputs["text"] for _, inputs in classes(graph, "CLIPTextEncode")])

    def test_split_flux_requires_all_roles_and_separates_cfg_from_embedded_guidance(self):
        components = {"vae": "vae.safetensors", "text_encoder": "clip.safetensors", "text_encoder_2": "clip2.safetensors"}
        graph = self.build("Flux.1 Krea", "denoiser.safetensors", components, cfg=1, guidance=3.75)
        self.assertEqual(classes(graph, "DualCLIPLoader")[0][1]["type"], "flux")
        self.assertEqual(classes(graph, "FluxGuidance")[0][1]["guidance"], 3.75)
        self.assertEqual(classes(graph, "KSampler")[0][1]["cfg"], 1)
        self.assertFalse(classes(graph, "CheckpointLoaderSimple"))
        validate_workflow(graph, self.objects)
        for missing in ("vae", "text_encoder", "text_encoder_2"):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.build("Flux.1 Krea", "denoiser.safetensors", {k: v for k, v in components.items() if k != missing})

    def test_component_replacement_does_not_change_checkpoint_denoiser(self):
        graph = self.build(components={"vae": "vae.safetensors", "text_encoder": "clip.safetensors", "text_encoder_2": "clip2.safetensors"})
        loader, _ = classes(graph, "CheckpointLoaderSimple")[0]
        self.assertEqual(classes(graph, "KSampler")[0][1]["model"], [loader, 0])
        self.assertEqual(classes(graph, "VAEDecode")[0][1]["vae"], [classes(graph, "VAELoader")[0][0], 0])

    def test_sd3_and_hidream_use_actual_encoder_arity(self):
        for name, counts in (("SD 3", (1, 2, 3)), ("HiDream", (1, 2, 4))):
            for count in counts:
                components = {"vae": "vae.safetensors", **{("text_encoder" if i == 1 else f"text_encoder_{i}"):
                             ("clip.safetensors" if i == 1 else f"clip{i}.safetensors") for i in range(1, count + 1)}}
                with self.subTest(name=name, count=count):
                    validate_workflow(self.build(name, "denoiser.safetensors", components), self.objects)

    def test_gguf_uses_optional_local_loader_and_never_standard_unet_loader(self):
        components = {"vae": "vae.safetensors", "text_encoder": "clip.safetensors"}
        graph = self.build("Chroma", "denoiser.gguf", components)
        self.assertEqual(len(classes(graph, "UnetLoaderGGUF")), 1)
        self.assertFalse(classes(graph, "UNETLoader"))
        del self.objects["UnetLoaderGGUF"]
        with self.assertRaisesRegex(ValueError, "missing required.*UnetLoaderGGUF"):
            self.build("Chroma", "denoiser.gguf", components, model_type="diffusion_model")

    def test_gguf_text_encoder_requires_gguf_clip_loader(self):
        components = {"vae": "vae.safetensors", "text_encoder": "clip.gguf"}
        with self.assertRaisesRegex(ValueError, "CLIPLoaderGGUF"):
            self.build("Chroma", "denoiser.safetensors", components)
        self.objects["CLIPLoaderGGUF"] = deepcopy(self.objects["CLIPLoader"])
        validate_workflow(self.build("Chroma", "denoiser.safetensors", components), self.objects)

    def test_noob_v_prediction_and_zero_terminal_snr_are_explicit(self):
        graph = self.build("NoobAI", prediction_type="v_prediction", zsnr=True, clip_skip=1)
        patch, values = classes(graph, "ModelSamplingDiscrete")[0]
        self.assertEqual(values["sampling"], "v_prediction")
        self.assertTrue(values["zsnr"])
        self.assertEqual(classes(graph, "KSampler")[0][1]["model"], [patch, 0])
        self.assertEqual(classes(graph, "CLIPSetLastLayer")[0][1]["stop_at_clip_layer"], -2)
        self.assertFalse(classes(self.build("NoobAI"), "ModelSamplingDiscrete"))

    def test_distilled_variants_keep_specific_sampling_defaults(self):
        for name, steps, cfg, width in (("SD 1.5 LCM", 4, 1, 512), ("SDXL Turbo", 1, 1, 512),
                                      ("Flux.1 S", 4, 1, 1024), ("ZImageTurbo", 8, 1, 1024),
                                      ("Flux.2 Klein 4B-base", 20, 5, 1024), ("SD 2.1 768", 20, 7, 768)):
            recipe = workflow_requirements(name)
            self.assertEqual((recipe["steps"], recipe["cfg"], recipe["width"]), (steps, cfg, width))
        self.assertFalse(classes(self.build("Flux.1 S"), "FluxGuidance"))

    def test_flux2_uses_128_channel_latent_and_native_schedule(self):
        graph = self.build("Flux.2 D", seed=42, width=1024, height=768)
        self.assertTrue(classes(graph, "EmptyFlux2LatentImage"))
        self.assertFalse(classes(graph, "KSampler"))
        sigmas, values = classes(graph, "Flux2Scheduler")[0]
        self.assertEqual(values, {"steps": 20, "width": 1024, "height": 768})
        self.assertEqual(classes(graph, "SamplerCustomAdvanced")[0][1]["sigmas"], [sigmas, 0])
        with self.assertRaisesRegex(ValueError, "requires its flux2 schedule"):
            self.build("Flux.2 D", scheduler="normal")

    def test_pixart_alpha_resolution_conditioning_is_not_applied_to_sigma(self):
        alpha = self.build("PixArt a", width=768, height=1024)
        self.assertEqual(classes(alpha, "CLIPTextEncodePixArtAlpha")[0][1]["width"], 768)
        self.assertFalse(classes(self.build("PixArt E"), "CLIPTextEncodePixArtAlpha"))

    def test_mage_uses_encoder_latent_and_two_conditioning_outputs(self):
        graph = self.build("MageFlow")
        encoded = classes(graph, "TextEncodeMageFlowEdit")[0][0]
        sampler = classes(graph, "KSampler")[0][1]
        self.assertEqual([sampler[k] for k in ("positive", "negative", "latent_image")], [[encoded, 0], [encoded, 1], [encoded, 2]])

    def test_cascade_decodes_stage_b_and_preserves_stage_c_conditioning(self):
        graph = self.build("Stable Cascade", components={"decoder": "decoder.safetensors"})
        stages = classes(graph, "KSampler")
        self.assertEqual(len(stages), 2)
        self.assertEqual(classes(graph, "VAEDecode")[0][1]["samples"], [stages[1][0], 0])
        self.assertEqual(classes(graph, "StableCascade_StageB_Conditioning")[0][1]["stage_c"], [stages[0][0], 0])
        self.assertEqual(stages[1][1]["latent_image"][1], 1)
        with self.assertRaisesRegex(ValueError, "stage-B"):
            self.build("Stable Cascade")

    def test_ideogram_requires_distinct_unconditional_model(self):
        with self.assertRaisesRegex(ValueError, "model_negative"):
            self.build("Ideogram 4.0")
        graph = self.build("Ideogram 4.0", components={"model_negative": "negative.safetensors"})
        guider = classes(graph, "DualModelGuider")[0][1]
        self.assertNotEqual(guider["model"], guider["model_negative"])
        with self.assertRaisesRegex(ValueError, "negative text"):
            self.build("Ideogram 4.0", components={"model_negative": "negative.safetensors"}, negative_prompt="blur")

    def test_model_task_cloud_unknown_components_and_ambiguous_inventory_fail(self):
        for name in ("Wan Video", "ACE Audio", "Hunyuan3D", "SD 2.1 Unclip", "Flux.1 Kontext", "Upscaler", "Other", "Kolors"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.build(name)
        with self.assertRaisesRegex(ValueError, "Components must use"):
            self.build(components={"text_encodre": "clip.safetensors"})
        self.objects["UNETLoader"]["input"]["required"]["unet_name"] = [["checkpoint.safetensors"]]
        with self.assertRaisesRegex(ValueError, "both checkpoint"):
            self.build()
        self.build(model_type="checkpoint")

    def test_live_schema_missing_nodes_unavailable_models_and_changed_inputs_fail(self):
        for mutate, message in ((lambda x: x.pop("VAEDecode"), "missing required"),
                                (lambda x: x["SaveImage"].update(api_node=True), "hosted"),
                                (lambda x: x["CheckpointLoaderSimple"]["input"]["required"].update(ckpt_name=[["other"]]), "absent"),
                                (lambda x: x["KSampler"]["input"]["required"].pop("seed"), "does not accept"),
                                (lambda x: x["CheckpointLoaderSimple"].update(output=["MODEL", "VAE", "CLIP"]), "Incompatible")):
            with self.subTest(message=message):
                self.objects = object_info()
                mutate(self.objects)
                with self.assertRaisesRegex(ValueError, message):
                    self.build()

    def test_native_v3_combos_are_enumerated_including_model_inventory(self):
        for specification in self.objects.values():
            for schema in specification["input"]["required"].values():
                if isinstance(schema[0], list):
                    choices = schema[0]
                    schema[:] = ["COMBO", {"options": choices, "multiselect": False}]
        self.build("Flux.2 D")
        with self.assertRaisesRegex(ValueError, "unsupported choice"):
            self.build("Flux.2 D", sampler_name="missing")
        with self.assertRaisesRegex(ValueError, "absent"):
            self.build(model="missing.safetensors")

    def test_native_v3_zero_minimum_autogrow_can_be_omitted(self):
        self.objects["TextEncodeMageFlowEdit"]["input"]["required"]["images"] = [
            "COMFY_AUTOGROW_V3", {"template": {"min": 0, "names": ["image_1"],
                                              "input": {"required": {"image": ["IMAGE"]}}}}]
        graph = self.build("MageFlow")
        self.assertNotIn("images", classes(graph, "TextEncodeMageFlowEdit")[0][1])
        self.objects["TextEncodeMageFlowEdit"]["input"]["required"]["images"][1]["template"]["min"] = 1
        with self.assertRaisesRegex(ValueError, "required.*images"):
            self.build("MageFlow")

    def test_invalid_dimensions_nonfinite_values_and_unconsumed_options_fail(self):
        for kwargs in ({"width": 1023}, {"height": True}, {"steps": 0}, {"cfg": float("nan")}, {"seed": -1},
                       {"batch_size": 0}, {"sampler_name": "unknown"}, {"prediction_type": "unknown"},
                       {"guidance": 3.5}, {"zsnr": True}, {"clip_skip": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_build_does_not_mutate_objects_or_component_inputs(self):
        before = deepcopy(self.objects)
        components = {"vae": "vae.safetensors"}
        self.build(components=components)
        self.assertEqual(self.objects, before)
        self.assertEqual(components, {"vae": "vae.safetensors"})

    def test_hires_omission_keeps_the_original_graph(self):
        self.assertEqual(self.build(), self.build(hires_fix=False))
        self.assertFalse(classes(self.build(), "ImageScale"))
        self.assertEqual(resolve_workflow_hires("Illustrious"), {"enabled": False, "passes": 0, "stages": []})

    def test_hires_repeat_uses_latest_refinement_as_next_input_and_saves_only_final(self):
        graph = self.build(width=64, height=64, hires_fix=True, hires_passes=3,
                           hires_scale=1.5, hires_steps=10, hires_strength=0.4, seed=17)
        decodes = classes(graph, "VAEDecode")
        scales = classes(graph, "ImageScale")
        encodes = classes(graph, "VAEEncode")
        refinements = classes(graph, "SamplerCustomAdvanced")
        splits = classes(graph, "SplitSigmasDenoise")
        self.assertEqual(len(decodes), 4)
        self.assertEqual([(x["width"], x["height"]) for _, x in scales], [(96, 96), (144, 144), (216, 216)])
        for index in range(3):
            self.assertEqual(scales[index][1]["image"], [decodes[index][0], 0])
            self.assertEqual(encodes[index][1]["pixels"], [scales[index][0], 0])
            self.assertEqual(refinements[index][1]["latent_image"], [encodes[index][0], 0])
            self.assertEqual(refinements[index][1]["sigmas"], [splits[index][0], 1])
            self.assertEqual(decodes[index + 1][1]["samples"], [refinements[index][0], 0])
            self.assertEqual(splits[index][1]["denoise"], 0.4)
        self.assertEqual(len(classes(graph, "SaveImage")), 1)
        self.assertEqual(classes(graph, "SaveImage")[0][1]["images"], [decodes[-1][0], 0])
        self.assertEqual([x["noise_seed"] for _, x in classes(graph, "RandomNoise")], [17, 17, 17])
        self.assertTrue(all(x["steps"] == 10 and x["denoise"] == 1 for _, x in classes(graph, "BasicScheduler")))
        validate_workflow(graph, self.objects)

    def test_hires_all_existing_image_families_keep_valid_refinement_graphs(self):
        for name in ("SD 1.5", "SD 2.0", "SDXL 1.0", "SD 3", "Flux.1 D", "Flux.2 D", "Flux.2 Klein 4B",
                     "AuraFlow", "Chroma", "PixArt a", "PixArt E", "Hunyuan 1", "HiDream", "HiDream-O1", "Qwen",
                     "Lumina", "ZImageBase", "Anima", "Ernie", "Boogu", "Krea 2", "Lens", "MageFlow", "Ideogram 4.0", "Stable Cascade"):
            with self.subTest(name=name):
                components = {"Ideogram 4.0": {"model_negative": "negative.safetensors"},
                              "Stable Cascade": {"decoder": "decoder.safetensors"}}.get(name, {})
                size = 256 if name in ("Ideogram 4.0", "Stable Cascade") else 64
                graph = self.build(name, components=components, width=size, height=size,
                                   hires_fix=True, hires_passes=2, hires_steps=10)
                validate_workflow(graph, self.objects)
                self.assertEqual(len(classes(graph, "ImageScale")), 2)
                self.assertEqual(len(classes(graph, "VAEEncode")), 2)

    def test_hires_flux2_and_ideogram_keep_size_aware_native_schedules(self):
        for name, schedule, components in (("Flux.2 D", "Flux2Scheduler", {}),
                                           ("Ideogram 4.0", "Ideogram4Scheduler", {"model_negative": "negative.safetensors"})):
            size = 256 if name == "Ideogram 4.0" else 64
            graph = self.build(name, components=components, width=size, height=size,
                               hires_fix=True, hires_passes=2, hires_steps=10)
            self.assertEqual([(x["width"], x["height"]) for _, x in classes(graph, schedule)],
                             [(size, size), (size * 2, size * 2), (size * 4, size * 4)])
            self.assertFalse(classes(graph, "BasicScheduler"))
            if name == "Ideogram 4.0":
                guiders = classes(graph, "DualModelGuider")
                self.assertEqual(len(guiders), 3)
                self.assertEqual(len({tuple(x["model_negative"]) for _, x in guiders}), 1)

    def test_hires_resolution_conditioning_updates_and_cascade_refines_stage_b(self):
        graph = self.build("PixArt a", width=64, height=64, hires_fix=True, hires_passes=2)
        self.assertEqual([x["width"] for _, x in classes(graph, "CLIPTextEncodePixArtAlpha")], [64, 64, 128, 128, 256, 256])
        graph = self.build("Lens", width=64, height=64, hires_fix=True, hires_passes=2)
        self.assertEqual([x["width"] for _, x in classes(graph, "ModelSamplingFlux")], [64, 128, 256])
        graph = self.build("Stable Cascade", components={"decoder": "decoder.safetensors"}, hires_fix=True, hires_passes=2)
        decoder = classes(graph, "UNETLoader")[0][0]
        self.assertEqual(len(classes(graph, "KSampler")), 2)
        self.assertTrue(all(x["model"] == [decoder, 0] for _, x in classes(graph, "BasicScheduler")))

    def test_hires_plan_defaults_rounding_effective_steps_and_invalid_values(self):
        plan = resolve_workflow_hires("Illustrious", width=64, height=64, hires_fix=True, steps=10)
        self.assertEqual((plan["passes"], plan["scale"], plan["strength"], plan["effective_steps"]), (1, 2.0, 0.35, 4))
        self.assertEqual(plan["stages"], [{"pass_index": 1, "width": 128, "height": 128}])
        for values in ({"hires_passes": 0}, {"hires_passes": -1}, {"hires_passes": True}, {"hires_passes": 1.5},
                       {"hires_scale": 1}, {"hires_scale": float("inf")}, {"hires_strength": 0},
                       {"hires_strength": float("nan")}, {"hires_strength": 1.1}, {"hires_steps": 0},
                       {"hires_steps": 1, "hires_strength": 0.1}, {"hires_scale": 1024}, {"hires_upscaler": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                resolve_workflow_hires("Illustrious", hires_fix=True, **values)
        with self.assertRaisesRegex(ValueError, "require --hires-fix"):
            self.build(hires_passes=2)

    def test_hires_missing_refinement_nodes_fail_without_silently_skipping(self):
        for name in ("ImageScale", "VAEEncode", "SplitSigmasDenoise", "BasicScheduler"):
            self.objects = object_info()
            del self.objects[name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.build(hires_fix=True)


if __name__ == "__main__":
    unittest.main()
