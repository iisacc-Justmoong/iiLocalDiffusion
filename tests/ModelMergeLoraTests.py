#!/usr/bin/env python3
"""Real adapter/checkpoint arithmetic, compatibility and preservation contracts."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from model_merge import merge_models

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "safetensors"))


@unittest.skipUnless(HAS_RUNTIME, "Requires the existing Torch/safetensors environment")
class ModelMergeLoraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from safetensors.torch import load_file, save_file
        cls.torch = torch
        cls.load = staticmethod(load_file)
        cls.save = staticmethod(save_file)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="merge-lora-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "base.safetensors"
        self.lora = self.root / "style.safetensors"
        self.output = self.root / "merged.safetensors"
        t = self.torch
        self.weight = t.arange(12, dtype=t.float32).reshape(3, 4)
        self.down = t.tensor([[1., 2., 3., 4.], [-1., 2., -3., 4.]])
        self.up = t.tensor([[1., 2.], [-3., 4.], [5., -6.]])
        self.save({"layer.weight": self.weight, "layer.bias": t.tensor([1., 2., 3.]),
                   "steps": t.tensor(42)}, str(self.base))
        self.adapter()

    def adapter(self, path=None, target="layer", style="peft", alpha=None, down=None, up=None):
        a, b = ("lora_A.weight", "lora_B.weight") if style == "peft" else ("lora_down.weight", "lora_up.weight")
        state = {f"{target}.{a}": self.down if down is None else down,
                 f"{target}.{b}": self.up if up is None else up}
        if alpha is not None:
            state[f"{target}.alpha"] = self.torch.tensor(alpha)
        self.save(state, str(path or self.lora))

    def merge(self, **options):
        return merge_models(self.base, self.lora, output=self.output, **options)

    def test_lora_is_first_required_material_and_defaults_to_unit_delta(self):
        before = [path.read_bytes() for path in (self.base, self.lora)]
        report = self.merge()
        state = self.load(self.output)
        self.torch.testing.assert_close(state["layer.weight"], self.weight + self.up @ self.down)
        self.assertEqual(state["layer.bias"].tolist(), [1, 2, 3])
        self.assertEqual(state["steps"].item(), 42)
        self.assertEqual(report["base_weight"], 1)
        self.assertEqual(report["weights"], [1])
        self.assertEqual(report["sources"][1]["kind"], "lora")
        self.assertEqual(report["lora_target_count"], 1)
        self.assertEqual(before, [path.read_bytes() for path in (self.base, self.lora)])

    def test_subtraction_subtracts_scaled_delta_and_alpha_over_rank(self):
        self.adapter(style="kohya", alpha=1.)
        self.merge(mode="weighted-difference", weights=0.3)
        self.torch.testing.assert_close(self.load(self.output)["layer.weight"],
                                       self.weight - 0.15 * (self.up @ self.down))

    def test_multiple_adapters_allow_different_ranks_and_strength_above_one(self):
        other = self.root / "detail.safetensors"
        down, up = self.down[:1].contiguous(), self.up[:, :1].contiguous()
        self.adapter(other, down=down, up=up)
        self.merge(additional_models=[other], weights=[1.5, 2.])
        self.torch.testing.assert_close(self.load(self.output)["layer.weight"],
                                       self.weight + 1.5 * (self.up @ self.down) + 2 * (up @ down))

    def test_checkpoint_and_lora_defaults_and_explicit_weights_in_both_orders(self):
        checkpoint = self.root / "other.safetensors"
        state = self.load(self.base)
        state["layer.weight"] += 10
        self.save(state, str(checkpoint))
        for first, rest in ((self.lora, checkpoint), (checkpoint, self.lora)):
            for mode in ("weighted-sum", "weighted-difference"):
                with self.subTest(first=first, mode=mode):
                    report = merge_models(self.base, first, additional_models=[rest],
                                          mode=mode, output=self.output)
                    expected = ((self.weight + state["layer.weight"]) / 2 + self.up @ self.down
                                if mode == "weighted-sum" else
                                self.weight - state["layer.weight"] / 2 - self.up @ self.down)
                    self.torch.testing.assert_close(self.load(self.output)["layer.weight"], expected)
                    self.assertEqual(report["base_weight"], 0.5 if mode == "weighted-sum" else 1)
                    self.output.unlink()
        self.merge(additional_models=[checkpoint], weights=[2., 0.25])
        self.torch.testing.assert_close(self.load(self.output)["layer.weight"],
                                       self.weight * 0.75 + state["layer.weight"] * 0.25 + 2 * (self.up @ self.down))

    def test_peft_directory_config_alpha_patterns_and_rslora(self):
        adapter = self.root / "adapter"
        adapter.mkdir()
        self.adapter(adapter / "adapter_model.safetensors", target="base_model.model.layer")
        (adapter / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "r": 2, "lora_alpha": 9, "alpha_pattern": {"layer": 4},
            "use_rslora": True, "bias": "none"}))
        merge_models(self.base, adapter, output=self.output)
        self.torch.testing.assert_close(self.load(self.output)["layer.weight"],
                                       self.weight + (4 / 2**0.5) * (self.up @ self.down))

    def test_directory_base_accepts_kohya_file_and_preserves_unaffected_components(self):
        base = self.root / "pipeline"
        (base / "unet").mkdir(parents=True)
        (base / "vae").mkdir()
        (base / "model_index.json").write_text('{"_class_name": "FixturePipeline"}')
        self.save({"layer_name.weight": self.weight}, str(base / "unet/model.safetensors"))
        self.save({"layer_name.weight": self.weight}, str(base / "vae/model.safetensors"))
        self.adapter(target="lora_unet_layer_name", style="kohya", alpha=2.)
        out = self.root / "pipeline-merged"
        merge_models(base, self.lora, output=out)
        self.torch.testing.assert_close(self.load(out / "unet/model.safetensors")["layer_name.weight"],
                                       self.weight + self.up @ self.down)
        self.torch.testing.assert_close(self.load(out / "vae/model.safetensors")["layer_name.weight"], self.weight)
        self.assertEqual((out / "model_index.json").read_bytes(), (base / "model_index.json").read_bytes())

    def test_standard_convolution_delta(self):
        t = self.torch
        base = t.arange(3*4*3*3, dtype=t.float32).reshape(3, 4, 3, 3)
        down = t.arange(2*4*3*3, dtype=t.float32).reshape(2, 4, 3, 3) / 100
        up = self.up[:, :, None, None].contiguous()
        self.save({"layer.weight": base}, str(self.base))
        self.adapter(down=down, up=up)
        self.merge()
        expected = base + t.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
        t.testing.assert_close(self.load(self.output)["layer.weight"], expected)

    def test_invalid_adapter_materials_fail_without_publishing(self):
        t = self.torch
        invalid = {
            "missing up": {"layer.lora_A.weight": self.down},
            "unknown target": {"absent.lora_A.weight": self.down, "absent.lora_B.weight": self.up},
            "rank mismatch": {"layer.lora_A.weight": self.down, "layer.lora_B.weight": self.up[:, :1].contiguous()},
            "extra tensor": {"layer.lora_A.weight": self.down, "layer.lora_B.weight": self.up, "bias": t.zeros(3)},
            "DoRA": {"layer.lora_A.weight": self.down, "layer.lora_B.weight": self.up,
                     "layer.lora_magnitude_vector": t.ones(3)},
            "LyCORIS": {"layer.hada_w1_a": self.down},
            "nonfinite": {"layer.lora_A.weight": self.down * float("nan"), "layer.lora_B.weight": self.up},
            "alpha": {"layer.lora_A.weight": self.down, "layer.lora_B.weight": self.up,
                      "layer.alpha": t.tensor(float("inf"))},
        }
        for case, state in invalid.items():
            with self.subTest(case=case):
                self.save(state, str(self.lora))
                with self.assertRaises((ValueError, RuntimeError)):
                    self.merge(weights=0)
                self.assertFalse(self.output.exists())
                self.assertFalse(list(self.root.glob(".merged.safetensors-*")))

    def test_lora_base_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "[Bb]ase.*checkpoint"):
            merge_models(self.lora, self.base, output=self.output)

    def test_diffusers_and_named_peft_key_conventions(self):
        self.save({"unet.block.attn1.to_q.weight": self.weight}, str(self.base))
        for down, up in (("lora_A.default", "lora_B.default"), ("lora.down", "lora.up"),
                         ("lora_linear_layer.down", "lora_linear_layer.up")):
            with self.subTest(down=down):
                self.save({f"unet.block.attn1.to_q.{down}.weight": self.down,
                           f"unet.block.attn1.to_q.{up}.weight": self.up}, str(self.lora))
                self.merge()
                self.torch.testing.assert_close(self.load(self.output)["unet.block.attn1.to_q.weight"],
                                               self.weight + self.up @ self.down)
                self.output.unlink()
        self.save({"unet.block.attn1.processor.to_q_lora.down.weight": self.down,
                   "unet.block.attn1.processor.to_q_lora.up.weight": self.up}, str(self.lora))
        self.merge()
        self.torch.testing.assert_close(self.load(self.output)["unet.block.attn1.to_q.weight"],
                                       self.weight + self.up @ self.down)

    @unittest.skipUnless(importlib.util.find_spec("diffusers"), "Requires existing Diffusers key converters")
    def test_sd_checkpoint_maps_kohya_unet_and_first_text_encoder(self):
        unet = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
        clip = "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        state = {unet: self.weight, clip: self.weight.clone(),
                 "model.diffusion_model.input_blocks.0.0.weight": self.torch.zeros(3, 4, 3, 3)}
        self.save(state, str(self.base))
        for target, key in (("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q", unet),
                            ("lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q", unet),
                            ("lora_te_text_model_encoder_layers_0_self_attn_q_proj", clip)):
            with self.subTest(target=target):
                self.adapter(target=target, style="kohya", alpha=1.)
                self.merge()
                for name, actual in self.load(self.output).items():
                    expected = state[name] + self.up @ self.down / 2 if name == key else state[name]
                    self.torch.testing.assert_close(actual, expected)
                self.output.unlink()

    @unittest.skipUnless(importlib.util.find_spec("diffusers"), "Requires existing OpenCLIP mapping")
    def test_sdxl_packed_qkv_updates_only_requested_projection(self):
        key = "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"
        base = self.torch.cat([self.weight, self.weight + 10, self.weight + 20])
        self.save({key: base}, str(self.base))
        self.adapter(target="lora_te2_text_model_encoder_layers_0_self_attn_k_proj", style="kohya")
        self.merge(mode="weighted-difference", weights=0.5)
        expected = base.clone()
        expected[3:6] -= 0.5 * (self.up @ self.down)
        self.torch.testing.assert_close(self.load(self.output)[key], expected)

    def test_unqualified_targets_and_flattened_kohya_collisions_are_rejected(self):
        base = self.root / "pipeline"
        base.mkdir()
        (base / "model_index.json").write_text('{}')
        for component in ("unet", "text_encoder"):
            (base / component).mkdir()
            self.save({"layer.weight": self.weight}, str(base / component / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            merge_models(base, self.lora, output=self.root / "merged")
        self.save({"unet.a_b.c.weight": self.weight, "unet.a.b_c.weight": self.weight.clone()}, str(self.base))
        self.adapter(target="lora_unet_a_b_c", style="kohya")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.merge()

    def test_dtype_precision_and_overflow(self):
        t = self.torch
        for dtype in (t.float16, t.bfloat16, t.float64):
            self.save({"layer.weight": self.weight.to(dtype)}, str(self.base))
            self.adapter(down=self.down.to(dtype), up=self.up.to(dtype))
            self.merge(weights=0.5)
            actual = self.load(self.output)["layer.weight"]
            self.assertEqual(actual.dtype, dtype)
            t.testing.assert_close(actual, (self.weight + 0.5 * (self.up @ self.down)).to(dtype))
            self.output.unlink()
        self.save({"layer.weight": self.weight.half()}, str(self.base))
        self.adapter()
        with self.assertRaisesRegex(ValueError, "overflow"):
            self.merge(weights=1e8)
        self.assertFalse(self.output.exists())

    def test_peft_sidecar_is_used_and_verified_before_publication(self):
        import model_merge_files
        adapter = self.root / "adapter_model.safetensors"
        self.adapter(adapter)
        config = self.root / "adapter_config.json"
        config.write_text(json.dumps({"r": 2, "lora_alpha": 1}))
        original_verify = model_merge_files.MergeModelFiles.verify
        def changed(model):
            config.write_text('{"r": 2, "lora_alpha": 2}')
            original_verify(model)
        with patch.object(model_merge_files.MergeModelFiles, "verify", changed):
            with self.assertRaises(RuntimeError):
                merge_models(self.base, adapter, output=self.output)
        self.assertFalse(self.output.exists())
        merge_models(self.base, adapter, output=self.output)
        self.torch.testing.assert_close(self.load(self.output)["layer.weight"], self.weight + self.up @ self.down)

    def test_adapter_directory_overlap_is_rejected_even_with_file_base(self):
        directory = self.root / "adapter"
        directory.mkdir()
        self.adapter(directory / "pytorch_lora_weights.safetensors")
        for options in ({"output": directory / "merged.safetensors"},
                        {"cache_dir": directory / "cache", "output": self.output}):
            with self.assertRaisesRegex(ValueError, "inside"):
                merge_models(self.base, directory, **options)

    def test_legacy_adapter_uses_existing_safe_checkpoint_conversion(self):
        legacy = self.root / "adapter.ckpt"
        self.torch.save({"state_dict": self.load(self.lora)}, legacy)
        merge_models(self.base, legacy, output=self.output, cache_dir=self.root / "cache")
        self.torch.testing.assert_close(self.load(self.output)["layer.weight"], self.weight + self.up @ self.down)

    @unittest.skipUnless(importlib.util.find_spec("diffusers"), "Requires pinned SGM converters")
    def test_diffusers_base_accepts_sgm_kohya_block_names(self):
        base = self.root / "pipeline"
        (base / "unet").mkdir(parents=True)
        (base / "model_index.json").write_text('{}')
        (base / "unet/config.json").write_text('{"layers_per_block": 2}')
        key = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight"
        self.save({key: self.weight}, str(base / "unet/model.safetensors"))
        self.adapter(target="lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q", style="kohya")
        out = self.root / "pipeline-merged"
        merge_models(base, self.lora, output=out)
        self.torch.testing.assert_close(self.load(out / "unet/model.safetensors")[key],
                                       self.weight + self.up @ self.down)

    def test_peft_flattened_clip_names_and_fan_in_fan_out(self):
        base = self.root / "pipeline"
        (base / "text_encoder").mkdir(parents=True)
        (base / "model_index.json").write_text('{}')
        key = "encoder.layers.0.self_attn.q_proj.weight"
        self.save({key: self.weight}, str(base / "text_encoder/model.safetensors"))
        self.adapter(target="base_model.model.text_model." + key.removesuffix(".weight"))
        out = self.root / "pipeline-merged"
        merge_models(base, self.lora, output=out)
        self.torch.testing.assert_close(self.load(out / "text_encoder/model.safetensors")[key],
                                       self.weight + self.up @ self.down)
        self.save({"layer.weight": self.weight.T.contiguous()}, str(self.base))
        adapter = self.root / "adapter_model.safetensors"
        self.adapter(adapter)
        (self.root / "adapter_config.json").write_text('{"r": 2, "lora_alpha": 2, "fan_in_fan_out": true}')
        merge_models(self.base, adapter, output=self.output)
        self.torch.testing.assert_close(self.load(self.output)["layer.weight"], (self.weight + self.up @ self.down).T)

    @unittest.skipUnless(importlib.util.find_spec("diffusers"), "Requires OpenCLIP mapping")
    def test_openclip_text_projection_transpose(self):
        key = "conditioner.embedders.1.model.text_projection"
        self.save({key: self.weight.T.contiguous()}, str(self.base))
        self.adapter(target="lora_te2_text_projection", style="kohya")
        self.merge()
        self.torch.testing.assert_close(self.load(self.output)[key], (self.weight + self.up @ self.down).T)


if __name__ == "__main__":
    unittest.main()
