#!/usr/bin/env python3
"""Offline contracts for exhaustive, conservative Civitai base-model routing."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import civitai_catalog


# Independent name fixture from Civitai ec49115e55d7c85722ec6223ec342c36df59c9d4.
# Hidden and disabled entries deliberately remain in the compatibility surface.
UPSTREAM_NAMES = set("""
Anima|AuraFlow|Chroma|CogVideoX|Ernie|Flux.1 S|Flux.1 D|Flux.1 Krea|Flux.1 Kontext
Flux.2 D|Flux.2 Klein 9B|Flux.2 Klein 9B-base|Flux.2 Klein 4B|Flux.2 Klein 4B-base
Flux 3 Video|Grok|HappyHorse|HiDream|HiDream-O1|Hunyuan 1|Hunyuan Video|Ideogram 4.0
Boogu|Illustrious|Imagen4|Kolors|Krea 2|LTXV|LTXV2|LTXV 2.3|LTXV 2.5|Lens|Lumina
MageFlow|MAI|Mochi|Nano Banana|NoobAI|ODOR|OpenAI|Upscaler|Other|PixArt a|PixArt E
Playground v2|Pony|Pony V7|Qwen|Qwen 2|Qwen 3|Stable Cascade|SD 1.4|SD 1.5
SD 1.5 LCM|SD 1.5 Hyper|SD 2.0|SD 2.0 768|SD 2.1|SD 2.1 768|SD 2.1 Unclip
SD 3|SD 3.5|SD 3.5 Large|SD 3.5 Large Turbo|SD 3.5 Medium|SDXL 0.9|SDXL 1.0
SDXL 1.0 LCM|SDXL Lightning|SDXL Hyper|SDXL Turbo|SDXL Distilled|Reve|Muse Image
Seedream|SVD|SVD XT|Sora 2|Veo 3|Wan Video|Wan Video 1.3B t2v|Wan Video 14B t2v
Wan Video 14B i2v 480p|Wan Video 14B i2v 720p|Wan Video 2.2 TI2V-5B
Wan Video 2.2 I2V-A14B|Wan Video 2.2 T2V-A14B|Wan Video 2.5 T2V|Wan Video 2.5 I2V
Wan Image 2.7|Wan Video 2.7|Wan Video 3.0|ZImageTurbo|ZImageBase|Vidu Q1|MiniMax H3
Kling|Seedance|ACE Audio|MiniMax Music 3|PolyGen|Tripo|Hunyuan3D|Pixal3D|Trellis.2
""".strip().replace("\n", "|").split("|"))


class CivitaiCatalogTests(unittest.TestCase):
    def test_every_upstream_name_has_one_record(self):
        rows = civitai_catalog.list_base_models()
        self.assertEqual(len(UPSTREAM_NAMES), 105)
        self.assertEqual(len(rows), 105)
        self.assertEqual({row["name"] for row in rows}, UPSTREAM_NAMES)

    def test_snapshot_is_pinned_and_auditable(self):
        source = civitai_catalog.CATALOG_SOURCE
        self.assertEqual(source["commit"], "ec49115e55d7c85722ec6223ec342c36df59c9d4")
        self.assertIn(source["commit"], source["url"])
        self.assertEqual(source["retrieved_at"], "2026-09-04")
        self.assertRegex(source["file_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(source["base_model_count"], len(UPSTREAM_NAMES))
        for source in civitai_catalog.UPSTREAM_SOURCES.values():
            self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
            self.assertIn(source["commit"], source["url"])

    def test_routing_records_have_explicit_status_and_evidence(self):
        for row in civitai_catalog.list_base_models():
            with self.subTest(name=row["name"]):
                self.assertIn(row["local_status"], {"local", "hosted", "unknown"})
                self.assertIn(row["preferred_backend"], {"preset", "diffusers", "comfyui", "unavailable"})
                self.assertTrue(row["family"])
                self.assertTrue(row["task"])
                self.assertTrue(row["notes"])
                self.assertTrue(row["sources"])
                self.assertTrue(all(url.startswith("https://") for url in row["sources"]))
                if row["local_status"] != "local":
                    self.assertEqual(row["preferred_backend"], "unavailable")
                    self.assertIsNone(row["preset"])
                    self.assertIsNone(row["pipeline_class"])
                elif row["preferred_backend"] == "preset":
                    self.assertIsNotNone(row["preset"])
                elif row["preferred_backend"] == "diffusers":
                    self.assertTrue(row["pipeline_class"].endswith("Pipeline"))

    def test_requested_families_route_to_compatible_named_presets(self):
        for name, preset, family in [
            ("Illustrious", "illustrious", "sdxl"),
            ("NoobAI", "noobai", "sdxl"),
            ("Pony", "pony", "sdxl"),
            ("Flux.1 Krea", "flux1-krea-dev", "flux1"),
        ]:
            with self.subTest(name=name):
                row = civitai_catalog.lookup_base_model(name)
                self.assertEqual(row["local_status"], "local")
                self.assertEqual(row["preset"], preset)
                self.assertEqual(row["family"], family)
        self.assertIn("v_prediction", civitai_catalog.lookup_base_model("NoobAI")["notes"])

    def test_related_names_do_not_collapse_distinct_architectures(self):
        self.assertEqual(civitai_catalog.lookup_base_model("Pony V7")["pipeline_class"], "AuraFlowPipeline")
        self.assertEqual(civitai_catalog.lookup_base_model("Krea 2")["pipeline_class"], "Krea2Pipeline")
        self.assertEqual(civitai_catalog.lookup_base_model("Flux.1 Kontext")["pipeline_class"], "FluxKontextPipeline")
        self.assertNotEqual(civitai_catalog.lookup_base_model("LTXV 2.5")["family"], civitai_catalog.lookup_base_model("LTXV2")["family"])

    def test_video_audio_and_3d_are_not_misrepresented_as_images(self):
        for name, task in [
            ("SVD", "image-to-video"), ("SVD XT", "image-to-video"),
            ("Wan Video 14B i2v 720p", "image-to-video"),
            ("ACE Audio", "text-to-audio"), ("MiniMax Music 3", "text-to-audio"),
            ("Hunyuan3D", "image-to-3d"), ("Pixal3D", "image-to-3d"),
            ("Trellis.2", "image-to-3d"), ("Upscaler", "image-upscale"),
        ]:
            with self.subTest(name=name):
                self.assertEqual(civitai_catalog.lookup_base_model(name)["task"], task)

    def test_hidden_disabled_local_models_are_retained(self):
        for name in ("SD 3", "SD 3.5 Medium", "SDXL Turbo", "SVD", "SVD XT"):
            with self.subTest(name=name):
                row = civitai_catalog.lookup_base_model(name)
                self.assertTrue(row["civitai_hidden"])
                self.assertTrue(row["civitai_disabled"])
                self.assertEqual(row["local_status"], "local")

    def test_modular_and_custom_models_require_explicit_workflows(self):
        for name in ("Anima", "MiniMax H3", "MiniMax Music 3", "Lens", "MageFlow", "HiDream-O1"):
            with self.subTest(name=name):
                row = civitai_catalog.lookup_base_model(name)
                self.assertEqual(row["preferred_backend"], "comfyui")
                self.assertIsNone(row["pipeline_class"])

    def test_hosted_and_unknown_do_not_acquire_local_routes(self):
        for name in ("OpenAI", "Sora 2", "Veo 3", "Nano Banana", "Qwen 3", "Other", "ODOR"):
            with self.subTest(name=name):
                self.assertEqual(civitai_catalog.lookup_base_model(name)["preferred_backend"], "unavailable")

    def test_lookup_normalizes_only_case_and_outer_whitespace(self):
        self.assertEqual(civitai_catalog.lookup_base_model("  ILLUSTRIOUS  ")["name"], "Illustrious")
        for name in ("", " ", "SDXL probably", "Flux.1 Kreaa", "Illustrious-v999", None, 15):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    civitai_catalog.lookup_base_model(name)

    def test_returned_records_cannot_mutate_future_routing(self):
        row = civitai_catalog.lookup_base_model("Illustrious")
        row["family"] = "flux1"
        row["sources"].clear()
        self.assertEqual(civitai_catalog.lookup_base_model("Illustrious")["family"], "sdxl")
        self.assertTrue(civitai_catalog.lookup_base_model("Illustrious")["sources"])
        rows = civitai_catalog.list_base_models()
        rows[0]["sources"].clear()
        rows.pop()
        self.assertEqual(len(civitai_catalog.list_base_models()), 105)
        self.assertTrue(civitai_catalog.list_base_models()[0]["sources"])


if __name__ == "__main__":
    unittest.main()
