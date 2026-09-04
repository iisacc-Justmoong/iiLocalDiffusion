#!/usr/bin/env python3
"""Downloaded-file routing, integrity and managed-process lifecycle contracts."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import local_image


class LocalImageTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.model = self.directory / "download.safetensors"
        self.model.write_bytes(b"fixture")
        self.inspection = {"role": "checkpoint", "weights_role": "checkpoint",
                           "architecture": "sdxl", "base_model": None, "task": "text-to-image"}

    def args(self, *tokens):
        return local_image.build_parser().parse_args(["--model", str(self.model), *tokens])

    def resolve(self, *tokens):
        with patch("downloaded_model.inspect_downloaded_model", return_value=self.inspection):
            return local_image.resolved_request(self.args(*tokens))

    def test_tensor_architecture_selects_sdxl_without_filename_guess(self):
        request = self.resolve()
        self.assertEqual(request["base_model"], "SDXL 1.0")
        self.assertEqual(request["model_type"], "checkpoint")

    def test_named_derivative_must_match_tensor_family(self):
        self.assertEqual(self.resolve("--base-model", "Illustrious")["base_model"], "Illustrious")
        with self.assertRaisesRegex(ValueError, "tensor architecture"):
            self.resolve("--base-model", "Flux.1 Krea")

    def test_adapters_are_not_accepted_as_standalone_models(self):
        for role in ("lora", "vae", "embedding", "controlnet"):
            self.inspection["role"] = role
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "base checkpoint"):
                self.resolve()

    def test_partial_and_quantized_downloads_choose_component_loader(self):
        self.inspection["weights_role"] = "denoiser"
        with self.assertRaisesRegex(ValueError, "requires --vae"):
            self.resolve()
        request = self.resolve("--vae", str(self.model), "--text-encoder", str(self.model))
        self.assertEqual(request["model_type"], "diffusion_model")

    def test_partially_bundled_checkpoint_preserves_embedded_components(self):
        self.inspection.update(weights_role="denoiser", available_components=["denoiser", "text_encoder"])
        request = self.resolve("--vae", str(self.model))
        self.assertEqual(request["model_type"], "checkpoint")
        self.assertEqual(set(request["components"]), {"vae"})

    def test_metadata_prediction_and_component_overrides_are_preserved(self):
        self.inspection.update(base_model="NoobAI", prediction_type="v_prediction")
        request = self.resolve("--components", json.dumps({"vae": str(self.model)}))
        self.assertEqual(request["prediction_type"], "v_prediction")
        self.assertEqual(request["components"]["vae"], str(self.model))

    def test_wrong_task_unknown_component_and_nonfinite_timeout_fail_before_startup(self):
        with self.assertRaisesRegex(ValueError, "Unknown component"):
            self.resolve("--components", '{"typo":"x"}')
        for timeout in ("nan", "inf", "0", "-1"):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "positive and finite"):
                self.resolve("--timeout", timeout)
        self.inspection["task"] = "inpainting"
        with self.assertRaisesRegex(ValueError, "input image"):
            self.resolve()

    def test_staged_files_keep_original_and_detect_later_mutation(self):
        request = self.resolve()
        staged, identities = local_image.stage_models(request, self.directory)
        target = self.directory / "models/checkpoints" / staged["model"]
        self.assertEqual(target.resolve(), self.model)
        local_image.verify_identities(identities)
        self.model.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed during generation"):
            local_image.verify_identities(identities)

    def test_retargeted_model_symlink_cannot_reuse_original_provenance(self):
        staged, identities = local_image.stage_models(self.resolve(), self.directory)
        alternate = self.directory / "other.safetensors"
        alternate.write_bytes(self.model.read_bytes())
        target = self.directory / "models/checkpoints" / staged["model"]
        target.unlink()
        target.symlink_to(alternate)
        with self.assertRaisesRegex(RuntimeError, "Staged model path changed"):
            local_image.verify_identities(identities)

    def runtime_args(self):
        source = self.directory / "runtime"
        source.mkdir()
        (source / "main.py").write_text("")
        executable = self.directory / "python"
        executable.write_text("fixture")
        return self.args("--runtime-source", str(source), "--runtime-python", str(executable), "--device", "cpu")

    def test_managed_server_is_loopback_isolated_and_stopped_even_on_error(self):
        args = self.runtime_args()
        venv_python = self.directory / "venv-python"
        venv_python.symlink_to(args.runtime_python)
        args.runtime_python = venv_python
        process = Mock()
        process.poll.return_value = None
        client = Mock(url="http://127.0.0.1:1234")
        client.json.side_effect = [{"SaveImage": {}}, {"devices": [{"type": "cpu"}]}]
        with patch("local_image.subprocess.Popen", return_value=process) as popen, patch("local_image.Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                with local_image.managed_server(args, self.directory):
                    raise RuntimeError("generation failed")
        command = popen.call_args.args[0]
        self.assertEqual(command[0], str(venv_python))
        self.assertIn("127.0.0.1", command)
        self.assertIn("--disable-api-nodes", command)
        self.assertIn("--disable-all-custom-nodes", command)
        self.assertTrue((self.directory / "custom_nodes").is_dir())
        process.terminate.assert_called_once()
        process.wait.assert_called_once()

    def test_requested_accelerator_mismatch_stops_process(self):
        args = self.runtime_args()
        args.device = "mps"
        process = Mock()
        process.poll.return_value = None
        client = Mock()
        client.json.side_effect = [{}, {"devices": [{"type": "cpu"}]}]
        with patch("local_image.subprocess.Popen", return_value=process), patch("local_image.Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "selected"):
                with local_image.managed_server(args, self.directory):
                    self.fail("Mismatched device must never execute the workflow")
        process.terminate.assert_called_once()

    def test_startup_exit_has_log_location_and_waits_for_owned_process(self):
        args = self.runtime_args()
        process = Mock()
        process.poll.return_value = 1
        with patch("local_image.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "runtime.log"):
                with local_image.managed_server(args, self.directory):
                    self.fail("Exited runtime cannot generate")
        process.terminate.assert_not_called()
        process.wait.assert_called_once()

    def test_plain_python_delegates_to_installed_runtime_before_ml_imports(self):
        runtime_python = self.directory / "managed-python"
        runtime_python.write_text("fixture")
        with patch("local_image.subprocess.run", return_value=Mock(returncode=0)) as run, patch("local_image.resolved_request") as resolve:
            self.assertEqual(local_image.main(["--model", str(self.model), "--runtime-python", str(runtime_python)]), 0)
        resolve.assert_not_called()
        self.assertEqual(run.call_args.args[0][0], str(runtime_python))

    def test_hires_cli_resolves_repeated_dimensions_without_starting_runtime(self):
        request = self.resolve("--hires-fix", "--hires-passes", "3", "--hires-scale", "1.5",
                               "--hires-strength", "0.4", "--hires-steps", "10", "--width", "64", "--height", "64")
        plan = request["hires"]
        self.assertEqual((plan["passes"], plan["scale"], plan["strength"], plan["steps"]), (3, 1.5, 0.4, 10))
        self.assertEqual([(x["width"], x["height"]) for x in plan["stages"]], [(96, 96), (144, 144), (216, 216)])
        self.assertFalse(self.resolve()["hires"]["enabled"])
        self.assertEqual(self.resolve("--hires-fix")["hires"]["passes"], 1)
        self.assertEqual(self.resolve("--hires-fix", "--hires-denoising-strength", "0.6")["hires"]["strength"], 0.6)

    def test_hires_bad_options_fail_before_managed_server_start(self):
        for flags in (("--hires-passes", "2"), ("--no-hires-fix", "--hires-scale", "2"),
                      ("--hires-fix", "--hires-passes", "0"), ("--hires-fix", "--hires-passes", "-1"),
                      ("--hires-fix", "--hires-strength", "nan"), ("--hires-fix", "--hires-scale", "1"),
                      ("--hires-fix", "--hires-steps", "0")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.resolve(*flags)

    def test_main_forwards_all_hires_inputs_to_the_generated_workflow(self):
        flags = ["--hires-fix", "--hires-passes", "2", "--hires-scale", "1.5", "--hires-strength", "0.6",
                 "--hires-steps", "10", "--hires-upscaler", "bicubic", "--width", "64", "--height", "64"]
        request = self.resolve(*flags)
        server = MagicMock()
        server.__enter__.return_value = ("http://127.0.0.1:1", {}, {})
        with patch("local_image.ROOT", self.directory), patch("local_image.resolved_request", return_value=request), \
                patch("local_image.stage_models", return_value=({"model": "checkpoint.safetensors"}, {})), \
                patch("local_image.managed_server", return_value=server), \
                patch("comfyui_image_workflow.build_workflow", return_value={}) as build, \
                patch("local_image.run_workflow", return_value={"status": "preflight_passed"}), \
                redirect_stdout(StringIO()):
            result = local_image.main(["--model", str(self.model), "--runtime-python", str(self.directory / "absent-python"),
                                       "--validate-only", *flags])
        self.assertEqual(result, 0)
        self.assertEqual({key: value for key, value in build.call_args.kwargs.items() if key.startswith("hires_")},
                         {"hires_fix": True, "hires_passes": 2, "hires_scale": 1.5, "hires_strength": 0.6,
                          "hires_steps": 10, "hires_upscaler": "bicubic"})

    def test_final_image_size_must_match_the_last_hires_stage(self):
        image = Mock(width=256, height=256, format="PNG")
        image.convert.return_value.getextrema.return_value = ((0, 255),) * 3
        opened = MagicMock()
        opened.__enter__.return_value = image
        image_api = SimpleNamespace(open=Mock(return_value=opened))
        with patch.dict(sys.modules, {"PIL": SimpleNamespace(Image=image_api)}):
            result = {"artifacts": [{"path": "final.png"}]}
            self.assertEqual(local_image.verify_images(result, 1, [256, 256])[0]["width"], 256)
            with self.assertRaisesRegex(RuntimeError, "Expected final image size"):
                local_image.verify_images(result, 1, [512, 512])


if __name__ == "__main__":
    unittest.main()
