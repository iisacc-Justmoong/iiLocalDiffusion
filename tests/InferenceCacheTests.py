#!/usr/bin/env python3
"""Resident inference must reuse unchanged inputs without hiding model edits."""

import io
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import weight_files
import inference_session
import inference_worker


class InferenceCacheTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="inference-cache-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "model.safetensors"
        self.model.write_bytes(b"original")
        weight_files.clear_model_hash_cache()

    def test_one_full_hash_across_resolution_loader_verification_and_next_request(self):
        with patch.object(weight_files, "file_sha256", wraps=weight_files.file_sha256) as digest:
            first = weight_files.resolve_weight_file(str(self.model), "--model")
            for _ in range(2):
                with weight_files.checked_safetensors_path(first, self.directory, "model"):
                    pass
            weight_files.verify_weight_file(first, "model")
            self.assertEqual(weight_files.resolve_weight_file(str(self.model), "--model"), first)
            self.assertEqual(digest.call_count, 1)

    def test_same_size_edit_with_restored_mtime_is_rehashed_and_rejected(self):
        before = self.model.stat()
        first = weight_files.resolve_weight_file(str(self.model), "--model")
        self.model.write_bytes(b"modified")
        os.utime(self.model, ns=(before.st_atime_ns, before.st_mtime_ns))
        with patch.object(weight_files, "file_sha256", wraps=weight_files.file_sha256) as digest:
            with self.assertRaisesRegex(RuntimeError, "changed"):
                weight_files.verify_weight_file(first, "model")
            replacement = weight_files.resolve_weight_file(str(self.model), "--model")
            self.assertNotEqual(first.sha256, replacement.sha256)
            weight_files.verify_weight_file(replacement, "model")
            self.assertEqual(digest.call_count, 1)

    def test_atomic_replacement_and_symlink_retarget_invalidate_identity(self):
        before = self.model.stat()
        first = weight_files.resolve_weight_file(str(self.model), "--model")
        replacement = self.directory / "replacement.safetensors"
        replacement.write_bytes(b"modified")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(self.model)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            weight_files.verify_weight_file(first, "model")
        alias = self.directory / "alias.safetensors"
        alias.symlink_to(self.model)
        selected = weight_files.resolve_weight_file(str(alias), "--model")
        replacement.write_bytes(b"modified")
        alias.unlink()
        alias.symlink_to(replacement)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            weight_files.verify_weight_file(selected, "model")

    def test_mutation_while_hashing_never_populates_cache(self):
        original = weight_files.file_sha256
        def mutate(path):
            digest = original(path)
            path.write_bytes(b"modified")
            return digest
        with patch.object(weight_files, "file_sha256", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                weight_files.resolve_weight_file(str(self.model), "--model")
        with patch.object(weight_files, "file_sha256", wraps=original) as digest:
            weight_files.resolve_weight_file(str(self.model), "--model")
            self.assertEqual(digest.call_count, 1)

    def test_generic_directory_hashes_are_shared_and_model_additions_are_detected(self):
        import generate_any
        with patch.object(weight_files, "file_sha256", wraps=weight_files.file_sha256) as digest:
            first = generate_any.model_identity(self.directory)
            generate_any.verify_identity(first)
            self.assertEqual(generate_any.model_identity(self.directory), first)
            self.assertEqual(digest.call_count, 1)
        (self.directory / "new.safetensors").write_bytes(b"new")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            generate_any.verify_identity(first)

    def test_metadata_and_controlnet_reuse_the_same_model_digest(self):
        import DownloadedModelTests
        import downloaded_model
        import controlnet
        fixture = DownloadedModelTests.DownloadedModelTests()
        fixture.directory = self.directory
        path = fixture.sd()
        fixture.info(path)
        with patch.object(weight_files, "file_sha256", wraps=weight_files.file_sha256) as digest:
            downloaded_model.inspect_downloaded_model(path)
            selected = weight_files.resolve_weight_file(str(path), "--model")
            identities = controlnet._identities([path])
            for identity in identities:
                weight_files.verify_weight_file(identity, "ControlNet")
            self.assertEqual(digest.call_count, 1)
            self.assertEqual(selected.sha256, identities[0].sha256)

    def test_single_resident_pipeline_reuses_and_evicts_on_model_or_options_change(self):
        loader = Mock(side_effect=object)
        with inference_session.InferenceSession() as session:
            first, hit = inference_session.cached_pipeline("cpu", [self.model], loader)
            self.assertFalse(hit)
            self.assertIs(inference_session.cached_pipeline("cpu", [self.model], loader)[0], first)
            self.assertEqual(loader.call_count, 1)
            self.model.write_bytes(b"modified")
            second, hit = inference_session.cached_pipeline("cpu", [self.model], loader)
            self.assertFalse(hit)
            self.assertIsNot(first, second)
            inference_session.cached_pipeline("mps", [self.model], loader)
            self.assertEqual(loader.call_count, 3)
            self.assertEqual(session.hits, 1)
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                inference_session.cached_pipeline("bad", [self.model], Mock(side_effect=RuntimeError("load failed")))
            inference_session.cached_pipeline("mps", [self.model], loader)
            self.assertEqual(loader.call_count, 4)
        self.assertIsNone(session.value)

    def test_directory_configuration_change_reloads_and_mid_inference_edit_fails(self):
        loader = Mock(side_effect=object)
        with inference_session.InferenceSession():
            inference_session.cached_pipeline("preset", [self.directory], loader)
            config = self.directory / "model_index.json"
            config.write_text("{}")
            inference_session.cached_pipeline("preset", [self.directory], loader)
            self.assertEqual(loader.call_count, 2)
            config.unlink()
            with self.assertRaisesRegex(RuntimeError, "changed"):
                inference_session.verify_pipeline_sources()

    def test_preset_reuses_weights_for_new_prompt_size_seed_but_resets_scheduler(self):
        import generate
        folder = self.directory / "diffusers"
        shutil.copytree(ROOT / "tests/fixtures/sd-v1-manifest", folder)
        class Scheduler:
            config = {}
            compatibles = ()
            @classmethod
            def from_config(cls, config):
                return cls()
        pipeline = SimpleNamespace(scheduler=Scheduler())
        def request(**values):
            return generate.resolve_request({"preset": "sd15", "model": str(folder), **values})
        def prepare(pipe, *unused, **options):
            return pipe, {"weight_storage": "ram"}, None, None
        with (inference_session.InferenceSession(),
              patch.object(generate, "load_generation_pipeline", return_value=(pipeline, {})) as load,
              patch.object(generate, "attach_controlnet", side_effect=lambda pipe, *args: (pipe, None)),
              patch.object(generate, "prepare_pipeline_with_adapters", side_effect=prepare) as initialize,
              patch.object(generate, "validate_clip_skip"), patch.object(generate, "validate_scheduler_values")):
            previous = pipeline.scheduler
            for index, values in enumerate(({}, {"prompt": "second", "seed": 22, "width": 64, "steps": 2,
                                                 "cache_dir": str(self.directory / "second-cache")},
                                           {"dtype": "float16"})):
                preset, args = request(**values)
                result = generate.prepared_pipeline(preset, args, {preset.pipeline_class: object}, {},
                                                    "cpu", args.dtype, False, None)
                self.assertEqual(result[-1], index == 1)
            self.assertIsNot(pipeline.scheduler, previous)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(initialize.call_count, 2)

    def test_worker_keeps_process_state_handles_errors_then_eof_without_persisted_queue(self):
        requests = [dict(schema="iild-worker-request-v1", id=str(i), arguments=[value])
                    for i, value in enumerate(("first", "second", "fail", "fourth"))]
        stream = io.BytesIO(("\n".join(json.dumps(item) for item in requests) + "\n").encode())
        output = io.StringIO()
        loader = Mock(side_effect=object)
        def generate(arguments):
            inference_session.cached_pipeline("cpu", [self.model], loader)
            weight_files.resolve_weight_file(str(self.model), "--model")
            if arguments == ["fail"]:
                raise SystemExit("fixture failure")
            return 0
        with patch("sys.stdout", output), patch("sys.stderr", io.StringIO()):
            self.assertEqual(inference_worker.serve(generate, stream), 0)
        events = [json.loads(line.split(" ", 1)[1]) for line in output.getvalue().splitlines()
                  if line.startswith("IILD_RESULT ")]
        self.assertEqual([event["id"] for event in events], ["0", "1", "2", "3"])
        self.assertEqual([event["ok"] for event in events], [True, True, False, True])
        self.assertEqual(events[1]["cache"]["model_hashes"], 0)
        self.assertEqual(events[1]["cache"]["pipeline_hits"], 1)
        self.assertEqual(loader.call_count, 2)  # Error evicts the possibly modified pipeline.
        self.assertEqual(list(self.directory.iterdir()), [self.model])

    def test_worker_rejects_malformed_and_recursive_requests(self):
        records = [b"{bad}\n", json.dumps({"schema": "iild-worker-request-v1", "id": "x",
                                         "arguments": ["--worker"]}).encode() + b"\n"]
        handler = Mock()
        with patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
            inference_worker.serve(handler, io.BytesIO(b"".join(records)))
        handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
