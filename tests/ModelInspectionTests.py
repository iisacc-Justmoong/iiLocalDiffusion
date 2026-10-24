"""Inspect metadata without loading weights; preserve manual Society classification."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

PROGRAM = Path(sys.argv.pop(1)).resolve()


def safetensors(path, metadata=None, names=(), payload=b"\0\0"):
    header = {"__metadata__": metadata or {}}
    for i, name in enumerate(names):
        header[name] = {"dtype": "BF16", "shape": [1], "data_offsets": [i * 2, i * 2 + 2]}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload * max(1, len(names)))


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="inspection-", dir=PROGRAM.parent)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_tool(self, path, operation="inspect", modalities=None, success=True):
        args = [str(PROGRAM), operation, "--input", str(path)]
        if modalities is not None:
            args += ["--modalities", modalities]
        completed = subprocess.run(args, text=True, capture_output=True, timeout=10)
        self.assertEqual(completed.returncode == 0, success, completed.stdout + completed.stderr)
        return json.loads(completed.stdout)

    def test_ltx2_pipeline_classes_are_video_outputs(self):
        for name in ("LTX2Pipeline", "LTX2ConditionPipeline", "LTX2ImageToVideoPipeline"):
            directory = self.root / name
            directory.mkdir()
            (directory / "model_index.json").write_text(json.dumps({"_class_name": name}))
            result = self.run_tool(directory)
            self.assertEqual(result["modalities"], ["audio", "video"])
            self.assertNotIn("image", result["modalities"])

    def test_renamed_single_checkpoint_reads_architecture(self):
        path = self.root / "nothing-to-do-with-video.safetensors"
        safetensors(path, {"model_version": "2.3.0", "config": json.dumps({
            "transformer": {"_class_name": "AVTransformer3DModel"},
            "vae": {"_class_name": "CausalVideoAutoencoder"}})})
        result = self.run_tool(path)
        self.assertEqual(result["modalities"], ["audio", "video"])
        self.assertEqual(result["version"], "2.3.0")
        self.assertTrue(result["recognized"])

    def test_package_manifest_and_tensor_signatures(self):
        path = self.root / "package.safetensors"
        safetensors(path, {"iild_package_manifest": json.dumps({"family": "LTX 2.3"})})
        self.assertEqual(self.run_tool(path)["modalities"], ["audio", "video"])
        safetensors(path, names=["model.diffusion_model.transformer_blocks.0.weight",
                                "audio_vae.decoder.weight", "text_embedding_projection.audio_aggregate_embed.weight"])
        self.assertEqual(self.run_tool(path)["modalities"], ["audio", "video"])

    def test_filename_is_not_model_evidence(self):
        path = self.root / "LTX2 VIDEO.safetensors"
        safetensors(path)
        result = self.run_tool(path)
        self.assertFalse(result["recognized"])
        self.assertEqual(result["modalities"], [])

    def test_existing_modality_override_and_multi_choice(self):
        path = self.root / "checkpoint.safetensors"
        safetensors(path, {"config": '{"_class_name":"LTX2Pipeline"}'})
        sidecar = Path(str(path) + ".model.json")
        sidecar.write_text(json.dumps({"society": {"modality": "image", "type": "Checkpoint"}, "license": "keep"}))
        self.assertEqual(self.run_tool(path)["modalities"], ["image"])
        original = path.read_bytes()
        result = self.run_tool(path, "classify", "video,image,video")
        self.assertEqual(result["modalities"], ["image", "video"])
        self.assertTrue(result["user_override"])
        saved = json.loads(sidecar.read_text())
        self.assertEqual(saved["society"]["type"], "Checkpoint")
        self.assertEqual(saved["license"], "keep")
        result = self.run_tool(path, "classify", "auto")
        self.assertEqual(result["modalities"], ["audio", "video"])
        self.assertFalse(result["user_override"])
        self.assertEqual(path.read_bytes(), original)

    def test_stem_sidecar_and_embedded_metadata_can_be_overridden(self):
        path = self.root / "checkpoint.safetensors"
        safetensors(path, {"config": '{"_class_name":"LTXPipeline"}', "society.modality": "image"})
        path.with_suffix(".model.json").write_text('{"society":{"modality":"language"}}')
        self.assertEqual(self.run_tool(path)["modalities"], ["language"])
        self.assertEqual(self.run_tool(path, "classify", "auto")["modalities"], ["video"])

    def test_legacy_ltx_and_image_directories(self):
        for name, modality in [("LTXPipeline", "video"), ("StableDiffusionXLPipeline", "image")]:
            directory = self.root / name
            directory.mkdir()
            (directory / "model_index.json").write_text(json.dumps({"_class_name": name}))
            self.assertEqual(self.run_tool(directory)["modalities"], [modality])

    def test_malformed_header_is_bounded_and_can_still_be_classified(self):
        path = self.root / "invalid.safetensors"
        path.write_bytes(struct.pack("<Q", 2**63) + b"{}")
        result = self.run_tool(path)
        self.assertFalse(result["recognized"])
        self.assertIn("header", result["diagnostic"])
        self.assertEqual(self.run_tool(path, "classify", "video")["modalities"], ["video"])
        self.run_tool(path, "classify", "banana", success=False)

    def test_header_only_large_sparse_file(self):
        path = self.root / "large.safetensors"
        safetensors(path, {"config": '{"_class_name":"LTX2Pipeline"}'})
        with path.open("r+b") as file:
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes
                import msvcrt
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                ioctl = kernel.DeviceIoControl
                ioctl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
                ioctl.restype = wintypes.BOOL
                returned = wintypes.DWORD()
                handle = msvcrt.get_osfhandle(file.fileno())
                if not ioctl(handle, 0x000900C4, None, 0, None, 0, ctypes.byref(returned), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                # The Windows CRT's truncate extension writes zero buffers even
                # for sparse files. Extend the sparse EOF without those writes.
                kernel.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
                kernel.SetFilePointerEx.restype = wintypes.BOOL
                kernel.SetEndOfFile.argtypes = [wintypes.HANDLE]
                kernel.SetEndOfFile.restype = wintypes.BOOL
                if not kernel.SetFilePointerEx(handle, 40 * 1024**3, None, 0) or not kernel.SetEndOfFile(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                file.truncate(40 * 1024**3)
        self.assertEqual(path.stat().st_size, 40 * 1024**3)
        if os.name == "nt":
            self.assertTrue(path.stat().st_file_attributes & 0x200, "large fixture must be sparse")
            kernel.GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
            kernel.GetCompressedFileSizeW.restype = wintypes.DWORD
            high = wintypes.DWORD()
            allocated = kernel.GetCompressedFileSizeW(str(path), ctypes.byref(high)) | (high.value << 32)
            self.assertLess(allocated, 1024**2, "sparse fixture must not allocate its 40 GiB logical range")
        result = self.run_tool(path)
        self.assertLess(result["metadata_bytes_read"], 4096)
        self.assertEqual(result["modalities"], ["audio", "video"])

    def test_directory_classification_preserves_metadata_and_weights(self):
        directory = self.root / "pipeline"
        directory.mkdir()
        (directory / "model_index.json").write_text('{"_class_name":"LTX2Pipeline"}')
        sidecar = directory / "society.model.json"
        sidecar.write_text('{"society":{"type":"Checkpoint"},"description":"retain"}')
        before = hashlib.sha256((directory / "model_index.json").read_bytes()).hexdigest()
        self.assertEqual(self.run_tool(directory, "classify", "video")["modalities"], ["video"])
        self.assertEqual(json.loads(sidecar.read_text())["description"], "retain")
        self.assertEqual(hashlib.sha256((directory / "model_index.json").read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
