"""Inspect metadata without loading weights; preserve manual Society classification."""
import resource_fixture
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import struct
import subprocess
import sys
import unittest

PROGRAM = Path(sys.argv.pop(1)).resolve()


def safetensors(path, metadata=None, names=(), payload=b"\0\0"):
    header = {"__metadata__": metadata or {}}
    for i, name in enumerate(names):
        header[name] = {"dtype": "BF16", "shape": [1], "data_offsets": [i * 2, i * 2 + 2]}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload * max(1, len(names)))


def safetensors_shaped(path, shapes, metadata=None):
    header, data = {"__metadata__": metadata or {}}, bytearray()
    for name, shape in shapes.items():
        size = 2
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [len(data), len(data) + size]}
        data.extend(bytes(size))
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def gguf(path, metadata=(), names=("weight",), tensor_type=0):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    data = b"GGUF" + struct.pack("<IQQ", 3, len(names), len(metadata))
    for key, value in metadata:
        if isinstance(value, list):
            data += string(key) + struct.pack("<IIQ", 9, 8, len(value))
            data += b"".join(string(item) for item in value)
        else:
            data += string(key) + struct.pack("<I", 8) + string(value)
    for index, name in enumerate(names):
        data += string(name) + struct.pack("<IQIQ", 1, 1, tensor_type, index * 32)
    data += b"\0" * (-len(data) % 32) + b"\0" * (32 * len(names))
    path.write_bytes(data)


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = resource_fixture.TemporaryDirectory(prefix="inspection-")
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

    def test_known_video_classes_and_wan_tensors_are_not_images(self):
        cases = {
            'Wan': ('WanPipeline', 'WanImageToVideoPipeline', 'WanVideoToVideoPipeline',
                    'WanVACEPipeline', 'WanTransformer3DModel'),
            'Hunyuan Video': ('HunyuanVideoPipeline', 'HunyuanVideoImageToVideoPipeline',
                              'HunyuanVideoTransformer3DModel'),
            'CogVideoX': ('CogVideoXPipeline', 'CogVideoXImageToVideoPipeline', 'CogVideoXTransformer3DModel'),
            'Mochi': ('MochiPipeline', 'MochiTransformer3DModel'),
            'Stable Video Diffusion': ('StableVideoDiffusionPipeline', 'UNetSpatioTemporalConditionModel'),
        }
        directory = self.root / 'video-pipeline'
        directory.mkdir()
        for family, classes in cases.items():
            for name in classes:
                with self.subTest(name=name):
                    (directory / 'model_index.json').write_text(json.dumps({'_class_name': name}))
                    result = self.run_tool(directory)
                    self.assertEqual((result['family'], result['modalities']), (family, ['video']))

        path = self.root / 'checkpoint.safetensors'
        for prefix in ('', 'model.diffusion_model.', 'diffusion_model.', 'transformer.'):
            with self.subTest(prefix=prefix):
                safetensors(path, names=[prefix + 'patch_embedding.weight',
                                         prefix + 'blocks.0.cross_attn.norm_k.weight'])
                result = self.run_tool(path)
                self.assertEqual((result['family'], result['modalities']), ('Wan', ['video']))
        safetensors(path, names=['patch_embedding.weight', 'transformer.blocks.0.cross_attn.norm_k.weight'])
        self.assertFalse(self.run_tool(path)['recognized'])
        safetensors_shaped(path, {'patch_embedding.weight': [1], 'blocks.0.cross_attn.norm_k.weight': [1],
                                  'video_embedder.weight': [2, 3], 'final_linear_video.weight': [3, 2],
                                  'layers.0.proj.qkv.weight': [6, 2]})
        conflicting = self.run_tool(path)
        self.assertFalse(conflicting['recognized'])
        self.assertIn('Conflicting', conflicting['diagnostic'])

        first, second = self.root / 'wan-part-a.safetensors', self.root / 'wan-part-b.safetensors'
        safetensors(first, names=['patch_embedding.weight'])
        safetensors(second, names=['blocks.0.cross_attn.norm_k.weight'])
        index = self.root / 'wan.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'patch_embedding.weight': first.name,
                                                    'blocks.0.cross_attn.norm_k.weight': second.name}}))
        self.assertEqual(self.run_tool(index)['modalities'], ['video'])
        Path(str(index) + '.model.json').write_text('{"society":{"modality":"image"}}')
        overridden = self.run_tool(index)
        self.assertEqual(overridden['detected_modalities'], ['video'])
        self.assertEqual(overridden['modalities'], ['image'])

        # Some image GGUFs also declare "wan" as their loader architecture.
        # The tag alone must not turn an unknown model into a video model.
        gguf_path = self.root / 'weights.gguf'
        gguf(gguf_path, metadata=[('general.architecture', 'wan')], names=['video_patch_proj.weight'])
        self.assertFalse(self.run_tool(gguf_path)['recognized'])
        gguf(gguf_path, metadata=[('general.architecture', 'wan')],
             names=['patch_embedding.weight', 'blocks.0.cross_attn.norm_k.weight'])
        self.assertEqual(self.run_tool(gguf_path)['modalities'], ['video'])

    def test_container_probe_is_available_to_cli_and_python(self):
        safefile = self.root / "weights.bin"
        safetensors(safefile, names=["model.diffusion_model.weight"])
        gguf = self.root / "weights.data"
        gguf.write_bytes(b"GGUF" + b"\0" * 8)
        plain = self.root / "tokenizer.bin"
        plain.write_bytes(b"not model weights")
        self.assertEqual(self.run_tool(safefile, "probe")["format"], "safetensors")
        self.assertEqual(self.run_tool(gguf, "probe")["format"], "gguf")
        self.assertEqual(self.run_tool(plain, "probe")["format"], "unknown")
        path = Path(__file__).resolve().parents[1] / "src/ModelPackaging/python/iild_model_packaging.py"
        spec = importlib.util.spec_from_file_location("iild_model_packaging", path)
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        self.assertEqual(api.detect_container(safefile, executable=PROGRAM)["format"], "safetensors")
        self.assertEqual(api.detect_container(gguf, executable=PROGRAM)["format"], "gguf")

    def test_diffusers_wan_signature_is_detected_in_files_and_shards(self):
        names = ['patch_embedding.weight', 'blocks.0.attn1.to_q.weight', 'blocks.0.attn2.norm_k.weight',
                 'condition_embedder.time_embedder.linear_1.weight']
        path = self.root / 'diffusers-wan.safetensors'
        for prefix in ('', 'model.diffusion_model.', 'diffusion_model.', 'transformer.'):
            with self.subTest(prefix=prefix):
                safetensors(path, names=[prefix + name for name in names])
                original = path.read_bytes()
                result = self.run_tool(path)
                self.assertEqual((result['family'], result['modalities']), ('Wan', ['video']))
                self.assertEqual(path.read_bytes(), original)
        safetensors(path, names=names[:-1])
        self.assertFalse(self.run_tool(path)['recognized'])
        safetensors(path, names=[names[0], names[1], 'transformer.' + names[2], names[3]])
        self.assertFalse(self.run_tool(path)['recognized'])
        gguf_path = self.root / 'diffusers-wan.gguf'
        gguf(gguf_path, metadata=[('general.architecture', 'wan')], names=names)
        self.assertEqual(self.run_tool(gguf_path)['family'], 'Wan')
        first, second = self.root / 'diffusers-a.safetensors', self.root / 'diffusers-b.safetensors'
        safetensors(first, names=names[:2]); safetensors(second, names=names[2:])
        index = self.root / 'diffusers.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {name: first.name if i < 2 else second.name
                                                   for i, name in enumerate(names)}}))
        self.assertEqual(self.run_tool(index)['family'], 'Wan')

    def test_packaged_primary_gguf_is_classified_from_its_own_tensor_header(self):
        directory = self.root / 'gguf-package-input'
        directory.mkdir()
        model = directory / 'main.gguf'
        gguf(model, metadata=[('general.architecture', 'wan')],
             names=['patch_embedding.weight', 'blocks.0.cross_attn.norm_k.weight'])
        safetensors(directory / 'text_encoder.safetensors', names=['encoder.weight'])
        package = self.root / 'wan-package.safetensors'
        process = subprocess.run([str(PROGRAM), 'create', '--input', str(directory), '--output', str(package),
                                  '--assign-position', 'main.gguf=main'], capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        before = hashlib.sha256(package.read_bytes()).hexdigest()
        info = self.run_tool(package)
        self.assertEqual((info['family'], info['modalities']), ('Wan', ['video']))
        self.assertLess(info['metadata_bytes_read'], 1024 * 1024)
        self.assertEqual(hashlib.sha256(package.read_bytes()).hexdigest(), before)
        Path(str(package) + '.model.json').write_text('{"society":{"modality":"image"}}')
        info = self.run_tool(package)
        self.assertEqual((info['detected_modalities'], info['modalities']), (['video'], ['image']))
        Path(str(package) + '.model.json').unlink()
        bounded = self.root / 'bounded-gguf.safetensors'
        shutil.copyfile(package, bounded)
        with bounded.open('r+b') as file:
            size, = struct.unpack('<Q', file.read(8))
            header = json.loads(file.read(size))
            start, end = header['__iild_assets__/model/main.gguf']['data_offsets']
            file.seek(8 + size + start + 24)  # first metadata string length
            file.write(struct.pack('<Q', end - start + 1))
        bounded_info = self.run_tool(bounded)
        self.assertFalse(bounded_info['recognized'])
        self.assertIn('asset or header limit', bounded_info['diagnostic'])
        with package.open('r+b') as file:
            size, = struct.unpack('<Q', file.read(8))
            header = json.loads(file.read(size))
            start = header['__iild_assets__/model/main.gguf']['data_offsets'][0]
            file.seek(8 + size + start)
            file.write(b'BAD!')
        corrupt = self.run_tool(package)
        self.assertFalse(corrupt['recognized'])
        self.assertTrue(corrupt['diagnostic'])

    def test_packaged_wan_tag_without_wan_signature_remains_unclassified(self):
        directory = self.root / 'unknown-gguf-input'
        directory.mkdir()
        gguf(directory / 'main.gguf', metadata=[('general.architecture', 'wan')], names=['video_patch_proj.weight'])
        safetensors(directory / 'text_encoder.safetensors',
                    names=['patch_embedding.weight', 'blocks.0.cross_attn.norm_k.weight'])
        package = self.root / 'unknown-package.safetensors'
        process = subprocess.run([str(PROGRAM), 'create', '--input', str(directory), '--output', str(package),
                                  '--assign-position', 'main.gguf=main'], capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        info = self.run_tool(package)
        self.assertFalse(info['recognized'])
        self.assertEqual(info['modalities'], [])

    def test_packaged_secondary_gguf_does_not_classify_the_primary_model(self):
        directory = self.root / 'secondary-gguf-input'
        directory.mkdir()
        safetensors(directory / 'main.safetensors', names=['unknown.weight'])
        gguf(directory / 'secondary.gguf', metadata=[('general.architecture', 'wan')],
             names=['patch_embedding.weight', 'blocks.0.cross_attn.norm_k.weight'])
        package = self.root / 'secondary-gguf-package.safetensors'
        process = subprocess.run([str(PROGRAM), 'create', '--input', str(directory), '--output', str(package),
                                  '--assign-position', 'main.safetensors=main'], capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        info = self.run_tool(package)
        self.assertFalse(info['recognized'])
        self.assertEqual(info['modalities'], [])

    def test_gguf_classification_uses_metadata_and_tensor_signatures(self):
        path = self.root / "renamed.bin"
        gguf(path, metadata=[("config", '{"_class_name":"LTX2Pipeline"}')])
        result = self.run_tool(path)
        self.assertEqual(result["modalities"], ["audio", "video"])
        self.assertTrue(result["recognized"])
        self.assertEqual(result["diagnostic"], "")
        self.assertLess(result["metadata_bytes_read"], path.stat().st_size)

        gguf(path, metadata=[("general.architecture", "ltxv")])
        self.assertEqual(self.run_tool(path)["modalities"], ["video"])
        gguf(path, names=["model.diffusion_model.transformer_blocks.0.weight",
                          "audio_vae.decoder.weight", "text_embedding_projection.audio_aggregate_embed.weight"])
        self.assertEqual(self.run_tool(path)["modalities"], ["audio", "video"])
        gguf(path, metadata=[("general.architecture", "flux")], tensor_type=999)
        result = self.run_tool(path)
        self.assertEqual(result["modalities"], ["image"])
        self.assertTrue(result["recognized"])
        self.assertEqual(result["diagnostic"], "")
        gguf(path, metadata=[("general.architecture", "unknown"), ("general.name", "LTX Video")])
        self.assertFalse(self.run_tool(path)["recognized"])

        gguf(path, metadata=[("general.architecture", "ltxv"), ("society.modalities", ["image", "video"])])
        result = self.run_tool(path)
        self.assertEqual(result["modalities"], ["image", "video"])
        self.assertTrue(result["user_override"])

    def test_gguf_malformed_metadata_is_bounded_and_manual_override_still_works(self):
        path = self.root / "corrupt.gguf"
        gguf(path, metadata=[("general.architecture", "flux"), ("general.architecture", "ltxv")])
        result = self.run_tool(path)
        self.assertFalse(result["recognized"])
        self.assertIn("duplicate", result["diagnostic"].lower())
        self.assertEqual(self.run_tool(path, "classify", "video")["modalities"], ["video"])
        path.with_suffix(".gguf.model.json").unlink()
        gguf(path, metadata=[("society.modalities", ["image"] * 5)])
        self.assertIn("modalities", self.run_tool(path)["diagnostic"].lower())
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + struct.pack("<Q", 2**63))
        result = self.run_tool(path)
        self.assertFalse(result["recognized"])
        self.assertTrue(result["diagnostic"])

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

    def test_moss_image_uses_declared_class_or_matching_tensor_shapes(self):
        directory = self.root / 'pipeline'
        directory.mkdir()
        (directory / 'model_index.json').write_text('{"_class_name":"MOSSImagePipeline"}')
        declared = self.run_tool(directory)
        self.assertTrue(declared['recognized'])
        self.assertEqual(declared['modalities'], ['image'])
        self.assertEqual(declared['family'], 'MOSS Image')

        model = self.root / 'neutral.safetensors'
        for prefix in ('', 'model.diffusion_model.', 'diffusion_model.', 'transformer.'):
            shapes = {prefix + 'video_embedder.weight': [2, 3],
                      prefix + 'final_linear_video.weight': [3, 2],
                      prefix + 'layers.0.proj.qkv.weight': [6, 2]}
            with self.subTest(prefix=prefix):
                safetensors_shaped(model, shapes)
                found = self.run_tool(model)
                self.assertTrue(found['recognized'])
                self.assertEqual((found['architecture'], found['modalities']), ('moss-image', ['image']))
        safetensors_shaped(model, shapes, {'_class_name': 'LTX2Pipeline'})
        self.assertEqual(self.run_tool(model)['modalities'], ['audio', 'video'])
        shapes['transformer.layers.0.proj.qkv.weight'] = [5, 2]
        safetensors_shaped(model, shapes)
        self.assertFalse(self.run_tool(model)['recognized'])
        safetensors_shaped(model, {'video_embedder.weight': [2, 3],
                                  'final_linear_video.weight': [3, 2],
                                  'transformer.layers.0.proj.qkv.weight': [6, 2]})
        self.assertFalse(self.run_tool(model)['recognized'])

        pieces = [{'video_embedder.weight': [2, 3]}, {'final_linear_video.weight': [3, 2]},
                  {'layers.0.proj.qkv.weight': [6, 2]}]
        index, mapping = self.root / 'sharded.safetensors.index.json', {}
        for number, piece in enumerate(pieces):
            shard = self.root / f'part-{number}.safetensors'
            safetensors_shaped(shard, piece)
            mapping.update({name: shard.name for name in piece})
        index.write_text(json.dumps({'weight_map': mapping}))
        self.assertEqual(self.run_tool(index)['architecture'], 'moss-image')
        Path(str(index) + '.model.json').write_text('{"society":{"modality":"video"}}')
        overridden = self.run_tool(index)
        self.assertEqual(overridden['detected_modalities'], ['image'])
        self.assertEqual(overridden['modalities'], ['video'])
        declared_gguf = self.root / 'declared.gguf'
        gguf(declared_gguf, metadata=[('_class_name', 'MOSSImagePipeline')])
        self.assertEqual(self.run_tool(declared_gguf)['modalities'], ['image'])

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
