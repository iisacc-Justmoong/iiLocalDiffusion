#!/usr/bin/env python3
"""Dependency-free learned-token validation, registration, and prompt expansion contracts."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import presets
from text_embedding_options import TextEmbeddingSelection
import text_embeddings as runtime
from weight_files import resolve_weight_file


def flatten(values):
    if isinstance(values, list):
        return [number for value in values for number in flatten(value)]
    return [values]


class Tensor:
    """Only the storage/inspection operations used at the loader boundary."""

    def __init__(self, values, dtype="float32", *, shape=None):
        self.values, self.dtype = deepcopy(values), dtype
        if shape is None:
            shape = []
            current = values
            while isinstance(current, list):
                shape.append(len(current))
                if not current:
                    break
                current = current[0]
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def is_floating_point(self):
        return self.dtype.startswith(("float", "bfloat"))

    def clone(self):
        return Tensor(self.values, self.dtype, shape=self.shape)

    def unsqueeze(self, axis):
        if axis != 0:
            raise AssertionError("This fixture only models a vector batch axis.")
        return Tensor([self.values], self.dtype, shape=(1, *self.shape))

    def to(self, device=None, dtype=None, copy=False):
        dtype = self.dtype if dtype is None else dtype
        values = deepcopy(self.values)
        if dtype == "float16":
            # Only overflow is relevant here; representable test vectors use exact small integers.
            def cast(value):
                if isinstance(value, list):
                    return [cast(item) for item in value]
                return math.copysign(math.inf, value) if abs(value) > 65504 else value
            values = cast(values)
        return Tensor(values, dtype, shape=self.shape)

    def detach(self):
        return self

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return Tensor([self.values[index] for index in indices], self.dtype)
        return Tensor(self.values[indices], self.dtype)

    def __setitem__(self, index, value):
        self.values[index] = deepcopy(value.values)


class Torch:
    @staticmethod
    def isfinite(tensor):
        return SimpleNamespace(all=lambda: SimpleNamespace(
            item=lambda: all(math.isfinite(number) for number in flatten(tensor.values))))

    @staticmethod
    def equal(left, right):
        return left.shape == right.shape and left.values == right.values


class Tokenizer:
    def __init__(self, vocabulary=None, max_length=16):
        self.vocabulary = {"[PAD]": 0, "[UNK]": 1} if vocabulary is None else dict(vocabulary)
        self.model_max_length = max_length

    def get_vocab(self):
        return dict(self.vocabulary)

    def convert_tokens_to_ids(self, token):
        return self.vocabulary.get(token, self.vocabulary.get("[UNK]", 0))

    def add_tokens(self, tokens):
        for token in tokens:
            self.vocabulary[token] = max(self.vocabulary.values(), default=-1) + 1

    def __len__(self):
        return len(self.vocabulary)


class Encoder:
    def __init__(self, dimension=3, rows=8, dtype="float32"):
        self.device, self.dtype = SimpleNamespace(type="cpu"), dtype
        self.embedding = SimpleNamespace(weight=Tensor([[row] * dimension for row in range(rows)], dtype))
        self.resize_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, size, *, mean_resizing):
        self.resize_calls.append((size, mean_resizing))
        weight = self.embedding.weight
        rows = deepcopy(weight.values[:size])
        rows.extend([[0] * weight.shape[-1] for _ in range(size - len(rows))])
        self.embedding.weight = Tensor(rows, self.dtype)
        return self.embedding


class Fixture:
    def __init__(self, root, preset=presets.SD15_PRESET, *, dimension=3, second_dimension=5, rows=8):
        self.root, self.preset = root, preset
        self.pipeline = SimpleNamespace(text_encoder=Encoder(dimension, rows), tokenizer=Tokenizer())
        if preset.name != "sd15":
            self.pipeline.text_encoder_2 = Encoder(second_dimension, rows)
            self.pipeline.tokenizer_2 = Tokenizer()
        self.files = {}
        self.reads = []
        self.loads = []
        self.mutate_on_read = None
        self.args = argparse.Namespace(text_embedding_selections=[], cache_dir=root / "cache")

    def add(self, raw, *, name=None, token=None, encoder="auto", metadata=None):
        path = self.root / (name or f"embedding-{len(self.files)}.safetensors")
        path.write_bytes(f"file-identity-{len(self.files)}".encode())
        self.files[str(path.resolve())] = (raw, metadata)
        selection = TextEmbeddingSelection(resolve_weight_file(str(path), "--text-embedding"), token, encoder)
        self.args.text_embedding_selections.append(selection)
        return selection

    @contextmanager
    def dependencies(self):
        fixture = self

        @contextmanager
        def safe_open(path, *, framework, device):
            if (framework, device) != ("pt", "cpu"):
                raise AssertionError("Embedding files must be read into CPU PyTorch tensors.")
            fixture.reads.append(Path(path))
            raw, metadata = fixture.files[str(Path(path).resolve())]
            yield SimpleNamespace(keys=lambda: raw.keys(), get_tensor=lambda key: raw[key], metadata=lambda: metadata)
            if fixture.mutate_on_read is not None:
                fixture.mutate_on_read()

        class Loader:
            def load_textual_inversion(self, state, *, token, tokenizer, text_encoder):
                fixture.loads.append((token, tokenizer, text_encoder))
                vectors = state[token]
                names = [token] + [f"{token}_{index}" for index in range(1, vectors.shape[0])]
                tokenizer.add_tokens(names)
                text_encoder.resize_token_embeddings(len(tokenizer))
                for index, name in enumerate(names):
                    text_encoder.get_input_embeddings().weight[tokenizer.convert_tokens_to_ids(name)] = vectors[index]

        with patch.dict(sys.modules, {
            "safetensors": SimpleNamespace(safe_open=safe_open),
            "diffusers.loaders": SimpleNamespace(TextualInversionLoaderMixin=Loader),
        }):
            yield

    def apply(self):
        with self.dependencies():
            return runtime.apply_text_embeddings(self.pipeline, self.preset, self.args, Torch())


class TextEmbeddingRuntimeTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="text-embedding-runtime-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, *args, **kwargs):
        return Fixture(self.root, *args, **kwargs)

    def test_no_input_is_a_complete_noop_without_optional_runtime_packages(self):
        args = argparse.Namespace(prompt="unchanged")
        self.assertIsNone(runtime.apply_text_embeddings(None, presets.SD15_PRESET, args, None))
        self.assertIsNone(runtime.validate_text_embeddings(None, None, None))
        with runtime.text_embedding_prompt_context(None, presets.SD15_PRESET, args) as result:
            self.assertIs(result, args)

    def test_missing_runtime_and_hooked_or_accelerator_components_fail_before_reading(self):
        fixture = self.fixture()
        fixture.add({"<concept>": Tensor([1, 2, 3])})
        with self.assertRaisesRegex(ValueError, "PyTorch"):
            runtime.apply_text_embeddings(fixture.pipeline, fixture.preset, fixture.args, None)
        for state in ("cuda", "hooked", "missing"):
            fixture = self.fixture()
            fixture.add({"<concept>": Tensor([1, 2, 3])})
            if state == "cuda":
                fixture.pipeline.text_encoder.device.type = "cuda"
            elif state == "hooked":
                fixture.pipeline.text_encoder._hf_hook = object()
            else:
                fixture.pipeline.tokenizer = None
            with self.subTest(state=state), self.assertRaises(ValueError):
                fixture.apply()
            self.assertEqual(fixture.reads, [])

    def test_file_formats_reject_prompt_tensors_empty_files_and_unrelated_multiple_tensors(self):
        for raw in ({}, {"prompt_embeds": Tensor([[1, 2, 3]])},
                    {"first": Tensor([1, 2, 3]), "second": Tensor([1, 2, 3])}):
            fixture = self.fixture()
            fixture.add(raw)
            with self.subTest(keys=list(raw)), self.assertRaises(ValueError):
                fixture.apply()
            self.assertEqual(fixture.loads, [])

    def test_existing_token_ids_must_fit_the_original_encoder_before_any_file_is_read(self):
        for invalid_id in (-1, 8, 100):
            fixture = self.fixture(rows=8)
            fixture.pipeline.tokenizer.vocabulary["mismatched-token"] = invalid_id
            fixture.add({"<concept>": Tensor([1, 2, 3])})
            with self.subTest(token_id=invalid_id), self.assertRaisesRegex(ValueError, "token IDs exceed"):
                fixture.apply()
            self.assertEqual(fixture.reads, [])
            self.assertEqual(fixture.loads, [])
            self.assertEqual(fixture.pipeline.text_encoder.resize_calls, [])

    def test_vectors_must_have_floating_finite_nonempty_supported_shapes(self):
        invalid = (Tensor([1, 2, 3], "int64"), Tensor(1.0), Tensor([[[1, 2, 3]]]),
                   Tensor([], shape=(0, 3)), Tensor([[]], shape=(1, 0)),
                   Tensor([1, math.inf, 3]), Tensor([1, math.nan, 3]))
        for vectors in invalid:
            fixture = self.fixture()
            fixture.add({"<concept>": vectors})
            with self.subTest(shape=vectors.shape, dtype=vectors.dtype), self.assertRaisesRegex(ValueError, "finite vectors"):
                fixture.apply()
            self.assertEqual(fixture.loads, [])

    def test_dimension_ambiguity_requires_an_explicit_encoder(self):
        fixture = self.fixture(presets.SDXL_BASE_PRESET, second_dimension=3)
        fixture.add({"<concept>": Tensor([1, 2, 3])})
        with self.assertRaisesRegex(ValueError, "uniquely"):
            fixture.apply()
        fixture.args.text_embedding_selections[0] = TextEmbeddingSelection(
            fixture.args.text_embedding_selections[0].file, None, "text_encoder_2")
        activation = fixture.apply()
        self.assertEqual(activation.registrations[0]["component"], "text_encoder_2")
        self.assertNotIn("<concept>", fixture.pipeline.tokenizer.get_vocab())

    def test_named_encoder_family_and_explicit_target_constraints_are_enforced(self):
        cases = (
            (presets.SD15_PRESET, "text_encoder_2", "auto", 3),
            (presets.FLUX1_SCHNELL_PRESET, "clip_g", "auto", 5),
            (presets.SDXL_BASE_PRESET, "t5xxl", "auto", 5),
            (presets.SDXL_BASE_PRESET, "clip_l", "text_encoder_2", 3),
            (presets.SD15_PRESET, "<concept>", "text_encoder", 4),
        )
        for preset, key, encoder, dimension in cases:
            fixture = self.fixture(preset)
            fixture.add({key: Tensor([1] * dimension)}, encoder=encoder)
            with self.subTest(preset=preset.name, key=key, encoder=encoder), self.assertRaises(ValueError):
                fixture.apply()
            self.assertEqual(fixture.loads, [])

    def test_vector_count_and_target_dtype_overflow_fail_before_registration(self):
        for mode in ("context", "overflow"):
            fixture = self.fixture()
            if mode == "context":
                fixture.pipeline.tokenizer.model_max_length = 1
                fixture.add({"<concept>": Tensor([[1, 2, 3], [4, 5, 6]])})
            else:
                fixture.pipeline.text_encoder = Encoder(dtype="float16")
                fixture.add({"<concept>": Tensor([1e20, 2, 3])})
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "context length|overflows"):
                fixture.apply()
            self.assertEqual(fixture.loads, [])

    def test_all_files_are_preflighted_before_any_vocabulary_or_weight_mutation(self):
        fixture = self.fixture()
        fixture.add({"<good>": Tensor([1, 2, 3])}, name="first.safetensors")
        fixture.add({"<bad>": Tensor([1, 2])}, name="second.safetensors")
        vocabulary = fixture.pipeline.tokenizer.get_vocab()
        weights = fixture.pipeline.text_encoder.embedding.weight.clone()
        with self.assertRaises(ValueError):
            fixture.apply()
        self.assertEqual(len(fixture.reads), 2)
        self.assertEqual(fixture.loads, [])
        self.assertEqual(fixture.pipeline.tokenizer.get_vocab(), vocabulary)
        self.assertTrue(Torch.equal(fixture.pipeline.text_encoder.embedding.weight, weights))

    def test_existing_base_suffix_and_planned_cross_file_token_collisions_are_rejected(self):
        for existing in ("<concept>", "<concept>_1"):
            fixture = self.fixture()
            fixture.pipeline.tokenizer.add_tokens([existing])
            fixture.add({"<concept>": Tensor([1, 2, 3])})
            with self.subTest(existing=existing), self.assertRaisesRegex(ValueError, "collision"):
                fixture.apply()
            self.assertEqual(fixture.loads, [])
        fixture = self.fixture()
        fixture.add({"<concept>": Tensor([[1, 2, 3], [4, 5, 6]])}, name="first.safetensors")
        fixture.add({"<concept>_1": Tensor([1, 2, 3])}, name="second.safetensors")
        with self.assertRaisesRegex(ValueError, "collision"):
            fixture.apply()
        self.assertEqual(fixture.loads, [])

    def test_registration_records_ids_rows_hashes_and_preserves_padded_vocabulary_rows(self):
        fixture = self.fixture(rows=8)
        selection = fixture.add({"<concept>": Tensor([[1, 2, 3], [4, 5, 6]])})
        padded = fixture.pipeline.text_encoder.embedding.weight.values[7][:]
        activation = fixture.apply()
        entry = activation.registrations[0]
        self.assertEqual(entry["tokens"], ["<concept>", "<concept>_1"])
        self.assertEqual(entry["token_ids"], [2, 3])
        self.assertEqual(entry["shape"], [2, 3])
        self.assertEqual((entry["embedding_rows_before"], entry["embedding_rows_after"]), (8, 8))
        self.assertEqual(fixture.pipeline.text_encoder.resize_calls, [(8, False)])
        self.assertEqual(fixture.pipeline.text_encoder.embedding.weight.values[7], padded)
        self.assertEqual(fixture.pipeline.text_encoder.embedding.weight.values[2:4], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(activation.metadata[0]["file"]["sha256"], selection.file.sha256)
        self.assertEqual(activation.metadata[0]["file"]["size_bytes"], selection.file.size_bytes)
        self.assertNotIn("vectors", activation.metadata[0]["registrations"][0])
        json.dumps(activation.metadata)

    def test_registration_grows_when_token_ids_exceed_the_existing_embedding_rows(self):
        fixture = self.fixture(rows=2)
        fixture.add({"<concept>": Tensor([[1, 2, 3], [4, 5, 6]])})
        activation = fixture.apply()
        self.assertEqual(fixture.pipeline.text_encoder.resize_calls, [(4, False)])
        self.assertEqual(activation.registrations[0]["embedding_rows_after"], 4)

    def test_named_dual_encoder_file_registers_matching_tokens_on_both_encoders(self):
        fixture = self.fixture(presets.SDXL_BASE_PRESET)
        fixture.add({"clip_l": Tensor([[1, 2, 3], [4, 5, 6]]),
                     "clip_g": Tensor([1, 2, 3, 4, 5])}, token="<shared>")
        activation = fixture.apply()
        self.assertEqual([entry["component"] for entry in activation.registrations],
                         ["text_encoder", "text_encoder_2"])
        self.assertEqual(activation.registrations[0]["vector_count"], 2)
        self.assertEqual(activation.registrations[1]["vector_count"], 1)
        self.assertIn("<shared>_1", fixture.pipeline.tokenizer.get_vocab())
        self.assertNotIn("<shared>_1", fixture.pipeline.tokenizer_2.get_vocab())

    def test_token_precedence_uses_override_metadata_key_then_filename_for_embedding_parameters(self):
        cases = (("<stored>", {"token": "<metadata>", "name": "<name>"}, "<override>", "<override>"),
                 ("<stored>", {"token": "<metadata>", "name": "<name>"}, None, "<metadata>"),
                 ("<stored>", {"name": "<name>"}, None, "<name>"),
                 ("<stored>", {}, None, "<stored>"), ("emb_params", {}, None, "concept"))
        for key, metadata, token, expected in cases:
            fixture = self.fixture()
            fixture.add({key: Tensor([1, 2, 3])}, name="concept.safetensors", token=token, metadata=metadata)
            with self.subTest(key=key, metadata=metadata, token=token):
                activation = fixture.apply()
                self.assertEqual(activation.registrations[0]["token"], expected)

    def test_file_tokens_follow_the_same_control_character_contract_as_cli_overrides(self):
        for token in ("bad token", "bad\x00token", "bad\u200btoken", "bad\u202etoken", "bad\ud800token"):
            fixture = self.fixture()
            fixture.add({"emb_params": Tensor([1, 2, 3])}, metadata={"token": token})
            with self.subTest(token=repr(token)), self.assertRaisesRegex(ValueError, "token"):
                fixture.apply()
            self.assertEqual(fixture.loads, [])

    def test_identity_changes_before_or_during_reading_are_rejected_before_registration(self):
        for timing in ("before", "during"):
            fixture = self.fixture()
            selection = fixture.add({"<concept>": Tensor([1, 2, 3])})
            mutate = lambda: Path(selection.file.path).write_bytes(b"changed after request resolution")
            if timing == "before":
                mutate()
            else:
                fixture.mutate_on_read = mutate
            with self.subTest(timing=timing), self.assertRaisesRegex(RuntimeError, "changed"):
                fixture.apply()
            self.assertEqual(fixture.loads, [])

    def test_singular_safetensor_suffix_uses_a_checked_temporary_loader_alias(self):
        fixture = self.fixture()
        fixture.add({"<concept>": Tensor([1, 2, 3])}, name="concept.safetensor")
        activation = fixture.apply()
        self.assertEqual(fixture.reads[0].suffix, ".safetensors")
        self.assertFalse(fixture.reads[0].exists())
        self.assertTrue(activation.metadata[0]["file"]["path"].endswith(".safetensor"))

    def test_stage_conversion_detects_changed_missing_duplicate_ids_and_modified_rows(self):
        for mutation in ("changed-id", "missing-token", "duplicate-id", "row", "nan"):
            fixture = self.fixture()
            fixture.add({"<concept>": Tensor([[1, 2, 3], [4, 5, 6]])})
            activation = fixture.apply()
            if mutation == "changed-id":
                fixture.pipeline.tokenizer.vocabulary["<concept>"] = 6
            elif mutation == "missing-token":
                del fixture.pipeline.tokenizer.vocabulary["<concept>"]
            elif mutation == "duplicate-id":
                fixture.pipeline.tokenizer.vocabulary["<concept>_1"] = 2
            else:
                fixture.pipeline.text_encoder.embedding.weight.values[2][0] = math.nan if mutation == "nan" else 99
            with self.subTest(mutation=mutation), self.assertRaisesRegex(RuntimeError, "changed"):
                runtime.validate_text_embeddings(fixture.pipeline, activation, Torch())

    def test_expected_vectors_can_follow_an_execution_dtype_change(self):
        fixture = self.fixture()
        fixture.add({"<concept>": Tensor([1, 2, 3])})
        activation = fixture.apply()
        fixture.pipeline.text_encoder.embedding.weight = fixture.pipeline.text_encoder.embedding.weight.to(dtype="float16")
        runtime.validate_text_embeddings(fixture.pipeline, activation, Torch())

    def test_prompt_expansion_uses_each_encoder_and_its_own_prompt_fallback(self):
        registrations = [
            {"component": "text_encoder", "token": "<style>", "tokens": ["<style>", "<style>_1"]},
            {"component": "text_encoder_2", "token": "<style>", "tokens": ["<style>", "<style>_1", "<style>_2"]},
        ]
        args = argparse.Namespace(prompt="portrait <style>", negative_prompt="avoid <style>",
                                  prompt_2=None, negative_prompt_2=None,
                                  text_embedding_activation=runtime.TextEmbeddingActivation([], registrations))
        pipeline = SimpleNamespace()
        for preset in (presets.SDXL_BASE_PRESET, presets.FLUX1_SCHNELL_PRESET):
            with self.subTest(preset=preset.name), runtime.text_embedding_prompt_context(pipeline, preset, args) as result:
                self.assertEqual(result.prompt, "portrait <style> <style>_1")
                self.assertEqual(result.negative_prompt, "avoid <style> <style>_1")
                self.assertEqual(result.prompt_2, "portrait <style> <style>_1 <style>_2")
                self.assertEqual(result.negative_prompt_2, "avoid <style> <style>_1 <style>_2")
                self.assertIsNot(result, args)
            self.assertIsNone(args.prompt_2)
            self.assertEqual(args.prompt, "portrait <style>")

    def test_explicit_secondary_prompts_and_suffix_or_prefix_tokens_are_not_rewritten_twice(self):
        registrations = [
            {"component": "text_encoder", "token": "style", "tokens": ["style", "style_1"]},
            {"component": "text_encoder", "token": "style_long", "tokens": ["style_long", "style_long_1"]},
            {"component": "text_encoder_2", "token": "<other>", "tokens": ["<other>", "<other>_1"]},
        ]
        args = argparse.Namespace(prompt="style style_1 style_long style_long_1", negative_prompt="",
                                  prompt_2="<other>", negative_prompt_2="skip <other>_1",
                                  text_embedding_activation=runtime.TextEmbeddingActivation([], registrations))
        original = lambda prompt, tokenizer: "unexpected double expansion"
        pipeline = SimpleNamespace(maybe_convert_prompt=original)
        with runtime.text_embedding_prompt_context(pipeline, presets.SDXL_BASE_PRESET, args) as result:
            self.assertEqual(result.prompt, "style style_1 style_1 style_long style_long_1 style_long_1")
            self.assertEqual(result.prompt_2, "<other> <other>_1")
            self.assertEqual(result.negative_prompt_2, "skip <other>_1")
            self.assertEqual(pipeline.maybe_convert_prompt(result.prompt, None), result.prompt)
        self.assertIs(pipeline.maybe_convert_prompt, original)

    def test_empty_secondary_prompts_fall_back_before_their_encoder_token_expansion(self):
        activation = runtime.TextEmbeddingActivation([], [
            {"component": "text_encoder_2", "token": "<token>", "tokens": ["<token>", "<token>_1"]}])
        args = argparse.Namespace(prompt="object <token>", negative_prompt="<token>",
                                  prompt_2="", negative_prompt_2="", text_embedding_activation=activation)
        for preset in (presets.SDXL_BASE_PRESET, presets.FLUX1_SCHNELL_PRESET):
            with self.subTest(preset=preset.name), runtime.text_embedding_prompt_context(
                    SimpleNamespace(), preset, args) as result:
                self.assertEqual(result.prompt, "object <token>")
                self.assertEqual(result.negative_prompt, "<token>")
                self.assertEqual(result.prompt_2, "object <token> <token>_1")
                self.assertEqual(result.negative_prompt_2, "<token> <token>_1")
            self.assertEqual((args.prompt_2, args.negative_prompt_2), ("", ""))

    def test_sd15_context_leaves_secondary_fields_unset_and_restores_after_errors(self):
        activation = runtime.TextEmbeddingActivation([], [
            {"component": "text_encoder", "token": "<concept>", "tokens": ["<concept>", "<concept>_1"]}])
        args = argparse.Namespace(prompt="<concept>", negative_prompt="", prompt_2=None, negative_prompt_2=None,
                                  text_embedding_activation=activation)
        for existing in (False, True):
            pipeline = SimpleNamespace()
            original = lambda prompt, tokenizer: prompt + " original"
            if existing:
                pipeline.maybe_convert_prompt = original
            with self.subTest(existing=existing), self.assertRaisesRegex(RuntimeError, "generation failure"):
                with runtime.text_embedding_prompt_context(pipeline, presets.SD15_PRESET, args) as result:
                    self.assertIsNone(result.prompt_2)
                    self.assertIsNone(result.negative_prompt_2)
                    raise RuntimeError("generation failure")
            self.assertEqual(hasattr(pipeline, "maybe_convert_prompt"), existing)
            if existing:
                self.assertIs(pipeline.maybe_convert_prompt, original)


if __name__ == "__main__":
    unittest.main()
