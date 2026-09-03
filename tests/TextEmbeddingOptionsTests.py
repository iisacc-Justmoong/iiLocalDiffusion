#!/usr/bin/env python3
"""Dependency-free Textual Inversion selection, configuration, and identity contracts."""

import argparse
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
from generation_config import ConfigurationArgumentParser
from text_embedding_options import (
    TextEmbeddingSelection, add_text_embedding_options, resolve_text_embedding_options,
)
import presets


def parser():
    result = ConfigurationArgumentParser(allow_abbrev=False)
    result.add_argument("--config", type=Path, default=None)
    result.add_argument("--embeddings", default=None)
    add_text_embedding_options(result)
    return result


def resolved(*arguments, preset=presets.SD15_PRESET):
    args = parser().parse_args(list(arguments))
    resolve_text_embedding_options(preset, args)
    return args


class TextEmbeddingOptionsTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="text-embedding-options-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def weight(self, name="concept.safetensors", contents=b"identity fixture; tensor validation happens at load"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def test_unselected_options_keep_neutral_json_values_for_every_family(self):
        for preset in presets.PRESETS.values():
            with self.subTest(preset=preset.name):
                args = resolved(preset=preset)
                self.assertIsNone(args.text_embedding)
                self.assertIsNone(args.text_embedding_token)
                self.assertIsNone(args.text_embedding_encoder)
                self.assertEqual(args.text_embedding_selections, [])

    def test_single_selection_records_file_identity_and_defers_default_token(self):
        path = self.weight()
        args = resolved("--text-embedding", str(path))
        selection = args.text_embedding_selections[0]
        self.assertIsInstance(selection, TextEmbeddingSelection)
        self.assertEqual(selection.file.path, str(path.absolute()))
        self.assertEqual(selection.file.resolved_file, str(path.resolve()))
        self.assertEqual(selection.file.size_bytes, path.stat().st_size)
        self.assertEqual(selection.file.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIsNone(selection.token)
        self.assertEqual(selection.encoder, "auto")
        self.assertEqual(args.text_embedding, [path.absolute()])
        self.assertEqual(args.text_embedding_encoder, ["auto"])
        self.assertIsNone(args.text_embedding_token)

    def test_textual_inversion_alias_has_identical_file_and_token_selection(self):
        path = self.weight()
        first = resolved("--text-embedding", str(path), "--text-embedding-token", "<concept>")
        alias = resolved("--textual-inversion", str(path), "--text-embedding-token", "<concept>")
        self.assertEqual(first.text_embedding_selections, alias.text_embedding_selections)
        self.assertEqual(alias.text_embedding_token, ["<concept>"])

    def test_multiple_files_keep_their_token_and_encoder_mapping_in_order(self):
        first, second = self.weight("first.safetensors"), self.weight("second.safetensor", b"second identity")
        args = resolved("--text-embedding", str(first), str(second),
                        "--text-embedding-token", "<first>", "<second>",
                        "--text-embedding-encoder", "text_encoder_2", "text_encoder",
                        preset=presets.SDXL_BASE_PRESET)
        self.assertEqual([(item.file.path, item.token, item.encoder) for item in args.text_embedding_selections],
                         [(str(first), "<first>", "text_encoder_2"), (str(second), "<second>", "text_encoder")])
        self.assertEqual(args.text_embedding_token, ["<first>", "<second>"])
        self.assertEqual(args.text_embedding_encoder, ["text_encoder_2", "text_encoder"])

    def test_each_file_receives_an_auto_encoder_when_encoders_are_omitted(self):
        first, second = self.weight("first.safetensors"), self.weight("second.safetensors")
        args = resolved("--text-embedding", str(first), str(second))
        self.assertEqual(args.text_embedding_encoder, ["auto", "auto"])
        self.assertTrue(all(item.encoder == "auto" and item.token is None
                            for item in args.text_embedding_selections))

    def test_dependent_options_require_a_selected_embedding(self):
        for arguments in (("--text-embedding-token", "<token>"),
                          ("--text-embedding-encoder", "auto")):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "--text-embedding"):
                resolved(*arguments)

    def test_file_token_and_encoder_counts_must_match_without_broadcasting(self):
        first, second = self.weight("first.safetensors"), self.weight("second.safetensors")
        cases = (
            ("--text-embedding-token", "<one>"),
            ("--text-embedding-token", "<one>", "<two>", "<three>"),
            ("--text-embedding-encoder", "auto"),
            ("--text-embedding-encoder", "auto", "text_encoder", "auto"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "count"):
                resolved("--text-embedding", str(first), str(second), *arguments)

    def test_precomputed_prompt_embeddings_cannot_be_combined_with_trainable_tokens(self):
        path = self.weight()
        with self.assertRaisesRegex(ValueError, "--embeddings"):
            resolved("--text-embedding", str(path), "--embeddings", str(path))

    def test_single_encoder_family_rejects_the_second_encoder_before_loading(self):
        path = self.weight()
        with self.assertRaisesRegex(ValueError, "text_encoder_2"):
            resolved("--text-embedding", str(path), "--text-embedding-encoder", "text_encoder_2")
        for preset in (presets.SDXL_BASE_PRESET, presets.FLUX1_SCHNELL_PRESET):
            with self.subTest(preset=preset.name):
                args = resolved("--text-embedding", str(path), "--text-embedding-encoder", "text_encoder_2",
                                preset=preset)
                self.assertEqual(args.text_embedding_selections[0].encoder, "text_encoder_2")

    def test_token_names_reject_empty_whitespace_control_and_invisible_format_characters(self):
        path = self.weight()
        for token in ("", " ", "two words", "<tab\t>", "<newline\n>", "\x00token", "token\x7f",
                      "<nonbreaking\u00a0space>", "<zero\u200bwidth>", "<bidi\u202econtrol>"):
            with self.subTest(token=repr(token)), self.assertRaisesRegex(ValueError, "token"):
                resolved("--text-embedding", str(path), "--text-embedding-token", token)

    def test_unicode_and_punctuated_token_names_remain_literal(self):
        path = self.weight()
        for token in ("<concept:1>", "한국어_질감", "style-with-dashes", "<têxt>" ):
            with self.subTest(token=token):
                args = resolved("--text-embedding", str(path), "--text-embedding-token", token)
                self.assertEqual(args.text_embedding_selections[0].token, token)

    def test_only_nonempty_local_safetensors_files_are_selected(self):
        directory = self.root / "directory.safetensors"
        directory.mkdir()
        invalid = [self.root / "missing.safetensors", directory, self.weight("empty.safetensors", b"")]
        invalid.extend(self.weight("unsafe" + suffix) for suffix in (".bin", ".pt", ".ckpt", ".txt"))
        for path in invalid:
            with self.subTest(path=path), self.assertRaises(ValueError):
                resolved("--text-embedding", str(path))
        with self.assertRaises(ValueError):
            resolved("--text-embedding", "organization/remote-repository")

    def test_symlink_source_and_resolved_identity_are_recorded_separately(self):
        actual = self.weight("stored.safetensors")
        alias = self.root / "selected.safetensors"
        alias.symlink_to(actual)
        args = resolved("--text-embedding", str(alias))
        self.assertEqual(args.text_embedding, [alias.absolute()])
        self.assertEqual(args.text_embedding_selections[0].file.path, str(alias.absolute()))
        self.assertEqual(args.text_embedding_selections[0].file.resolved_file, str(actual.resolve()))

    def test_json_configuration_resolves_each_relative_file_against_the_config_directory(self):
        first = self.weight("weights/first.safetensors")
        second = self.weight("weights/second.safetensors")
        config = self.root / "request.json"
        config.write_text(json.dumps({"text_embedding": ["weights/first.safetensors", "weights/second.safetensors"],
                                     "text_embedding_token": ["<first>", "<second>"]}))
        args = resolved("--config", str(config))
        self.assertEqual(args.text_embedding, [first.resolve(), second.resolve()])
        self.assertEqual([item.token for item in args.text_embedding_selections], ["<first>", "<second>"])

    def test_cli_values_override_the_corresponding_json_arrays(self):
        original, override = self.weight("original.safetensors"), self.weight("override.safetensors")
        config = self.root / "request.json"
        config.write_text(json.dumps({"text_embedding": [original.name], "text_embedding_token": ["<original>"],
                                     "text_embedding_encoder": ["auto"]}))
        args = resolved("--config", str(config), "--text-embedding", str(override),
                        "--text-embedding-token", "<override>", "--text-embedding-encoder", "text_encoder")
        self.assertEqual(args.text_embedding_selections[0].file.path, str(override))
        self.assertEqual(args.text_embedding_selections[0].token, "<override>")
        self.assertEqual(args.text_embedding_selections[0].encoder, "text_encoder")

    def test_selected_and_unselected_configuration_arrays_roundtrip(self):
        path = self.weight()
        for arguments in ((), ("--text-embedding", str(path)),
                          ("--text-embedding", str(path), "--text-embedding-token", "<concept>")):
            with self.subTest(arguments=arguments):
                args = resolved(*arguments)
                values = {name: getattr(args, name) for name in args._argument_names}
                serialized = json.loads(json.dumps(values, default=str))
                replay = parser().parse_values(serialized, base_directory=self.root)
                resolve_text_embedding_options(presets.SD15_PRESET, replay)
                self.assertEqual(replay.text_embedding_selections, args.text_embedding_selections)
                self.assertEqual(replay.text_embedding_encoder, args.text_embedding_encoder)
                self.assertEqual(replay.text_embedding_token, args.text_embedding_token)

    def test_configuration_rejects_scalar_paths_empty_arrays_and_wrong_member_types(self):
        path = self.weight()
        cases = ({"text_embedding": str(path)}, {"text_embedding": []}, {"text_embedding": [7]},
                 {"text_embedding": [str(path)], "text_embedding_token": [None]},
                 {"text_embedding": [str(path)], "text_embedding_encoder": [True]},
                 {"text_embedding": [str(path)], "text_embedding_encoder": ["unknown"]})
        for values in cases:
            with self.subTest(values=values), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_values(values)

    def test_cli_requires_values_and_known_encoder_names(self):
        for arguments in (("--text-embedding",), ("--text-embedding-token",),
                          ("--text-embedding-encoder", "text_encoder_3")):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                resolved(*arguments)

    def test_resolver_keeps_backward_compatible_unselected_namespaces(self):
        args = argparse.Namespace()
        resolve_text_embedding_options(presets.SD15_PRESET, args)
        self.assertEqual(args.text_embedding_selections, [])
        self.assertIsNone(args.text_embedding)

    def test_selected_resolution_is_idempotent(self):
        path = self.weight()
        args = resolved("--text-embedding", str(path), "--text-embedding-token", "<concept>")
        previous = dict(vars(args))
        resolve_text_embedding_options(presets.SD15_PRESET, args)
        self.assertEqual(vars(args), previous)


if __name__ == "__main__":
    unittest.main()
