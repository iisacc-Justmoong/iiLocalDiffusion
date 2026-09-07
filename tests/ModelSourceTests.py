#!/usr/bin/env python3
"""Explicit location types, configuration aliases and credential-free metadata."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from generation_config import ConfigurationArgumentParser
from model_sources import add_model_arguments, resolve_model_input


class ModelSourceTests(unittest.TestCase):
    def parser(self):
        parser = ConfigurationArgumentParser(allow_abbrev=False)
        parser.add_argument("--config", type=Path)
        add_model_arguments(parser)
        return parser

    def source(self, *tokens):
        return resolve_model_input(self.parser().parse_args(list(tokens)))

    def test_three_distinct_locations(self):
        local = self.source("--model-path", str(ROOT / "tests/fixtures/sd-v1-manifest"))
        api = self.source("--model-api", "https://inference.example/model")
        cloud = self.source("--model-cloud", "owner/model", "--model-provider", "fal-ai")
        self.assertEqual([local.kind, api.kind, cloud.kind], ["local", "api", "cloud"])
        self.assertEqual(cloud.provider, "fal-ai")
        self.assertEqual(api.location, "https://inference.example/model")

    def test_model_remains_a_local_alias(self):
        path = str(ROOT / "tests/fixtures/sd-v1-manifest")
        self.assertEqual(self.source("--model", path), self.source("--model-path", path))
        with self.assertRaisesRegex(ValueError, "local"):
            self.source("--model", "owner/model")

    def test_location_inputs_are_mutually_exclusive(self):
        for fields in (("model_path", "model_api"), ("model_api", "model_cloud"),
                       ("model", "model_cloud")):
            with self.subTest(fields=fields), self.assertRaises(SystemExit):
                self.parser().parse_values({name: "example" for name in fields})
        with self.assertRaisesRegex(ValueError, "--model-path"):
            self.source()

    def test_json_local_alias_is_relative_to_configuration(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            root = Path(temporary)
            (root / "model").mkdir()
            path = root / "request.json"
            path.write_text(json.dumps({"model_path": "./model"}))
            source = resolve_model_input(self.parser().parse_args(["--config", str(path)]))
            self.assertEqual(source.location, str(root / "model"))

    def test_remote_identifiers_are_not_local_paths(self):
        for field, value in (("model_api", "https://host.example/infer"), ("model_cloud", "owner/model")):
            values = {field: value}
            if field == "model_cloud":
                values["model_provider"] = "fal-ai"
            self.assertEqual(resolve_model_input(self.parser().parse_values(values)).location, value)

    def test_invalid_or_secret_bearing_urls_are_rejected(self):
        for address in ("ftp://host/model", "http://host/model", "https://u:p@host/model",
                        "https://host/model?token=secret", "https://host/model#secret", "https://"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                self.source("--model-api", address)
        self.assertEqual(self.source("--model-api", "http://127.0.0.1:8080/infer").kind, "api")

    def test_cloud_requires_an_explicit_provider_and_model_id(self):
        for tokens in (("--model-cloud", "owner/model"),
                       ("--model-cloud", "https://host/model", "--model-provider", "fal-ai"),
                       ("--model-cloud", "../model", "--model-provider", "fal-ai")):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.source(*tokens)

    def test_remote_options_cannot_be_silently_used_for_local_input(self):
        for option, value in (("--model-provider", "fal-ai"), ("--model-token-env", "TOKEN")):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.source("--model-path", str(ROOT), option, value)
        with self.assertRaises(ValueError):
            self.source("--model-api", "https://host/model", "--model-token-env", "raw-secret-value")

    def test_local_alias_is_accepted_by_all_weight_loading_parsers(self):
        import generate
        import generate_any
        import local_image
        import video_options
        path = str(ROOT / "tests/fixtures/sd-v1-manifest")
        for module in (generate, generate_any, local_image, video_options):
            with self.subTest(module=module.__name__):
                self.assertEqual(str(module.build_parser().parse_args(["--model-path", path]).model), path)

    def test_mixed_config_and_cli_sources_fail_before_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            path = Path(temporary) / "request.json"
            path.write_text(json.dumps({"model_path": str(ROOT)}))
            with self.assertRaises(SystemExit):
                self.parser().parse_args(["--config", str(path), "--model-api", "https://host/infer"])


if __name__ == "__main__":
    unittest.main()
