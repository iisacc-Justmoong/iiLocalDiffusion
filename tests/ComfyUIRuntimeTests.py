#!/usr/bin/env python3
"""Local HTTP contract tests; no ComfyUI installation or model weights required."""

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import comfyui_runtime as runtime


GRAPH = {"1": {"class_type": "Loader", "inputs": {"model": "local.safetensors"}},
         "2": {"class_type": "Output", "inputs": {"source": ["1", 0]}}}
OBJECTS = {"Loader": {"input": {"required": {"model": [["local.safetensors"]]}}, "output": ["IMAGE"]},
           "Output": {"input": {"required": {"source": ["IMAGE"]}}, "output": [], "output_node": True}}


class ComfyUIRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="comfy-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_loopback_urls_only(self):
        for value in ("http://127.0.0.1:8188", "http://[::1]:8188/", "http://localhost:8188"):
            self.assertTrue(runtime.local_url(value).startswith("http:"))
        for value in ("https://example.org", "file:///etc/passwd", "http://user@localhost:8188",
                      "http://127.0.0.1:8188/foo", "http://127.0.0.1:8188?token=secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.local_url(value)

    def test_overrides_copy_graph_and_reject_typos(self):
        graph = runtime.prepare_workflow(GRAPH, {"1": {"model": "other.safetensors"}})
        self.assertEqual(GRAPH["1"]["inputs"]["model"], "local.safetensors")
        self.assertEqual(graph["1"]["inputs"]["model"], "other.safetensors")
        for value in ({"3": {}}, {"1": {"typo": 0}}):
            with self.assertRaises(ValueError):
                runtime.prepare_workflow(GRAPH, value)

    def test_editor_format_and_empty_graph_are_rejected(self):
        for graph in ({}, {"nodes": []}):
            with self.assertRaises(ValueError):
                runtime.prepare_workflow(graph, {})

    def test_unavailable_node_model_and_api_nodes_fail_preflight(self):
        runtime.validate_workflow(GRAPH, OBJECTS)
        missing_model = runtime.prepare_workflow(GRAPH, {"1": {"model": "missing.gguf"}})
        for graph, objects in ((missing_model, OBJECTS), (GRAPH, {"Output": OBJECTS["Output"]}),
                               (GRAPH, {**OBJECTS, "Loader": {**OBJECTS["Loader"], "api_node": True}})):
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                runtime.validate_workflow(graph, objects)

    def test_cycles_and_invalid_links_are_rejected(self):
        for link in (["missing", 0], ["1", 12], ["1", -1], ["1", True]):
            with self.subTest(link=link), self.assertRaises(ValueError):
                runtime.validate_workflow(runtime.prepare_workflow(GRAPH, {"2": {"source": link}}), OBJECTS)
        cyclic = {"1": {"class_type": "Output", "inputs": {"source": ["1", 0]}}}
        with self.assertRaisesRegex(ValueError, "cycle"):
            runtime.validate_workflow(cyclic, {"Output": {**OBJECTS["Output"], "output": ["IMAGE"]}})

    def test_missing_required_input_and_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "required"):
            runtime.validate_workflow({"1": {"class_type": "Loader", "inputs": {}}}, OBJECTS)
        with self.assertRaisesRegex(ValueError, "output node"):
            runtime.validate_workflow({"1": GRAPH["1"]}, OBJECTS)

    def test_v3_choices_and_optional_zero_reference_image_inputs(self):
        objects = json.loads(json.dumps(OBJECTS))
        required = objects["Loader"]["input"]["required"]
        required["model"] = ["COMBO", {"options": ["local.safetensors"]}]
        required["images"] = ["COMFY_AUTOGROW_V3", {"template": {"min": 0}}]
        runtime.validate_workflow(GRAPH, objects)
        with self.assertRaisesRegex(ValueError, "unsupported choice"):
            runtime.validate_workflow(runtime.prepare_workflow(GRAPH, {"1": {"model": "absent"}}), objects)
        required["images"][1]["template"]["min"] = 1
        with self.assertRaisesRegex(ValueError, "required"):
            runtime.validate_workflow(GRAPH, objects)

    def test_incompatible_link_types_are_rejected(self):
        objects = {**OBJECTS, "Loader": {**OBJECTS["Loader"], "output": ["MODEL"]}}
        with self.assertRaisesRegex(ValueError, "Incompatible workflow link"):
            runtime.validate_workflow(GRAPH, objects)
        for source in ("*", "IMAGE,MASK"):
            runtime.validate_workflow(GRAPH, {**OBJECTS, "Loader": {**OBJECTS["Loader"], "output": [source]}})

    def test_outputs_include_video_audio_mesh_and_are_deduplicated(self):
        values = [{"filename": name} for name in ("frame.png", "clip.mp4", "sound.flac", "mesh.glb")]
        outputs = {"1": {"images": values, "again": values}}
        self.assertEqual(len(runtime.artifact_descriptors(outputs)), 4)
        for descriptor in ({"filename": "../secret"}, {"filename": "ok", "subfolder": "../private"},
                           {"filename": "ok", "type": "input"}, {"filename": "ok", "subfolder": "/private"}):
            with self.assertRaises(ValueError):
                runtime.artifact_descriptors({"images": [descriptor]})

    @contextmanager
    def server(self, *, failed=False, payload=b"generated fixture", redirect=False, reject=False):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                calls.append(("GET", self.path))
                if redirect:
                    self.send_response(302)
                    self.send_header("Location", "https://example.org")
                    self.end_headers()
                    return
                if self.path == "/object_info":
                    data = json.dumps(OBJECTS).encode()
                elif self.path.startswith("/history/"):
                    data = json.dumps({"job": {"status": {"completed": not failed,
                        "status_str": "error" if failed else "success", "messages": []},
                        "outputs": {"2": {"images": [{"filename": "result.png", "type": "output"}]}}}}).encode()
                else:
                    data = payload
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(("POST", self.path, body))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "invalid graph"} if reject else {"prompt_id": "job"}).encode())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", calls
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def arguments(self, url, *flags):
        workflow = self.root / "workflow.json"
        workflow.write_text(json.dumps(GRAPH))
        return runtime.build_parser().parse_args(["--workflow", str(workflow), "--comfyui-url", url,
                                                  "--output-dir", str(self.root / "out"), *flags])

    def test_actual_http_submit_wait_download_and_complete_manifest(self):
        with self.server() as (url, calls):
            result = runtime.run(self.arguments(url))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(Path(result["artifacts"][0]["path"]).read_bytes(), b"generated fixture")
        self.assertEqual(len(result["artifacts"][0]["sha256"]), 64)
        self.assertTrue((self.root / "out" / "generation.json").exists())
        self.assertEqual(len([call for call in calls if call[0] == "POST"]), 1)

    def test_preflight_never_queues(self):
        with self.server() as (url, calls):
            result = runtime.run(self.arguments(url, "--validate-only"))
        self.assertFalse(result["generation_verified"])
        self.assertFalse(any(call[0] == "POST" for call in calls))

    def test_print_config_is_offline(self):
        with patch.object(runtime, "Client", side_effect=AssertionError("network")):
            result = runtime.run(self.arguments("http://127.0.0.1:1", "--print-config"))
        self.assertEqual(result["workflow"], GRAPH)

    def test_failed_job_has_submission_but_no_complete_manifest(self):
        with self.server(failed=True) as (url, calls):
            with self.assertRaisesRegex(RuntimeError, "execution failed"):
                runtime.run(self.arguments(url))
        self.assertTrue((self.root / "out" / "submission.json").exists())
        self.assertFalse((self.root / "out" / "generation.json").exists())

    def test_empty_or_oversized_artifact_never_publishes(self):
        for payload in (b"", b"oversized"):
            with self.subTest(payload=payload), self.server(payload=payload) as (url, _):
                with self.assertRaises(RuntimeError):
                    runtime.Client(url).download({"filename": "result.png", "type": "output"}, self.root / "fail.png", 1)
                self.assertFalse((self.root / "fail.png").exists())

    def test_redirect_is_blocked(self):
        with self.server(redirect=True) as (url, calls):
            with self.assertRaisesRegex(ValueError, "redirect"):
                runtime.Client(url).json("/object_info")
        self.assertEqual(len(calls), 1)

    def test_existing_output_directory_does_not_queue(self):
        (self.root / "out").mkdir()
        (self.root / "out" / "user-file").write_text("preserve")
        with self.server() as (url, calls):
            with self.assertRaises(FileExistsError):
                runtime.run(self.arguments(url))
        self.assertFalse(any(call[0] == "POST" for call in calls))

    def test_timeout_records_id_and_does_not_retry_submit(self):
        client = type("Pending", (), {"json": lambda self, route: {}})()
        with self.assertRaisesRegex(TimeoutError, "job-id"):
            runtime.wait_for_history(client, "job-id", 0.001, 0.001)


if __name__ == "__main__":
    unittest.main()
