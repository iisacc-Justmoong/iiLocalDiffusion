"""Run explicit local ComfyUI API graphs without coupling to model architectures."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from generation_config import json_object
from generation_output import publish_file


ROOT = Path(__file__).resolve().parents[2]
MAX_JSON = 32 * 1024 * 1024


def read_object(value: str) -> dict:
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if path.stat().st_size > MAX_JSON:
            raise ValueError("JSON input exceeds 32 MiB.")
        value = path.read_text(encoding="utf-8")
    return json_object(value)


def local_url(value: str) -> str:
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in ("http", "https") or parts.username or parts.password:
        raise ValueError("ComfyUI URL must be a local HTTP(S) server without credentials.")
    try:
        is_local = parts.hostname == "localhost" or ipaddress.ip_address(parts.hostname or "").is_loopback
        port = parts.port
    except ValueError as error:
        raise ValueError("ComfyUI requires localhost or a loopback IP address.") from error
    if not is_local or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ValueError("ComfyUI requires a loopback URL without a path, query, or fragment.")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "", "", ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("The local ComfyUI server must not redirect requests.")


class Client:
    def __init__(self, url: str, request_timeout: float = 30):
        self.url = local_url(url)
        self.timeout = request_timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _open(self, route: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.url + route, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            return self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            body = error.read(8192).decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI {route} rejected the request ({error.code}): {body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Cannot reach local ComfyUI at {self.url}: {error.reason}") from error

    def json(self, route: str, payload: dict | None = None) -> dict:
        with self._open(route, payload) as response:
            body = response.read(MAX_JSON + 1)
        if len(body) > MAX_JSON:
            raise RuntimeError("ComfyUI JSON response exceeds 32 MiB.")
        return json_object(body.decode("utf-8"))

    def download(self, descriptor: dict, destination: Path, max_bytes: int) -> dict:
        query = urllib.parse.urlencode({key: descriptor.get(key, "")
                                        for key in ("filename", "subfolder", "type")})
        digest, size = hashlib.sha256(), 0
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as stream:
                temporary = Path(stream.name)
                with self._open("/view?" + query) as response:
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > max_bytes:
                            raise RuntimeError("Generated artifact exceeds --max-artifact-bytes.")
                        stream.write(block)
                        digest.update(block)
                if not size:
                    raise RuntimeError("ComfyUI returned an empty artifact.")
                stream.flush()
                os.fsync(stream.fileno())
            publish_file(temporary, destination, overwrite=False)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {"path": str(destination.resolve()), "size_bytes": size, "sha256": digest.hexdigest(),
                "server_file": descriptor}


def prepare_workflow(workflow: dict, overrides: dict) -> dict:
    graph = deepcopy(workflow)
    if not graph or "nodes" in graph or "links" in graph:
        raise ValueError("Export the ComfyUI workflow in API format (node-id to class_type/inputs).")
    for node_id, node in graph.items():
        if (not node_id or not isinstance(node, dict) or not isinstance(node.get("class_type"), str)
                or not node["class_type"] or not isinstance(node.get("inputs"), dict)):
            raise ValueError(f"Invalid ComfyUI API node: {node_id}")
    for node_id, values in overrides.items():
        if node_id not in graph or not isinstance(values, dict):
            raise ValueError(f"Workflow override must name an existing node: {node_id}")
        for name, value in values.items():
            if name not in graph[node_id]["inputs"]:
                raise ValueError(f"Workflow override names an absent input: {node_id}.{name}")
            graph[node_id]["inputs"][name] = value
    # JSON roundtrip rejects non-finite/duplicate values before any network operation.
    return json_object(json.dumps(graph, allow_nan=False))


def validate_workflow(graph: dict, objects: dict) -> None:
    dependencies = {node_id: [] for node_id in graph}
    outputs = 0
    for node_id, node in graph.items():
        class_name = node["class_type"]
        if class_name not in objects:
            raise ValueError(f"ComfyUI is missing node {class_name} (node {node_id}).")
        specification = objects[class_name]
        if specification.get("api_node") or str(specification.get("python_module", "")).startswith("comfy_api_nodes"):
            raise ValueError(f"Local generation cannot use a hosted API node: {class_name}.")
        outputs += bool(specification.get("output_node"))
        inputs = specification.get("input", {})
        required, optional = inputs.get("required", {}), inputs.get("optional", {})
        for name, schema in required.items():
            if name not in node["inputs"]:
                if (len(schema) > 1 and schema[0] == "COMFY_AUTOGROW_V3"
                        and isinstance(schema[1], dict) and schema[1].get("template", {}).get("min") == 0):
                    continue
                raise ValueError(f"Missing required workflow input: {node_id}.{name}")
        for name, value in node["inputs"].items():
            schema = required.get(name, optional.get(name))
            if schema is None:
                # Dynamic custom-node inputs are finally checked by /prompt.
                continue
            kind = schema[0] if schema else None
            if (kind == "COMBO" and len(schema) > 1 and isinstance(schema[1], dict)
                    and isinstance(schema[1].get("options"), list)):
                kind = schema[1]["options"]
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int):
                source, slot = value
                if source not in graph or isinstance(slot, bool) or slot < 0:
                    raise ValueError(f"Invalid node link: {node_id}.{name}")
                source_type = graph[source]["class_type"]
                if source_type not in objects:
                    raise ValueError(f"ComfyUI is missing node {source_type}.")
                source_outputs = objects[source_type].get("output", [])
                if slot >= len(source_outputs):
                    raise ValueError(f"Invalid output slot: {node_id}.{name}")
                supplied = source_outputs[slot]
                if isinstance(kind, str) and isinstance(supplied, str):
                    expected_types = {part.strip() for part in kind.split(",")}
                    supplied_types = {part.strip() for part in supplied.split(",")}
                    if ("*" not in expected_types | supplied_types
                            and not expected_types.intersection(supplied_types)):
                        raise ValueError(f"Incompatible workflow link {node_id}.{name}: {supplied} -> {kind}")
                dependencies[node_id].append(source)
            elif isinstance(kind, list) and value not in kind:
                raise ValueError(f"Unavailable model or unsupported choice at {node_id}.{name}: {value!r}")
    if not outputs:
        raise ValueError("Workflow requires an output node that saves an artifact.")
    visited, active = set(), set()
    def visit(node_id):
        if node_id in active:
            raise ValueError("Workflow contains a cycle.")
        if node_id in visited:
            return
        active.add(node_id)
        for dependency in dependencies[node_id]:
            visit(dependency)
        active.remove(node_id)
        visited.add(node_id)
    for node_id in graph:
        visit(node_id)


def artifact_descriptors(outputs: dict) -> list[dict]:
    results, seen = [], set()
    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("filename"), str):
                filename = value["filename"]
                folder = value.get("subfolder", "")
                kind = value.get("type", "output")
                if (not filename or filename in (".", "..") or "/" in filename or "\\" in filename
                        or not isinstance(folder, str) or "\\" in folder
                        or PurePosixPath(folder).is_absolute() or ".." in PurePosixPath(folder).parts
                        or kind not in ("output", "temp")):
                    raise ValueError("ComfyUI returned an unsafe output artifact descriptor.")
                identity = (filename, folder, kind)
                if identity not in seen:
                    seen.add(identity)
                    results.append({"filename": filename, "subfolder": folder, "type": kind})
            else:
                for member in value.values():
                    collect(member)
        elif isinstance(value, list):
            for member in value:
                collect(member)
    collect(outputs)
    return results


def wait_for_history(client: Client, prompt_id: str, timeout: float, poll_interval: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        response = client.json("/history/" + urllib.parse.quote(prompt_id, safe=""))
        entry = response.get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution failed ({prompt_id}): {status.get('messages', [])}")
            if status.get("completed") is True and status.get("status_str") == "success":
                return entry
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"ComfyUI job {prompt_id} has not completed; the local job may still run. See submission.json.")
        time.sleep(min(poll_interval, remaining))


def write_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".metadata-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        publish_file(temporary, path, overwrite=False)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--workflow", type=Path, required=True, help="Local workflow exported in ComfyUI API format")
    parser.add_argument("--workflow-inputs", type=read_object, default={}, help='JSON or @file: {"node_id":{"input_name":value}}')
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8188")
    parser.add_argument("--output-dir", type=Path, default=None, help="New or empty destination directory")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument("--max-artifact-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--print-config", action="store_true", help="Resolve workflow without contacting the server")
    parser.add_argument("--validate-only", action="store_true", help="Check live node/model availability without queueing")
    return parser


def run(args: argparse.Namespace) -> dict:
    url = local_url(args.comfyui_url)
    for name in ("timeout", "poll_interval"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if args.max_artifact_bytes <= 0:
        raise ValueError("--max-artifact-bytes must be positive.")
    graph = prepare_workflow(read_object("@" + str(args.workflow)), args.workflow_inputs)
    graph_hash = hashlib.sha256(json.dumps(graph, sort_keys=True, allow_nan=False).encode()).hexdigest()
    base = None
    if args.base_model:
        from civitai_catalog import lookup_base_model
        base = lookup_base_model(args.base_model)
        if base["local_status"] == "hosted":
            raise ValueError(f"{base['name']} is a hosted model, not a local generation target.")
    metadata = {"backend": "comfyui", "base_model": base, "server": url,
                "workflow": graph, "workflow_sha256": graph_hash,
                "verification": "workflow execution; model family is declared, not inferred from tensors"}
    if args.print_config:
        return metadata
    client = Client(url, request_timeout=min(args.timeout, 30))
    objects = client.json("/object_info")
    validate_workflow(graph, objects)
    metadata["nodes"] = {name: {key: objects[name].get(key) for key in ("python_module", "name", "api_node")}
                         for name in sorted({node["class_type"] for node in graph.values()})}
    if args.validate_only:
        return {**metadata, "status": "preflight_passed", "generation_verified": False}
    destination = (args.output_dir or ROOT / "build" / "reference" / ("comfyui-" + uuid.uuid4().hex)).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {destination}")
    submitted = client.json("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    if submitted.get("error") or submitted.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected workflow: {submitted}")
    prompt_id = submitted.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError("ComfyUI did not return a prompt_id; submission state is unknown.")
    metadata.update(prompt_id=prompt_id, status="submitted", output_directory=str(destination))
    write_json(destination / "submission.json", metadata)
    print(f"ComfyUI prompt: {prompt_id}", flush=True)
    entry = wait_for_history(client, prompt_id, args.timeout, args.poll_interval)
    descriptors = artifact_descriptors(entry.get("outputs", {}))
    if not descriptors:
        raise RuntimeError("ComfyUI completed without downloadable image/video/audio/mesh artifacts.")
    artifacts = []
    for index, descriptor in enumerate(descriptors):
        path = destination / f"{index:04d}-{descriptor['filename']}"
        artifacts.append(client.download(descriptor, path, args.max_artifact_bytes))
    metadata.update(status="complete", artifacts=artifacts, execution_status=entry["status"])
    write_json(destination / "generation.json", metadata)
    return metadata


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError, argparse.ArgumentTypeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
