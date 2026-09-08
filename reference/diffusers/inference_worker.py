"""NDJSON transport for the SDK's in-memory sequential inference session."""

import json
import os
import sys
import time

from generation_config import json_object
from inference_session import InferenceSession
from weight_files import model_hash_statistics

MAX_REQUEST_BYTES = 1024 * 1024


def serve(generate, stream=None):
    """Read one request at a time. EOF releases the model; no job is persisted."""
    stream = sys.stdin.buffer if stream is None else stream
    print('IILD_READY ' + json.dumps({"schema": "iild-worker-v1", "pid": os.getpid(),
                                    "capabilities": ["foreground-residency"]}), flush=True)
    with InferenceSession() as session:
        while line := stream.readline(MAX_REQUEST_BYTES + 1):
            started = time.monotonic()
            identifier, error, code, action = None, "", 0, "generate"
            before = {**model_hash_statistics(), **session.statistics()}
            try:
                if len(line) > MAX_REQUEST_BYTES:
                    raise ValueError("Worker request exceeds the 1 MiB limit.")
                request = json_object(line.decode("utf-8"))
                identifier = request.get("id")
                if not isinstance(identifier, str) or not 1 <= len(identifier) <= 128:
                    identifier = None
                    raise ValueError("Worker request requires a string id of 1 to 128 characters.")
                action = request.get("action", "generate")
                arguments = request.get("arguments", [])
                if (request.get("schema") != "iild-worker-request-v1"
                        or action not in ("generate", "foreground")
                        or not isinstance(arguments, list) or (action == "generate" and not arguments)
                        or any(not isinstance(token, str) or "\0" in token for token in arguments)
                        or any(token.split("=", 1)[0] == "--worker" for token in arguments)):
                    raise ValueError("Invalid worker request schema or arguments.")
                if action == "foreground":
                    foreground = request.get("foreground")
                    if not isinstance(foreground, bool) or (not foreground and arguments):
                        raise ValueError("Foreground control requires a boolean; background control cannot contain arguments.")
                    code = session.set_foreground(foreground, (lambda: generate(arguments)) if arguments else None)
                else:
                    code = generate(arguments) or 0
                if code:
                    raise RuntimeError(f"Inference returned exit code {code}.")
                session.verify()
            except SystemExit as failure:
                code = failure.code if isinstance(failure.code, int) else 1
                error = "" if code == 0 else str(failure)
            except Exception as failure:
                code, error = 1, str(failure)
            if code:
                session.clear()
            after = {**model_hash_statistics(), **session.statistics()}
            result = {"schema": "iild-worker-result-v1", "id": identifier, "ok": code == 0,
                      "exit_code": code, "error": error[-4000:], "pid": os.getpid(),
                      "elapsed_seconds": time.monotonic() - started,
                      "action": action, "residency": session.residency(),
                      "cache": {name: value - before[name] for name, value in after.items()}}
            print("\nIILD_RESULT " + json.dumps(result, ensure_ascii=True), flush=True)
            if len(line) > MAX_REQUEST_BYTES:
                return 2
    return 0
