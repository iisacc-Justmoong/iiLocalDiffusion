"""Three explicit model locations; parsing never contacts a provider or reads a token."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ipaddress
import math
from pathlib import Path
import re
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ModelInput:
    kind: str
    location: str
    provider: str | None = None
    token_env: str | None = None
    family: str | None = None
    timeout: float = 300

    def metadata(self):
        # Only the environment variable's name belongs in a replayable request.
        return asdict(self)


def add_model_arguments(parser, *, local_help="Local model directory or weight file"):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", "--model-path", dest="model", help=local_help)
    group.add_argument("--model-api", help="Direct Hugging Face-compatible inference endpoint URL")
    group.add_argument("--model-cloud", help="Remote Hugging Face model ID; requires --model-provider")
    parser.add_argument("--model-provider", help="Hugging Face Inference Provider, e.g. fal-ai or hf-inference")
    parser.add_argument("--model-token-env", help="Environment variable containing the remote token; never a token literal")
    parser.add_argument("--model-family", choices=("image", "ltx"),
                        help="Remote model architecture declaration; LTX video requires ltx")
    parser.add_argument("--model-timeout", type=float, help="Remote request timeout in seconds (default: 300)")


def is_remote(args):
    return getattr(args, "model_api", None) is not None or getattr(args, "model_cloud", None) is not None


def resolve_model_input(args):
    selected = [(kind, getattr(args, field, None))
                for kind, field in (("local", "model"), ("api", "model_api"), ("cloud", "model_cloud"))
                if getattr(args, field, None) is not None]
    if len(selected) != 1:
        raise ValueError("Select exactly one --model-path (--model), --model-api or --model-cloud.")
    kind, location = selected[0]
    if not isinstance(location, str) or not location.strip() or location != location.strip():
        raise ValueError("Model location must be a non-empty string without surrounding whitespace.")
    provider = getattr(args, "model_provider", None)
    token_env = getattr(args, "model_token_env", None)
    family = getattr(args, "model_family", None)
    timeout = getattr(args, "model_timeout", None)
    if kind == "local":
        if any(value is not None for value in (provider, token_env, family, timeout)):
            raise ValueError("Remote model options require --model-api or --model-cloud.")
        path = Path(location).expanduser()
        if not path.is_file() and not path.is_dir():
            raise ValueError("--model-path must be an existing local model path.")
        return ModelInput(kind, str(path.resolve()))
    timeout = 300 if timeout is None else timeout
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise ValueError("--model-timeout must be finite and in (0,86400].")
    if token_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env):
        raise ValueError("--model-token-env requires an environment variable name, not a token.")
    if kind == "api":
        if provider is not None:
            raise ValueError("--model-provider is only used with --model-cloud.")
        try:
            url = urlsplit(location)
            loopback = url.hostname == "localhost"
            if url.hostname:
                try:
                    loopback = loopback or ipaddress.ip_address(url.hostname).is_loopback
                except ValueError:
                    pass
            valid = (url.scheme == "https" or url.scheme == "http" and loopback)
            valid = valid and url.hostname and url.username is None and url.password is None
            valid = valid and not url.query and not url.fragment and url.port != 0
            valid = valid and not any(character.isspace() or ord(character) < 32 for character in location)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("--model-api requires HTTPS (HTTP only on loopback), without URL credentials, query or fragment.")
    else:
        if not provider or not re.fullmatch(r"[a-z][a-z0-9-]*", provider) or provider == "auto":
            raise ValueError("--model-cloud requires an explicit --model-provider.")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", location)
                or ".." in location or "--" in location):
            raise ValueError("--model-cloud requires a Hugging Face namespace/model ID, not a URL or local path.")
        token_env = token_env or "HF_TOKEN"
    return ModelInput(kind, location, provider, token_env, family, timeout)
