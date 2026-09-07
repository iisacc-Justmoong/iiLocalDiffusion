"""Direct inference endpoints and model-ID based Inference Providers."""

from __future__ import annotations

from io import BytesIO
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_RESPONSE = 512 * 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class RemoteInference:
    def __init__(self, source):
        self.source = source
        self.token = None
        if source.token_env:
            self.token = os.environ.get(source.token_env)
            if not self.token or any(ord(char) < 32 for char in self.token):
                raise ValueError(f"Set the token environment variable {source.token_env} before remote inference.")
        self.client = None
        if source.kind == "cloud":
            from huggingface_hub import InferenceClient
            # Explicit token and provider selection; no default model/download.
            self.client = InferenceClient(model=source.location, provider=source.provider,
                                          token=self.token, timeout=source.timeout)

    def _endpoint(self, prompt, parameters, media_type):
        headers = {"Content-Type": "application/json", "Accept": media_type}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request(self.source.location, data=json.dumps(
            {"inputs": prompt, "parameters": parameters}, allow_nan=False).encode(), headers=headers, method="POST")
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.source.timeout) as response:
                if response.status != 200 or response.headers.get_content_type() != media_type:
                    raise RuntimeError(f"Inference endpoint must return HTTP 200 with {media_type} bytes.")
                data = response.read(MAX_RESPONSE + 1)
                if not data or len(data) > MAX_RESPONSE:
                    raise RuntimeError("Inference response must be non-empty and at most 512 MiB.")
                return data
        except HTTPError as error:
            # Bodies, URLs and provider exception strings can echo credentials.
            status = error.code
            error.close()
            raise RuntimeError(f"Inference endpoint returned HTTP {status}; no generation retry was submitted.") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Inference endpoint request failed or timed out; no generation retry was submitted.") from None

    def image(self, args, seed):
        parameters = {"width": args.width, "height": args.height, "num_inference_steps": args.steps,
                      "guidance_scale": args.guidance_scale, "negative_prompt": args.negative_prompt, "seed": seed}
        if self.client is None:
            from PIL import Image
            with Image.open(BytesIO(self._endpoint(args.prompt, parameters, "image/png"))) as image:
                image.load()
                if image.format != "PNG":
                    raise RuntimeError("Inference endpoint response must contain a PNG image.")
                return image.convert("RGB")
        try:
            return self.client.text_to_image(args.prompt, **parameters)
        except Exception:
            raise RuntimeError("Cloud image inference failed; verify the provider, model task and credentials.") from None

    def video(self, args, shot):
        parameters = {"width": args.width, "height": args.height, "num_frames": shot["sample_frames"],
                      "frame_rate": shot["sampling_fps"], "num_inference_steps": args.steps,
                      "guidance_scale": args.guidance_scale, "negative_prompt": shot["negative_prompt"],
                      "seed": shot["seed"]}
        if self.client is None:
            return self._endpoint(shot["effective_prompt"], parameters, "video/mp4")
        try:
            # Spatial size and rate are provider-specific; the caller can select a
            # deployment with the required output contract. Always inspect its
            # actual video before accepting it into the local interpolation stage.
            data = self.client.text_to_video(
                shot["effective_prompt"], num_frames=shot["sample_frames"], seed=shot["seed"],
                num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                negative_prompt=[shot["negative_prompt"]])
        except Exception:
            raise RuntimeError("Cloud video inference failed; verify that this provider serves the selected LTX model.") from None
        if not isinstance(data, bytes) or not data or len(data) > MAX_RESPONSE:
            raise RuntimeError("Cloud video response must contain at most 512 MiB of video bytes.")
        return data
