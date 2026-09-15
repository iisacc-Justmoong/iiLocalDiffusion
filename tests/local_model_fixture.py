"""Metadata-only baseline fixtures; real bundled defaults have their own tests.

These synthetic pipelines do not contain the trained encoders/adapter targets.
Disable bundled modifiers explicitly while testing independent option contracts.
"""

from pathlib import Path

MODEL = str(Path(__file__).resolve().parent / "fixtures/sd-v1-manifest")


def local_parser(factory):
    parser = factory()
    parser.set_defaults(model=MODEL, default_modifiers=False, hires_fix=False)
    return parser


def local_request(values=None):
    import generate
    return generate.resolve_request({"model": MODEL, "default_modifiers": False,
                                     "hires_fix": False, **(values or {})})
