"""Explicit local metadata fixture for tests of generation options, not inference."""

from pathlib import Path

MODEL = str(Path(__file__).resolve().parent / "fixtures/sd-v1-manifest")


def local_parser(factory):
    parser = factory()
    parser.set_defaults(model=MODEL)
    return parser


def local_request(values=None):
    import generate
    return generate.resolve_request({"model": MODEL, **(values or {})})
