"""Choose a replayable base seed once per generation request."""

import secrets


def resolve_seed(seed: int | None) -> int:
    """Preserve explicit seeds, including zero; omission uses OS randomness."""
    return secrets.randbits(32) if seed is None else seed
