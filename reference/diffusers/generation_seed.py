"""Choose a replayable base seed once per generation request."""

import secrets
import hashlib


def resolve_seed(seed: int | None) -> int:
    """Preserve explicit seeds, including zero; omission uses OS randomness."""
    return secrets.randbits(32) if seed is None else seed


def derive_generator(generator, domain: bytes, torch):
    """Derive a deterministic named stream without consuming the source RNG."""
    state = bytes(generator.get_state().tolist())
    digest = hashlib.sha256(domain + state).digest()
    return torch.Generator(device=generator.device).manual_seed(int.from_bytes(digest[:8], 'big'))
