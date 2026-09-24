"""Verify serialized output before atomic publication, not just in-memory math."""


def verify_saved_tensors(path, expected, base_reader, safe_open, torch):
    counts = {"tensor_count": 0, "changed_tensor_count": 0, "finite_tensor_count": 0}
    with safe_open(str(path), framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(expected):
            raise ValueError("Saved tensor keys differ from the expected base layout.")
        for key, value in expected.items():
            actual = saved.get_tensor(key)
            if actual.shape != value.shape or actual.dtype != value.dtype:
                raise ValueError(f"Saved tensor structure differs from the expected base layout: {key}")
            # PyTorch does not implement equal/isfinite for every FP8 storage.
            observed = actual.float() if "float8" in str(actual.dtype) else actual
            wanted = value.float() if "float8" in str(value.dtype) else value
            if not torch.equal(observed, wanted):
                raise ValueError(f"Saved tensor values differ from computed merge: {key}")
            if actual.is_floating_point():
                if not torch.isfinite(observed).all().item():
                    raise ValueError(f"Saved tensor contains nonfinite values: {key}")
                counts["finite_tensor_count"] += 1
            original = base_reader.get_tensor(key)
            original = original.float() if "float8" in str(original.dtype) else original
            counts["changed_tensor_count"] += int(not torch.equal(observed, original))
            counts["tensor_count"] += 1
    return counts
