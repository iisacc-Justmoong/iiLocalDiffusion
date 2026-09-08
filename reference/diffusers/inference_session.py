"""SDK-owned, single-pipeline memory cache for sequential inference requests."""

from contextvars import ContextVar
from collections import OrderedDict
from copy import deepcopy
import gc
from pathlib import Path

from weight_files import file_signature

_current = ContextVar("iild_inference_session", default=None)


def source_signature(sources, *, missing_ok=False):
    records = []
    for source in sources:
        path = Path(source).expanduser().absolute()
        if missing_ok and not path.exists():
            records.append((str(path), None))
            continue
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file()
                           and not any(part.startswith(".") or part == "__pycache__"
                                       for part in p.relative_to(path).parts))
            records.append((str(path), str(path.resolve()),
                            tuple((str(p.relative_to(path)), file_signature(p)) for p in files)))
        else:
            records.append((str(path), file_signature(path)))
    return tuple(records)


class SchedulerConfiguration:
    """Keep the model's initial scheduler configuration, never sampling history."""

    def __init__(self, pipeline):
        self.scheduler_type = type(pipeline.scheduler)
        self.values = deepcopy(dict(pipeline.scheduler.config))

    def restore(self, pipeline):
        pipeline.scheduler = self.scheduler_type.from_config(deepcopy(self.values))


class InferenceSession:
    """One loaded model per worker; never owns a queue or a disk cache."""

    def __init__(self):
        self.value = None
        self.key = None
        self.sources = ()
        self.signature = None
        self.hits = 0
        self.loads = 0
        self.placement_key = self.placement_value = None
        self.device_placements = self.device_placement_hits = 0
        self.configurations = OrderedDict()
        self.configuration_reads = self.configuration_hits = 0
        self.foreground = self.preparing = False
        self.execution = None
        self.preparation_recorded = False

    def __enter__(self):
        self.token = _current.set(self)
        return self

    def __exit__(self, *exception):
        self.clear()
        self.configurations.clear()
        _current.reset(self.token)

    def clear(self):
        had_value = self.value is not None
        self.value = self.key = self.signature = None
        self.placement_key = self.placement_value = None
        self.sources = ()
        self.execution = None
        if had_value:
            gc.collect()

    def pipeline(self, key, sources, loader):
        signature = source_signature(sources) if key is not None else None
        if key is not None and self.value is not None and self.key == key and self.signature == signature:
            self.hits += 1
            return self.value, True
        # Release the previous model before allocating its replacement.
        self.clear()
        self.loads += 1
        value = loader()
        if key is not None:
            if source_signature(sources) != signature:
                raise RuntimeError("Model/configuration changed while initializing inference.")
            self.key, self.sources, self.signature, self.value = key, tuple(sources), signature, value
        return value, False

    def placement(self, key, prepare):
        if key is not None and self.value is not None and self.placement_key == key:
            self.device_placement_hits += 1
            return self.placement_value, True
        previous = self.placement_value
        self.placement_key = self.placement_value = None
        try:
            value = prepare(previous)
        except BaseException:
            self.clear()  # A partially moved model cannot be reused.
            raise
        if key is not None and self.value is not None:
            self.placement_key, self.placement_value = key, value
        return value, False

    def configuration(self, key, sources, read):
        signature = source_signature(sources, missing_ok=True)
        previous = self.configurations.get(key)
        if previous is not None and previous[0] == signature:
            self.configurations.move_to_end(key)
            self.configuration_hits += 1
            return deepcopy(previous[1])
        self.configurations.pop(key, None)
        self.configuration_reads += 1
        value = read()
        if source_signature(sources, missing_ok=True) != signature:
            raise RuntimeError("Model configuration changed while reading.")
        self.configurations[key] = (signature, deepcopy(value))
        if len(self.configurations) > 16:
            self.configurations.popitem(last=False)
        return value

    def statistics(self):
        return {"pipeline_hits": self.hits, "pipeline_loads": self.loads,
                "configuration_reads": self.configuration_reads, "configuration_hits": self.configuration_hits,
                "device_placements": self.device_placements, "device_placement_hits": self.device_placement_hits}

    def verify(self):
        if self.value is not None and source_signature(self.sources) != self.signature:
            self.clear()
            raise RuntimeError("Model/configuration changed while generating.")

    def set_foreground(self, enabled, prepare=None):
        self.foreground = enabled
        if not enabled:
            return 0
        if prepare is None:
            self.clear()  # Foreground without a selected model must not expose the old model as ready.
            return 0
        self.preparing, self.preparation_recorded = True, False
        try:
            code = prepare() or 0
            if not code and (not self.preparation_recorded or self.value is None):
                raise ValueError("Foreground preparation requires a supported, retained local image pipeline.")
            if code:
                self.clear()
            return code
        except BaseException:
            self.clear()
            raise
        finally:
            self.preparing = False

    def residency(self):
        ready = self.value is not None and self.execution is not None
        execution = self.execution or {}
        return {"foreground": self.foreground, "ready": ready,
                "state": ("ready" if self.foreground else "cached") if ready else "waiting-model",
                "gpu_resident": ready and execution.get("device") in ("mps", "cuda")
                                and execution.get("offload") == "none", **execution}


def cached_pipeline(key, sources, loader):
    session = _current.get()
    return (loader(), False) if session is None else session.pipeline(key, sources, loader)


def verify_pipeline_sources():
    session = _current.get()
    if session is not None:
        session.verify()


def cached_placement(key, prepare):
    session = _current.get()
    return (prepare(None), False) if session is None else session.placement(key, prepare)


def record_device_placement():
    session = _current.get()
    if session is not None:
        session.device_placements += 1


def cached_configuration(key, sources, read):
    session = _current.get()
    return read() if session is None else session.configuration(key, sources, read)


def is_preparing():
    session = _current.get()
    return session is not None and session.preparing


def record_execution(device, offload, model):
    session = _current.get()
    if session is not None and session.value is not None:
        session.execution = {"device": device, "offload": offload, "model": str(model)}
        session.preparation_recorded = True
