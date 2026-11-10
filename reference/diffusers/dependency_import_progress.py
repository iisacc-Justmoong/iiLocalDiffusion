"""Observe successful new imports while preserving normal Python import behavior."""
from __future__ import annotations

import builtins
import importlib.util
import sys
import time


class DependencyImportProgress:
    """Report cumulative completed module imports, never periodic heartbeats."""

    def __init__(self, callback, *, minimum_interval=0.25):
        if minimum_interval < 0:
            raise ValueError('The dependency progress interval must be nonnegative.')
        self._callback = callback
        self._minimum_interval = minimum_interval
        self._known = set(sys.modules)
        self._count = 0
        self._reported = 0
        self._pending_name = ''
        self._last_report = time.monotonic()
        self._original = None
        self._wrapper = None
        self._original_dynamic = None
        self._dynamic_wrapper = None
        self._reporting = False
        self._active = False

    def __enter__(self):
        if self._original is not None:
            raise RuntimeError('Dependency progress cannot be entered recursively.')
        self._known.update(sys.modules)
        self._original = builtins.__import__
        original = self._original
        self._original_dynamic = importlib.import_module
        original_dynamic = self._original_dynamic
        self._active = True

        def tracked(name, globals=None, locals=None, fromlist=(), level=0):
            result = original(name, globals, locals, fromlist, level)
            if not self._active or self._reporting:
                return result
            absolute = name
            if level:
                package = (globals or {}).get('__package__')
                if not package:
                    return result
                try:
                    absolute = importlib.util.resolve_name('.' * level + name, package)
                except (ImportError, ValueError):
                    return result
            candidates = [absolute]
            candidates.extend(absolute + '.' + item for item in (fromlist or ())
                              if isinstance(item, str) and item != '*')
            self._observe(candidates)
            return result

        def tracked_dynamic(name, package=None):
            result = original_dynamic(name, package)
            if self._active and not self._reporting:
                self._observe([result.__name__])
            return result

        self._wrapper = tracked
        self._dynamic_wrapper = tracked_dynamic
        builtins.__import__ = tracked
        importlib.import_module = tracked_dynamic
        return self

    def _observe(self, candidates):
        for candidate in candidates:
            module = sys.modules.get(candidate)
            if candidate in self._known or module is None:
                continue
            try:
                spec = getattr(module, '__spec__', None)
            except Exception:
                continue
            if getattr(spec, '_initializing', False):
                continue
            self._known.add(candidate)
            self._count += 1
            self._pending_name = candidate
        self._emit(force=False)

    def _emit(self, *, force):
        if self._reported == self._count:
            return
        now = time.monotonic()
        if not force and now - self._last_report < self._minimum_interval:
            return
        self._reported = self._count
        self._last_report = now
        self._reporting = True
        try:
            self._callback(self._count, self._pending_name)
        finally:
            self._reporting = False

    def __exit__(self, exc_type, exc_value, traceback):
        self._active = False
        if builtins.__import__ is self._wrapper:
            builtins.__import__ = self._original
        if importlib.import_module is self._dynamic_wrapper:
            importlib.import_module = self._original_dynamic
        self._original = None
        self._wrapper = None
        self._original_dynamic = None
        self._dynamic_wrapper = None
        self._emit(force=True)
        return False
