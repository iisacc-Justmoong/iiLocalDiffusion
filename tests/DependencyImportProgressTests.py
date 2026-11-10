"""Dependency progress reflects completed imports, not periodic heartbeats."""
import builtins
import importlib
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
from dependency_import_progress import DependencyImportProgress


class DependencyImportProgressTests(unittest.TestCase):
    def test_new_nested_imports_are_counted_once_and_cached_imports_are_ignored(self):
        events = []
        leaf = 'iild_leaf_' + uuid.uuid4().hex
        parent = 'iild_parent_' + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(dir=ROOT / 'resources') as directory:
            root = Path(directory)
            (root / (leaf + '.py')).write_text('value = 1\n')
            (root / (parent + '.py')).write_text('import ' + leaf + '\n')
            sys.path.insert(0, directory)
            try:
                before = builtins.__import__
                with DependencyImportProgress(lambda count, name: events.append((count, name)), minimum_interval=0):
                    __import__(parent)
                    count = len(events)
                    __import__(parent)
                    self.assertEqual(len(events), count)
                self.assertIs(builtins.__import__, before)
                self.assertEqual([name for _, name in events if name in (leaf, parent)], [leaf, parent])
                self.assertEqual([count for count, _ in events], sorted({count for count, _ in events}))
            finally:
                sys.path.remove(directory)
                sys.modules.pop(parent, None)
                sys.modules.pop(leaf, None)

    def test_failed_import_does_not_report_completion_and_hook_is_restored(self):
        events = []
        missing = 'iild_missing_' + uuid.uuid4().hex
        before = builtins.__import__
        with self.assertRaises(ModuleNotFoundError):
            with DependencyImportProgress(lambda count, name: events.append((count, name)), minimum_interval=0):
                __import__(missing)
        self.assertIs(builtins.__import__, before)
        self.assertFalse(any(name == missing for _, name in events))

    def test_dynamic_imports_are_reported_and_cached_imports_are_ignored(self):
        events = []
        name = 'iild_dynamic_' + uuid.uuid4().hex
        original = importlib.import_module
        with tempfile.TemporaryDirectory(dir=ROOT / 'resources') as directory:
            (Path(directory) / (name + '.py')).write_text('value = 1\n')
            sys.path.insert(0, directory)
            try:
                with DependencyImportProgress(lambda count, module: events.append((count, module)), minimum_interval=0):
                    importlib.import_module(name)
                    count = len(events)
                    importlib.import_module(name)
                    self.assertEqual(len(events), count)
                self.assertIs(importlib.import_module, original)
                self.assertEqual([module for _, module in events if module == name], [name])
            finally:
                sys.path.remove(directory)
                sys.modules.pop(name, None)

    def test_pending_success_is_reported_when_scope_ends(self):
        events = []
        name = 'iild_throttled_' + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(dir=ROOT / 'resources') as directory:
            (Path(directory) / (name + '.py')).write_text('value = 1\n')
            sys.path.insert(0, directory)
            try:
                with DependencyImportProgress(lambda count, module: events.append((count, module)),
                                              minimum_interval=3600):
                    __import__(name)
                self.assertTrue(any(module == name for _, module in events))
            finally:
                sys.path.remove(directory)
                sys.modules.pop(name, None)

    def test_other_import_wrapper_is_not_overwritten_or_left_broken(self):
        before = builtins.__import__
        dynamic_before = importlib.import_module
        events = []
        try:
            with DependencyImportProgress(lambda count, name: events.append((count, name)), minimum_interval=0):
                observed_import = builtins.__import__
                observed_dynamic = importlib.import_module
                def replacement(*args, **kwargs):
                    return observed_import(*args, **kwargs)
                def dynamic_replacement(*args, **kwargs):
                    return observed_dynamic(*args, **kwargs)
                builtins.__import__ = replacement
                importlib.import_module = dynamic_replacement
            self.assertIs(builtins.__import__, replacement)
            self.assertIs(importlib.import_module, dynamic_replacement)
            self.assertIs(builtins.__import__('sys'), sys)
            self.assertIs(importlib.import_module('sys'), sys)
        finally:
            builtins.__import__ = before
            importlib.import_module = dynamic_before


if __name__ == '__main__':
    unittest.main()
