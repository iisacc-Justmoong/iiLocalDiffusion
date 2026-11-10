"""Actual, monotonic tensor-loading progress without synthetic heartbeats."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))


class MossLoadingProgressTests(unittest.TestCase):
    def test_reports_only_completed_tensors_with_accurate_byte_totals(self):
        from moss_loading_progress import TensorLoadingProgress
        now = [0.0]
        events = []
        header = {
            'first': {'shape': [2, 4], 'dtype': 'I8'},
            'second': {'shape': [2, 4], 'dtype': 'F32'},
            'ignored': {'shape': [100], 'dtype': 'F32'},
        }
        progress = TensorLoadingProgress(header, ['first', 'second'], 'transformer',
                                         lambda *event: events.append(event), clock=lambda: now[0])
        self.assertEqual(events, [])
        progress.completed('first')
        self.assertEqual(events, [('transformer', 8, 40)])
        now[0] = 0.1
        progress.completed('second')
        self.assertEqual(events[-1], ('transformer', 40, 40))
        with self.assertRaises(ValueError):
            progress.completed('second')
        self.assertEqual(len(events), 2)

    def test_throttling_never_creates_progress_for_unchanged_work(self):
        from moss_loading_progress import TensorLoadingProgress
        now = [0.0]
        events = []
        header = {str(n): {'shape': [1], 'dtype': 'BF16'} for n in range(4)}
        progress = TensorLoadingProgress(header, header.keys(), 'text_encoder',
                                         lambda *event: events.append(event), clock=lambda: now[0])
        progress.completed('0')
        now[0] = 0.1
        progress.completed('1')
        self.assertEqual(events, [('text_encoder', 2, 8)])
        now[0] = 0.5
        progress.completed('2')
        progress.completed('3')
        self.assertEqual(events, [('text_encoder', 2, 8), ('text_encoder', 6, 8),
                                  ('text_encoder', 8, 8)])

    def test_forward_progress_counts_real_completed_layers_and_resets_each_pass(self):
        from moss_loading_progress import attach_quantized_forward_progress

        class Weight:
            shape = (2, 4)

        class Module:
            def __init__(self, quantized=False):
                self._iild_int8 = Weight() if quantized else None
                self.pre_hooks = []
                self.hooks = []
            def register_forward_pre_hook(self, hook): self.pre_hooks.append(hook)
            def register_forward_hook(self, hook): self.hooks.append(hook)

        root, a, b = Module(), Module(True), Module(True)
        root.named_modules = lambda: iter([('', root), ('a', a), ('b', b)])
        events = []
        attach_quantized_forward_progress(root, 'text_encoder_compute', lambda *v: events.append(v))
        self.assertEqual(events, [])
        root.pre_hooks[0](root, ())
        a.hooks[0](a, (), None)
        a.hooks[0](a, (), None)  # Repeated module calls cannot manufacture progress.
        b.hooks[0](b, (), None)
        self.assertEqual(events, [('text_encoder_compute', 8, 16), ('text_encoder_compute', 16, 16)])
        root.pre_hooks[0](root, ())
        b.hooks[0](b, (), None)
        self.assertEqual(events[-1], ('text_encoder_compute', 8, 16))


if __name__ == '__main__':
    unittest.main()
