"""Offline contracts for the MOSS data-space few-step adapter."""
from pathlib import Path
import sys
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import resource_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference/diffusers'))
import moss_rapid


class MossRapidTests(unittest.TestCase):
    def test_published_header_matches_the_moss_sft_architecture(self):
        root = Path(__file__).resolve().parents[1]
        fixture = json.loads((root / 'resources/test-fixtures/moss-rapid-header.json').read_text())
        self.assertEqual(fixture['revision'], 'c88f33bcb3251eb39943d0635e74b650732977c8')
        def shapes(name):
            return {k: v['shape'] for k, v in fixture['files'][name]['header'].items() if k != '__metadata__'}
        projections = {'q': (3072, 3072), 'k': (3072, 3072), 'v': (3072, 3072),
                       'o': (3072, 3072), 'ffn_up_gate': (16384, 3072), 'ffn_down': (3072, 8192)}
        modules = {f'layers.{i}.proj.{name}': shape for i in range(42) for name, shape in projections.items()}
        time = {'time_embedding.0.weight': (3072, 256), 'time_embedding.0.bias': (3072,),
                'time_embedding.2.weight': (3072, 3072), 'time_embedding.2.bias': (3072,),
                'shared_block_modulation.bias': (18432,), 'video_head_modulation.bias': (6144,)}
        targets = moss_rapid.plan_weights(*(shapes(n) for n in moss_rapid.FILES), modules, time, 128, 42)
        self.assertEqual(len(targets), 252)

    def test_request_selects_four_steps_and_rejects_incompatible_sampling(self):
        with resource_fixture.TemporaryDirectory(prefix='moss-rapid-options-') as temporary:
            def args(**changes):
                values = dict(moss_rapid_adapter=temporary, offload='none', seed=1, dtype='bfloat16',
                              preview_dir=None, inputs={})
                values.update(changes)
                return SimpleNamespace(**values)
            with patch.object(moss_rapid, 'resolve_weight_file', side_effect=lambda path, role: path):
                request = args()
                selected = moss_rapid.resolve(request, 'MOSSImagePipeline')
                self.assertEqual(len(selected.weights), 3)
                self.assertEqual(request.inputs, {'num_inference_steps': 4, 'guidance_scale': 1.})
                for change in ({'inputs': {'num_inference_steps': 5}}, {'inputs': {'guidance_scale': 3}},
                               {'inputs': {'negative_prompt': 'bad'}}, {'inputs': {'slg_scale': 1}},
                               {'seed': None}, {'offload': 'model'}, {'dtype': 'float16'}):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        moss_rapid.resolve(args(**change), 'MOSSImagePipeline')
                with self.assertRaises(ValueError):
                    moss_rapid.resolve(args(), 'FluxPipeline')

    def test_shape_plan_requires_complete_lora_time_and_dataspace(self):
        modules = {f'layers.0.proj.{name}': (12, 8) for name in moss_rapid.TARGETS}
        lora = {}
        for name in modules:
            key = 'diffusion_model.' + name
            lora.update({key + '.lora_down.weight': (2, 8),
                         key + '.lora_up.weight': (12, 2), key + '.alpha': ()})
        time = {'time_embedding.0.weight': (12, 8), 'time_embedding.0.bias': (12,),
                'time_embedding.2.weight': (12, 12), 'time_embedding.2.bias': (12,),
                'shared_block_modulation.bias': (72,), 'video_head_modulation.bias': (24,)}
        dataspace = {'latent_mix.Q': (8, 8), 'step_codes': (4,)}
        targets = moss_rapid.plan_weights(lora, time, dataspace, modules, time, 8, 1)
        self.assertEqual(set(targets), set(modules))
        for source, key, value in ((lora, 'diffusion_model.layers.0.proj.q.alpha', None),
                                   (time, 'time_embedding.2.bias', (8,)),
                                   (dataspace, 'latent_mix.Q', (8, 7))):
            broken = dict(source)
            if value is None: broken.pop(key)
            else: broken[key] = value
            args = [lora, time, dataspace]
            args[(lora, time, dataspace).index(source)] = broken
            with self.assertRaises(ValueError):
                moss_rapid.plan_weights(*args, modules, time, 8, 1)


if __name__ == '__main__':
    unittest.main()
