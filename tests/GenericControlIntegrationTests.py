"""Exercise control preprocessing through the generic runner's dispatch."""
from contextlib import ExitStack, nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import resource_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/diffusers'))
import depth_preprocessing
import flux2_controlnet
import generate_any as runner
import generic_controlnet
import krea2_controlnet
import krea2_color_control
import krea2_ostris_control
import moss_controlnet


class GenericControlIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = resource_fixture.TemporaryDirectory(prefix='control-routing-')
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / 'output'

    def run_request(self, family, method, auto_bucket=False, inpaint=False, rapid=False, reference_resize='none',
                    map_preprocess='semantic', regions=False):
        image = 'photo.png'
        color = 'color.png'
        if family == 'krea':
            selection = krea2_controlnet.Selection(None, image, 1., auto_bucket=auto_bucket)
        elif family == 'krea-ostris':
            selection = krea2_ostris_control.Selection(None, image, 1.)
        elif family == 'krea-color':
            selection = krea2_color_control.Selection(None, image, color, 1., 0., 1., 1., auto_bucket)
        elif family == 'flux2':
            selection = flux2_controlnet.Selection(None, image, 1., 0., 1.,
                inpaint_image='inpaint.png' if inpaint else None, mask_image='mask.png' if inpaint else None)
        elif family == 'klein-reference':
            selection = generic_controlnet.KleinReferenceSelection(None, image, 'reference.png', 1., reference_resize)
        elif family == 'moss':
            selection = moss_controlnet.Selection(None, image, color, 1., 0., 1., 1., map_preprocess)
        elif family == 'sd3':
            selection = generic_controlnet.Selection('control', 'StableDiffusion3Pipeline',
                'StableDiffusion3ControlNetPipeline', 'SD3ControlNetModel', 'sd3', 'control_image', None)
        else:
            selection = generic_controlnet.Selection('control', 'FluxPipeline', 'FluxControlNetPipeline',
                                                      'FluxControlNetModel', 'flux1-schnell', 'control_image', None)
        inputs = {'width': 16, 'height': 16}
        if family in ('flux1', 'sd3'):
            inputs['control_image'] = {'image_path': image, 'mode': 'RGB'}
        if family == 'klein-reference':
            inputs['image'] = [{'image_path': image}, {'image_path': 'reference.png'}]
        args = SimpleNamespace(print_config=False, output_dir=self.output,
            overwrite=False, device='cpu', dtype='float32', inputs=inputs, generic_controlnet=selection,
            control_preprocess=method, control_preprocessor_library=None,
            depth_preprocessor=depth_preprocessing.Selection('depth-model'), offload='none', model='model',
            seed=7, preview_dir=None, krea2=({'variant': 'raw', 'mu_override': None, 'resolved_mu': None}
                                             if auto_bucket else None), lora_activation=None, lora_selection=None,
            vae_selection=None, text_encoder_selection=None, tokenizer_selection=None, model_configuration_cache_hit=False,
            device_placement_cache_hit=False, moss_rapid=SimpleNamespace(weights=[]) if rapid else None)
        args.moss_regions = SimpleNamespace(map_path='region.png') if regions else None
        calls = {}
        fingerprints = []
        class Pipeline:
            def __call__(self, **kwargs):
                calls['pipeline'] = kwargs.copy()
                return {'images': []}
        pipeline = Pipeline()
        def decode(descriptors, unused_image):
            def value_for(key, value):
                if isinstance(value, list):
                    return [value_for(key, item) for item in value]
                if isinstance(value, dict) and 'image_path' in value:
                    return (SimpleNamespace(size=(900, 1600)) if auto_bucket and key == 'control_image'
                            else 'decoded:' + value['image_path'])
                return value
            return ({key: value_for(key, value) for key, value in descriptors.items()}, [])
        def preprocess(original, selection, torch, np, Image, device, crop, *, preserve_float):
            calls['depth'] = (original, crop, preserve_float)
            return 'depth-output', {'name': 'depth-anything-relative', 'output': 'fixture-format'}
        def execution(pipeline, control, inputs, selection, torch, np, **kwargs):
            calls['control'] = control
            calls['control_options'] = kwargs
            return nullcontext({'adapter': family})
        def rapid_generate(pipeline, inputs, adapter, trace, torch):
            calls['rapid'] = (inputs.copy(), adapter, trace)
            return {'images': []}
        def publish(result, pipeline, args, Image, np, report):
            calls['report'] = report
            return Path('report.json'), []
        class Generator:
            def __init__(self, device): pass
            def manual_seed(self, seed): return self
        torch = SimpleNamespace(float32='float32', Generator=Generator, no_grad=nullcontext)
        pil = ModuleType('PIL')
        pil.Image = object()
        modules = {'torch': torch, 'diffusers': SimpleNamespace(), 'numpy': SimpleNamespace(), 'PIL': pil}
        identities = {'directory': 'depth-model', 'files': []}
        mocked = {'resolve_arguments': Mock(return_value=args), 'is_preparing': Mock(return_value=False),
                  'select_device': Mock(return_value='cpu'), 'accelerator_preflight': Mock(return_value={}),
                  'prepare_inputs': Mock(side_effect=decode), 'prepared_pipeline': Mock(return_value=(pipeline, {'identity': identities}, False)),
                  'model_identity': Mock(return_value=identities), 'verify_identity': Mock(),
                  'record_execution': Mock(), 'validate_call_arguments': Mock(), 'attach_preview': Mock(),
                  'validate_execution_device': Mock(), 'validate_lora_activation': Mock(), 'verify_lora_identity': Mock(),
                  'verify_vae_selection': Mock(), 'verify_input_files': Mock(), 'verify_pipeline_sources': Mock(),
                  'configuration': Mock(return_value={}), 'lora_metadata': Mock(return_value=None),
                  'publish_generation': Mock(side_effect=publish), 'file_identity': Mock(return_value={'path': 'native'})}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.multiple(runner, **mocked))
            stack.enter_context(patch.object(runner.depth_preprocessing, 'apply', side_effect=preprocess))
            stack.enter_context(patch.object(runner.krea2_contract, 'execution', return_value=nullcontext(None)))
            stack.enter_context(patch.object(runner.krea2_text_encoder, 'verify_selection'))
            stack.enter_context(patch.object(runner.generic_controlnet, 'verify_selection'))
            seed_context = stack.enter_context(patch.object(runner.generic_controlnet, 'sd3_control_seed',
                                                           return_value=nullcontext()))
            stack.enter_context(patch('weight_files.verify_weight_file'))
            stack.enter_context(patch.object(runner.native_control_preprocessing, 'library_path', return_value=Path('native')))
            stack.enter_context(patch.object(runner.native_control_preprocessing, 'load_library', return_value=object()))
            stack.enter_context(patch.object(runner.native_control_preprocessing, 'apply_canny',
                                             side_effect=lambda original, Image, lib: 'canny:' + original))
            stack.enter_context(patch.object(runner.control_hint_identity, 'fingerprint',
                side_effect=lambda processed, np: fingerprints.append(processed) or {'sha256': 'hint-digest'}))
            stack.enter_context(patch.object(runner.klein_reference_resize, 'prepare',
                side_effect=lambda images, torch, np, Image: (['resized:' + image for image in images], {'mode': 'comfy-1mp'})))
            stack.enter_context(patch.object(runner.klein_reference_resize, 'execution', return_value=nullcontext()))
            stack.enter_context(patch.object(runner.klein_reference_resize, 'comfy_vae_execution', return_value=nullcontext()))
            def region_execution(pipeline, inputs, selection, image, torch, np, *, map_fit):
                calls['region_map_fit'] = map_fit
                return nullcontext({'map_fit': map_fit})
            stack.enter_context(patch.object(runner.moss_regions, 'execution', side_effect=region_execution))
            stack.enter_context(patch.object(runner.moss_regions, 'verify'))
            for module in (runner.krea2_controlnet, runner.krea2_color_control, runner.krea2_ostris_control,
                           runner.flux2_controlnet, runner.moss_controlnet):
                stack.enter_context(patch.object(module, 'execution', side_effect=execution))
            stack.enter_context(patch.object(runner.moss_rapid, 'execution', return_value=nullcontext('rapid-adapter')))
            stack.enter_context(patch.object(runner.moss_rapid, 'generate', side_effect=rapid_generate))
            stack.enter_context(patch('sys.stdout'))
            self.assertEqual(runner.main(['--model', 'unused']), 0)
            self.assertIs(seed_context.call_args.args[1], selection)
            self.assertEqual(seed_context.call_args.args[2], 7)
            self.assertEqual(fingerprints, ['depth-output' if method == 'depth' else 'canny:decoded:photo.png'])
            self.assertEqual(calls['report']['control_preprocessor']['output_identity'], {'sha256': 'hint-digest'})
            if method == 'depth':
                self.assertEqual(calls['report']['control_preprocessor']['output'], 'fixture-format')
        return calls

    def test_krea_ostris_receives_canny_and_rgb_depth_preprocessing(self):
        canny = self.run_request('krea-ostris', 'canny')
        self.assertEqual(canny['control'], 'canny:decoded:photo.png')
        self.assertNotIn('control_image', canny['pipeline'])
        depth = self.run_request('krea-ostris', 'depth')
        self.assertEqual(depth['depth'], ('decoded:photo.png', None, False))
        self.assertEqual(depth['control'], 'depth-output')

    def test_moss_rapid_receives_the_preprocessed_control_context(self):
        calls = self.run_request('moss', 'depth', rapid=True)
        self.assertEqual(calls['control'], {'depth': 'depth-output', 'color': 'decoded:color.png'})
        self.assertEqual(calls['rapid'][1:], ('rapid-adapter', {'adapter': 'moss'}))
        self.assertNotIn('pipeline', calls)

    def test_moss_training_map_preprocessing_aligns_regional_geometry(self):
        for mode, expected in (('training', 'center'), ('semantic', 'stretch')):
            with self.subTest(mode=mode):
                calls = self.run_request('moss', 'depth', map_preprocess=mode, regions=True)
                self.assertEqual(calls['region_map_fit'], expected)
                self.assertEqual(calls['report']['regional_execution']['map_fit'], expected)

    def test_krea_bucket_is_selected_before_depth_estimation(self):
        calls = self.run_request('krea', 'depth', auto_bucket=True)
        self.assertEqual(calls['depth'][0].size, (900, 1600))
        self.assertEqual(calls['depth'][1:], ((768, 1344), True))
        self.assertEqual((calls['pipeline']['width'], calls['pipeline']['height']), (768, 1344))

    def test_krea_color_and_depth_route_together(self):
        calls = self.run_request('krea-color', 'depth')
        self.assertEqual(calls['depth'], ('decoded:photo.png', None, False))
        self.assertEqual(calls['control'], {'depth': 'depth-output', 'color': 'decoded:color.png'})

    def test_flux2_inpaint_and_mask_reach_control_execution(self):
        calls = self.run_request('flux2', 'depth', inpaint=True)
        self.assertEqual(calls['control'], 'depth-output')
        self.assertEqual(calls['control_options'], {'inpaint_image': 'decoded:inpaint.png',
                                                    'mask_image': 'decoded:mask.png'})
        self.assertNotIn('control_inpaint_image', calls['pipeline'])
        self.assertNotIn('control_mask_image', calls['pipeline'])

    def test_klein_reference_preprocesses_first_image_only(self):
        for method, expected in (('depth', 'depth-output'), ('canny', 'canny:decoded:photo.png')):
            with self.subTest(method=method):
                calls = self.run_request('klein-reference', method)
                self.assertEqual(calls['pipeline']['image'], [expected, 'decoded:reference.png'])
                calls = self.run_request('klein-reference', method, reference_resize='comfy-1mp')
                self.assertEqual(calls['pipeline']['image'], ['resized:' + expected, 'resized:decoded:reference.png'])
                self.assertEqual(calls['report']['control_reference_resize'],
                                 {'mode': 'comfy-1mp', 'vae_normalization': 'batch-norm-inference'})

    def test_krea_color_depth_auto_bucket_routes_both_maps(self):
        calls = self.run_request('krea-color', 'depth', auto_bucket=True)
        self.assertEqual(calls['depth'][0].size, (900, 1600))
        self.assertEqual(calls['depth'][1:], (None, False))
        self.assertEqual(calls['control'], {'depth': 'depth-output', 'color': 'decoded:color.png'})
        self.assertEqual((calls['pipeline']['width'], calls['pipeline']['height']), (768, 1344))

    def test_depth_is_routed_to_each_model_and_moss_color_is_preserved(self):
        for family in ('krea', 'flux1', 'flux2', 'moss', 'sd3'):
            with self.subTest(family=family):
                calls = self.run_request(family, 'depth')
                self.assertEqual(calls['depth'], ('decoded:photo.png', (16, 16) if family == 'krea' else None,
                                                   family == 'krea'))
                if family in ('flux1', 'sd3'):
                    self.assertEqual(calls['pipeline']['control_image'], 'depth-output')
                elif family == 'moss':
                    self.assertEqual(calls['control'], {'depth': 'depth-output', 'color': 'decoded:color.png'})
                else:
                    self.assertEqual(calls['control'], 'depth-output')
                self.assertEqual(calls['report']['control_preprocessor']['name'], 'depth-anything-relative')

    def test_native_canny_is_routed_without_replacing_moss_color(self):
        for family in ('krea', 'flux1', 'flux2', 'moss', 'sd3'):
            with self.subTest(family=family):
                calls = self.run_request(family, 'canny')
                if family in ('flux1', 'sd3'):
                    self.assertEqual(calls['pipeline']['control_image'], 'canny:decoded:photo.png')
                elif family == 'moss':
                    self.assertEqual(calls['control'], {'depth': 'canny:decoded:photo.png', 'color': 'decoded:color.png'})
                else:
                    self.assertEqual(calls['control'], 'canny:decoded:photo.png')
                self.assertEqual(calls['report']['control_preprocessor']['name'], 'native-canny')


if __name__ == '__main__':
    unittest.main()
