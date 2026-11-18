#!/usr/bin/env python3
"""Verify resource installation excludes private state without installing models."""
from pathlib import Path
import os
import shutil
import subprocess
import re
import sys
import unittest
import resource_fixture
import resource_runner

ROOT = Path(__file__).resolve().parents[1]


class InstallResourcesTests(unittest.TestCase):
    def test_runtime_installs_standalone_vae_conversion_helper(self):
        generated = (ROOT / 'build/cmake_install.cmake').read_text(encoding='utf-8')
        rules = [line.strip() for line in generated.splitlines()
                 if 'TYPE DIRECTORY FILES' in line and '/reference/diffusers" FILES_MATCHING' in line]
        self.assertEqual(len(rules), 1)
        with resource_fixture.TemporaryDirectory(prefix='install-vae-helper-') as temporary:
            workspace = Path(temporary)
            stage = workspace / 'stage'
            script = workspace / 'install-vae-helper.cmake'
            script.write_text(rules[0] + '\n', encoding='utf-8')
            subprocess.run(['cmake', f'-DCMAKE_INSTALL_PREFIX={stage}', '-P', str(script)],
                           check=True, capture_output=True, text=True)
            installed = stage / 'share/iiLocalDiffusion/reference/diffusers/vae_legacy_conversion.py'
            self.assertEqual(installed.read_bytes(), (ROOT / 'reference/diffusers/vae_legacy_conversion.py').read_bytes())

    def test_component_inspection_headers_are_installable_without_model_resources(self):
        generated = (ROOT / 'build/cmake_install.cmake').read_text(encoding='utf-8')
        rules = re.findall(r'(file\(INSTALL DESTINATION "[^"\n]*/include(?:/Generation)?" '
                           r'TYPE FILE FILES.*?\))', generated, flags=re.DOTALL)
        selected = [rule for rule in rules if 'NativeTextEncoderInspection.hpp' in rule or '/src/Export.hpp' in rule]
        self.assertEqual(len(selected), 2)
        with resource_fixture.TemporaryDirectory(prefix='install-inspection-headers-') as temporary:
            stage = Path(temporary) / 'stage'
            script = Path(temporary) / 'install-headers.cmake'
            script.write_text('\n'.join(selected) + '\n', encoding='utf-8')
            subprocess.run(['cmake', f'-DCMAKE_INSTALL_PREFIX={stage}', '-P', str(script)],
                           check=True, capture_output=True, text=True)
            for relative in ('Export.hpp', 'Generation/NativeVaeInspection.hpp',
                             'Generation/NativeTextEncoderInspection.hpp', 'Generation/ImagePackageBridge.hpp'):
                self.assertTrue((stage / 'include' / relative).is_file(), relative)

    def test_runtime_component_installs_native_shared_library(self):
        # Inspect the generated installer: running the complete Runtime install
        # here would copy the multi-gigabyte user-provisioned model resources.
        script = (ROOT / 'build/cmake_install.cmake').read_text(encoding='utf-8')
        rules = re.findall(r'if\(CMAKE_INSTALL_COMPONENT STREQUAL "([^"]+)" OR NOT CMAKE_INSTALL_COMPONENT\)\s*'
                           r'file\(INSTALL[^\)]*TYPE (?:SHARED_LIBRARY|EXECUTABLE) FILES([^\)]*)\)', script)
        shared = [(component, files) for component, files in rules
                  if re.search(r'(?:lib)?iiLocalDiffusion(?:\.[0-9]+)*\.(?:dylib|so|dll)', files)]
        self.assertTrue(shared, 'Generated installer must contain the iiLocalDiffusion runtime library')
        self.assertEqual({component for component, _ in shared}, {'Runtime'})
        provision_rules = re.findall(
            r'if\(CMAKE_INSTALL_COMPONENT STREQUAL "Runtime" OR NOT CMAKE_INSTALL_COMPONENT\)\s*'
            r'file\(INSTALL DESTINATION "[^"\n]*/iiLocalDiffusion/scripts" TYPE FILE FILES(.*?)\)\s*endif\(\)',
            script, flags=re.DOTALL)
        self.assertEqual(len(provision_rules), 1, 'Provisioners must be installed as Runtime scripts')
        provisioned = set(re.findall(r'"[^"\n]*/scripts/([^/"\n]+\.py)"', provision_rules[0]))
        self.assertEqual(provisioned, {
            'provision_flux1_text_encoders.py', 'provision_flux1_schnell.py', 'provision_flux1_canny.py',
            'provision_anima_text_encoder.py',
            'provision_krea2.py', 'provision_depth_anything_v2.py', 'provision_moss_text_encoder.py',
            'provision_moss_transformer.py', 'provision_moss_control.py',
        })
        for model in ('moss-image-sft', 'krea2-raw', 'krea2-turbo', 'flux1-schnell'):
            for config in ('model_index.json', 'scheduler/scheduler_config.json', 'transformer/config.json'):
                self.assertIn(f'model-configs/{model}/{config}', script)
        self.assertIn('model-configs/flux1-schnell/transformer-source-shapes.json', script)

    def test_installed_provisioners_import_and_resolve_installed_resource_root(self):
        generated = (ROOT / 'build/cmake_install.cmake').read_text(encoding='utf-8')
        rules = re.findall(r'(file\(INSTALL DESTINATION "[^"\n]*/iiLocalDiffusion/scripts" '
                           r'TYPE FILE FILES.*?\))', generated, flags=re.DOTALL)
        self.assertEqual(len(rules), 1)
        with resource_fixture.TemporaryDirectory(prefix='install-provisioners-') as temporary:
            workspace = Path(temporary)
            stage = workspace / 'stage'
            install_script = workspace / 'install-provisioners.cmake'
            install_script.write_text(rules[0] + '\n', encoding='utf-8')
            subprocess.run(['cmake', f'-DCMAKE_INSTALL_PREFIX={stage}', '-P', str(install_script)],
                           check=True, capture_output=True, text=True)
            installed = stage / 'share/iiLocalDiffusion/scripts'
            expected_root = stage / 'share/iiLocalDiffusion/resources'
            modules = ('provision_flux1_text_encoders', 'provision_flux1_schnell', 'provision_flux1_canny',
                       'provision_anima_text_encoder',
                       'provision_krea2', 'provision_depth_anything_v2', 'provision_moss_text_encoder',
                       'provision_moss_transformer', 'provision_moss_control')
            code = ('import importlib, pathlib, sys\n'
                    'expected = pathlib.Path(sys.argv[1]).resolve()\n'
                    'for name in sys.argv[2:]:\n'
                    '    module = importlib.import_module(name)\n'
                    '    assert module.ROOT == expected, (name, module.ROOT, expected)\n')
            env = {**os.environ, 'PYTHONPATH': str(installed), 'PYTHONDONTWRITEBYTECODE': '1'}
            subprocess.run([sys.executable, '-c', code, str(expected_root), *modules],
                           env=env, check=True, capture_output=True, text=True)

    def test_only_public_resource_inputs_are_installed(self):
        with resource_fixture.TemporaryDirectory(prefix="install-resources-") as temporary:
            workspace = Path(temporary)
            provider = resource_runner._provider()
            source = workspace / "source"
            resources = source / "fixtures"
            stage = workspace / "stage"
            target = ROOT / "build" / "install-resources-tests" / workspace.name
            if target.exists() or target.is_symlink():
                self.fail("Refusing to replace an existing build")
            def cleanup():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                try:
                    target.parent.rmdir()
                except OSError:
                    pass
            self.addCleanup(cleanup)
            files = {
                "vae/model.safetensors": "public model", "vae/config.json": "{}",
                "text_encoders/qwen3vl-4b/config.json": "{}",
                "text_encoders/qwen3vl-4b/tokenizer_config.json": "{}",
                "text_encoders/qwen3vl-4b/vocab.json": "{}",
                "text_encoders/qwen3vl-4b/merges.txt": "merges",
                "text_encoders/qwen35-9b/config.json": "{}",
                "text_encoders/qwen35-9b/tokenizer_config.json": "{}",
                "text_encoders/qwen35-9b/vocab.json": "{}",
                "text_encoders/qwen35-9b/merges.txt": "merges",
                "text_encoders/flux1-clip/vocab.json": "{}",
                "text_encoders/flux1-clip/merges.txt": "merges",
                "text_encoders/flux1-clip/tokenizer_config.json": "{}",
                "text_encoders/flux1-clip/special_tokens_map.json": "{}",
                "text_encoders/flux1-t5/tokenizer.json": "{}",
                "text_encoders/flux1-t5/tokenizer_config.json": "{}",
                "text_encoders/flux1-t5/special_tokens_map.json": "{}",
                "text_encoders/flux1-clip-config/config.json": "{}",
                "text_encoders/flux1-t5-config/config.json": "{}",
                "vae/NOTICE.md": "public notice", "vae/.cache/private.safetensors": "cache",
                ".state/private.safetensors": "state", ".recovery/private.safetensors": "recovery",
                "dependencies/third-party/README.md": "dependency", "outputs/private.safetensors": "user output",
                "test-workspaces/private/config.json": "test", "arbitrary.txt": "unmatched",
            }
            for name, content in files.items():
                p = resources / name
                provider.mkdir(p.parent, parents=True, exist_ok=True)
                provider.write_text(p, content)
            provider.write_text(source / "CMakeLists.txt",
                "cmake_minimum_required(VERSION 3.24)\nproject(ResourceInstall NONE)\n"
                "include(\"${IILD_PROJECT_ROOT}/cmake/IildInstallResources.cmake\")\n"
                "iild_install_resources(\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures\" resources)\n")
            subprocess.run(["cmake", "-S", str(source), "-B", str(target),
                            f"-DIILD_PROJECT_ROOT={ROOT}"], check=True, capture_output=True, text=True)
            subprocess.run(["cmake", "--install", str(target), "--component", "Runtime",
                            "--prefix", str(stage)], check=True, capture_output=True, text=True)
            installed = stage / "resources"
            actual = {p.relative_to(installed).as_posix() for p in installed.rglob("*") if p.is_file()}
            self.assertEqual(actual, {"vae/model.safetensors", "vae/config.json", "vae/NOTICE.md",
                                      "text_encoders/qwen3vl-4b/config.json",
                                      "text_encoders/qwen3vl-4b/tokenizer_config.json",
                                      "text_encoders/qwen3vl-4b/vocab.json", "text_encoders/qwen3vl-4b/merges.txt",
                                      "text_encoders/qwen35-9b/config.json", "text_encoders/qwen35-9b/tokenizer_config.json",
                                      "text_encoders/qwen35-9b/vocab.json", "text_encoders/qwen35-9b/merges.txt",
                                      "text_encoders/flux1-clip/vocab.json", "text_encoders/flux1-clip/merges.txt",
                                      "text_encoders/flux1-clip/tokenizer_config.json", "text_encoders/flux1-clip/special_tokens_map.json",
                                      "text_encoders/flux1-t5/tokenizer.json", "text_encoders/flux1-t5/tokenizer_config.json",
                                      "text_encoders/flux1-t5/special_tokens_map.json",
                                      "text_encoders/flux1-clip-config/config.json", "text_encoders/flux1-t5-config/config.json"})


if __name__ == "__main__":
    unittest.main()
