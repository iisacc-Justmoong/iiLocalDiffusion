#!/usr/bin/env python3
"""Inspect registered test workspaces without starting apps or model runtimes."""
from pathlib import Path
import os
import subprocess
import sys
import unittest
import resource_fixture
import resource_runner

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(sys.argv.pop(1)) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else ROOT/'build'


class BuildFixtureLayoutTests(unittest.TestCase):
    def test_cuda_license_download_is_a_resource_input(self):
        source = (ROOT/'cmake/IildMlx.cmake').read_text(encoding='utf-8')
        self.assertIn('set(cccl_license "${CMAKE_CURRENT_SOURCE_DIR}/resources/dependencies/cccl-3.1.3-LICENSE")', source)
        self.assertNotIn('set(cccl_license "${CMAKE_CURRENT_BINARY_DIR}/dependencies/', source)

    def test_cuda_license_preserves_mismatched_and_linked_cache_entries(self):
        source = (ROOT/'cmake/IildMlx.cmake').read_text(encoding='utf-8')
        start = source.index('    set(cccl_license_hash ')
        end = source.index('    install(FILES "${cccl_license}"', start)
        block = source[start:end]
        with resource_fixture.TemporaryDirectory(prefix='cccl-license-policy-') as temporary:
            owner = Path(temporary)
            provider = resource_runner._provider()
            license_path = owner/'LICENSE'
            driver = owner/'verify.cmake'
            provider.write_text(driver, f'set(cccl_license "{license_path.as_posix()}")\n{block}')
            provider.write_text(license_path, 'preserve mismatched content')
            environment = dict(os.environ, TMPDIR=str(owner))
            mismatch = subprocess.run(['cmake', '-P', str(driver)], env=environment,
                                      capture_output=True, text=True, check=False)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn('does not match the pinned checksum', mismatch.stderr)
            self.assertEqual(license_path.read_text(), 'preserve mismatched content')
            if os.name != 'nt':
                license_path.unlink()
                other = owner/'other'
                provider.write_text(other, 'preserve link target')
                license_path.symlink_to(other)
                linked = subprocess.run(['cmake', '-P', str(driver)], env=environment,
                                        capture_output=True, text=True, check=False)
                self.assertNotEqual(linked.returncode, 0)
                self.assertIn('cannot be a symlink', linked.stderr)
                self.assertEqual(other.read_text(), 'preserve link target')

    def test_registered_install_and_cli_fixtures_use_owned_workspaces(self):
        lines = (BUILD/'CTestTestfile.cmake').read_text(encoding='utf-8').splitlines()
        expected = {
            'iiLocalDiffusion.ModelPackagingComponentInstall': 'STAGE_DIRECTORY=@WORK@/',
            'iiLocalDiffusion.InstallScript': 'IILD_INSTALL_SCRIPT_TEST_DIRECTORY=@WORK@/',
            'CliContractTests': 'WORK_DIRECTORY=@WORK@/',
            'InstallConsumerTests': 'WORK_DIRECTORY=@WORK@/',
        }
        cache = (BUILD/'CMakeCache.txt').read_text(encoding='utf-8')
        if 'IILD_ENABLE_NATIVE_DIFFUSION:BOOL=ON' in cache:
            expected.update({
                'NativeIPResourceTests': '@WORK@/native-ip-resource-contract',
                'NativeMappedMetalStorage': '@WORK@/native-mapped-shared-contract',
                'NativeMappedPrivateStorage': '@WORK@/native-mapped-private-contract',
            })
        for name, argument in expected.items():
            with self.subTest(test=name):
                matching = [line for line in lines if line.startswith(f'add_test("{name}" ')]
                self.assertEqual(len(matching), 1)
                self.assertIn('tests/resource_runner.py', matching[0])
                self.assertIn(argument, matching[0])
                if name == 'InstallConsumerTests':
                    self.assertIn('CONSUMER_BUILD_DIRECTORY=@BUILD_WORK@', matching[0])
        if 'IILD_ENABLE_NATIVE_DIFFUSION:BOOL=ON' in cache:
            private = [line for line in lines if line.startswith('set_tests_properties("NativeMappedPrivateStorage" ')]
            self.assertEqual(len(private), 1)
            self.assertIn('GGML_METAL_SHARED_BUFFERS_DISABLE=1', private[0])

    def test_existing_test_workspaces_are_never_removed(self):
        with resource_fixture.TemporaryDirectory(prefix='fixture-preservation-') as temporary:
            owner = Path(temporary)
            provider = resource_runner._provider()
            existing = owner/'existing'
            provider.mkdir(existing)
            marker = existing/'user.txt'
            provider.write_text(marker, 'preserve')
            cases = {
                'ModelPackagingInstallTests.cmake': {
                    'BUILD_DIRECTORY': str(BUILD), 'STAGE_DIRECTORY': str(existing), 'BIN_DIRECTORY': 'bin'},
                'CliContractTests.cmake': {
                    'CLI': 'unused', 'FIXTURE': 'unused', 'SDXL_FIXTURE': 'unused',
                    'FLUX_FIXTURE': 'unused', 'WORK_DIRECTORY': str(existing)},
                'InstallConsumerTests.cmake': {
                    'BUILD_DIRECTORY': str(BUILD), 'WORK_DIRECTORY': str(existing),
                    'CONSUMER_BUILD_DIRECTORY': str(owner/'compiled'), 'GENERATOR': 'Ninja',
                    'CONFIG': 'Release', 'EXECUTABLE_SUFFIX': '', 'FIXTURE': 'unused',
                    'SDXL_FIXTURE': 'unused', 'FLUX_FIXTURE': 'unused', 'BUILD_TOOLS': 'OFF',
                    'PYTHON_REFERENCE_ENABLED': 'OFF', 'BIN_DIRECTORY': 'bin',
                    'DOC_DIRECTORY': 'docs', 'REFERENCE_DIRECTORY': 'reference'},
            }
            for script, arguments in cases.items():
                with self.subTest(script=script):
                    command = ['cmake', *[f'-D{name}={value}' for name, value in arguments.items()],
                               '-P', str(ROOT/'tests'/script)]
                    result = subprocess.run(command, env=dict(os.environ, TMPDIR=str(owner)),
                                            capture_output=True, text=True, check=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('already exists', result.stderr)
                    self.assertEqual(marker.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
