import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .cuda_environment import inspect_environment


class CudaEnvironmentTests(unittest.TestCase):
    def test_cpu_diagnostics_from_real_environment(self):
        report = inspect_environment(full=False)
        self.assertIn('status', report)
        self.assertIn('torch', report)
        self.assertIn('environment', report)
        self.assertIsInstance(report['diagnosis'], list)

    def test_real_environment_endpoint_request(self):
        from .app import environment
        body = environment()
        self.assertIn(body['status'], {'cpu_wheel', 'available', 'no_gpu', 'masked', 'driver_or_gpu_missing', 'cuda_error', 'torch_missing', 'torch_broken'})
        self.assertIn('interpreter', body)

    def test_cpu_wheel_is_distinguished(self):
        torch = SimpleNamespace(__version__='2.0.0+cpu', version=SimpleNamespace(cuda=None), cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
        report = inspect_environment(full=False, torch_module=torch, environ={})
        self.assertEqual(report['status'], 'cpu_wheel')
        self.assertEqual(report['diagnosis'][0]['code'], 'CPU_WHEEL')

    def test_masking_is_reported(self):
        torch = SimpleNamespace(__version__='2.0.0+cu128', version=SimpleNamespace(cuda='12.8'), cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
        report = inspect_environment(full=False, torch_module=torch, environ={'CUDA_VISIBLE_DEVICES': '-1'})
        self.assertEqual(report['status'], 'masked')
        self.assertEqual(report['diagnosis'][0]['code'], 'CUDA_VISIBLE_DEVICES_MASKED')

    def test_empty_visibility_mask_is_reported(self):
        torch = SimpleNamespace(__version__='2.0.0+cu128', version=SimpleNamespace(cuda='12.8'), cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
        report = inspect_environment(full=False, torch_module=torch, environ={'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(report['status'], 'masked')
        self.assertEqual(report['environment']['CUDA_VISIBLE_DEVICES'], '')

    def test_torch_loader_exception_is_broken_not_missing(self):
        report = inspect_environment(full=False, torch_module=RuntimeError('DLL load failed'), environ={})
        self.assertEqual(report['status'], 'torch_broken')
        self.assertEqual(report['diagnosis'][0]['code'], 'TORCH_BROKEN')

    def test_invalid_smi_rows_and_nonzero_status_are_not_ready(self):
        torch = SimpleNamespace(__version__='2.0.0+cu128', version=SimpleNamespace(cuda='12.8'), cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda index: 'GPU'))
        report = inspect_environment(full=True, torch_module=torch, nvidia_smi=lambda: ('GPU, bad, 0', '', 1), environ={})
        self.assertEqual(report['status'], 'available')
        self.assertFalse(report['gpu']['present'])

    def test_gpu_and_driver_fields_are_validated(self):
        torch = SimpleNamespace(__version__='2.0.0+cu128', version=SimpleNamespace(cuda='12.8'), cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
        report = inspect_environment(full=True, torch_module=torch, nvidia_smi=lambda: ('NVIDIA RTX, 591.74, 16384\nmalformed', '', 0), environ={})
        self.assertTrue(report['gpu']['present'])
        self.assertEqual(report['gpu']['driver'], '591.74')
        self.assertEqual(report['gpu']['devices'][0]['memoryMiB'], '16384')
        self.assertEqual(report['diagnosis'][0]['code'], 'CUDA_UNAVAILABLE')

    def test_torch_import_failure_is_safe(self):
        report = inspect_environment(full=True, torch_module=None, nvidia_smi=lambda: ('', 'not found', 127), environ={})
        self.assertEqual(report['status'], 'torch_missing')
        self.assertEqual(report['diagnosis'][0]['code'], 'TORCH_MISSING')

    def test_driver_inspection_does_not_override_usable_cuda(self):
        torch = SimpleNamespace(__version__='2.0.0+cu128', version=SimpleNamespace(cuda='12.8'), cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda index: 'GPU'))
        for output, code in (('', 0), ('GPU, 591.74, 16384', 1), ('GPU, NaN, Infinity', 0)):
            report = inspect_environment(full=True, torch_module=torch, nvidia_smi=lambda: (output, '', code), environ={})
            self.assertEqual(report['status'], 'available')
            self.assertFalse(report['gpu']['present'])

    def test_failed_cuda_initialization_is_explicit(self):
        def unavailable(): raise RuntimeError('driver initialization failed')
        torch = SimpleNamespace(__version__='2.0.0+cu128', version=SimpleNamespace(cuda='12.8'), cuda=SimpleNamespace(is_available=unavailable))
        report = inspect_environment(full=False, torch_module=torch, environ={})
        self.assertEqual(report['status'], 'cuda_error')
        self.assertIn('driver initialization failed', report['cuda']['error'])

    def test_health_does_not_launch_driver_process(self):
        from .app import health
        with patch('backend.cuda_environment._run_nvidia_smi', side_effect=AssertionError('health launched driver process')):
            self.assertIn('cuda', health())

    def test_training_lock_blocks_environment_and_smoke_and_is_released(self):
        from fastapi import HTTPException
        from .app import environment, environment_smoke, training_lock
        self.assertTrue(training_lock.acquire(blocking=False))
        try:
            for endpoint in (environment, environment_smoke):
                with self.assertRaises(HTTPException) as error: endpoint()
                self.assertEqual(error.exception.status_code, 409)
        finally: training_lock.release()
        with patch('backend.app.inspect_environment', return_value={'cuda': {'available': False}}):
            with self.assertRaises(HTTPException) as error: environment_smoke()
            self.assertEqual(error.exception.status_code, 422)
        self.assertFalse(training_lock.locked())

    def test_missing_internal_torch_dependency_is_reported_as_broken(self):
        from .cuda_environment import _torch_module
        import builtins
        original = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == 'torch': raise ModuleNotFoundError('missing internal dependency', name='torch_dependency')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=missing):
            self.assertIsInstance(_torch_module(), ModuleNotFoundError)
            self.assertEqual(inspect_environment(full=False)['status'], 'torch_broken')


if __name__ == '__main__':
    unittest.main()
