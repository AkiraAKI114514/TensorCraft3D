"""Bounded, read-only diagnostics for the local PyTorch/CUDA environment."""
from __future__ import annotations
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping

ENVIRONMENT_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_PATH', 'CUDA_HOME', 'CUDA_PATH_V12_8', 'CUDA_PATH_V12_6', 'CUDA_PATH_V11_8')
PYTORCH_INSTALL_URL = 'https://pytorch.org/get-started/locally/'
NVIDIA_CUDA_URL = 'https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/'
_AUTO_TORCH = object()


def _torch_module() -> Any | Exception | None:
    try:
        import torch
        return torch
    except ModuleNotFoundError as error:
        return None if error.name == 'torch' else error
    except Exception as error:
        return error


def _run_nvidia_smi() -> tuple[str, str, int]:
    executable = shutil.which('nvidia-smi')
    if not executable:
        return '', 'nvidia-smi was not found', 127
    try:
        completed = subprocess.run([executable, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=False)
        return completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired:
        return '', 'nvidia-smi timed out', 124
    except OSError as error:
        return '', str(error), 1


def _parse_gpus(output: str) -> list[dict[str, str]]:
    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 3 or not fields[0]:
            continue
        if not re.fullmatch(r'\d+(?:\.\d+)+', fields[1]):
            continue
        try:
            memory = float(fields[2])
            if not math.isfinite(memory) or memory <= 0:
                continue
        except ValueError:
            continue
        devices.append({'name': fields[0][:160], 'driver': fields[1][:64], 'memoryMiB': fields[2][:32]})
    return devices


def inspect_environment(*, full: bool = True, torch_module: Any | None = _AUTO_TORCH, nvidia_smi: Callable[[], tuple[str, str, int]] | None = None, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return safe diagnostics without executing any request-provided command."""
    env = os.environ if environ is None else environ
    torch = _torch_module() if torch_module is _AUTO_TORCH else torch_module
    result: dict[str, Any] = {'ok': True, 'status': 'unavailable', 'torch': {'installed': False, 'version': None, 'cudaBuild': None}, 'cuda': {'available': False, 'deviceCount': 0, 'devices': [], 'error': None}, 'gpu': {'present': False, 'devices': [], 'driver': None}, 'interpreter': sys.executable, 'platform': platform.platform(), 'environment': {key: env[key] for key in ENVIRONMENT_KEYS if key in env}, 'diagnosis': [], 'setup': {'pytorch': PYTORCH_INSTALL_URL, 'nvidia': NVIDIA_CUDA_URL, 'note': '环境变量和系统 CUDA toolkit 仅供参考；PyTorch wheel 的 CUDA runtime 与驱动共同决定可用性。'}}
    if torch is None:
        result['status'] = 'torch_missing'
        result['diagnosis'].append({'code': 'TORCH_MISSING', 'level': 'error', 'message': '未找到可导入的 PyTorch。训练功能不可用，请在项目 .venv 中安装。'})
    elif isinstance(torch, Exception):
        result['status'] = 'torch_broken'
        result['torch']['error'] = str(torch)[:500]
        result['diagnosis'].append({'code': 'TORCH_BROKEN', 'level': 'error', 'message': f'PyTorch 找到但无法加载：{str(torch)[:400]}'})
    else:
        try:
            cuda_build = getattr(getattr(torch, 'version', None), 'cuda', None)
            result['torch'] = {'installed': True, 'version': str(getattr(torch, '__version__', 'unknown')), 'cudaBuild': str(cuda_build) if cuda_build else None}
            visible = result['environment'].get('CUDA_VISIBLE_DEVICES')
            if visible is not None and visible.strip() in ('', '-1'):
                result['status'] = 'masked'
                result['diagnosis'].append({'code': 'CUDA_VISIBLE_DEVICES_MASKED', 'level': 'warning', 'message': 'CUDA_VISIBLE_DEVICES 隐藏了 GPU；清除该变量后重启 Python 服务。'})
            try:
                available = bool(torch.cuda.is_available())
                count = int(torch.cuda.device_count()) if available else 0
                result['cuda'].update({'available': available, 'deviceCount': count})
                if available and count:
                    result['cuda']['devices'] = [{'name': str(torch.cuda.get_device_name(index))[:160], 'index': index} for index in range(min(count, 16))]
                    result['status'] = 'available'
                elif result['status'] != 'masked':
                    result['status'] = 'cpu_wheel' if not cuda_build else 'no_gpu'
                    result['diagnosis'].append({'code': 'CPU_WHEEL' if not cuda_build else 'CUDA_UNAVAILABLE', 'level': 'warning', 'message': '当前 PyTorch 是 CPU wheel（torch.version.cuda 为 None），CUDA_PATH 不会把它变成 CUDA wheel。' if not cuda_build else 'PyTorch 含 CUDA runtime，但当前没有可用 GPU。'})
            except Exception as error:
                result['status'] = 'cuda_error'
                result['cuda']['error'] = str(error)[:500]
                result['diagnosis'].append({'code': 'CUDA_INIT_ERROR', 'level': 'error', 'message': f'CUDA 初始化失败：{str(error)[:400]}'})
        except Exception as error:
            result['status'] = 'torch_broken'
            result['torch']['error'] = str(error)[:500]
            result['diagnosis'].append({'code': 'TORCH_BROKEN', 'level': 'error', 'message': f'PyTorch 可以找到但无法正常加载：{str(error)[:400]}'})
    if full:
        stdout, stderr, returncode = (nvidia_smi or _run_nvidia_smi)()
        devices = _parse_gpus(stdout) if returncode == 0 else []
        result['gpu'] = {'present': bool(devices), 'devices': devices, 'driver': devices[0]['driver'] if devices else None, 'nvidiaSmi': {'ok': bool(devices) and returncode == 0, 'returnCode': returncode}}
        if not devices and result['status'] == 'no_gpu':
            result['status'] = 'driver_or_gpu_missing'
            result['diagnosis'].append({'code': 'GPU_OR_DRIVER_MISSING', 'level': 'warning', 'message': 'PyTorch CUDA 不可用，nvidia-smi 也未确认 GPU/驱动；请检查硬件及官方驱动。'})
        elif devices and result['status'] == 'no_gpu':
            result['diagnosis'].append({'code': 'DRIVER_PRESENT_RUNTIME_UNAVAILABLE', 'level': 'warning', 'message': 'nvidia-smi 可见 GPU，但 PyTorch CUDA 不可用，通常是 CPU wheel、驱动/runtime 不匹配或环境变量问题。'})
        if not devices and stderr and result['status'] == 'cuda_error':
            result['cuda']['error'] = f"{result['cuda']['error']}; nvidia-smi: {stderr[:250]}"
    if not result['diagnosis']:
        result['diagnosis'].append({'code': 'CUDA_READY', 'level': 'info', 'message': 'PyTorch CUDA 已可用。'})
    return result
