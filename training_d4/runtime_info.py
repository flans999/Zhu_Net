import argparse
import json
import platform
import importlib.metadata
import torch
from contract import atomic_json


def information():
    arch = getattr(torch._C, '_cuda_getArchFlags', lambda: '')() or ''
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'machine': platform.machine(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'compiled_cuda_architectures': arch,
            'numpy': importlib.metadata.version('numpy'), 'pillow': importlib.metadata.version('pillow'),
            'cuda_available': torch.cuda.is_available()}


def gpu_smoke():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; GPU acceptance has not passed')
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(f'Expected RTX5090 sm_120, received {capability}')
    if 'sm_120' not in (getattr(torch._C, '_cuda_getArchFlags', lambda: '')() or ''):
        raise RuntimeError('This PyTorch binary does not declare sm_120')
    x = torch.randn(1024, 1024, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    if not torch.isfinite(y).all():
        raise RuntimeError('Non-finite GPU smoke output')
    print('GPU_SMOKE_PASS', list(y.shape), flush=True)
    return {'name': torch.cuda.get_device_name(), 'capability': list(capability)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--require-build', action='store_true')
    args = p.parse_args()
    result = information()
    if args.require_build:
        assert result['torch'].startswith('2.7.1'), result
        assert result['cuda_runtime'] == '12.8', result
        assert 'sm_120' in result['compiled_cuda_architectures'], result
    if args.gpu:
        result['gpu_smoke'] = gpu_smoke()
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        atomic_json(args.output, result)
