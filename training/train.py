"""Noninteractive paired Zhu-Net training for the K-Lab contract."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from contract import atomic_json, validate_index
from data import PairDataset, validate_pixels
from model import Zhu_Net
from runtime_info import information, gpu_smoke

STOP = False


class StopRequested(Exception):
    pass


def stop_handler(signum, frame):
    global STOP
    STOP = True
    print(f'STOP_REQUESTED signal={signum}; checkpoint at next safe boundary', flush=True)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def loader(dataset, order, batch_pairs, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, order), batch_size=batch_pairs, shuffle=False,
                      num_workers=workers, drop_last=False, generator=generator)


def flatten(batch, device):
    pairs = batch.shape[0]
    x = batch.reshape(pairs * 2, 1, 256, 256).to(device)
    y = torch.tensor([0, 1], device=device).repeat(pairs)
    return x, y


def evaluate(model, dataset, device, batch_pairs, workers, max_batches=None):
    model.eval()
    total_loss = total = correct = 0
    confusion = torch.zeros(2, 2, dtype=torch.int64)
    with torch.no_grad():
        for index, batch in enumerate(loader(dataset, list(range(len(dataset))), batch_pairs, workers, 0)):
            if STOP:
                raise StopRequested()
            x, y = flatten(batch, device)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite evaluation logits')
            total_loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
            predicted = logits.argmax(1)
            correct += int((predicted == y).sum().item())
            total += y.numel()
            confusion += torch.bincount((y * 2 + predicted).cpu(), minlength=4).reshape(2, 2)
            if max_batches and index + 1 >= max_batches:
                break
    tn, fp, fn, tp = confusion.flatten().tolist()
    return {'loss': total_loss / total, 'accuracy': correct / total, 'images': total,
            'confusion_matrix': confusion.tolist(),
            'precision_stego': tp / (tp + fp) if tp + fp else 0.0,
            'recall_stego': tp / (tp + fn) if tp + fn else 0.0,
            'f1_stego': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def select_batch(model, dataset, device, requested, config):
    if device.type == 'cpu':
        return requested
    free, total = torch.cuda.mem_get_info()
    budget = min(config['max_vram_gb'] * 1024**3, free * 0.85)
    torch.cuda.set_per_process_memory_fraction(min(budget / total, 0.95), device)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    random_state = rng_state()
    size = requested
    while size >= 1:
        try:
            model.train()
            batch = torch.stack([dataset[i % len(dataset)] for i in range(size)])
            x, y = flatten(batch, device)
            nn.functional.cross_entropy(model(x), y).backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f'GPU_BATCH_PROBE_PASS batch_pairs={size} allocated_peak_gb={peak:.3f}', flush=True)
            del batch, x, y
            break
        except torch.cuda.OutOfMemoryError:
            print(f'GPU_BATCH_PROBE_OOM batch_pairs={size}; reducing', flush=True)
            size //= 2
        finally:
            model.zero_grad(set_to_none=True)
            model.load_state_dict(state)
            restore_rng(random_state)
            torch.cuda.empty_cache()
    if size < 1:
        raise RuntimeError('Model cannot fit even one cover/stego pair; request more VRAM')
    return size


def run(args, output):
    global STOP
    STOP = False
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    for key, value in {'HOME': str(scratch), 'XDG_CACHE_HOME': str(scratch / 'cache'),
                       'TORCH_HOME': str(scratch / 'torch')}.items():
        os.environ[key] = value
    torch.set_num_threads(config.get('cpu_threads', 4))
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    smoke = args.smoke_steps > 0
    pairs, meta, dataset_fingerprint = validate_index(args.dataset_dir, config['expected_pair_counts'], allow_fixture=smoke)
    validate_pixels(args.dataset_dir, pairs)
    datasets = {split: PairDataset(args.dataset_dir, rows) for split, rows in pairs.items()}
    if args.validate_only:
        atomic_json(output / 'training_result.json', {'status': 'validation_pass', 'success': True,
                    'training_completed': False, 'dataset_sha256': dataset_fingerprint})
        print('DATASET_VALIDATION_PASS', flush=True)
        return 0
    device = torch.device(args.device)
    gpu = gpu_smoke() if device.type == 'cuda' else None
    runtime = information()
    atomic_json(output / 'runtime_info.json', runtime)
    model = Zhu_Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=config['momentum'], weight_decay=config['weight_decay'])
    # Every declared trainable layer, including SRM kernels, is optimized.
    signature_data = {key: config[key] for key in ['seed', 'learning_rate', 'momentum', 'weight_decay', 'milestones', 'gamma', 'expected_pair_counts']}
    signature_data.update({'dataset': dataset_fingerprint, 'model_sha256': hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(), 'smoke': smoke})
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    last = checkpoints / 'last.pt'
    best = checkpoints / 'best.pt'
    state = {'epoch': 0, 'offset': 0, 'global_step': 0, 'history': [], 'best_accuracy': -1.0,
             'epoch_loss_sum': 0.0, 'epoch_correct': 0, 'epoch_images': 0}
    if args.resume == 'auto' and last.exists():
        # Only load checkpoints created by this job in its own writable output directory.
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('Checkpoint does not match dataset, model, optimizer settings or smoke mode')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']
        restore_rng(saved['rng'])
        print(f'RESUME_PASS global_step={state["global_step"]} epoch={state["epoch"]} pair_offset={state["offset"]}', flush=True)
    requested = 1 if smoke else config['batch_pairs']
    batch_pairs = select_batch(model, datasets['train'], device, requested, config)
    workers = 0 if smoke else config.get('num_workers', 0)
    atomic_json(output / 'config_used.json', {'config': config, 'batch_pairs_actual': batch_pairs,
                 'runtime': runtime, 'gpu': gpu, 'mode': 'smoke' if smoke else config['phase'],
                 'signature': signature, 'dataset_metadata': {k: v for k, v in meta.items() if k != 'files'}})
    started = time.monotonic()
    invocation_steps = 0
    last_save = started

    def checkpoint():
        nonlocal last_save
        atomic_torch(last, {'signature': signature, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'state': copy.deepcopy(state), 'rng': rng_state(), 'batch_pairs': batch_pairs,
                     'scheduler': {'kind': 'explicit_multistep', 'epoch': state['epoch'],
                                   'milestones': config['milestones'], 'gamma': config['gamma'],
                                   'current_lr': optimizer.param_groups[0]['lr']}})
        last_save = time.monotonic()
        print(f'CHECKPOINT_SAVED global_step={state["global_step"]}', flush=True)

    def export_model(path):
        atomic_torch(path, {'model_state_dict': model.state_dict(), 'model': 'author-released-Zhu-Net',
                     'input_shape': [None, 1, 256, 256], 'input_scale': 'uint8 / 255',
                     'classes': {'0': 'cover', '1': 'stego'}, 'signature': signature,
                     'completed_epochs': len(state['history']), 'global_step': state['global_step']})

    checkpoint()
    final_evaluation = False
    try:
        while state['epoch'] < config['epochs']:
            epoch = state['epoch']
            lr = config['learning_rate'] * config['gamma'] ** sum(epoch + 1 >= m for m in config['milestones'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            order = torch.randperm(len(datasets['train']), generator=torch.Generator().manual_seed(config['seed'] + epoch)).tolist()
            remaining = order[state['offset']:]
            epoch_start = time.monotonic()
            model.train()
            for batch in loader(datasets['train'], remaining, batch_pairs, workers, config['seed'] + epoch):
                if STOP:
                    raise StopRequested()
                n_pairs = batch.shape[0]
                x, y = flatten(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                state['offset'] += n_pairs
                state['global_step'] += 1
                state['epoch_loss_sum'] += loss.item() * y.numel()
                state['epoch_correct'] += int((logits.argmax(1) == y).sum().item())
                state['epoch_images'] += y.numel()
                invocation_steps += 1
                if state['global_step'] % config['log_every_steps'] == 0 or smoke:
                    print(f'STEP epoch={epoch+1} global_step={state["global_step"]} pairs={state["offset"]}/{len(order)} loss={loss.item():.6f} lr={lr}', flush=True)
                if STOP:
                    raise StopRequested()
                if time.monotonic() - last_save >= config['checkpoint_seconds']:
                    checkpoint()
                if smoke and invocation_steps >= args.smoke_steps:
                    checkpoint()
                    validation = evaluate(model, datasets['validation'], device, 1, 0, max_batches=2)
                    export_model(output / 'model.pt')
                    atomic_json(output / 'metrics.json', {'mode': 'smoke', 'global_step': state['global_step'], 'smoke_validation': validation})
                    atomic_json(output / 'training_result.json', {'status': 'smoke_pass', 'success': True,
                                'training_completed': False, 'completed_epochs': len(state['history']),
                                'global_step': state['global_step'], 'invocation_steps': invocation_steps,
                                'device': str(device), 'gpu_acceptance': 'PASS' if gpu else 'NOT TESTED'})
                    print('TRAINING_SMOKE_PASS', flush=True)
                    return 0
            validation = evaluate(model, datasets['validation'], device, batch_pairs, workers)
            entry = {'epoch': epoch + 1, 'global_step': state['global_step'], 'learning_rate': lr,
                     'train_loss': state['epoch_loss_sum'] / state['epoch_images'],
                     'train_accuracy': state['epoch_correct'] / state['epoch_images'],
                     'validation': validation, 'seconds_this_invocation': time.monotonic() - epoch_start}
            state['history'].append(entry)
            improved = validation['accuracy'] > state['best_accuracy']
            if improved:
                state['best_accuracy'] = validation['accuracy']
                export_model(best)
            state.update(epoch=epoch+1, offset=0, epoch_loss_sum=0.0, epoch_correct=0, epoch_images=0)
            checkpoint()
            atomic_json(output / 'metrics.json', {'mode': config['phase'], 'history': state['history'], 'test': None})
            print('EPOCH_FINISHED ' + json.dumps(entry), flush=True)
        if STOP:
            raise StopRequested()
        final_evaluation = True
        best_state = torch.load(best, map_location=device, weights_only=True)
        model.load_state_dict(best_state['model_state_dict'])
        test = evaluate(model, datasets['test'], device, batch_pairs, workers) if config.get('evaluate_test') else None
        # Copy best validation-selected artifact; keep last.pt as resumable final training state.
        atomic_torch(output / 'model.pt', best_state)
        atomic_json(output / 'metrics.json', {'mode': 'smoke' if smoke else config['phase'], 'history': state['history'],
                    'best_validation_accuracy': state['best_accuracy'], 'test': test})
        atomic_json(output / 'training_result.json', {'status': 'smoke_pass' if smoke else 'success', 'success': True,
                    'training_completed': not smoke, 'phase': 'smoke' if smoke else config['phase'], 'completed_epochs': len(state['history']),
                    'global_step': state['global_step'], 'invocation_seconds': time.monotonic()-started,
                    'batch_pairs': batch_pairs, 'gpu_acceptance': 'PASS' if gpu else 'NOT TESTED',
                    'cuda_peak_allocated_gb': torch.cuda.max_memory_allocated()/1024**3 if gpu else None,
                    'cuda_peak_reserved_gb': torch.cuda.max_memory_reserved()/1024**3 if gpu else None})
        print('TRAINING_SMOKE_PASS' if smoke else 'TRAINING_SUCCESS', flush=True)
        return 0
    except StopRequested:
        if not final_evaluation:
            checkpoint()
        atomic_json(output / 'training_result.json', {'status': 'interrupted', 'success': False,
                    'training_completed': False, 'completed_epochs': len(state['history']),
                    'global_step': state['global_step'], 'resume_supported': True})
        print('RECOVERY_CHECKPOINT_SAVED', flush=True)
        return 143


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--scratch-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--resume', choices=['auto', 'never'], default='auto')
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume == 'never' and (output / 'checkpoints/last.pt').exists():
        p.error('Existing checkpoint found; use a new output directory or --resume auto')
    if args.smoke_steps < 0:
        p.error('--smoke-steps must be non-negative')
    try:
        return run(args, output)
    except Exception as exc:
        atomic_json(output / 'training_result.json', {'status': 'failed', 'success': False,
                    'training_completed': False, 'error': f'{type(exc).__name__}: {exc}',
                    'resume_checkpoint': 'checkpoints/last.pt' if (output / 'checkpoints/last.pt').exists() else None})
        raise


if __name__ == '__main__':
    sys.exit(main())
