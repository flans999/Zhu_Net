"""Verify publication files and, optionally, the saved local Docker reports."""
import argparse
import csv
import hashlib
import math
import tarfile
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args, input_text=None):
    return subprocess.run(
        ['git', *args], cwd=ROOT, input=input_text, text=True,
        encoding='utf-8', capture_output=True, check=True,
    ).stdout


def publication_files():
    candidates = set(filter(None, git('ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0')))
    docs = ('README.md', 'docs/tasks.md', 'docs/datasets.md', 'docs/results.md', 'docs/d4-training.md')
    for name in docs:
        text = (ROOT / name).read_text(encoding='utf-8')
        assert 'PLACEHOLDER' not in text and 'DOCKER_RESULTS' not in text, name
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith('#') or '://' in target:
                continue
            path = (ROOT / name).parent / target.split('#')[0]
            assert path.is_file(), (name, target)
            assert path.resolve().relative_to(ROOT).as_posix() in candidates, (name, target, 'excluded from Git')
    tasks = (ROOT / 'docs/tasks.md').read_text(encoding='utf-8')
    assert len(re.findall(r'^## \d+\.', tasks, re.M)) == 10
    assert tasks.count('**Описание:**') == tasks.count('**Критерий готовности:**') == 10
    training = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'training').rglob('*') if p.is_file()}
    assert len(training) == 15, sorted(training)
    assert training <= candidates, sorted(training - candidates)
    required = {'Dockerfile.dataset', '.gitignore', '.dockerignore', 'requirements.txt',
                'scripts/describe_dataset.py', 'scripts/verify_publication.py',
                'tests/test_describe_dataset.py', 'tests/test_d4_data.py', '.gitattributes', *docs}
    assert required <= candidates, sorted(required - candidates)
    d4_source_files(candidates)
    excluded = [
        'data/image.png', 'dataset/image.jpg', 'datasets/manifest.csv',
        'nested/dataset/metadata.json', 'sample.pgm', 'model.pt', 'model.pth',
        'model.ckpt', 'model.safetensors', 'model.onnx', 'weights.h5',
        'cache.npy', 'cache.npz', 'zhuk2-20261107-results.tar.gz',
        'zhuk2-20261107-results.tar(1).gz', 'training_d4/checkpoints/best.pt',
        'training_d4/artifacts/metrics.json',
        'bundle.zip', 'nested/checkpoints/best.pt', 'nested/artifacts/metrics.json',
        'nested/outputs/result.json', 'results/test_metrics.json',
        'reports/docker/verification.json', '.env', '__pycache__/module.pyc',
    ]
    ignored = set(git('check-ignore', '--no-index', '-z', '--stdin', input_text='\0'.join(excluded) + '\0').rstrip('\0').split('\0'))
    assert set(excluded) <= ignored, sorted(set(excluded) - ignored)
    # Check existing candidates too, including anything explicitly staged in the future.
    if candidates:
        result = subprocess.run(
            ['git', 'check-ignore', '--no-index', '-z', '--stdin'], cwd=ROOT,
            input='\0'.join(sorted(candidates)) + '\0', text=True, encoding='utf-8',
            capture_output=True,
        )
        assert result.returncode in (0, 1), result.stderr
        assert not result.stdout.strip(), ('ignored files in commit candidates', result.stdout)
    print('PASS: UTF-8 docs, publication links, 10 issue headings, 10 descriptions, 10 criteria')
    print('PASS: 15 baseline training files; D4 source, configs, Dockerfiles and SRM filters included')
    print('PASS: datasets, weights, archives and generated reports excluded by Git')
    print(f'COMMIT CANDIDATES ({len(candidates)}):')
    for name in sorted(candidates):
        print(name)


def local_reports():
    verification = json.loads((ROOT / 'reports/docker/verification.json').read_text(encoding='utf-8'))
    assert len(verification['runs']) == 2
    assert {r['dataset'] for r in verification['runs']} == {'baseline', 'expanded'}
    assert all(r['container_exit_code'] == 0 for r in verification['runs'])
    for name in ('baseline', 'expanded'):
        folder = ROOT / 'reports/docker' / (name + '-20')
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        with (folder / 'images.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 40 and summary['inspected']['pairs'] == 20
        assert summary['inspected']['images'] == 40
        assert summary['complete'] is False and summary['healthy'] is True
        assert all(
            (r['format'], r['pgm_variant'], r['mode'], r['width'], r['height'], r['status'])
            == ('PGM', 'P5', 'L', '256', '256', 'ok') for r in rows
        )
        with (ROOT / 'reports' / name / 'trial-20/images.csv').open(encoding='utf-8', newline='') as stream:
            host_rows = list(csv.DictReader(stream))
        assert [(r['pair_id'], r['label'], r['sha256']) for r in rows] == [
            (r['pair_id'], r['label'], r['sha256']) for r in host_rows
        ]
        run = next(r for r in verification['runs'] if r['dataset'] == name)
        assert run['pairs'] == 20 and run['images'] == run['csv_rows'] == 40
        assert run['complete'] is False and run['healthy'] is True
        assert run['summary_path'] == f'reports/docker/{name}-20/summary.json'
        assert run['images_path'] == f'reports/docker/{name}-20/images.csv'
        print(f'PASS: {name}: 20 pairs, 40 images, 40 CSV rows, complete=false, healthy=true, host hashes match')
    print('PASS: verification.json: 2 successful container runs')



def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def d4_source_files(candidates):
    root = ROOT / 'training_d4'
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['source_files'].items():
        assert file_sha256(root / name) == expected, ('D4 source hash', name)
    expected = set(manifest['source_files']) | {'.dockerignore', 'SOURCE_MANIFEST.json', 'config/full-archive.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual == expected, ('D4 file list', sorted(actual ^ expected))
    assert {'training_d4/' + name for name in actual} <= candidates
    full = json.loads((root / 'config/full.json').read_text(encoding='utf-8'))
    archived = json.loads((root / 'config/full-archive.json').read_text(encoding='utf-8'))
    assert full['evaluate_test'] is False and archived['evaluate_test'] is True
    assert {k: v for k, v in full.items() if k != 'evaluate_test'} == {
        k: v for k, v in archived.items() if k != 'evaluate_test'
    }
    assert json.loads((root / 'config/expanded.json').read_text(encoding='utf-8')) == full
    print(f'PASS: D4 source manifest: {len(manifest["source_files"])} unchanged source files; archive config differs only in evaluate_test')


def external_d4_source(path):
    manifest = json.loads((ROOT / 'training_d4/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}
    assert actual == set(manifest['source_files']), ('External D4 source file list', sorted(actual ^ set(manifest['source_files'])))
    for name, expected in manifest['source_files'].items():
        assert file_sha256(path / name) == expected, ('External D4 source hash', name)
    print(f'PASS: {len(actual)} external D4 source files match the published copy and manifest')


def archived_results(path, dataset_metadata=None):
    manifest = json.loads((ROOT / 'training_d4/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    assert file_sha256(path) == manifest['archive']['sha256'], 'Archive SHA-256 differs'
    values = {}
    with tarfile.open(path, 'r:gz') as archive:
        for name, expected in manifest['archive']['members'].items():
            matches = [m for m in archive.getmembers() if m.name.removeprefix('./') == name]
            assert len(matches) == 1 and matches[0].isfile(), name
            assert matches[0].size < 2 * 1024 * 1024, ('Unexpected JSON size', name)
            raw = archive.extractfile(matches[0]).read()
            assert hashlib.sha256(raw).hexdigest() == expected, ('Archive member hash', name)
            values[name] = json.loads(raw.decode('utf-8'))
    metrics = values['outputs/train/metrics.json']
    result = values['outputs/train/training_result.json']
    used = values['outputs/train/config_used.json']
    history = metrics['history']
    assert [row['epoch'] for row in history] == list(range(1, 401))
    best = max(history, key=lambda row: row['validation']['accuracy'])
    assert best['epoch'] == 368
    assert math.isclose(best['validation']['accuracy'], 0.9064064064064065, rel_tol=0, abs_tol=1e-15)
    assert metrics['best_validation_accuracy'] == best['validation']['accuracy']
    assert best['validation']['images'] == 1998
    assert math.isclose(metrics['test']['accuracy'], 0.9000600240096038, rel_tol=0, abs_tol=1e-15)
    assert metrics['test']['images'] == 9996
    assert metrics['test']['confusion_matrix'] == [[4500, 498], [501, 4497]]
    assert result['status'] == 'success' and result['success'] is True and result['training_completed'] is True
    assert result['completed_epochs'] == 400
    assert metrics['mode'] == result['phase'] == used['mode'] == 'expanded-d4'
    assert result['global_step'] == history[-1]['global_step'] == 5598800
    config = json.loads((ROOT / 'training_d4/config/full-archive.json').read_text(encoding='utf-8'))
    assert used['config'] == config
    assert used['augmentation'] == 'd4-exhaustive-v1'
    assert used['physical_train_pairs'] == 13997
    assert used['virtual_train_pairs'] == 8 * used['physical_train_pairs'] == 111976
    assert used['batch_pairs_actual'] == 8
    steps_per_epoch = math.ceil(used['virtual_train_pairs'] / used['batch_pairs_actual'])
    for row in history:
        assert row['global_step'] == row['epoch'] * steps_per_epoch
        expected_lr = config['learning_rate'] * config['gamma'] ** sum(
            row['epoch'] >= milestone for milestone in config['milestones']
        )
        assert math.isclose(row['learning_rate'], expected_lr, rel_tol=1e-12)
    for entry in [row['validation'] for row in history] + [metrics['test']]:
        (tn, fp), (fn, tp) = entry['confusion_matrix']
        assert tn + fp + fn + tp == entry['images']
        for key, expected in {
            'accuracy': (tn + tp) / entry['images'],
            'precision_stego': tp / (tp + fp),
            'recall_stego': tp / (tp + fn),
            'f1_stego': 2 * tp / (2 * tp + fp + fn),
        }.items():
            assert math.isclose(entry[key], expected, rel_tol=0, abs_tol=1e-15), (key, entry)
    if dataset_metadata is not None:
        assert file_sha256(dataset_metadata) == manifest['local_dataset_metadata_sha256'], 'Local metadata SHA-256 differs'
        metadata = json.loads(dataset_metadata.read_text(encoding='utf-8'))
        assert {k: v for k, v in metadata.items() if k != 'files'} == used['dataset_metadata']
        print('PASS: local dataset metadata hash and archived metadata without files')
    doc = (ROOT / 'docs/results.md').read_text(encoding='utf-8')
    for entry in (best['validation'], metrics['test']):
        for key in ('accuracy', 'loss', 'precision_stego', 'recall_stego', 'f1_stego'):
            assert str(entry[key]) in doc, ('Missing documented metric', key)
    data = {key: config[key] for key in [
        'seed', 'learning_rate', 'momentum', 'weight_decay', 'milestones', 'gamma', 'expected_pair_counts'
    ]}
    data.update({
        'augmentation': used['augmentation'],
        'data_sha256': file_sha256(ROOT / 'training_d4/data.py'),
        'training_sha256': file_sha256(ROOT / 'training_d4/train.py'),
        'model_sha256': file_sha256(ROOT / 'training_d4/model.py'),
        'dataset': manifest['local_dataset_metadata_sha256'], 'smoke': False,
    })
    signature = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    assert signature == manifest['candidate_signature']
    assert used['signature'] == manifest['archive']['signature']
    print('PASS: archive: validation=90.6406%, epoch=368; test=90.0060%, images=9996; completed epochs=400')
    print('PASS: archive JSON hashes, documented metrics, D4 counts and full-archive.json')
    print(f'PASS: all 400 epochs: {steps_per_epoch} steps per epoch, learning rates and confusion-derived metrics')
    if signature != used['signature']:
        print('LIMIT: local source/dataset signature differs from archive; exact historical code identity is unconfirmed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-reports', action='store_true', help='also require and check saved reports excluded from Git')
    parser.add_argument('--results-archive', type=Path, help='also verify saved expanded D4 JSON inside an external archive')
    parser.add_argument('--d4-source', type=Path, help='also compare the external D4 source directory with its manifest')
    parser.add_argument('--dataset-metadata', type=Path, help='also compare local metadata.json with the archive (requires --results-archive)')
    args = parser.parse_args()
    if args.dataset_metadata and not args.results_archive:
        parser.error('--dataset-metadata requires --results-archive')
    publication_files()
    if args.local_reports:
        local_reports()
    if args.results_archive:
        archived_results(args.results_archive, args.dataset_metadata)
    if args.d4_source:
        external_d4_source(args.d4_source)


if __name__ == '__main__':
    main()
