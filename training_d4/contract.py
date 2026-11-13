"""Portable dataset and output contract; standard library only."""
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def safe_file(root, name):
    root = Path(root).resolve()
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(part in ('..', '.') for part in p.parts) or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe relative path: {name}')
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError(f'Links are not accepted: {name}')
    if not current.is_file() or not current.resolve().is_relative_to(root):
        raise ValueError(f'Missing or external file: {name}')
    return current


def read_pairs(root):
    path = safe_file(root, 'splits/manifest.csv')
    grouped, seen_files = {}, set()
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = csv.DictReader(f)
        if rows.fieldnames != ['pair_id', 'split', 'label', 'path']:
            raise ValueError('Unexpected manifest columns')
        for row in rows:
            pair, split, label, name = (row[k] for k in rows.fieldnames)
            if split not in ('train', 'validation', 'test') or label not in ('0', '1'):
                raise ValueError(f'Invalid split or label: {row}')
            prefix = 'data/cover_256/' if label == '0' else 'data/wow_04/'
            if not name.startswith(prefix) or not name.endswith('.pgm') or PurePosixPath(name).stem != pair:
                raise ValueError(f'Invalid paired path: {row}')
            safe_file(root, name)
            if name in seen_files:
                raise ValueError(f'Repeated file: {name}')
            seen_files.add(name)
            item = grouped.setdefault(pair, {'pair_id': pair, 'split': split})
            if item['split'] != split or label in item:
                raise ValueError(f'Pair crosses splits or repeats label: {pair}')
            item[label] = name
    result = {'train': [], 'validation': [], 'test': []}
    for item in grouped.values():
        if '0' not in item or '1' not in item:
            raise ValueError(f'Incomplete pair: {item["pair_id"]}')
        result[item['split']].append(item)
    if any(not value for value in result.values()):
        raise ValueError('All three splits must be nonempty')
    return result


def validate_index(root, expected_counts, allow_fixture=False):
    root = Path(root)
    meta_path = safe_file(root, 'metadata.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('schema') != 'zhu-wow-dataset/v1':
        raise ValueError('Unsupported dataset metadata schema')
    if meta.get('test_fixture') and not allow_fixture:
        raise ValueError('Synthetic fixture may only be used with --smoke-steps')
    pairs = read_pairs(root)
    counts = {k: len(v) for k, v in pairs.items()}
    if not meta.get('test_fixture') and counts != expected_counts:
        raise ValueError(f'Unexpected split counts: {counts}; expected {expected_counts}')
    if counts != meta['split_pair_counts']:
        raise ValueError('Metadata split counts differ from the manifest')
    required = {'splits/manifest.csv'} | {row[label] for group in pairs.values() for row in group for label in ('0', '1')}
    if not required.issubset(meta['files']):
        raise ValueError('Dataset hash index is incomplete')
    for i, (name, spec) in enumerate(meta['files'].items(), 1):
        p = safe_file(root, name)
        if p.stat().st_size != spec['bytes'] or sha256(p) != spec['sha256']:
            raise ValueError(f'Dataset integrity mismatch: {name}')
        if i % 5000 == 0:
            print(f'DATASET_HASH_CHECK {i}/{len(meta["files"])}', flush=True)
    return pairs, meta, sha256(meta_path)
