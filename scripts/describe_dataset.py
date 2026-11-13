#!/usr/bin/env python3
"""Describe manifest-listed images without modifying the dataset (requires Pillow)."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, __version__ as pillow_version

SPLITS = ('train', 'validation', 'test')
CLASSES = {'0': 'cover', '1': 'stego'}
FIELDS = ('pair_id', 'split', 'label', 'class', 'path', 'status', 'error',
          'format', 'pgm_variant', 'mode', 'width', 'height', 'bytes', 'sha256',
          'source', 'source_evidence')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_manifest(root):
    with (root / 'splits/manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'pair_id', 'split', 'label', 'path'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Manifest must contain pair_id, split, label, path')
        rows = list(reader)
    for line, row in enumerate(rows, 2):
        if not row['pair_id'] or not row['path']:
            raise ValueError(f'Manifest line {line}: empty pair_id or path')
        if row['split'] not in SPLITS or row['label'] not in CLASSES:
            raise ValueError(f'Manifest line {line}: invalid split or label')
        target = (root / row['path']).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f'Manifest line {line}: path escapes dataset root')
    return rows


def group_pairs(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row['pair_id']].append(row)
    return pairs


def pair_problems(rows):
    problems = []
    if Counter(r['label'] for r in rows) != Counter({'0': 1, '1': 1}):
        problems.append('expected exactly one cover and one stego')
    if len({r['split'] for r in rows}) != 1:
        problems.append('pair spans multiple splits')
    if len({r['path'] for r in rows}) != len(rows):
        problems.append('duplicate path within pair')
    return problems


def manifest_stats(rows):
    pairs = group_pairs(rows)
    return {
        'images': len(rows), 'pairs': len(pairs),
        'images_by_class': dict(Counter(CLASSES[r['label']] for r in rows)),
        'splits': {s: {
            'images': sum(r['split'] == s for r in rows),
            'pairs': sum(any(r['split'] == s for r in group) for group in pairs.values()),
        } for s in SPLITS},
    }


def source_from_record(root, pair_id):
    # A filename alone is never used as proof of a dataset's origin.
    if Path(pair_id).name != pair_id or '/' in pair_id or '\\' in pair_id:
        return 'unknown', ''
    record = root / 'provenance/pair_metadata' / (pair_id + '.pgm.json')
    if record.is_file():
        source = str(read_json(record).get('source', '')).lower()
        if 'bossbase' in source:
            return 'BOSSBase', record.relative_to(root).as_posix() + ':source'
        if 'bows2' in source:
            return 'BOWS2', record.relative_to(root).as_posix() + ':source'
    return 'unknown', ''


class Sources:
    def __init__(self, root, metadata, reference):
        self.root = root
        self.metadata = metadata
        self.reference = reference
        self.reference_files = {}
        self.reference_rows = {}
        self.comparison = None
        self.inputs = {item.get('sha256') for item in
                       metadata.get('preparation', {}).get('inputs', [])}
        if reference:
            expected = metadata.get('preparation', {}).get('baseline_metadata_sha256')
            actual = hashlib.sha256((reference / 'metadata.json').read_bytes()).hexdigest()
            if not expected or actual != expected:
                raise ValueError('Reference metadata hash does not match baseline_metadata_sha256')
            self.reference_files = read_json(reference / 'metadata.json').get('files', {})
            ref_rows = read_manifest(reference)
            self.reference_rows = {(r['pair_id'], r['split'], r['label'], r['path']): r
                                   for r in ref_rows}
            current = {(r['pair_id'], r['split'], r['label'], r['path'])
                       for r in read_manifest(root)}
            self.comparison = {
                'dataset_root': str(reference), 'metadata_sha256_verified': True,
                'reference_manifest_images': len(ref_rows),
                'inherited_manifest_rows': len(current & self.reference_rows.keys()),
                'missing_or_changed_manifest_rows': len(self.reference_rows.keys() - current),
            }

    def for_pair(self, pair_id, rows):
        source, evidence = source_from_record(self.root, pair_id)
        if source != 'unknown':
            return source, evidence
        record_path = self.root / 'records' / (pair_id + '.json')
        if Path(pair_id).name == pair_id and record_path.is_file():
            record = read_json(record_path)
            paths = {r['label']: r['path'] for r in rows}
            if ('BOWS2' in self.metadata.get('description', '')
                    and record.get('pair_id') == pair_id
                    and record.get('raw_sha256') in self.inputs
                    and record.get('cover') == paths.get('0')
                    and record.get('stego') == paths.get('1')):
                return 'BOWS2', record_path.relative_to(self.root).as_posix() + ';metadata.json:description,preparation.inputs'
        if self.reference and all(
                (r['pair_id'], r['split'], r['label'], r['path']) in self.reference_rows
                for r in rows):
            source, evidence = source_from_record(self.reference, pair_id)
            return source, 'reference:' + evidence + ';metadata.json:preparation.baseline_metadata_sha256'
        description = self.metadata.get('description', '')
        if 'BOSSBase' in description and 'BOWS2' not in description:
            return 'BOSSBase', 'metadata.json:description (dataset-wide)'
        return 'unknown', ''


def inspect_image(root, row):
    result = {key: '' for key in FIELDS}
    result.update({k: row[k] for k in ('pair_id', 'split', 'label', 'path')})
    result['class'] = CLASSES[row['label']]
    result['status'] = 'unreadable'
    try:
        payload = (root / row['path']).read_bytes()
        result['bytes'] = len(payload)
        result['sha256'] = hashlib.sha256(payload).hexdigest()
        with Image.open(io.BytesIO(payload)) as image:
            result.update(format=image.format, mode=image.mode,
                          width=image.width, height=image.height)
            image.load()  # Detect truncated raster data, not just a readable header.
        if payload[:2] not in (b'P2', b'P5') or result['format'] != 'PPM':
            raise ValueError('File contents are not PGM (P2/P5)')
        result['format'] = 'PGM'
        result['pgm_variant'] = payload[:2].decode('ascii')
        result['status'] = 'ok'
    except FileNotFoundError as exc:
        result.update(status='missing', error=str(exc))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('pair limit must be positive')
    return number


def describe(root, report_dir, limit_pairs=None, reference=None):
    root, report_dir = root.resolve(), report_dir.resolve()
    reference = reference.resolve() if reference else None
    for protected in (root, reference):
        if protected and report_dir.is_relative_to(protected):
            raise ValueError('Report directory must be outside the read-only datasets')
    rows = read_manifest(root)
    all_pairs = group_pairs(rows)
    selected_ids = set(list(all_pairs)[:limit_pairs]) if limit_pairs else set(all_pairs)
    selected = [r for r in rows if r['pair_id'] in selected_ids]
    pairs = group_pairs(selected)
    metadata_path = root / 'metadata.json'
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    sources = Sources(root, metadata, reference)
    source_by_pair = {pair_id: sources.for_pair(pair_id, group)
                      for pair_id, group in pairs.items()}
    report_dir.mkdir(parents=True, exist_ok=True)
    images = []
    reference_mismatches = []
    with (report_dir / 'images.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(selected, 1):
            result = inspect_image(root, row)
            source, evidence = source_by_pair[row['pair_id']]
            key = (row['pair_id'], row['split'], row['label'], row['path'])
            if reference and key in sources.reference_rows:
                expected = sources.reference_files.get(row['path'], {}).get('sha256')
                if not expected or result['sha256'] != expected:
                    reference_mismatches.append(row['path'])
                    if evidence.startswith('reference:'):
                        source, evidence = 'unknown', 'reference image hash mismatch'
            result.update(source=source, source_evidence=evidence)
            writer.writerow(result)
            images.append(result)
            if i % 5000 == 0:
                print(f'Inspected {i}/{len(selected)} images', flush=True)
    inspected_pairs = group_pairs(images)
    incomplete = []
    for pair_id, group in inspected_pairs.items():
        problems = pair_problems(group)
        if any(r['status'] != 'ok' for r in group):
            problems.append('missing or unreadable image')
        if problems:
            incomplete.append({'pair_id': pair_id, 'reasons': problems})
    duplicate_paths = [p for p, n in Counter(r['path'] for r in rows).items() if n > 1]
    manifest_pair_errors = [{'pair_id': p, 'reasons': pair_problems(g)}
                            for p, g in all_pairs.items() if pair_problems(g)]
    splits = {}
    for split in SPLITS:
        group = [r for r in images if r['split'] == split]
        groups = group_pairs(group)
        splits[split] = {
            'images': len(group), 'pairs': len(groups),
            'complete_readable_pairs': sum(
                not pair_problems(g) and all(r['status'] == 'ok' for r in g)
                for g in groups.values()),
            'images_by_class': {name: sum(r['class'] == name for r in group)
                                for name in CLASSES.values()},
            'images_by_status': dict(Counter(r['status'] for r in group)),
            'images_by_source': dict(Counter(r['source'] for r in group)),
        }
    ok = [r for r in images if r['status'] == 'ok']
    size_counts = Counter((r['width'], r['height'], r['format'], r['pgm_variant'], r['mode'])
                          for r in ok)
    expected_counts = metadata.get('split_pair_counts', {})
    full_stats = manifest_stats(rows)
    if sources.comparison is not None:
        sources.comparison.update(
            inspected_inherited_images=sum(
                (r['pair_id'], r['split'], r['label'], r['path']) in sources.reference_rows
                for r in images),
            image_hash_mismatches=reference_mismatches,
            complete=limit_pairs is None)
    summary = {
        'schema': 'zhu-net-dataset-description/v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_root': str(root), 'manifest': 'splits/manifest.csv',
        'manifest_sha256': hashlib.sha256((root / 'splits/manifest.csv').read_bytes()).hexdigest(),
        'pillow_version': pillow_version,
        'complete': limit_pairs is None,
        'trial': limit_pairs is not None, 'limit_pairs': limit_pairs,
        'selection': 'first N distinct pair_id values in manifest order; all rows of each pair',
        'manifest_totals': full_stats,
        'inspected': {'images': len(images), 'pairs': len(pairs),
                      'complete_readable_pairs': len(pairs) - len(incomplete),
                      'images_by_class': dict(Counter(r['class'] for r in images)),
                      'images_by_status': {s: sum(r['status'] == s for r in images)
                                           for s in ('ok', 'missing', 'unreadable')},
                      'images_by_source': dict(Counter(r['source'] for r in images))},
        'splits': splits,
        'sizes': [{'width': w, 'height': h, 'format': fmt, 'pgm_variant': variant,
                   'mode': mode, 'images': count}
                  for (w, h, fmt, variant, mode), count in sorted(size_counts.items())],
        'issues': {
            'missing_files': [r['path'] for r in images if r['status'] == 'missing'],
            'unreadable_files': [{'path': r['path'], 'error': r['error']}
                                 for r in images if r['status'] == 'unreadable'],
            'incomplete_pairs': incomplete,
            'manifest_pair_errors': manifest_pair_errors,
            'duplicate_manifest_paths': duplicate_paths,
        },
        'metadata_checks': {
            'declared_pairs': metadata.get('valid_pairs'),
            'pairs_match': (full_stats['pairs'] == metadata['valid_pairs'])
                           if 'valid_pairs' in metadata else None,
            'declared_split_pair_counts': expected_counts,
            'split_pairs_match': all(full_stats['splits'][s]['pairs'] == n
                                     for s, n in expected_counts.items())
                                 if expected_counts else None,
        },
        'reference_comparison': sources.comparison,
    }
    summary['healthy'] = not (incomplete or manifest_pair_errors or duplicate_paths
                              or reference_mismatches
                              or summary['metadata_checks']['pairs_match'] is False
                              or summary['metadata_checks']['split_pairs_match'] is False
                              or (sources.comparison and sources.comparison['missing_or_changed_manifest_rows']))
    (report_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'complete': summary['complete'], 'healthy': summary['healthy'],
                      'inspected': summary['inspected'], 'report_dir': str(report_dir)},
                     ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('report_dir', type=Path)
    parser.add_argument('--limit-pairs', type=positive_int)
    parser.add_argument('--reference-dataset', type=Path,
                        help='Baseline dataset for verified inherited source attribution')
    args = parser.parse_args()
    try:
        summary = describe(args.dataset, args.report_dir, args.limit_pairs, args.reference_dataset)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Error: {exc}\n')
    return 0 if summary['healthy'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
