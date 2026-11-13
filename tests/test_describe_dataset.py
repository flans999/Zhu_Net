import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/describe_dataset.py'
spec = importlib.util.spec_from_file_location('describe_dataset', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DescriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'dataset'
        (self.root / 'splits').mkdir(parents=True)
        self.report = self.base / 'report'

    def manifest(self, rows):
        with (self.root / 'splits/manifest.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pair_id', 'split', 'label', 'path'])
            writer.writerows(rows)

    def image(self, name, fmt='PPM'):
        Image.new('L', (7, 5), 10).save(self.root / name, format=fmt)

    def test_trial_selects_whole_nonadjacent_pair(self):
        self.manifest([['a', 'train', 0, 'a.pgm'], ['b', 'test', 0, 'b.pgm'],
                       ['a', 'train', 1, 'aa.pgm'], ['b', 'test', 1, 'bb.pgm']])
        self.image('a.pgm')
        self.image('aa.pgm')
        result = module.describe(self.root, self.report, 1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['inspected']['images'], 2)
        self.assertTrue(result['healthy'])
        self.assertEqual(result['sizes'][0]['width'], 7)
        self.assertEqual(result['sizes'][0]['height'], 5)
        self.assertEqual(result['sizes'][0]['format'], 'PGM')

    def test_missing_unreadable_and_incomplete_pairs(self):
        self.manifest([['a', 'train', 0, 'a.pgm'], ['a', 'train', 1, 'missing.pgm'],
                       ['b', 'test', 0, 'bad.pgm']])
        self.image('a.pgm')
        (self.root / 'bad.pgm').write_bytes(b'P5\n7 5\n255\n\x01')
        result = module.describe(self.root, self.report)
        self.assertEqual(result['inspected']['images_by_status'],
                         {'ok': 1, 'missing': 1, 'unreadable': 1})
        self.assertEqual(len(result['issues']['incomplete_pairs']), 2)
        self.assertEqual(len(result['issues']['manifest_pair_errors']), 1)
        self.assertFalse(result['healthy'])
        self.assertTrue((self.report / 'images.csv').is_file())
        self.assertEqual(json.loads((self.report / 'summary.json').read_text())['complete'], True)

    def test_split_leak_and_duplicate_class(self):
        self.manifest([['a', 'train', 0, 'a.pgm'], ['a', 'test', 0, 'aa.pgm']])
        self.image('a.pgm')
        self.image('aa.pgm')
        result = module.describe(self.root, self.report)
        errors = result['issues']['manifest_pair_errors'][0]['reasons']
        self.assertIn('pair spans multiple splits', errors)
        self.assertIn('expected exactly one cover and one stego', errors)

    def test_png_disguised_as_pgm_is_unreadable(self):
        self.image('fake.pgm', 'PNG')
        row = {'pair_id': 'a', 'split': 'train', 'label': '0', 'path': 'fake.pgm'}
        self.assertEqual(module.inspect_image(self.root, row)['status'], 'unreadable')

    def test_unknown_source_without_metadata(self):
        self.assertEqual(module.Sources(self.root, {}, None).for_pair('bows2_00001', []),
                         ('unknown', ''))

    def test_reject_escaping_path_and_dataset_output(self):
        self.manifest([['a', 'train', 0, '../outside.pgm']])
        with self.assertRaises(ValueError):
            module.describe(self.root, self.report)
        self.manifest([])
        with self.assertRaises(ValueError):
            module.describe(self.root, self.root / 'reports')


if __name__ == '__main__':
    unittest.main()
