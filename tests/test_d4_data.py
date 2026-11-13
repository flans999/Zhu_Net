"""CPU checks of the D4 pixel function without importing PyTorch."""
import ast
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError:
    np = None

SOURCE = Path(__file__).resolve().parents[1] / 'training_d4/data.py'


@unittest.skipIf(np is None, 'D4 pixel checks require NumPy from the training environment')
class D4PixelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'), filename=str(SOURCE))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'd4_pixels')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        cls.transform = staticmethod(namespace['d4_pixels'])

    def test_all_eight_symmetries(self):
        pixels = np.arange(9, dtype=np.uint8).reshape(3, 3)
        expected = [
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[2, 5, 8], [1, 4, 7], [0, 3, 6]],
            [[8, 7, 6], [5, 4, 3], [2, 1, 0]],
            [[6, 3, 0], [7, 4, 1], [8, 5, 2]],
            [[2, 1, 0], [5, 4, 3], [8, 7, 6]],
            [[0, 3, 6], [1, 4, 7], [2, 5, 8]],
            [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            [[8, 5, 2], [7, 4, 1], [6, 3, 0]],
        ]
        variants = [self.transform(pixels, variant) for variant in range(8)]
        for actual, wanted in zip(variants, expected):
            np.testing.assert_array_equal(actual, wanted)
        self.assertEqual(len({v.tobytes() for v in variants}), 8)

    def test_cover_stego_alignment(self):
        cover = np.arange(9, dtype=np.uint8).reshape(3, 3)
        pair = np.stack([cover, cover + 7])
        for variant in range(8):
            actual = self.transform(pair, variant)
            self.assertEqual(actual.shape, (2, 3, 3))
            np.testing.assert_array_equal(actual[0], self.transform(cover, variant))
            np.testing.assert_array_equal(actual[1] - actual[0], np.full((3, 3), 7))

    def test_returns_independent_contiguous_pixels(self):
        pixels = np.arange(9, dtype=np.uint8).reshape(3, 3)
        for variant in range(8):
            actual = self.transform(pixels, variant)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertFalse(np.shares_memory(actual, pixels))
            self.assertEqual(actual.dtype, pixels.dtype)
            actual[:] = 0
            np.testing.assert_array_equal(pixels, np.arange(9).reshape(3, 3))

    def test_rejects_invalid_variants(self):
        for variant in (-1, 8, 9):
            with self.assertRaises(ValueError):
                self.transform(np.zeros((3, 3)), variant)


if __name__ == '__main__':
    unittest.main()
