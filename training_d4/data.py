from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def read_gray(path):
    with Image.open(path) as im:
        if im.mode != 'L' or im.size != (256, 256):
            raise ValueError(f'Expected grayscale uint8 256x256: {path}')
        return np.array(im)


def validate_pixels(root, pairs):
    fingerprints = {}
    number = 0
    for split, rows in pairs.items():
        for row in rows:
            cover, stego = (read_gray(Path(root) / row[key]) for key in ('0', '1'))
            delta = stego.astype(np.int16) - cover.astype(np.int16)
            if not np.isin(delta, [-1, 0, 1]).all() or not np.any(delta):
                raise ValueError(f'Invalid WOW pair: {row["pair_id"]}')
            fingerprint = hashlib.sha256(cover.tobytes()).hexdigest()
            if fingerprint in fingerprints:
                raise ValueError(f'Duplicate cover: {row["pair_id"]} and {fingerprints[fingerprint]}')
            fingerprints[fingerprint] = row['pair_id']
            number += 1
            if number % 2000 == 0:
                print(f'DATASET_PIXEL_CHECK {number}', flush=True)


class PairDataset(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = np.stack([read_gray(self.root / row[key]) for key in ('0', '1')])
        # Identical scaling to torchvision.transforms.ToTensor for uint8 input.
        return torch.from_numpy(x.copy()).unsqueeze(1).float().div_(255.0)


def d4_pixels(pixels, variant):
    """D4: four exact rotations and the same rotations after a reflection."""
    if not 0 <= variant < 8:
        raise ValueError('D4 variant must be in 0..7')
    if variant >= 4:
        pixels = np.flip(pixels, axis=-1)
    return np.rot90(pixels, variant % 4, axes=(-2, -1)).copy()


class EightWayPairDataset(PairDataset):
    """Every epoch visits each physical pair in all eight orientations."""
    def __len__(self):
        return 8 * len(self.rows)

    def __getitem__(self, index):
        physical, variant = divmod(index, 8)
        row = self.rows[physical]
        pixels = np.stack([read_gray(self.root / row[key]) for key in ('0', '1')])
        pixels = d4_pixels(pixels, variant)
        return torch.from_numpy(pixels).unsqueeze(1).float().div_(255.0)
