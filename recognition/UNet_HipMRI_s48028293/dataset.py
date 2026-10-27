import glob
import math
import os
import random
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Constants
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
NUM_CLASSES = len(CLASS_NAMES)
PROSTATE = 5  # label index of the prostate

IMG_SIZE = (256, 128)

SPLIT_DIRS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "val": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}

# 3D volumes
DATA_ROOT_2D = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
DATA_ROOT_3D = "/home/groups/comp3710/HipMRI_Study_open"
VOLUME_DIRS = ("semantic_MRs", "semantic_labels_only")
DOWNSAMPLE = (2, 2, 1)      # downsampling factor per axis of the native volume
VOL_SIZE = (128, 128, 128)   # after downsampling; each axis divisible by 16

# File discovery and pairing
def parse_ids(path):
    """Return (case_id, week, slice_idx) from the numbers in a file name."""
    nums = re.findall(r"\d+", os.path.basename(path))
    if len(nums) < 3:
        raise ValueError(
            f"Expected case, week and slice numbers in '{os.path.basename(path)}'. "
            "Update parse_ids() to match the real file naming."
        )
    return int(nums[0]), int(nums[1]), int(nums[2])


def list_pairs(img_dir, seg_dir):
    """Pair every image with its mask by (case, week, slice).

    Returns a sorted list of (img_path, seg_path, (case, week, slice))."""
    imgs = {parse_ids(p): p for p in glob.glob(os.path.join(img_dir, "*.nii*"))}
    segs = {parse_ids(p): p for p in glob.glob(os.path.join(seg_dir, "*.nii*"))}
    if not imgs or not segs:
        raise FileNotFoundError(f"No Nifti files found in {img_dir} or {seg_dir}")
    if imgs.keys() != segs.keys():
        unmatched = len(imgs.keys() ^ segs.keys())
        raise ValueError(f"{unmatched} images/masks have no partner in {img_dir} / {seg_dir}")
    return [(imgs[k], segs[k], k) for k in sorted(imgs)]

def parse_volume_ids(path):
    """Return (case_id, week) from a 3D file name, e.g. Case_004_Week0_LFOV.nii.gz."""
    nums = re.findall(r"\d+", os.path.basename(path))
    if len(nums) < 2:
        raise ValueError(
            f"Expected case and week numbers in '{os.path.basename(path)}'. "
            "Update parse_volume_ids() to match the real file naming."
        )
    return int(nums[0]), int(nums[1])

def list_volume_pairs(img_dir, seg_dir):
    """Pair every 3D scan with its label volume by (case, week).

    Returns a sorted list of (img_path, seg_path, (case, week))."""
    imgs = {parse_volume_ids(p): p for p in glob.glob(os.path.join(img_dir, "*.nii*"))}
    segs = {parse_volume_ids(p): p for p in glob.glob(os.path.join(seg_dir, "*.nii*"))}
    if not imgs or not segs:
        raise FileNotFoundError(f"No Nifti files found in {img_dir} or {seg_dir}")
    if imgs.keys() != segs.keys():
        unmatched = len(imgs.keys() ^ segs.keys())
        raise ValueError(f"{unmatched} images/masks have no partner in {img_dir} / {seg_dir}")
    return [(imgs[k], segs[k], k) for k in sorted(imgs)]

# Leakage-free splitting
def patients_of(pairs):
    return {key[0] for _, _, key in pairs}

def assert_no_leakage(splits):
    """Raise if any patient appears in more than one split."""
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = patients_of(splits[names[i]]) & patients_of(splits[names[j]])
            if shared:
                raise ValueError(
                    f"Patient leakage between {names[i]} and {names[j]}: "
                    f"cases {sorted(shared)}. Re-run with --resplit."
                )

# Not being implemented, currently using given splits
def split_by_patient(pairs, fractions=(0.7, 0.15, 0.15), seed=42):
    """Split slices into train/val/test so that every slice (and every weekly
    scan) of a patient lands in exactly one split."""
    patients = sorted(patients_of(pairs))
    random.Random(seed).shuffle(patients)
    n_train = round(fractions[0] * len(patients))
    n_val = round(fractions[1] * len(patients))
    groups = {
        "train": set(patients[:n_train]),
        "val": set(patients[n_train:n_train + n_val]),
        "test": set(patients[n_train + n_val:]),
    }
    return {name: [p for p in pairs if p[2][0] in ids] for name, ids in groups.items()}

def get_splits(data_root, resplit=False, seed=42):
    """Return {'train': pairs, 'val': pairs, 'test': pairs}.

    By default the provided folders are used and verified to be disjoint at
    the patient level. With resplit=True all slices are pooled and re-split
    by patient ID instead."""
    splits = {
        name: list_pairs(os.path.join(data_root, img), os.path.join(data_root, seg))
        for name, (img, seg) in SPLIT_DIRS.items()
    }
    if resplit:
        pooled = [p for pairs in splits.values() for p in pairs]
        splits = split_by_patient(pooled, seed=seed)
    assert_no_leakage(splits)
    return splits

def get_volume_splits(data_root=DATA_ROOT_3D, resplit=False, seed=42, slices_root=DATA_ROOT_2D):
    """3D volumes: return {'train': pairs, 'val': pairs, 'test': pairs}.

    The 3D data has no provided split, so by default each patient goes in the
    split it occupies in the 2D slice folders. This makes the 3D test set the
    same patients as the 2D baseline's. Patients with no 2D slices go to train.
    With resplit=True patients are split 70/15/15 by split_by_patient instead."""
    pairs = list_volume_pairs(*(os.path.join(data_root, d) for d in VOLUME_DIRS))
    if resplit:
        splits = split_by_patient(pairs, seed=seed)
    else:
        slice_splits = get_splits(slices_root, resplit=False)
        owner = {case: name for name, sp in slice_splits.items() for case in patients_of(sp)}
        missing = sorted(patients_of(pairs) - owner.keys())
        if missing:
            print(f"  patients with no 2D slices, assigned to train: {missing}")
        splits = {name: [p for p in pairs if owner.get(p[2][0], "train") == name]
                    for name in SPLIT_DIRS}
    assert_no_leakage(splits)
    return splits

# Preprocessing
def load_slice(path):
    """Load one 2D Nifti slice as a NumPy array (drops a trailing extra dim)."""
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr

def fit_to_shape(arr, shape):
    """Centre-crop and/or zero-pad a 2D array to the target shape."""
    out = np.zeros(shape, dtype=arr.dtype)
    src, dst = [], []
    for have, want in zip(arr.shape, shape):
        if have >= want:  # crop
            start = (have - want) // 2
            src.append(slice(start, start + want))
            dst.append(slice(0, want))
        else:  # pad
            start = (want - have) // 2
            src.append(slice(0, have))
            dst.append(slice(start, start + have))
    out[tuple(dst)] = arr[tuple(src)]
    return out

def preprocess_image(arr, img_size):
    arr = fit_to_shape(arr.astype(np.float32), img_size)
    return (arr - arr.mean()) / (arr.std() + 1e-8)  # per-slice z-score


def preprocess_mask(arr, img_size):
    return fit_to_shape(np.rint(arr).astype(np.uint8), img_size)

def load_volume(path):
    """Load one 3D Nifti volume as a NumPy array (drops a trailing extra dim)."""
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 4:
        arr = arr[:, :, :, 0]
    return arr

def downsampled_shape(shape, factor):
    return tuple(max(1, round(s / f)) for s, f in zip(shape, factor))

def downsample_image(arr, factor):
    """Area (block-average) downsampling, which also avoids aliasing."""
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.interpolate(t, size=downsampled_shape(arr.shape, factor), mode="area")
    return t[0, 0].numpy()

def downsample_mask(arr, factor):
    """Majority-vote downsampling: average each class's one-hot map over the
    block and keep the most common class. Nearest-neighbour sampling would
    drop thin structures such as the rectal wall."""
    labels = torch.from_numpy(np.rint(arr).astype(np.int64))
    onehot = F.one_hot(labels, NUM_CLASSES).movedim(-1, 0).float()
    onehot = F.interpolate(onehot[None], size=downsampled_shape(arr.shape, factor), mode="area")
    return onehot[0].argmax(0).numpy().astype(np.uint8)

def preprocess_volume(img, seg, vol_size=VOL_SIZE, factor=DOWNSAMPLE):
    """Native scan + labels -> downsampled, cropped/padded, z-scored volume."""
    img = fit_to_shape(downsample_image(img, factor), vol_size)
    seg = fit_to_shape(downsample_mask(seg, factor), vol_size)
    img = (img - img.mean()) / (img.std() + 1e-8)  # per-volume z-score
    return img.astype(np.float32), seg

# Augmentation (applied to training slices only)
class Augment:
    """Random small rotation, scaling and translation applied identically to
    the image (bilinear) and mask (nearest), plus intensity jitter on the image.

    Flips are deliberately left out: pelvic anatomy has a fixed orientation,
    so a flipped slice is not a realistic scan."""

    def __init__(self, max_rot_deg=10.0, max_scale=0.1, max_shift=0.05, intensity=0.1):
        self.max_rot = math.radians(max_rot_deg)
        self.max_scale = max_scale
        self.max_shift = max_shift
        self.intensity = intensity

    def __call__(self, img, mask):
        # img: (1, H, W) float, mask: (H, W) long
        h, w = mask.shape
        angle = random.uniform(-self.max_rot, self.max_rot)
        scale = 1.0 + random.uniform(-self.max_scale, self.max_scale)
        tx = random.uniform(-self.max_shift, self.max_shift) * 2
        ty = random.uniform(-self.max_shift, self.max_shift) * 2
        cos, sin = math.cos(angle) * scale, math.sin(angle) * scale
        # The h/w factors keep the rotation rigid on a non-square image.
        theta = torch.tensor([[cos, -sin * h / w, tx],
                              [sin * w / h, cos, ty]], dtype=torch.float32)
        grid = F.affine_grid(theta[None], (1, 1, h, w), align_corners=False)
        img = F.grid_sample(img[None], grid, mode="bilinear",
                            padding_mode="border", align_corners=False)[0]
        mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)[0, 0].long()

        # Intensity: random contrast and brightness (images are z-scored).
        img = img * (1.0 + random.uniform(-self.intensity, self.intensity))
        img = img + random.uniform(-self.intensity, self.intensity)
        return img, mask

class Augment3D(Augment):
    """3D version of Augment: the same in-plane rotation, scaling and shift
    applied to every slice, plus a small scaling along the slice axis.
    Out-of-plane rotations are left out because the slices are thick."""

    def __call__(self, img, mask):
        # img: (1, X, Y, Z) float, mask: (X, Y, Z) long; Z is the slice axis.
        # affine_grid treats the volume as (D, H, W) = (X, Y, Z), and the rows
        # of theta are ordered (W, H, D) = (z, y, x).
        nx, ny, nz = mask.shape
        angle = random.uniform(-self.max_rot, self.max_rot)
        s = 1.0 + random.uniform(-self.max_scale, self.max_scale)
        sz = 1.0 + random.uniform(-self.max_scale, self.max_scale) / 2
        tx = random.uniform(-self.max_shift, self.max_shift) * 2
        ty = random.uniform(-self.max_shift, self.max_shift) * 2
        cos, sin = math.cos(angle) * s, math.sin(angle) * s
        # Size ratios keep the rotation rigid on a non-square plane (as in Augment).
        theta = torch.tensor([[sz, 0.0, 0.0, 0.0],
                              [0.0, cos, -sin * nx / ny, ty],
                              [0.0, sin * ny / nx, cos, tx]], dtype=torch.float32)
        grid = F.affine_grid(theta[None], (1, 1, nx, ny, nz), align_corners=False)
        img = F.grid_sample(img[None], grid, mode="bilinear",
                            padding_mode="border", align_corners=False)[0]
        mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)[0, 0].long()

        img = img * (1.0 + random.uniform(-self.intensity, self.intensity))
        img = img + random.uniform(-self.intensity, self.intensity)
        return img, mask

# Dataset
class HipMRI2D(Dataset):
    """HipMRI 2D slices.

    Each item is (image, mask, case_id, week, slice_idx):
        image (1, H, W) float32, z-scored
        mask  (H, W) int64 class indices
    The IDs let the evaluation code group slices back into patient volumes.

    preload=True reads everything into RAM once (about 2 GB for the full
    training set), which makes each epoch much faster than re-reading
    compressed Nifti files.
    """

    def __init__(self, pairs, img_size=IMG_SIZE, augment=None, preload=True):
        self.pairs = pairs
        self.img_size = tuple(img_size)
        self.augment = augment
        self.cache = None
        if preload:
            n = len(pairs)
            images = np.zeros((n,) + self.img_size, dtype=np.float32)
            masks = np.zeros((n,) + self.img_size, dtype=np.uint8)
            for i, (img_path, seg_path, _) in enumerate(pairs):
                images[i] = preprocess_image(load_slice(img_path), self.img_size)
                masks[i] = preprocess_mask(load_slice(seg_path), self.img_size)
                if (i + 1) % 2000 == 0:
                    print(f"  loaded {i + 1}/{n} slices", flush=True)
            if masks.max() >= NUM_CLASSES:
                raise ValueError(f"Found label {masks.max()} but NUM_CLASSES is {NUM_CLASSES}")
            self.cache = (images, masks)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, seg_path, (case, week, slice_idx) = self.pairs[idx]
        if self.cache is not None:
            img, mask = self.cache[0][idx], self.cache[1][idx]
        else:
            img = preprocess_image(load_slice(img_path), self.img_size)
            mask = preprocess_mask(load_slice(seg_path), self.img_size)
        img = torch.from_numpy(np.array(img))[None]             # (1, H, W)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # (H, W)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return img, mask, case, week, slice_idx

class HipMRI3D(Dataset):
    """HipMRI 3D weekly scans, downsampled.

    Each item is (image, mask, case_id, week, -1):
        image (1, X, Y, Z) float32, z-scored per volume
        mask  (X, Y, Z) int64 class indices
    The -1 keeps the same tuple layout as HipMRI2D (a volume has no single slice index).
    preload=True downsamples every volume once and keeps it in RAM (~4 MB each).
    """

    def __init__(self, pairs, vol_size=VOL_SIZE, downsample=DOWNSAMPLE, augment=None, preload=True):
        self.pairs = pairs
        self.vol_size = tuple(vol_size)
        self.downsample = tuple(downsample)
        self.augment = augment
        self.cache = None
        if preload:
            n = len(pairs)
            images = np.zeros((n,) + self.vol_size, dtype=np.float32)
            masks = np.zeros((n,) + self.vol_size, dtype=np.uint8)
            for i in range(n):
                images[i], masks[i] = self._load(i)
                if (i + 1) % 25 == 0:
                    print(f"  loaded {i + 1}/{n} volumes", flush=True)
            if masks.max() >= NUM_CLASSES:
                raise ValueError(f"Found label {masks.max()} but NUM_CLASSES is {NUM_CLASSES}")
            self.cache = (images, masks)

    def _load(self, idx):
        img_path, seg_path, _ = self.pairs[idx]
        return preprocess_volume(load_volume(img_path), load_volume(seg_path),
                                self.vol_size, self.downsample)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        case, week = self.pairs[idx][2]
        if self.cache is not None:
            img, mask = self.cache[0][idx], self.cache[1][idx]
        else:
            img, mask = self._load(idx)
        img = torch.from_numpy(np.array(img))[None]             # (1, X, Y, Z)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # (X, Y, Z)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return img, mask, case, week, -1
    
def get_loaders(data_root, batch_size=32, img_size=IMG_SIZE, workers=4,
                resplit=False, limit=None, preload=True, seed=42, dims=2):
    """Build the train/val/test loaders for the 2D slices (dims=2) or the
    3D volumes (dims=3).

    limit: keep only the first N slices / volumes of each split, for quick smoke tests."""
    if dims == 2:
        splits = get_splits(data_root, resplit=resplit, seed=seed)
    else:
        splits = get_volume_splits(data_root, resplit=resplit, seed=seed)
    if limit:
        splits = {name: pairs[:limit] for name, pairs in splits.items()}
    unit = "slices" if dims == 2 else "volumes"
    for name, pairs in splits.items():
        print(f"{name:5s}: {len(pairs):6d} {unit} from {len(patients_of(pairs))} patients")

    if dims == 2:
        sets = {
            "train": HipMRI2D(splits["train"], img_size, augment=Augment(), preload=preload),
            "val": HipMRI2D(splits["val"], img_size, preload=preload),
            "test": HipMRI2D(splits["test"], img_size, preload=preload),
        }
    else:
        sets = {
            "train": HipMRI3D(splits["train"], img_size, augment=Augment3D(), preload=preload),
            "val": HipMRI3D(splits["val"], img_size, preload=preload),
            "test": HipMRI3D(splits["test"], img_size, preload=preload),
        }
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0)
    return {
        "train": DataLoader(sets["train"], shuffle=True, drop_last=True, **common),
        "val": DataLoader(sets["val"], shuffle=False, **common),
        "test": DataLoader(sets["test"], shuffle=False, **common),
    }

# Quick data check: python dataset.py   (optionally followed by a data path)
def inspect_data(data_root, n_files=300):
    """Print patient counts, leakage status, image shapes and label values."""
    splits = {
        name: list_pairs(os.path.join(data_root, img), os.path.join(data_root, seg))
        for name, (img, seg) in SPLIT_DIRS.items()
    }
    for name, pairs in splits.items():
        print(f"{name:5s}: {len(pairs):6d} slices, patients {sorted(patients_of(pairs))}")
        print(f"       example file: {os.path.basename(pairs[0][0])}")
    try:
        assert_no_leakage(splits)
        print("Provided split is leakage-free at the patient level.")
    except ValueError as err:
        print("WARNING:", err)

    pairs = splits["train"]
    step = max(1, len(pairs) // n_files)
    shapes, labels = {}, set()
    for img_path, seg_path, _ in pairs[::step]:
        shape = nib.load(img_path).shape
        shapes[shape] = shapes.get(shape, 0) + 1
        labels.update(np.unique(load_slice(seg_path)).tolist())
    print("Image shapes in sample:", shapes)
    print("Label values in sample:", sorted(labels), f"(expecting 0..{NUM_CLASSES - 1})")

if __name__ == "__main__":
    import sys
    default_root = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    inspect_data(sys.argv[1] if len(sys.argv) > 1 else default_root)