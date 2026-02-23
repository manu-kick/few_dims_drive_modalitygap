import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ---------------------------
# 1) Reproducibility (SEED)
# ---------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic-ish (can slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int):
    # make dataloader workers deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
# ---------------------------
# 2) Parse captions (5 per image)
# ---------------------------
def load_flickr30k_captions(captions_txt_path: str) -> Dict[str, List[str]]:
    """
    captions.txt format (as in your screenshot):
    first row header: image,caption
    then: filename.jpg, caption text ...
    """
    cap_dict = defaultdict(list)

    with open(captions_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # skip header if present
    start_idx = 1 if lines and lines[0].strip().lower().startswith("image,caption") else 0

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        # split only on first comma because caption can contain commas
        fname, caption = line.split(",", 1)
        cap_dict[fname.strip()].append(caption.strip())

    # sanity check: keep only images that have at least 1 caption
    cap_dict = {k: v for k, v in cap_dict.items() if len(v) > 0}
    return cap_dict

# ---------------------------
# 3) Split by IMAGE (80/10/10)
# ---------------------------
def split_filenames(
    filenames: List[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.01,
    test_ratio: float = 0.19,
) -> Tuple[List[str], List[str], List[str]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    rng = random.Random(seed)
    filenames = list(filenames)
    rng.shuffle(filenames)

    n = len(filenames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # remainder goes to test (handles rounding)
    n_test = n - n_train - n_val

    train_files = filenames[:n_train]
    val_files = filenames[n_train:n_train + n_val]
    test_files = filenames[n_train + n_val:]

    assert len(train_files) == n_train
    assert len(val_files) == n_val
    assert len(test_files) == n_test

    return train_files, val_files, test_files



# ---------------------------
# 4) Dataset: (image, random caption) without repeats for same image
#    Implementation trick:
#    Expand each image into 5 distinct (image, caption_i) pairs.
#    Shuffle pairs in DataLoader => caption "chosen at random"
#    and cannot repeat for same image because each caption index appears once.
# ---------------------------
class Flickr30kNoRepeatCaptionDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        captions_by_file: Dict[str, List[str]],
        filenames: List[str],
        seed: int = 42,
        transform=None,
        require_n_captions: int = 5,
    ):
        """
        - filenames: list of image filenames included in this split
        - captions_by_file: dict filename -> list of captions
        - require_n_captions: if 5, we enforce exactly 5 captions (skip images that don't have 5)
        """
        self.images_dir = images_dir
        self.transform = transform
        self.seed = seed

        # filter to files that exist and have enough captions
        kept = []
        for fn in filenames:
            if fn not in captions_by_file:
                continue
            if require_n_captions is not None and len(captions_by_file[fn]) < require_n_captions:
                continue
            img_path = os.path.join(images_dir, fn)
            if os.path.isfile(img_path):
                kept.append(fn)

        self.filenames = kept
        self.captions_by_file = captions_by_file

        # Build expanded index list: each (filename, caption_idx) appears once
        rng = random.Random(seed)
        self.pairs: List[Tuple[str, int]] = []
        for fn in self.filenames:
            caps = captions_by_file[fn]
            # choose 5 captions, but shuffle their order (random choice without repetition)
            idxs = list(range(len(caps)))
            rng.shuffle(idxs)
            idxs = idxs[:require_n_captions] if require_n_captions is not None else idxs
            for ci in idxs:
                self.pairs.append((fn, ci))

        # Note: global shuffle will be handled by DataLoader(shuffle=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        fn, cap_idx = self.pairs[idx]
        img_path = os.path.join(self.images_dir, fn)

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        caption = self.captions_by_file[fn][cap_idx]
        return image, caption, fn, cap_idx  # fn/cap_idx are handy for debugging


class Flickr30k_TestDataset(Dataset):
    """
    One item = one image + ONE random caption (deterministic).
    """

    def __init__(
        self,
        images_dir: str,
        captions_by_file: Dict[str, List[str]],
        filenames: List[str],
        seed: int = 42,
        transform=None,
        require_n_captions: int = 5,
    ):
        self.images_dir = images_dir
        self.captions_by_file = captions_by_file
        self.transform = transform

        rng = random.Random(seed)

        kept = []
        chosen_caption_idx = {}

        for fn in filenames:
            if fn not in captions_by_file:
                continue
            if require_n_captions is not None and len(captions_by_file[fn]) < require_n_captions:
                continue

            img_path = os.path.join(images_dir, fn)
            if not os.path.isfile(img_path):
                continue

            kept.append(fn)

            # scegli UNA caption random per immagine
            cap_idx = rng.randint(0, len(captions_by_file[fn]) - 1)
            chosen_caption_idx[fn] = cap_idx

        self.filenames = kept
        self.chosen_caption_idx = chosen_caption_idx

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fn = self.filenames[idx]
        img_path = os.path.join(self.images_dir, fn)

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        cap_idx = self.chosen_caption_idx[fn]
        caption = self.captions_by_file[fn][cap_idx]

        return image, caption, fn, cap_idx


def get_dataloaders(cf):
    captions_txt = os.path.join(cf.dataset_root, "captions.txt")
    images_dir = os.path.join(cf.dataset_root, "Images")
    captions_by_file = load_flickr30k_captions(captions_txt)
    
    all_files = sorted(list(captions_by_file.keys()))
    train_files, val_files, test_files = split_filenames(all_files, seed=cf.seed)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    train_ds = Flickr30kNoRepeatCaptionDataset(
        images_dir=images_dir,
        captions_by_file=captions_by_file,
        filenames=train_files,
        seed=cf.seed,
        transform=transform,
        require_n_captions=5,
    )
    val_ds = Flickr30kNoRepeatCaptionDataset(
        images_dir=images_dir,
        captions_by_file=captions_by_file,
        filenames=val_files,
        seed=cf.seed + 1,   # different but deterministic
        transform=transform,
        require_n_captions=5,
    )
    test_ds = Flickr30k_TestDataset(
        images_dir=images_dir,
        captions_by_file=captions_by_file,
        filenames=test_files,
        seed=cf.seed + 2,   # different but deterministic
        transform=transform,
        require_n_captions=5,
    )

    # DataLoaders
    g = torch.Generator()
    print("batch size:", cf.batch_size)
    g.manual_seed(cf.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cf.batch_size,
        shuffle=True,              # shuffle pairs => "caption scelta a caso"
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cf.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cf.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    
    print("Images (unique) in splits:")
    print("  train:", len(train_files), "val:", len(val_files), "test:", len(test_files))
    print("Pairs (image,caption) in splits (x5 per image):")
    print("  train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))
    
    return train_loader, val_loader, test_loader