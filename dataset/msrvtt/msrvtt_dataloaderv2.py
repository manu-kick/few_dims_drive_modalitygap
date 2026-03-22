"""
MSRVTT dataloader v2 — one sample per unique video_id, always using the first
annotation (caption index 0).

Loads the same NPZ shards produced by ``precompute_embeddings_msrvtt.py`` but
de-duplicates rows so that each ``video_id`` appears exactly once and the
text embedding comes from the first caption associated with that video.
"""

import os
import json
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MSRVTTEmbeddingsDatasetV2(Dataset):
    """One-sample-per-video_id dataset for precomputed CLIP embeddings.

    For every unique ``video_id`` found across all shards the dataset keeps
    exactly one entry and always selects caption-index **0** for the text
    embedding.

    Parameters
    ----------
    precomputed_dir : str
        Directory containing ``.npz`` shard files.
    split_name : str
        Substring used to select the correct shard files (e.g.
        ``"train_shard"`` or ``"test_shard"``).
    return_metadata : bool
        If *True* each ``__getitem__`` returns
        ``(text_emb, vision_emb, label, video_id, caption_id, file_name)``
        instead of the default ``(text_emb, vision_emb, label)``.
    """

    def __init__(
        self,
        precomputed_dir: str,
        split_name: str = "train_shard",
        return_metadata: bool = False,
    ) -> None:
        self.return_metadata = return_metadata

        # video_id -> category mapping from official MSRVTT metadata
        self.videoid_to_category = self._load_videoid_to_category(split_name)

        # Accumulators — one entry per *unique* video_id
        text_list: List[np.ndarray] = []       # each (D,)
        vision_list: List[np.ndarray] = []     # each (D,)
        video_id_list: List[str] = []
        label_list: List[int] = []
        caption_id_list: List[int] = []
        file_name_list: List[str] = []

        seen_video_ids: Dict[str, bool] = {}

        found = 0
        for fn in sorted(os.listdir(precomputed_dir)):
            if not (fn.endswith(".npz") and split_name in fn):
                continue
            found += 1
            path = os.path.join(precomputed_dir, fn)
            data = np.load(path, allow_pickle=True)

            for key in ("vision_emb", "text_emb", "video_ids"):
                if key not in data:
                    raise RuntimeError(f"Missing key '{key}' in {path}")

            vision_emb = data["vision_emb"]           # (N, D)
            text_emb = data["text_emb"]                # (N, D) or object(N,)
            video_ids = data["video_ids"]              # (N,) str/object
            caption_ids = (
                data["caption_ids"]
                if "caption_ids" in data
                else np.full(len(video_ids), -1, dtype=np.int64)
            )
            file_names = (
                data["fns"]
                if "fns" in data
                else np.array([""] * len(video_ids), dtype=object)
            )

            for i in range(len(video_ids)):
                vid = str(video_ids[i])

                # ---- keep only the first occurrence of each video_id ----
                if vid in seen_video_ids:
                    continue

                # ---- category / label (skip if not in metadata) ----
                if vid not in self.videoid_to_category:
                    continue
                seen_video_ids[vid] = True
                label_list.append(self.videoid_to_category[vid])

                # ---- vision (always a simple 1-D vector) ----
                vision_list.append(vision_emb[i])

                # ---- text — always take the *first* caption embedding ----
                t = text_emb[i]
                if isinstance(t, np.ndarray) and t.dtype != object:
                    if t.ndim == 1:
                        # (D,) — single caption, already flat
                        text_list.append(t)
                    elif t.ndim == 2:
                        # (K, D) — multiple captions, take index 0
                        text_list.append(t[0])
                    else:
                        raise RuntimeError(
                            f"Unexpected text_emb shape {t.shape} at idx {i} in {path}"
                        )
                elif isinstance(t, np.ndarray) and t.dtype == object:
                    # object array wrapping variable-length captions
                    arr = np.asarray(t.tolist(), dtype=np.float32)
                    if arr.ndim == 1:
                        text_list.append(arr)
                    elif arr.ndim == 2:
                        text_list.append(arr[0])
                    else:
                        raise RuntimeError(
                            f"Unexpected text_emb object shape at idx {i} in {path}"
                        )
                elif isinstance(t, (list, tuple)):
                    arr = np.asarray(t, dtype=np.float32)
                    text_list.append(arr[0] if arr.ndim == 2 else arr)
                else:
                    raise RuntimeError(
                        f"Unsupported text embedding type {type(t)} at idx {i} in {path}"
                    )

                # ---- caption id ----
                cid = caption_ids[i]
                if isinstance(cid, np.ndarray):
                    caption_id_list.append(int(cid.flat[0]) if cid.size > 0 else 0)
                elif isinstance(cid, (list, tuple)):
                    caption_id_list.append(int(cid[0]) if len(cid) > 0 else 0)
                else:
                    try:
                        caption_id_list.append(int(cid))
                    except Exception:
                        caption_id_list.append(0)

                video_id_list.append(vid)
                file_name_list.append(str(file_names[i]))

        if found == 0:
            raise RuntimeError(
                f"No .npz files found in {precomputed_dir} for split '{split_name}'"
            )

        # Convert to contiguous arrays
        self.text_embeddings = np.stack(text_list, axis=0).astype(np.float32)      # (N_unique, D)
        self.vision_embeddings = np.stack(vision_list, axis=0).astype(np.float32)  # (N_unique, D)
        self.labels = np.array(label_list, dtype=np.int64)
        self.video_ids = np.array(video_id_list, dtype=object)
        self.caption_ids = np.array(caption_id_list, dtype=np.int64)
        self.file_names = np.array(file_name_list, dtype=object)

        assert len(self.text_embeddings) == len(self.vision_embeddings)
        assert len(self.labels) == len(self.vision_embeddings)

        print(
            f"[MSRVTTv2] {len(self)} unique videos from {precomputed_dir} | "
            f"vision_emb={self.vision_embeddings.shape} | "
            f"text_emb={self.text_embeddings.shape} | "
            f"num_classes={len(np.unique(self.labels))}"
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _load_videoid_to_category(self, split_name: str) -> Dict[str, int]:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        msrvtt_root = os.path.join(this_dir, "MSRVTT")

        if "train" in split_name:
            metadata_path = os.path.join(msrvtt_root, "train_val_videodatainfo.json")
        elif "test" in split_name:
            metadata_path = os.path.join(msrvtt_root, "test_videodatainfo.json")
        else:
            raise RuntimeError(
                f"Unsupported split_name='{split_name}'. Expected 'train' or 'test'."
            )

        if not os.path.exists(metadata_path):
            raise RuntimeError(f"MSRVTT metadata not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if "videos" not in metadata:
            raise RuntimeError(f"No 'videos' field in {metadata_path}")

        mapping: Dict[str, int] = {}
        for item in metadata["videos"]:
            vid = item.get("video_id")
            cat = item.get("category")
            if vid is not None and cat is not None:
                mapping[str(vid)] = int(cat)

        if not mapping:
            raise RuntimeError(f"Empty video_id->category map from {metadata_path}")
        return mapping

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.vision_embeddings)

    def __getitem__(self, idx: int):
        text_emb = torch.as_tensor(self.text_embeddings[idx]).float()
        vision_emb = torch.as_tensor(self.vision_embeddings[idx]).float()
        label = int(self.labels[idx])

        if self.return_metadata:
            return (
                text_emb,
                vision_emb,
                label,
                str(self.video_ids[idx]),
                int(self.caption_ids[idx]),
                str(self.file_names[idx]),
            )
        return text_emb, vision_emb, label


# ======================================================================
# Collate
# ======================================================================
def msrvtt_v2_collate_fn(batch):
    """Collate returning stacked tensors.

    Supports both the 3-tuple and the 6-tuple format from
    ``MSRVTTEmbeddingsDatasetV2``.
    """
    n = len(batch[0])

    if n == 3:
        texts, visions, labels = zip(*batch)
        return (
            torch.stack(texts),
            torch.stack(visions),
            torch.tensor(labels, dtype=torch.long),
        )

    if n == 6:
        texts, visions, labels, vids, cids, fns = zip(*batch)
        return (
            torch.stack(texts),
            torch.stack(visions),
            torch.tensor(labels, dtype=torch.long),
            list(vids),
            torch.tensor(cids, dtype=torch.long),
            list(fns),
        )

    raise RuntimeError(f"Unsupported batch item length: {n}")


# ======================================================================
# Convenience loader factory
# ======================================================================
def make_loaders_msrvtt_v2(
    precomputed_dir: str,
    precomputed_dir_test: str,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 123,
):
    """Build train / test ``DataLoader`` pair (one sample per video_id)."""
    g = torch.Generator().manual_seed(seed)

    ds_train = MSRVTTEmbeddingsDatasetV2(
        precomputed_dir, split_name="train_shard",
    )
    ds_test = MSRVTTEmbeddingsDatasetV2(
        precomputed_dir_test, split_name="test_shard",
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=msrvtt_v2_collate_fn,
        generator=g,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=msrvtt_v2_collate_fn,
    )
    return train_loader, test_loader
