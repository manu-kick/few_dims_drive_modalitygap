# mscoco_dataloader_with_labels.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MSCOCOEmbeddingsDatasetWithLabels(Dataset):
    def __init__(
        self,
        precomputed_dir,
        split_name="train_shard",
        return_metadata=False,
        allow_multi_caption=False,
        text_index=0,
        require_labels=False,
    ):
        """
        Supported NPZ formats:

        New format:
          - vision_emb:  (N, D)
          - text_emb:    (N, D) OR object array (N,), each item (K_i, D)
          - img_ids:     (N,)
          - caption_ids: (N,) optional
          - label_ids:   object array (N,), each item (L_i,)
          - label_names: object array (N,), each item (L_i,)

        Legacy format:
          - label_ids:   (N,)
          - label_names: (N,)

        Args:
            precomputed_dir: directory with NPZ shards
            split_name: substring used to select split files
            return_metadata: if True returns img_id/caption_id/label_names too
            allow_multi_caption: if True supports variable number of captions per sample
            text_index: caption index used when a sample has multiple captions
            require_labels: if True, fail if label_ids / label_names are missing
        """
        self.text_embeddings = []
        self.vision_embeddings = []
        self.img_ids = []
        self.caption_ids = []
        self.label_ids = []
        self.label_names = []

        self.return_metadata = return_metadata
        self.allow_multi_caption = allow_multi_caption
        self.text_index = text_index
        self.require_labels = require_labels

        found = 0

        for fn in os.listdir(precomputed_dir):
            if fn.endswith(".npz") and (split_name in fn):
                found += 1
                path = os.path.join(precomputed_dir, fn)
                data = np.load(path, allow_pickle=True)

                for k in ["vision_emb", "text_emb", "img_ids"]:
                    if k not in data:
                        raise RuntimeError(f"Missing key '{k}' in {path}")

                v = data["vision_emb"]
                t = data["text_emb"]
                img_ids = data["img_ids"]

                caption_ids = (
                    data["caption_ids"]
                    if "caption_ids" in data
                    else np.full(len(img_ids), -1, dtype=np.int64)
                )

                if self.require_labels and ("label_ids" not in data or "label_names" not in data):
                    raise RuntimeError(f"{path} is missing COCO labels required for clustering.")

                if "label_ids" in data:
                    label_ids = data["label_ids"]
                else:
                    # fallback: one empty label list per sample
                    label_ids = np.array(
                        [np.array([], dtype=np.int64) for _ in range(len(img_ids))],
                        dtype=object
                    )

                if "label_names" in data:
                    label_names = data["label_names"]
                else:
                    label_names = np.array(
                        [np.array([], dtype=object) for _ in range(len(img_ids))],
                        dtype=object
                    )

                # -------------------------
                # Validate text embeddings
                # -------------------------
                if isinstance(t, np.ndarray) and t.dtype != object:
                    # standard dense array
                    if t.ndim == 2:
                        # (N, D)
                        pass
                    elif t.ndim == 3 and self.allow_multi_caption:
                        # old dense multi-caption format (rare)
                        pass
                    else:
                        raise RuntimeError(
                            f"Unexpected dense text_emb shape {t.shape} in {path}. "
                            f"Expected (N,D) or (N,K,D) with allow_multi_caption=True."
                        )
                elif isinstance(t, np.ndarray) and t.dtype == object:
                    # variable number of captions per sample: object array (N,)
                    if not self.allow_multi_caption:
                        raise RuntimeError(
                            f"{path} contains variable-length multi-caption text_emb "
                            f"but allow_multi_caption=False."
                        )
                else:
                    raise RuntimeError(f"Unsupported text_emb format in {path}")

                if len(v) != len(img_ids):
                    raise RuntimeError(f"vision_emb and img_ids length mismatch in {path}")
                if len(t) != len(img_ids):
                    raise RuntimeError(f"text_emb and img_ids length mismatch in {path}")
                if len(caption_ids) != len(img_ids):
                    raise RuntimeError(f"caption_ids and img_ids length mismatch in {path}")
                if len(label_ids) != len(img_ids):
                    raise RuntimeError(f"label_ids and img_ids length mismatch in {path}")
                if len(label_names) != len(img_ids):
                    raise RuntimeError(f"label_names and img_ids length mismatch in {path}")

                self.vision_embeddings.append(v)
                self.text_embeddings.extend(list(t))
                self.img_ids.append(img_ids)
                self.caption_ids.append(caption_ids)
                self.label_ids.extend(list(label_ids))
                self.label_names.extend(list(label_names))

        if found == 0:
            raise RuntimeError(f"No .npz files found in {precomputed_dir} for split '{split_name}'")

        self.vision_embeddings = np.concatenate(self.vision_embeddings, axis=0)
        self.img_ids = np.concatenate(self.img_ids, axis=0)
        self.caption_ids = np.concatenate(self.caption_ids, axis=0)

        assert len(self.text_embeddings) == len(self.vision_embeddings) == len(self.img_ids)
        assert len(self.label_ids) == len(self.vision_embeddings)
        assert len(self.label_names) == len(self.vision_embeddings)

        print(
            f"[Loaded COCO] {len(self)} samples from {precomputed_dir} | "
            f"vision_emb shape={self.vision_embeddings.shape}"
        )

    def __len__(self):
        return len(self.vision_embeddings)

    def _normalize_label_ids(self, x):
        """
        Converts legacy scalar label or new array-of-labels into a 1D LongTensor.
        """
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return torch.tensor([int(x.item())], dtype=torch.long)
            return torch.as_tensor(x.astype(np.int64), dtype=torch.long)

        if isinstance(x, (list, tuple)):
            return torch.as_tensor(np.array(x, dtype=np.int64), dtype=torch.long)

        # scalar legacy case
        return torch.tensor([int(x)], dtype=torch.long)

    def _normalize_label_names(self, x):
        """
        Converts legacy scalar label_name or new array-of-label_names into a Python list[str].
        """
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return [str(x.item())]
            return [str(y) for y in x.tolist()]

        if isinstance(x, (list, tuple)):
            return [str(y) for y in x]

        return [str(x)]

    def _get_text_embedding(self, idx):
        t = self.text_embeddings[idx]

        # case 1: single caption embedding, shape (D,)
        if isinstance(t, np.ndarray) and t.dtype != object and t.ndim == 1:
            return torch.as_tensor(t).float()

        # case 2: dense multi-caption embedding, shape (K, D)
        if isinstance(t, np.ndarray) and t.dtype != object and t.ndim == 2:
            k = min(self.text_index, t.shape[0] - 1)
            return torch.as_tensor(t[k]).float()

        # case 3: object item containing (K_i, D)
        if isinstance(t, np.ndarray) and t.dtype == object:
            t = np.array(t.tolist(), dtype=np.float32)
            k = min(self.text_index, t.shape[0] - 1)
            return torch.as_tensor(t[k]).float()

        # fallback for object entries loaded as Python list
        if isinstance(t, (list, tuple)):
            t = np.array(t, dtype=np.float32)
            if t.ndim == 1:
                return torch.as_tensor(t).float()
            k = min(self.text_index, t.shape[0] - 1)
            return torch.as_tensor(t[k]).float()

        raise RuntimeError(f"Unsupported text embedding format at idx={idx}: {type(t)}")

    def __getitem__(self, idx):
        text_emb = self._get_text_embedding(idx)
        vision_emb = torch.as_tensor(self.vision_embeddings[idx]).float()

        label_ids = self._normalize_label_ids(self.label_ids[idx])
        label_names = self._normalize_label_names(self.label_names[idx])

        if self.return_metadata:
            return (
                text_emb,
                vision_emb,
                label_ids,                  # 1D tensor, variable length
                int(self.img_ids[idx]),
                int(self.caption_ids[idx]),
                label_names,                # list[str]
            )

        return text_emb, vision_emb, label_ids


def mscoco_collate_fn(batch):
    """
    Keeps variable-length label tensors as a Python list.
    This is necessary because COCO labels are multi-label.
    """
    if len(batch[0]) == 3:
        texts, visions, label_ids = zip(*batch)
        return torch.stack(texts, dim=0), torch.stack(visions, dim=0), list(label_ids)

    texts, visions, label_ids, img_ids, caption_ids, label_names = zip(*batch)
    return (
        torch.stack(texts, dim=0),
        torch.stack(visions, dim=0),
        list(label_ids),
        torch.tensor(img_ids, dtype=torch.long),
        torch.tensor(caption_ids, dtype=torch.long),
        list(label_names),
    )


def make_loaders_mscoco(precomputed_dir, precomputed_dir_test, batch_size=256, num_workers=0):
    ds_train = MSCOCOEmbeddingsDatasetWithLabels(
        precomputed_dir,
        split_name="train_shard",
        require_labels=True,
    )
    ds_test = MSCOCOEmbeddingsDatasetWithLabels(
        precomputed_dir_test,
        split_name="val_shard",
        require_labels=True,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=mscoco_collate_fn,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mscoco_collate_fn,
    )
    return train_loader, test_loader