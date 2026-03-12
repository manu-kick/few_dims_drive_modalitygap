import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EmbeddingsDatasetWithLabels(Dataset):
    def __init__(
        self,
        precomputed_dir,
        split_name="train_shard",  # substring to match in NPZ filenames for this split
        return_label_name=False,
        allow_legacy_flickr_format=False,  # True se vuoi caricare anche (N,5,D)
        text_index=0,                      # usato solo per legacy
    ):
        """
        Args:
            precomputed_dir (str): Directory containing precomputed .npz files.
            split_name (str): Substring used to match NPZ shard filenames.
            return_label_name (bool): If True, also return string label.
            allow_legacy_flickr_format (bool): If True, supports old text_emb shapes (N,5,D).
            text_index (int): Which caption to use if legacy text_emb is (N,5,D).
        """

        self.text_embeddings = []
        self.vision_embeddings = []
        self.label_ids = []
        self.label_names = []

        self.return_label_name = return_label_name
        self.allow_legacy_flickr_format = allow_legacy_flickr_format
        self.text_index = text_index

        found = 0

        # Load all matching shards
        for fn in os.listdir(precomputed_dir):
            if fn.endswith(".npz") and (split_name in fn):
                found += 1
                path = os.path.join(precomputed_dir, fn)
                data = np.load(path, allow_pickle=True)

                # Required keys
                for k in ["vision_emb", "text_emb", "label_ids", "label_names"]:
                    if k not in data:
                        raise RuntimeError(f"Missing key '{k}' in {path}")

                v = data["vision_emb"]   # (N, D)
                t = data["text_emb"]     # (N, D) new OR (N,5,D) legacy
                y = data["label_ids"]    # (N,)
                n = data["label_names"]  # (N,)

                # Validate text_emb shape
                if t.ndim == 2:
                    # New format: (N, D)
                    pass
                elif t.ndim == 3 and self.allow_legacy_flickr_format:
                    # Legacy: (N, 5, D)
                    pass
                else:
                    raise RuntimeError(
                        f"Unexpected text_emb shape {t.shape} in {path}. "
                        f"Expected (N,D) or legacy (N,5,D) with allow_legacy_flickr_format=True."
                    )

                self.vision_embeddings.append(v)
                self.text_embeddings.append(t)
                self.label_ids.append(y)
                self.label_names.append(n)

        if found == 0:
            raise RuntimeError(f"No .npz files found in {precomputed_dir} for split '{split_name}'")

        # Concatenate shards
        self.vision_embeddings = np.concatenate(self.vision_embeddings, axis=0)  # (N, D)
        self.label_ids = np.concatenate(self.label_ids, axis=0)                  # (N,)
        self.label_names = np.concatenate(self.label_names, axis=0)              # (N,)

        # text embeddings may be (N,D) or (N,5,D)
        # if shards mix formats, this will error: that's good (keeps things consistent).
        self.text_embeddings = np.concatenate(self.text_embeddings, axis=0)

        # Basic length check
        assert len(self.vision_embeddings) == len(self.label_ids) == len(self.label_names)
        assert self.text_embeddings.shape[0] == len(self.vision_embeddings)

        print(f"[Loaded] {len(self)} samples from {precomputed_dir} | text_emb shape={self.text_embeddings.shape}")

    def __len__(self):
        return len(self.vision_embeddings)

    def __getitem__(self, idx):
        # text_emb new: (D,), legacy: (5, D)
        t = self.text_embeddings[idx]
        if t.ndim == 1:
            text_emb = t
        else:
            # legacy (5, D) -> pick one caption
            text_emb = t[self.text_index]

        vision_emb = self.vision_embeddings[idx]
        label_id = self.label_ids[idx]
        label_name = self.label_names[idx]

        text_emb = torch.as_tensor(text_emb).float()
        vision_emb = torch.as_tensor(vision_emb).float()
        label_id = torch.as_tensor(label_id).long()

        if self.return_label_name:
            return text_emb, vision_emb, label_id, label_name
        else:
            return text_emb, vision_emb, label_id
        

def make_loaders_cifar10(batch_size=256, precomputed_train_dir=None, precomputed_test_dir=None, seed=0, num_workers=0):
    ds_train = EmbeddingsDatasetWithLabels(precomputed_train_dir, split_name="train_shard")
    ds_test = EmbeddingsDatasetWithLabels(precomputed_test_dir, split_name="test_shard")
    n = len(ds_train)
    n_train = int(0.8 * n)
    n_test = n - n_train

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader