# precompute_embeddings.py
import os
import argparse
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import sys

# Add parent directory to Python path so we can import `models.*`
import open_clip

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(".."))
from dataloader import load_flickr30k_captions
from config_loader import load_configs_from_dir


# -----------------------------------
# ------ DATASET UTILS FUNCTIONS ----
# -----------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _save_npz(
    out_path: str,
    text_emb: np.ndarray,      # (N, 5, D)
    vision_emb: np.ndarray,    # (N, D)
    fns: List[str],            # len N
    meta: Dict[str, Any],
) -> None:
    fns_arr = np.array(fns, dtype=object)
    np.savez_compressed(
        out_path,
        text_emb=text_emb,
        vision_emb=vision_emb,
        fns=fns_arr,
        meta=np.array([meta], dtype=object),
    )


@torch.no_grad()
def precompute_split(
    cf,
    split_name: str,
    images_dir: str,
    captions_by_file: Dict[str, List[str]],
    text_encoder: Tuple[Any, Any],
    vision_encoder,
    device: torch.device,
    out_dir: str,
    normalize: bool,
    shard_size: Optional[int] = None,   # shard_size = NUM IMMAGINI per shard
    preprocess=None,                    # CLIP preprocess (consigliato)
) -> None:
    """
    Precompute embeddings per immagine, salvando 1 record per file:

      - fns[i]        = filename immagine
      - vision_emb[i] = embedding immagine (D,)
      - text_emb[i]   = embedding testo per 5 caption (5, D)

    Output shard:
      fns:        (N,)
      vision_emb: (N, D)
      text_emb:   (N, 5, D)
    """

    shard_fns: List[str] = []
    shard_vis: List[np.ndarray] = []      # list of (D,)
    shard_text: List[np.ndarray] = []     # list of (5, D)
    shard_idx = 0

    tokenizer = text_encoder[0]
    encode_text_fn = text_encoder[1]

    def flush_shard():
        nonlocal shard_idx, shard_fns, shard_vis, shard_text
        if len(shard_fns) == 0:
            return

        vis_np = np.stack(shard_vis, axis=0)          # (N, D)
        text_np = np.stack(shard_text, axis=0)        # (N, 5, D)

        meta = {
            "dataset_name": getattr(cf, "dataset_name", None),
            "embedding_dim": int(getattr(cf, "embedding_dim", -1)),
            "normalization": bool(normalize),
            "reproject_with_shared_head": bool(getattr(cf, "reproject_with_shared_head", False)),
            "split": split_name,
            "num_items": int(vis_np.shape[0]),
            "text_per_image": 5,
        }

        if shard_size is None:
            out_path = os.path.join(out_dir, f"{split_name}.npz")
        else:
            out_path = os.path.join(out_dir, f"{split_name}_shard{shard_idx:03d}.npz")

        _save_npz(out_path, text_np, vis_np, shard_fns, meta)
        print(f"[SAVED] {out_path} | N_images={vis_np.shape[0]} | text shape={text_np.shape}")

        shard_idx += 1
        shard_fns = []
        shard_vis = []
        shard_text = []

    print(f"\n=== Precomputing embeddings: {split_name} ===")

    num_images_in_current_shard = 0
    skipped_missing_caps = 0
    skipped_bad_images = 0
    skipped_no_caption = 0

    # iterazione deterministica
    for fn in tqdm(
        sorted(os.listdir(images_dir)),
        desc=f"{split_name} images",
        total=len(os.listdir(images_dir)),
    ):
        if fn not in captions_by_file:
            skipped_no_caption += 1
            continue

        caps = captions_by_file[fn]
        if len(caps) < 5:
            skipped_missing_caps += 1
            continue

        caps5 = caps[:5]

        img_path = os.path.join(images_dir, fn)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped_bad_images += 1
            continue

        if preprocess is None:
            raise ValueError("preprocess is None. Pass CLIP preprocess from open_clip.create_model_and_transforms().")

        img_tensor = preprocess(img).unsqueeze(0).to(device)  # (1,3,H,W)

        tokens = tokenizer(caps5).to(device)                  # (5, ...)
        text_emb = encode_text_fn(tokens)                     # (5, D)
        vision_emb = vision_encoder(img_tensor)               # (1, D)

        if normalize:
            text_emb = F.normalize(text_emb, dim=-1)
            vision_emb = F.normalize(vision_emb, dim=-1)

        text_np = _to_numpy(text_emb)         # (5, D)
        vis_np = _to_numpy(vision_emb)[0]     # (D,)

        shard_fns.append(fn)
        shard_vis.append(vis_np)
        shard_text.append(text_np)

        num_images_in_current_shard += 1
        if shard_size is not None and num_images_in_current_shard >= shard_size:
            flush_shard()
            num_images_in_current_shard = 0

    flush_shard()

    if skipped_no_caption:
        print(f"[INFO] {split_name}: skipped {skipped_no_caption} files without captions entry.")
    if skipped_missing_caps:
        print(f"[WARN] {split_name}: skipped {skipped_missing_caps} images with <5 captions.")
    if skipped_bad_images:
        print(f"[WARN] {split_name}: skipped {skipped_bad_images} unreadable/corrupt images.")


def build_models(cf, device: torch.device, use_clip=True):
    if use_clip:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        clip_model = clip_model.to(device)
        return clip_model, preprocess

    raise NotImplementedError(
        "Non-CLIP model building not implemented in this script. "
        "Please set use_clip=True or implement your own model loading logic here."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default="../config_dir/precompute_embedding_with_clip",
        help="Directory containing config files (same as main.py)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./precomputed_embeddings",
        help="Output directory."
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=4096,
        help="Number of images per shard. If set, saves multiple npz files."
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="flickr30k",
        help="Name used in output filenames/metadata. (No dataset split filtering here.)"
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    configs = load_configs_from_dir(args.config_dir)
    if len(configs) == 0:
        raise RuntimeError(f"No configs found in {args.config_dir}")

    cfg_path, cf = configs[0]
    print(f"[CONFIG] Using: {cfg_path}")

    captions_txt = os.path.join(cf.dataset_root, "captions.txt")
    images_dir = os.path.join(cf.dataset_root, "Images")
    captions_by_file = load_flickr30k_captions(captions_txt)

    device = torch.device(cf.device)
    _ensure_dir(args.out_dir)

    # models + preprocess CLIP
    model, preprocess = build_models(cf, device=device, use_clip=True)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # normalization decision
    normalize = True

    # output dir per run/config
    run_name = getattr(cf, "run", None) or os.path.splitext(os.path.basename(cfg_path))[0]
    out_dir = os.path.join(args.out_dir, run_name)
    _ensure_dir(out_dir)

    # freeze CLIP
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    precompute_split(
        cf=cf,
        split_name=args.split_name,
        images_dir=images_dir,
        captions_by_file=captions_by_file,
        text_encoder=(tokenizer, model.encode_text),
        vision_encoder=model.encode_image,
        device=device,
        out_dir=out_dir,
        normalize=normalize,
        shard_size=args.shard_size,
        preprocess=preprocess,
    )

    print(f"\n✅ Done. Embeddings saved under: {out_dir}")


if __name__ == "__main__":
    main()