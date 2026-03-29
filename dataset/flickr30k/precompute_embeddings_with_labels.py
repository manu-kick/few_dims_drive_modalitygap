# precompute_embeddings_with_labels.py
import os
import sys
import argparse
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip
from transformers import ConvNextImageProcessor, ConvNextForImageClassification

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

os.chdir(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloader import load_flickr30k_captions
from config_loader import load_configs_from_dir


# -----------------------------------
# ------ DATASET UTILS FUNCTIONS ----
# -----------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(os.path.join(PROJECT_ROOT, path)),
        os.path.abspath(os.path.join(SCRIPT_DIR, path)),
        os.path.abspath(path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _save_npz(
    out_path: str,
    text_emb: np.ndarray,         # (N, 5, D)
    vision_emb: np.ndarray,       # (N, D)
    fns: List[str],               # len N
    label_ids: np.ndarray,        # (N,)
    label_names: List[str],       # len N
    meta: Dict[str, Any],
) -> None:
    fns_arr = np.array(fns, dtype=object)
    label_names_arr = np.array(label_names, dtype=object)

    np.savez_compressed(
        out_path,
        text_emb=text_emb,
        vision_emb=vision_emb,
        fns=fns_arr,
        label_ids=label_ids.astype(np.int64),
        label_names=label_names_arr,
        meta=np.array([meta], dtype=object),
    )


# -----------------------------------
# -------- MODEL BUILDERS -----------
# -----------------------------------

def build_clip(device: torch.device, model_name: str, pretrained: str):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model, preprocess


def build_model_output_name(model_name: str, pretrained: str) -> str:
    return f"{model_name}___{pretrained}".replace("/", "_")


def build_convnext(
    model_name: str,
    device: torch.device,
):
    processor = ConvNextImageProcessor.from_pretrained(model_name)
    model = ConvNextForImageClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model


# -----------------------------------
# --------- MAIN PRECOMPUTE ---------
# -----------------------------------

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
    preprocess=None,                    # CLIP preprocess (required)
    convnext_processor=None,
    convnext_model=None,
    convnext_device: Optional[torch.device] = None,
    store_logits: bool = False,         # optional: store top-1 logit (confidence-ish)
) -> None:
    """
    Precompute, per image (1 record per file):
      - fns[i]        = filename immagine
      - vision_emb[i] = CLIP image embedding (D,)
      - text_emb[i]   = CLIP text embedding per 5 caption (5, D)
      - label_ids[i]  = ConvNeXt top-1 class id (int)
      - label_names[i]= ConvNeXt top-1 class label (string)
      - (optional) label_logits[i] = ConvNeXt top-1 logit (float)

    Output shard:
      fns:        (N,)
      vision_emb: (N, D)
      text_emb:   (N, 5, D)
      label_ids:  (N,)
      label_names:(N,)
      (+ optional) label_logits:(N,)
    """

    if preprocess is None:
        raise ValueError("preprocess is None. Pass CLIP preprocess from open_clip.create_model_and_transforms().")
    if convnext_processor is None or convnext_model is None:
        raise ValueError("ConvNeXt processor/model not provided.")
    if convnext_device is None:
        convnext_device = device

    shard_fns: List[str] = []
    shard_vis: List[np.ndarray] = []         # list of (D,)
    shard_text: List[np.ndarray] = []        # list of (5, D)
    shard_label_ids: List[int] = []
    shard_label_names: List[str] = []
    shard_label_logits: List[float] = []

    shard_idx = 0
    tokenizer = text_encoder[0]
    encode_text_fn = text_encoder[1]

    def flush_shard():
        nonlocal shard_idx, shard_fns, shard_vis, shard_text
        nonlocal shard_label_ids, shard_label_names, shard_label_logits

        if len(shard_fns) == 0:
            return

        vis_np = np.stack(shard_vis, axis=0)          # (N, D)
        text_np = np.stack(shard_text, axis=0)        # (N, 5, D)
        label_ids_np = np.array(shard_label_ids, dtype=np.int64)  # (N,)

        meta = {
            "dataset_name": getattr(cf, "dataset_name", None),
            "embedding_dim": int(getattr(cf, "embedding_dim", -1)),
            "normalization": bool(normalize),
            "reproject_with_shared_head": bool(getattr(cf, "reproject_with_shared_head", False)),
            "split": split_name,
            "num_items": int(vis_np.shape[0]),
            "text_per_image": 5,
            "clip_model": getattr(cf, "clip_model_name", None),
            "clip_pretrained": getattr(cf, "clip_pretrained_name", None),
            "label_model": getattr(cf, "label_model_name", None),
            "label_space": "ImageNet-1k",
            "stores_label_logits": bool(store_logits),
        }

        if shard_size is None:
            out_path = os.path.join(out_dir, f"{split_name}.npz")
        else:
            out_path = os.path.join(out_dir, f"{split_name}_shard{shard_idx:03d}.npz")

        _save_npz(
            out_path=out_path,
            text_emb=text_np,
            vision_emb=vis_np,
            fns=shard_fns,
            label_ids=label_ids_np,
            label_names=shard_label_names,
            meta=meta,
        )

        # If you also want to store logits, write a sidecar npz or extend the format.
        # Here we extend by re-saving with logits (still compressed), without breaking your core arrays.
        if store_logits:
            with np.load(out_path, allow_pickle=True) as data:
                tmp = dict(data.items())
            tmp["label_logits"] = np.array(shard_label_logits, dtype=np.float32)
            np.savez_compressed(out_path, **tmp)

        print(
            f"[SAVED] {out_path} | N_images={vis_np.shape[0]} | "
            f"text={text_np.shape} | labels={label_ids_np.shape}"
            + (" | logits=(N,)" if store_logits else "")
        )

        shard_idx += 1
        shard_fns = []
        shard_vis = []
        shard_text = []
        shard_label_ids = []
        shard_label_names = []
        shard_label_logits = []

    print(f"\n=== Precomputing embeddings + labels: {split_name} ===")

    num_images_in_current_shard = 0
    skipped_missing_caps = 0
    skipped_bad_images = 0
    skipped_no_caption = 0

    # deterministic iteration
    all_files = sorted(os.listdir(images_dir))

    for fn in tqdm(all_files, desc=f"{split_name} images", total=len(all_files)):
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

        # ---- CLIP embeddings ----
        img_tensor = preprocess(img).unsqueeze(0).to(device)  # (1,3,H,W)

        tokens = tokenizer(caps5).to(device)                  # (5, ...)
        text_emb = encode_text_fn(tokens)                     # (5, D)
        vision_emb = vision_encoder(img_tensor)               # (1, D)

        if normalize:
            text_emb = F.normalize(text_emb, dim=-1)
            vision_emb = F.normalize(vision_emb, dim=-1)

        text_np = _to_numpy(text_emb)         # (5, D)
        vis_np = _to_numpy(vision_emb)[0]     # (D,)

        # ---- ConvNeXt label ----
        # ConvNeXt processor expects PIL image or numpy, returns tensors
        label_inputs = convnext_processor(images=img, return_tensors="pt")
        label_inputs = {k: v.to(convnext_device) for k, v in label_inputs.items()}

        logits = convnext_model(**label_inputs).logits  # (1, 1000)
        pred_id = int(torch.argmax(logits, dim=-1).item())
        pred_name = convnext_model.config.id2label.get(pred_id, str(pred_id))

        if store_logits:
            top1_logit = float(logits[0, pred_id].detach().cpu().item())

        # ---- Append to shard ----
        shard_fns.append(fn)
        shard_vis.append(vis_np)
        shard_text.append(text_np)
        shard_label_ids.append(pred_id)
        shard_label_names.append(pred_name)
        if store_logits:
            shard_label_logits.append(top1_logit)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default="config_dir/precompute_embedding_with_labels",
        help="Directory containing config files (same as main.py)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./precomputed_embeddings_with_labels",
        help="Output root directory. Default layout: <out_dir>/<model_name>/"
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
        default="flickr30k_labels",
        help="Name used in output filenames/metadata. (No dataset split filtering here.)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, L2-normalize CLIP embeddings."
    )
    parser.add_argument(
        "--convnext_name",
        type=str,
        default="facebook/convnext-xlarge-384-22k-1k",
        help="HuggingFace ConvNeXt checkpoint for ImageNet-1k classification."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for CLIP (e.g. 'cuda:0' or 'cpu'). Default: cf.device.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained tag.",
    )
    parser.add_argument(
        "--label_device",
        type=str,
        default=None,
        help="Device for ConvNeXt (e.g. 'cuda:0' or 'cpu'). Default: same as cf.device."
    )
    parser.add_argument(
        "--store_logits",
        action="store_true",
        default=True,
        help="If set, also store label_logits (top-1 logit) in the NPZ."
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    config_dir = _resolve_path(args.config_dir)
    configs = load_configs_from_dir(config_dir)
    if len(configs) == 0:
        raise RuntimeError(f"No configs found in {config_dir}")

    cfg_path, cf = configs[0]
    print(f"[CONFIG] Using: {cfg_path}")

    # dataset
    captions_txt = os.path.join(cf.dataset_root, "captions.txt")
    images_dir = os.path.join(cf.dataset_root, "Images")
    captions_by_file = load_flickr30k_captions(captions_txt)

    # devices
    device = torch.device(args.device) if args.device is not None else torch.device(cf.device)
    label_device = torch.device(args.label_device) if args.label_device is not None else device

    _ensure_dir(args.out_dir)

    # CLIP
    clip_model, preprocess = build_clip(
        device=device,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    # ConvNeXt
    convnext_processor, convnext_model = build_convnext(args.convnext_name, device=label_device)

    # Save the label model name in meta
    setattr(cf, "clip_model_name", args.clip_model)
    setattr(cf, "clip_pretrained_name", args.clip_pretrained)
    setattr(cf, "label_model_name", args.convnext_name)

    # normalization
    normalize = bool(args.normalize) if "normalize" in args else True
    # If you want to preserve old behavior (always normalize), force:
    # normalize = True

    model_output_name = build_model_output_name(args.clip_model, args.clip_pretrained)
    out_dir = os.path.join(args.out_dir, model_output_name)
    _ensure_dir(out_dir)

    print(f"[CLIP] model={args.clip_model}")
    print(f"[CLIP] pretrained={args.clip_pretrained}")
    print(f"[CLIP] device={device}")
    print(f"[LABEL] device={label_device}")
    print(f"[OUT] model_output_name={model_output_name}")

    precompute_split(
        cf=cf,
        split_name=args.split_name,
        images_dir=images_dir,
        captions_by_file=captions_by_file,
        text_encoder=(tokenizer, clip_model.encode_text),
        vision_encoder=clip_model.encode_image,
        device=device,
        out_dir=out_dir,
        normalize=normalize,
        shard_size=args.shard_size,
        preprocess=preprocess,
        convnext_processor=convnext_processor,
        convnext_model=convnext_model,
        convnext_device=label_device,
        store_logits=args.store_logits,
    )

    print(f"\n✅ Done. Embeddings+labels saved under: {out_dir}")
    print("NPZ keys: text_emb, vision_emb, fns, label_ids, label_names, meta" + (", label_logits" if args.store_logits else ""))


if __name__ == "__main__":
    main()
