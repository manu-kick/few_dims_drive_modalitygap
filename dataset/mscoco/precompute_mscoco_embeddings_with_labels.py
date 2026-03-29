# precompute_mscoco_embeddings_with_labels.py
import os
import zipfile
import urllib.request
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

import open_clip
from pycocotools.coco import COCO


# =========================================================
# Utils
# =========================================================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _download_file(url: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"[OK] Already downloaded: {dst}")
        return

    print(f"[DOWNLOAD] {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"[SAVED] {dst}")


def _extract_zip(zip_path: str, extract_to: str) -> None:
    print(f"[EXTRACT] {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"[DONE] Extracted {zip_path}")


def ensure_mscoco_split(root_dir: str, split: str) -> Dict[str, str]:
    """
    Ensures that COCO images + annotations exist locally for the selected split.

    Supported splits:
      - "train" -> train2017
      - "val"   -> val2017

    Returns:
        dict with:
          - image_dir
          - instances_json
          - captions_json
          - split_folder
    """
    _ensure_dir(root_dir)

    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'val'.")

    split_folder = f"{split}2017"
    image_dir = os.path.join(root_dir, split_folder)

    annotations_dir = os.path.join(root_dir, "annotations")
    instances_json = os.path.join(annotations_dir, f"instances_{split_folder}.json")
    captions_json = os.path.join(annotations_dir, f"captions_{split_folder}.json")

    images_ok = os.path.isdir(image_dir) and len(os.listdir(image_dir)) > 0
    anns_ok = os.path.exists(instances_json) and os.path.exists(captions_json)

    if images_ok and anns_ok:
        print(f"[OK] MSCOCO {split_folder} already available.")
        return {
            "image_dir": image_dir,
            "instances_json": instances_json,
            "captions_json": captions_json,
            "split_folder": split_folder,
        }

    split_url = f"http://images.cocodataset.org/zips/{split_folder}.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    split_zip = os.path.join(root_dir, f"{split_folder}.zip")
    ann_zip = os.path.join(root_dir, "annotations_trainval2017.zip")

    if not images_ok:
        _download_file(split_url, split_zip)
        _extract_zip(split_zip, root_dir)

    if not anns_ok:
        _download_file(annotations_url, ann_zip)
        _extract_zip(ann_zip, root_dir)

    if not os.path.isdir(image_dir):
        raise RuntimeError(f"Could not find image directory after extraction: {image_dir}")
    if not os.path.exists(instances_json):
        raise RuntimeError(f"Could not find instances annotation file: {instances_json}")
    if not os.path.exists(captions_json):
        raise RuntimeError(f"Could not find captions annotation file: {captions_json}")

    print(f"[OK] MSCOCO {split_folder} is ready.")
    return {
        "image_dir": image_dir,
        "instances_json": instances_json,
        "captions_json": captions_json,
        "split_folder": split_folder,
    }


def _save_npz(
    out_path: str,
    text_emb: np.ndarray,
    vision_emb: np.ndarray,
    img_ids: np.ndarray,
    caption_ids: np.ndarray,
    label_ids: np.ndarray,
    label_names: np.ndarray,
    fns: List[str],
    meta: Dict[str, Any],
) -> None:
    np.savez_compressed(
        out_path,
        text_emb=text_emb,
        vision_emb=vision_emb,
        img_ids=img_ids.astype(np.int64),
        caption_ids=caption_ids.astype(np.int64),
        label_ids=label_ids,       # object array of variable-length int arrays
        label_names=label_names,   # object array of variable-length str arrays
        fns=np.array(fns, dtype=object),
        meta=np.array([meta], dtype=object),
    )


def build_clip(device: torch.device, model_name: str, pretrained: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def build_model_output_name(model_name: str, pretrained: str) -> str:
    return f"{model_name}___{pretrained}".replace("/", "_")


# =========================================================
# Precompute
# =========================================================
@torch.no_grad()
def precompute_coco_split(
    *,
    split_name: str,
    image_dir: str,
    instances_json: str,
    captions_json: str,
    clip_model,
    preprocess,
    tokenizer,
    device: torch.device,
    out_dir: str,
    normalize: bool,
    shard_size: Optional[int] = 4096,
    text_mode: str = "first_caption",   # "first_caption" | "all_captions"
    single_object_only: bool = False,
):
    _ensure_dir(out_dir)

    coco_instances = COCO(instances_json)
    coco_captions = COCO(captions_json)

    cat_id_to_name = {
        c["id"]: c["name"]
        for c in coco_instances.loadCats(coco_instances.getCatIds())
    }
    img_ids = sorted(coco_captions.getImgIds())

    shard_idx = 0
    shard_text, shard_vis = [], []
    shard_img_ids, shard_caption_ids = [], []
    shard_label_ids, shard_label_names = [], []
    shard_fns = []

    def flush():
        nonlocal shard_idx

        if len(shard_img_ids) == 0:
            return

        # text embeddings:
        # - first_caption  -> all (D,) so stack -> (N, D)
        # - all_captions   -> variable number of captions, keep object array (N,)
        if text_mode == "first_caption":
            text_np = np.stack(shard_text, axis=0)  # (N, D)
        elif text_mode == "all_captions":
            text_np = np.array(shard_text, dtype=object)  # each item is (K_i, D)
        else:
            raise ValueError(f"Unknown text_mode={text_mode}")

        vis_np = np.stack(shard_vis, axis=0)  # (N, D)
        img_ids_np = np.array(shard_img_ids, dtype=np.int64)
        caption_ids_np = np.array(shard_caption_ids, dtype=np.int64)

        # variable-length label arrays per image
        label_ids_np = np.array(shard_label_ids, dtype=object)
        label_names_np = np.array(shard_label_names, dtype=object)

        meta = {
            "dataset_name": "mscoco",
            "split": split_name,
            "num_items": int(len(img_ids_np)),
            "embedding_dim": int(vis_np.shape[-1]),
            "text_mode": text_mode,
            "normalization": bool(normalize),
            "single_object_only": bool(single_object_only),
            "instances_json": instances_json,
            "captions_json": captions_json,
            "labels_format": "variable_length_arrays",
        }

        out_path = os.path.join(out_dir, f"{split_name}_shard{shard_idx:03d}.npz")
        _save_npz(
            out_path=out_path,
            text_emb=text_np,
            vision_emb=vis_np,
            img_ids=img_ids_np,
            caption_ids=caption_ids_np,
            label_ids=label_ids_np,
            label_names=label_names_np,
            fns=shard_fns,
            meta=meta,
        )

        print(
            f"[SAVED] {out_path} | "
            f"N={len(img_ids_np)} | vision={vis_np.shape} | "
            f"text_type={type(text_np)}"
        )

        shard_idx += 1
        shard_text.clear()
        shard_vis.clear()
        shard_img_ids.clear()
        shard_caption_ids.clear()
        shard_label_ids.clear()
        shard_label_names.clear()
        shard_fns.clear()

    print(f"\n=== Precomputing MSCOCO embeddings: {split_name} ===")

    for img_id in tqdm(img_ids, desc=f"COCO {split_name}"):
        img_info = coco_captions.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(image_dir, file_name)

        if not os.path.exists(img_path):
            print(f"[WARNING] Missing image: {img_path}")
            continue

        # -------------------------
        # labels: keep ALL labels
        # -------------------------
        ann_ids = coco_instances.getAnnIds(imgIds=img_id)
        anns = coco_instances.loadAnns(ann_ids)

        unique_cats = sorted(set(int(a["category_id"]) for a in anns))
        unique_cat_names = [cat_id_to_name[cid] for cid in unique_cats]

        if single_object_only and len(unique_cats) != 1:
            continue

        # -------------------------
        # captions
        # -------------------------
        cap_ids = coco_captions.getAnnIds(imgIds=img_id)
        caps = coco_captions.loadAnns(cap_ids)
        captions = [c["caption"] for c in caps]
        caption_id = int(caps[0]["id"]) if len(caps) > 0 else -1

        if len(captions) == 0:
            continue

        # -------------------------
        # image embedding
        # -------------------------
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        vision_emb = clip_model.encode_image(img_tensor)

        if normalize:
            vision_emb = F.normalize(vision_emb, dim=-1)

        if text_mode == "first_caption":
            tokens = tokenizer([captions[0]]).to(device)
            text_emb = clip_model.encode_text(tokens)

            if normalize:
                text_emb = F.normalize(text_emb, dim=-1)

            shard_text.append(_to_numpy(text_emb)[0])   # (D,)

        elif text_mode == "all_captions":
            tokens = tokenizer(captions).to(device)
            text_emb = clip_model.encode_text(tokens)   # (K, D)

            if normalize:
                text_emb = F.normalize(text_emb, dim=-1)

            shard_text.append(_to_numpy(text_emb))      # (K, D), variable K

        else:
            raise ValueError(f"Unknown text_mode={text_mode}")

        shard_vis.append(_to_numpy(vision_emb)[0])      # (D,)
        shard_img_ids.append(int(img_id))
        shard_caption_ids.append(caption_id)

        # store ALL labels for this image
        shard_label_ids.append(np.array(unique_cats, dtype=np.int64))
        shard_label_names.append(np.array(unique_cat_names, dtype=object))

        shard_fns.append(file_name)

        if shard_size is not None and len(shard_img_ids) >= shard_size:
            flush()

    flush()
    print(f"\n✅ Done. Saved under: {out_dir}")


# =========================================================
# Main without CLI arguments
# =========================================================
def main():
    # -------------------------
    # CONFIG
    # -------------------------
    DATA_ROOT = "/mnt/media/emanuele/few_dimensions/dataset/mscoco/data/mscoco"
    SPLIT = "val"
    DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

    NORMALIZE = True
    SHARD_SIZE = 2048
    TEXT_MODE = "first_caption"   # "first_caption" or "all_captions"
    SINGLE_OBJECT_ONLY = False

    # [(ViT-B-32, laion2b_s34b_b79k), (ViT-B-16, laion2b_s34b_b88k), (RN50, laion2b_s34b_b82k)]
    CLIP_MODEL = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"

    MODEL_OUTPUT_NAME = build_model_output_name(CLIP_MODEL, CLIP_PRETRAINED)
    OUT_DIR = os.path.join(
        DATA_ROOT,
        MODEL_OUTPUT_NAME,
        f"precomputed_{SPLIT}2017_clip",
    )

    # -------------------------
    # Ensure dataset exists
    # -------------------------
    paths = ensure_mscoco_split(DATA_ROOT, SPLIT)
    IMAGE_DIR = paths["image_dir"]
    INSTANCES_JSON = paths["instances_json"]
    CAPTIONS_JSON = paths["captions_json"]

    print("\n[CONFIG]")
    print(f"DATA_ROOT       : {DATA_ROOT}")
    print(f"SPLIT           : {SPLIT}")
    print(f"IMAGE_DIR       : {IMAGE_DIR}")
    print(f"INSTANCES_JSON  : {INSTANCES_JSON}")
    print(f"CAPTIONS_JSON   : {CAPTIONS_JSON}")
    print(f"MODEL_OUTPUT    : {MODEL_OUTPUT_NAME}")
    print(f"OUT_DIR         : {OUT_DIR}")
    print(f"DEVICE          : {DEVICE}")
    print(f"TEXT_MODE       : {TEXT_MODE}")
    print(f"SINGLE_OBJECT   : {SINGLE_OBJECT_ONLY}")

    # -------------------------
    # Build CLIP
    # -------------------------
    device = torch.device(DEVICE)
    clip_model, preprocess, tokenizer = build_clip(device, CLIP_MODEL, CLIP_PRETRAINED)

    # -------------------------
    # Run precompute
    # -------------------------
    precompute_coco_split(
        split_name=SPLIT,
        image_dir=IMAGE_DIR,
        instances_json=INSTANCES_JSON,
        captions_json=CAPTIONS_JSON,
        clip_model=clip_model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        out_dir=OUT_DIR,
        normalize=NORMALIZE,
        shard_size=SHARD_SIZE,
        text_mode=TEXT_MODE,
        single_object_only=SINGLE_OBJECT_ONLY,
    )


if __name__ == "__main__":
    main()
