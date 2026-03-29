"""
Precompute MSCOCO CLIP embeddings and assign ImageNet top-1 labels
using a ConvNeXt classifier (similar to the Flickr30k pipeline).
"""

import os
import argparse
import zipfile
import urllib.request
import sys
from typing import Optional, Dict, Any, List
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(".."))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

import open_clip
from pycocotools.coco import COCO
from transformers import ConvNextImageProcessor, ConvNextForImageClassification


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
	label_logits: Optional[np.ndarray] = None,
) -> None:
	payload = {
		"text_emb": text_emb,
		"vision_emb": vision_emb,
		"img_ids": img_ids.astype(np.int64),
		"caption_ids": caption_ids.astype(np.int64),
		"label_ids": label_ids.astype(np.int64),
		"label_names": label_names,
		"fns": np.array(fns, dtype=object),
		"meta": np.array([meta], dtype=object),
	}
	if label_logits is not None:
		payload["label_logits"] = label_logits.astype(np.float32)

	np.savez_compressed(out_path, **payload)


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


# =========================================================
# Precompute
# =========================================================
@torch.no_grad()
def precompute_coco_split_with_imagenet_labels(
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
	convnext_processor=None,
	convnext_model=None,
	convnext_device: Optional[torch.device] = None,
	store_logits: bool = True,
) -> None:
	_ensure_dir(out_dir)

	if convnext_processor is None or convnext_model is None:
		raise ValueError("ConvNeXt processor/model not provided.")
	if convnext_device is None:
		convnext_device = device

	coco_captions = COCO(captions_json)
	img_ids = sorted(coco_captions.getImgIds())

	shard_idx = 0
	shard_text, shard_vis = [], []
	shard_img_ids, shard_caption_ids = [], []
	shard_label_ids, shard_label_names = [], []
	shard_label_logits: List[float] = []
	shard_fns = []

	def flush():
		nonlocal shard_idx

		if len(shard_img_ids) == 0:
			return

		if text_mode == "first_caption":
			text_np = np.stack(shard_text, axis=0)  # (N, D)
		elif text_mode == "all_captions":
			text_np = np.array(shard_text, dtype=object)  # each item is (K_i, D)
		else:
			raise ValueError(f"Unknown text_mode={text_mode}")

		vis_np = np.stack(shard_vis, axis=0)  # (N, D)
		img_ids_np = np.array(shard_img_ids, dtype=np.int64)
		caption_ids_np = np.array(shard_caption_ids, dtype=np.int64)
		label_ids_np = np.array(shard_label_ids, dtype=np.int64)
		label_names_np = np.array(shard_label_names, dtype=object)

		meta = {
			"dataset_name": "mscoco",
			"split": split_name,
			"num_items": int(len(img_ids_np)),
			"embedding_dim": int(vis_np.shape[-1]),
			"text_mode": text_mode,
			"normalization": bool(normalize),
			"instances_json": instances_json,
			"captions_json": captions_json,
			"label_model": getattr(convnext_model, "name_or_path", None),
			"label_space": "ImageNet-1k",
			"stores_label_logits": bool(store_logits),
		}

		out_path = os.path.join(out_dir, f"{split_name}_shard{shard_idx:03d}.npz")
		label_logits_np = None
		if store_logits:
			label_logits_np = np.array(shard_label_logits, dtype=np.float32)

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
			label_logits=label_logits_np,
		)

		print(
			f"[SAVED] {out_path} | "
			f"N={len(img_ids_np)} | vision={vis_np.shape} | "
			f"text_type={type(text_np)}"
			+ (" | logits=(N,)" if store_logits else "")
		)

		shard_idx += 1
		shard_text.clear()
		shard_vis.clear()
		shard_img_ids.clear()
		shard_caption_ids.clear()
		shard_label_ids.clear()
		shard_label_names.clear()
		shard_label_logits.clear()
		shard_fns.clear()

	print(f"\n=== Precomputing MSCOCO embeddings + ImageNet labels: {split_name} ===")

	for img_id in tqdm(img_ids, desc=f"COCO {split_name}"):
		img_info = coco_captions.loadImgs(img_id)[0]
		file_name = img_info["file_name"]
		img_path = os.path.join(image_dir, file_name)

		if not os.path.exists(img_path):
			print(f"[WARNING] Missing image: {img_path}")
			continue

		# captions
		cap_ids = coco_captions.getAnnIds(imgIds=img_id)
		caps = coco_captions.loadAnns(cap_ids)
		captions = [c["caption"] for c in caps]
		caption_id = int(caps[0]["id"]) if len(caps) > 0 else -1

		if len(captions) == 0:
			continue

		# image embedding
		img = Image.open(img_path).convert("RGB")
		img_tensor = preprocess(img).unsqueeze(0).to(device)
		vision_emb = clip_model.encode_image(img_tensor)

		if normalize:
			vision_emb = F.normalize(vision_emb, dim=-1)

		# text embedding
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
			shard_text.append(_to_numpy(text_emb))      # (K, D)
		else:
			raise ValueError(f"Unknown text_mode={text_mode}")

		# ImageNet label
		label_inputs = convnext_processor(images=img, return_tensors="pt")
		label_inputs = {k: v.to(convnext_device) for k, v in label_inputs.items()}
		logits = convnext_model(**label_inputs).logits  # (1, 1000)
		pred_id = int(torch.argmax(logits, dim=-1).item())
		pred_name = convnext_model.config.id2label.get(pred_id, str(pred_id))

		shard_vis.append(_to_numpy(vision_emb)[0])      # (D,)
		shard_img_ids.append(int(img_id))
		shard_caption_ids.append(caption_id)
		shard_label_ids.append(pred_id)
		shard_label_names.append(pred_name)
		if store_logits:
			shard_label_logits.append(float(logits[0, pred_id].detach().cpu().item()))

		shard_fns.append(file_name)

		if shard_size is not None and len(shard_img_ids) >= shard_size:
			flush()

	flush()
	print(f"\n✅ Done. Saved under: {out_dir}")


# =========================================================
# Main
# =========================================================
def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_root",
		type=str,
		default="./data/mscoco",
		help="Root directory where COCO data is stored/downloaded.",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="val",
		choices=["train", "val"],
		help="COCO split.",
	)
	parser.add_argument(
		"--out_dir",
		type=str,
		default=None,
		help="Output directory. Default: <data_root>/<model_name>/precomputed_<split>2017_clip_imagenet",
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help="Device for CLIP (e.g. 'cuda:0' or 'cpu'). Default: auto.",
	)
	parser.add_argument(
		"--label_device",
		type=str,
		default=None,
		help="Device for ConvNeXt (e.g. 'cuda:0' or 'cpu'). Default: same as CLIP.",
	)
	parser.add_argument(
		"--normalize",
		action="store_true",
		help="If set, L2-normalize CLIP embeddings.",
	)
	parser.add_argument(
		"--shard_size",
		type=int,
		default=2048,
		help="Images per shard.",
	)
	parser.add_argument(
		"--text_mode",
		type=str,
		default="first_caption",
		choices=["first_caption", "all_captions"],
		help="Use the first caption or all captions per image.",
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
		"--convnext_name",
		type=str,
		default="facebook/convnext-xlarge-384-22k-1k",
		help="HuggingFace ConvNeXt ImageNet-1k checkpoint.",
	)
	parser.add_argument(
		"--store_logits",
		action="store_true",
		default=True,
		help="If set, store label_logits (top-1 logit) in the NPZ.",
	)
	args = parser.parse_args()

	device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
	device = torch.device(device_str)
	label_device = torch.device(args.label_device) if args.label_device else device

	# Ensure dataset exists
	paths = ensure_mscoco_split(args.data_root, args.split)
	image_dir = paths["image_dir"]
	instances_json = paths["instances_json"]
	captions_json = paths["captions_json"]

	out_dir = args.out_dir
	if out_dir is None:
		model_output_name = build_model_output_name(args.clip_model, args.clip_pretrained)
		out_dir = os.path.join(
			args.data_root,
			model_output_name,
			f"precomputed_{args.split}2017_clip_imagenet",
		)
	else:
		model_output_name = build_model_output_name(args.clip_model, args.clip_pretrained)

	print("\n[CONFIG]")
	print(f"DATA_ROOT       : {args.data_root}")
	print(f"SPLIT           : {args.split}")
	print(f"IMAGE_DIR       : {image_dir}")
	print(f"INSTANCES_JSON  : {instances_json}")
	print(f"CAPTIONS_JSON   : {captions_json}")
	print(f"MODEL_OUTPUT    : {model_output_name}")
	print(f"OUT_DIR         : {out_dir}")
	print(f"DEVICE          : {device}")
	print(f"LABEL_DEVICE    : {label_device}")
	print(f"TEXT_MODE       : {args.text_mode}")

	# Build models
	clip_model, preprocess, tokenizer = build_clip(device, args.clip_model, args.clip_pretrained)
	convnext_processor, convnext_model = build_convnext(args.convnext_name, device=label_device)

	precompute_coco_split_with_imagenet_labels(
		split_name=args.split,
		image_dir=image_dir,
		instances_json=instances_json,
		captions_json=captions_json,
		clip_model=clip_model,
		preprocess=preprocess,
		tokenizer=tokenizer,
		device=device,
		out_dir=out_dir,
		normalize=bool(args.normalize),
		shard_size=args.shard_size,
		text_mode=args.text_mode,
		convnext_processor=convnext_processor,
		convnext_model=convnext_model,
		convnext_device=label_device,
		store_logits=bool(args.store_logits),
	)


if __name__ == "__main__":
	main()
