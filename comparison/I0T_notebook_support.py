import json
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from analysis.modality_gap import compute_gap
from comparison.I0T_implementation import (
    apply_i0t_with_statistics,
    fit_i0t_statistics,
    mean_rmg_over_batches,
)
from dataset.flickr30k.dataloader_embeddings_with_labels import EmbeddingsDatasetWithLabels
from dataset.mscoco.mscoco_dataloader_with_imagenet_labels import (
    MSCOCOEmbeddingsDatasetWithImageNetLabels,
    mscoco_imagenet_collate_fn,
)
from dataset.msrvtt.msrvtt_dataloaderv2 import (
    MSRVTTEmbeddingsDatasetV2,
    msrvtt_v2_collate_fn,
)
from metrics.clustering import (
    _clustering_metrics_two_modalities_flickr30k,
    clustering_metrics_two_modalities_mscoco_imagenet_labels,
    clustering_metrics_two_modalities_msrvtt,
)
from metrics.retrieval import compute_retrieval, retrieval


DEFAULT_SEED = 123

DEFAULT_PATHS = {
    "flickr30k": "/mnt/media/emanuele/few_dimensions/dataset/flickr30k/precomputed_embeddings_with_labels/clip_vit_b_32___laion2b_s34b_b79k",
    "mscoco_train": "/mnt/media/emanuele/few_dimensions/dataset/mscoco/data/mscoco/precomputed_train2017_clip_imagenet",
    "mscoco_val": "/mnt/media/emanuele/few_dimensions/dataset/mscoco/data/mscoco/precomputed_val2017_clip_imagenet",
    "msrvtt_train": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/ViT-B-32___laion2b_s34b_b79k_v2/precomputed_train",
    "msrvtt_test": "/mnt/media/emanuele/few_dimensions/dataset/msrvtt/ViT-B-32___laion2b_s34b_b79k_v2/precomputed_test",
}

DEFAULT_MODEL_NAME = "ViT-B-32___laion2b_s34b_b79k"


def build_embedding_paths(model_name=DEFAULT_MODEL_NAME):
    normalized_model_name = resolve_embedding_model_name(model_name=model_name, path_hints=[])
    return {
        "flickr30k": f"/mnt/media/emanuele/few_dimensions/dataset/flickr30k/precomputed_embeddings_with_labels/{model_name}",
        "mscoco_train": f"/mnt/media/emanuele/few_dimensions/dataset/mscoco/data/mscoco/{model_name}/precomputed_train2017_clip_imagenet",
        "mscoco_val": f"/mnt/media/emanuele/few_dimensions/dataset/mscoco/data/mscoco/{model_name}/precomputed_val2017_clip_imagenet",
        "msrvtt_train": f"/mnt/media/emanuele/few_dimensions/dataset/msrvtt/{normalized_model_name}_v2/precomputed_train",
        "msrvtt_test": f"/mnt/media/emanuele/few_dimensions/dataset/msrvtt/{normalized_model_name}_v2/precomputed_test",
    }


DEFAULT_CONFIGS = {
    "flickr30k": {
        "seed": DEFAULT_SEED,
        "n_fit": 10_000,
        "batch_size": 2048,
        "num_workers": 0,
        "min_samples_per_class": 10,
        "test_size": 0.20,
        "plot_pca": True,
        "max_cluster_samples": 5_000,
    },
    "mscoco_imagenet": {
        "seed": DEFAULT_SEED,
        "n_fit": 10_000,
        "batch_size": 256,
        "num_workers": 0,
        "min_train_samples_per_class": 10,
        "max_eval_batches": None,
        "plot_pca": True,
        "max_cluster_samples": 3_000,
    },
    "msrvtt_v2": {
        "seed": DEFAULT_SEED,
        "n_fit": 10_000,
        "batch_size": 256,
        "num_workers": 0,
        "max_eval_batches": None,
        "plot_pca": True,
        "max_cluster_samples": 3_000,
        "n_clusters": 20,
    },
}


def set_global_seed(seed=DEFAULT_SEED, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    return torch.Generator().manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _to_scalar(value):
    if isinstance(value, dict):
        value = value.get("text_vision", next(iter(value.values())))
    if torch.is_tensor(value):
        value = value.item()
    return float(value)


def _normalize_pair(x, y):
    return F.normalize(x, dim=-1), F.normalize(y, dim=-1)


def _parse_loader_batch(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("Expected batches in the form (text, vision, ...).")

    text = batch[0]
    vision = batch[1]
    labels = batch[2] if len(batch) >= 3 else None
    return text, vision, labels


def collect_paired_embeddings(loader, max_samples=None, device="cpu", max_batches=None):
    text_buf, vision_buf, label_buf = [], [], []
    seen = 0
    batch_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Collect paired embeddings"):
            text_b, vision_b, labels_b = _parse_loader_batch(batch)
            text_b = text_b.to(device)
            vision_b = vision_b.to(device)

            if max_samples is not None:
                remaining = max_samples - seen
                if remaining <= 0:
                    break
                take = min(text_b.shape[0], remaining)
                text_b = text_b[:take]
                vision_b = vision_b[:take]
                if labels_b is not None:
                    labels_b = labels_b[:take]
                seen += take

            text_buf.append(text_b.detach().cpu())
            vision_buf.append(vision_b.detach().cpu())

            if labels_b is not None:
                if not torch.is_tensor(labels_b):
                    labels_b = torch.as_tensor(labels_b)
                label_buf.append(labels_b.detach().cpu())

            batch_idx += 1
            if max_batches is not None and batch_idx >= max_batches:
                break

    if not text_buf:
        raise ValueError("No embeddings were collected from the dataloader.")

    text_all = torch.cat(text_buf, dim=0)
    vision_all = torch.cat(vision_buf, dim=0)
    labels_all = torch.cat(label_buf, dim=0) if label_buf else None
    return text_all, vision_all, labels_all


def fit_i0t_statistics_from_loader(loader, n_fit=None, device="cpu"):
    text_all, vision_all, _ = collect_paired_embeddings(
        loader,
        max_samples=n_fit,
        device=device,
    )

    statistics = fit_i0t_statistics(text_all, vision_all)
    statistics["n_fit"] = int(text_all.shape[0])
    statistics["embedding_dim"] = int(text_all.shape[1])
    statistics["mean_x_norm"] = float(torch.norm(statistics["mean_x"], p=2).item())
    statistics["mean_y_norm"] = float(torch.norm(statistics["mean_y"], p=2).item())
    return statistics


def apply_i0t_from_statistics(text_embeddings, vision_embeddings, statistics):
    text_norm, vision_norm = _normalize_pair(text_embeddings, vision_embeddings)
    text_i0t, vision_i0t = apply_i0t_with_statistics(
        text_embeddings,
        vision_embeddings,
        statistics,
    )
    return text_norm, vision_norm, text_i0t, vision_i0t


def _compute_gap_bundle(text_embeddings, vision_embeddings):
    metrics = ["L2M", "L2I", "cosineTP"]
    out = {metric: float(_to_scalar(compute_gap(metric, text_embeddings, vision_embeddings, iterations=None))) for metric in metrics}
    return out


def plot_pca_modalities(embeddings_2n, labels_2n, title, seed=DEFAULT_SEED, max_points=4000):
    modality = np.concatenate(
        [
            np.zeros(embeddings_2n.shape[0] // 2, dtype=np.int32),
            np.ones(embeddings_2n.shape[0] - embeddings_2n.shape[0] // 2, dtype=np.int32),
        ]
    )

    if embeddings_2n.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(embeddings_2n.shape[0], size=max_points, replace=False)
        embeddings_2n = embeddings_2n[idx]
        labels_2n = labels_2n[idx]
        modality = modality[idx]

    z = PCA(n_components=2, random_state=seed).fit_transform(embeddings_2n)

    plt.figure(figsize=(6, 5))
    sc_vis = plt.scatter(
        z[modality == 1, 0],
        z[modality == 1, 1],
        c=labels_2n[modality == 1],
        s=20,
        marker="o",
        alpha=0.7,
        label="vision",
    )
    plt.scatter(
        z[modality == 0, 0],
        z[modality == 0, 1],
        c=labels_2n[modality == 0],
        s=20,
        marker="x",
        alpha=0.8,
        cmap=sc_vis.cmap,
        norm=sc_vis.norm,
        label="text",
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.colorbar(sc_vis, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def _subset_label_counts(dataset):
    counts = {}
    for idx in range(len(dataset)):
        _, _, label = dataset[idx]
        label = int(label.item()) if torch.is_tensor(label) else int(label)
        counts[label] = counts.get(label, 0) + 1
    return counts


def build_flickr30k_i0t_experiment(
    seed=DEFAULT_SEED,
    batch_size=2048,
    num_workers=0,
    min_samples_per_class=10,
    test_size=0.20,
    precomputed_dir=None,
):
    precomputed_dir = precomputed_dir or DEFAULT_PATHS["flickr30k"]
    generator = torch.Generator().manual_seed(seed)

    dataset = EmbeddingsDatasetWithLabels(
        precomputed_dir=precomputed_dir,
        split_name="flickr30k",
    )

    class_counts = _subset_label_counts(dataset)
    keep_classes = {label for label, count in class_counts.items() if count >= min_samples_per_class}
    filtered_indices = [
        idx for idx in range(len(dataset))
        if int(dataset[idx][2].item()) in keep_classes
    ]
    filtered_dataset = Subset(dataset, filtered_indices)

    filtered_labels = np.array([int(dataset[idx][2].item()) for idx in filtered_indices], dtype=np.int64)
    all_indices = np.arange(len(filtered_dataset))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_size,
        stratify=filtered_labels,
        random_state=seed,
    )

    train_dataset = Subset(filtered_dataset, train_idx)
    test_dataset = Subset(filtered_dataset, test_idx)

    train_class_counts = _subset_label_counts(train_dataset)
    test_class_counts = _subset_label_counts(test_dataset)
    assert set(train_class_counts.keys()) == set(test_class_counts.keys())

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        generator=generator,
        **loader_kwargs,
    )

    return {
        "dataset_name": "flickr30k",
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "n_clusters": len(train_class_counts),
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
        "kept_classes": sorted(train_class_counts.keys()),
    }


def build_mscoco_i0t_experiment(
    seed=DEFAULT_SEED,
    batch_size=256,
    num_workers=0,
    min_train_samples_per_class=10,
    train_dir=None,
    val_dir=None,
):
    train_dir = train_dir or DEFAULT_PATHS["mscoco_train"]
    val_dir = val_dir or DEFAULT_PATHS["mscoco_val"]
    generator = torch.Generator().manual_seed(seed)

    train_dataset = MSCOCOEmbeddingsDatasetWithImageNetLabels(
        train_dir,
        split_name="train_shard",
        return_label_name=False,
    )
    test_dataset = MSCOCOEmbeddingsDatasetWithImageNetLabels(
        val_dir,
        split_name="val_shard",
        return_label_name=False,
    )

    train_counts = _subset_label_counts(train_dataset)
    test_counts = _subset_label_counts(test_dataset)

    keep_classes = {
        label for label in test_counts
        if train_counts.get(label, 0) >= min_train_samples_per_class
    }

    filtered_train = Subset(
        train_dataset,
        [idx for idx in range(len(train_dataset)) if int(train_dataset[idx][2].item()) in keep_classes],
    )
    filtered_test = Subset(
        test_dataset,
        [idx for idx in range(len(test_dataset)) if int(test_dataset[idx][2].item()) in keep_classes],
    )

    filtered_train_counts = _subset_label_counts(filtered_train)
    filtered_test_counts = _subset_label_counts(filtered_test)
    assert set(filtered_train_counts.keys()) == set(filtered_test_counts.keys())

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": mscoco_imagenet_collate_fn,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        filtered_train,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        filtered_test,
        shuffle=False,
        generator=generator,
        **loader_kwargs,
    )

    return {
        "dataset_name": "mscoco_imagenet_labels",
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_size": len(filtered_train),
        "test_size": len(filtered_test),
        "n_clusters": len(filtered_train_counts),
        "train_class_counts": filtered_train_counts,
        "test_class_counts": filtered_test_counts,
        "kept_classes": sorted(filtered_train_counts.keys()),
    }


def build_msrvtt_v2_i0t_experiment(
    seed=DEFAULT_SEED,
    batch_size=256,
    num_workers=0,
    train_dir=None,
    test_dir=None,
):
    train_dir = train_dir or DEFAULT_PATHS["msrvtt_train"]
    test_dir = test_dir or DEFAULT_PATHS["msrvtt_test"]
    generator = torch.Generator().manual_seed(seed)

    train_dataset = MSRVTTEmbeddingsDatasetV2(
        train_dir,
        split_name="train_shard",
        return_metadata=False,
    )
    test_dataset = MSRVTTEmbeddingsDatasetV2(
        test_dir,
        split_name="test_shard",
        return_metadata=False,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": msrvtt_v2_collate_fn,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        generator=generator,
        **loader_kwargs,
    )

    train_counts = _subset_label_counts(train_dataset)
    test_counts = _subset_label_counts(test_dataset)

    return {
        "dataset_name": "msrvtt",
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "n_clusters": len(train_counts),
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
    }


def evaluate_i0t_flickr30k(
    test_loader,
    statistics,
    device="cpu",
    max_cluster_samples=5_000,
    plot_pca=True,
    seed=DEFAULT_SEED,
):
    text_all, vision_all, labels_all = collect_paired_embeddings(test_loader, device="cpu")
    labels_all = labels_all.long()

    text_norm, vision_norm, text_i0t, vision_i0t = apply_i0t_from_statistics(
        text_all.to(device),
        vision_all.to(device),
        statistics,
    )
    text_norm = text_norm.cpu()
    vision_norm = vision_norm.cpu()
    text_i0t = text_i0t.cpu()
    vision_i0t = vision_i0t.cpu()

    retrieval_orig = {
        1: float(retrieval(text_norm, vision_norm, top_k=1)),
        5: float(retrieval(text_norm, vision_norm, top_k=5)),
        10: float(retrieval(text_norm, vision_norm, top_k=10)),
    }
    retrieval_i0t = {
        1: float(retrieval(text_i0t, vision_i0t, top_k=1)),
        5: float(retrieval(text_i0t, vision_i0t, top_k=5)),
        10: float(retrieval(text_i0t, vision_i0t, top_k=10)),
    }

    gaps_orig = _compute_gap_bundle(text_norm, vision_norm)
    gaps_i0t = _compute_gap_bundle(text_i0t, vision_i0t)
    mean_rmg = mean_rmg_over_batches(test_loader, statistics, device=device)
    gaps_orig["RMG"] = mean_rmg["orig"]
    gaps_i0t["RMG"] = mean_rmg["i0t"]

    cluster_limit = min(int(max_cluster_samples), text_norm.shape[0])
    clustering_orig = _clustering_metrics_two_modalities_flickr30k(
        text_norm[:cluster_limit],
        vision_norm[:cluster_limit],
        labels_all[:cluster_limit],
        n_clusters=len(torch.unique(labels_all)),
        random_state=seed,
    )
    clustering_i0t = _clustering_metrics_two_modalities_flickr30k(
        text_i0t[:cluster_limit],
        vision_i0t[:cluster_limit],
        labels_all[:cluster_limit],
        n_clusters=len(torch.unique(labels_all)),
        random_state=seed,
    )

    if plot_pca:
        labels_np = labels_all[:cluster_limit].numpy()
        labels_2n = np.concatenate([labels_np, labels_np], axis=0)
        emb_orig = torch.cat([text_norm[:cluster_limit], vision_norm[:cluster_limit]], dim=0).numpy()
        emb_i0t = torch.cat([text_i0t[:cluster_limit], vision_i0t[:cluster_limit]], dim=0).numpy()
        plot_pca_modalities(emb_orig, labels_2n, "Flickr30k ImageNet labels PCA 2D (original)", seed=seed)
        plot_pca_modalities(emb_i0t, labels_2n, "Flickr30k ImageNet labels PCA 2D (I0T)", seed=seed)

    return {
        "method": "I0T",
        "dataset": "flickr30k",
        "retrieval_orig": retrieval_orig,
        "retrieval_i0t": retrieval_i0t,
        "gaps_orig": gaps_orig,
        "gaps_i0t": gaps_i0t,
        "mean_rmg_over_batches": mean_rmg,
        "statistics": statistics,
        "clustering_orig": clustering_orig,
        "clustering_i0t": clustering_i0t,
    }


def evaluate_i0t_mscoco_imagenet(
    test_loader,
    statistics,
    device="cpu",
    max_cluster_samples=3_000,
    max_eval_batches=None,
    plot_pca=True,
    seed=DEFAULT_SEED,
):
    text_all, vision_all, labels_all = collect_paired_embeddings(
        test_loader,
        device="cpu",
        max_batches=max_eval_batches,
    )
    labels_all = labels_all.long()

    text_norm, vision_norm, text_i0t, vision_i0t = apply_i0t_from_statistics(
        text_all.to(device),
        vision_all.to(device),
        statistics,
    )
    text_norm = text_norm.cpu()
    vision_norm = vision_norm.cpu()
    text_i0t = text_i0t.cpu()
    vision_i0t = vision_i0t.cpu()

    retrieval_orig = {
        1: float(_to_scalar(compute_retrieval("mscoco_imagenet_labels", (text_norm, vision_norm), top_k=1))),
        5: float(_to_scalar(compute_retrieval("mscoco_imagenet_labels", (text_norm, vision_norm), top_k=5))),
        10: float(_to_scalar(compute_retrieval("mscoco_imagenet_labels", (text_norm, vision_norm), top_k=10))),
    }
    retrieval_i0t = {
        1: float(_to_scalar(compute_retrieval("mscoco_imagenet_labels", (text_i0t, vision_i0t), top_k=1))),
        5: float(_to_scalar(compute_retrieval("mscoco_imagenet_labels", (text_i0t, vision_i0t), top_k=5))),
        10: float(_to_scalar(compute_retrieval("mscoco_imagenet_labels", (text_i0t, vision_i0t), top_k=10))),
    }

    gaps_orig = _compute_gap_bundle(text_norm, vision_norm)
    gaps_i0t = _compute_gap_bundle(text_i0t, vision_i0t)
    mean_rmg = mean_rmg_over_batches(
        test_loader,
        statistics,
        device=device,
        max_eval_batches=max_eval_batches,
    )
    gaps_orig["RMG"] = mean_rmg["orig"]
    gaps_i0t["RMG"] = mean_rmg["i0t"]

    if max_cluster_samples is not None and text_norm.shape[0] > max_cluster_samples:
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(text_norm.shape[0], generator=rng)[:max_cluster_samples]
        text_norm_cl = text_norm[idx]
        vision_norm_cl = vision_norm[idx]
        text_i0t_cl = text_i0t[idx]
        vision_i0t_cl = vision_i0t[idx]
        labels_cl = labels_all[idx]
    else:
        text_norm_cl = text_norm
        vision_norm_cl = vision_norm
        text_i0t_cl = text_i0t
        vision_i0t_cl = vision_i0t
        labels_cl = labels_all

    n_clusters = len(torch.unique(labels_all))
    clustering_orig = clustering_metrics_two_modalities_mscoco_imagenet_labels(
        text_norm_cl,
        vision_norm_cl,
        labels_cl,
        n_clusters=n_clusters,
        random_state=seed,
    )
    clustering_i0t = clustering_metrics_two_modalities_mscoco_imagenet_labels(
        text_i0t_cl,
        vision_i0t_cl,
        labels_cl,
        n_clusters=n_clusters,
        random_state=seed,
    )

    if plot_pca:
        labels_np = labels_cl.numpy()
        labels_2n = np.concatenate([labels_np, labels_np], axis=0)
        emb_orig = torch.cat([text_norm_cl, vision_norm_cl], dim=0).numpy()
        emb_i0t = torch.cat([text_i0t_cl, vision_i0t_cl], dim=0).numpy()
        plot_pca_modalities(emb_orig, labels_2n, "MSCOCO ImageNet labels PCA 2D (original)", seed=seed)
        plot_pca_modalities(emb_i0t, labels_2n, "MSCOCO ImageNet labels PCA 2D (I0T)", seed=seed)

    return {
        "method": "I0T",
        "dataset": "mscoco_imagenet",
        "retrieval_orig": retrieval_orig,
        "retrieval_i0t": retrieval_i0t,
        "gaps_orig": gaps_orig,
        "gaps_i0t": gaps_i0t,
        "mean_rmg_over_batches": mean_rmg,
        "statistics": statistics,
        "clustering_orig": clustering_orig,
        "clustering_i0t": clustering_i0t,
    }


def evaluate_i0t_msrvtt_v2(
    test_loader,
    statistics,
    device="cpu",
    max_cluster_samples=3_000,
    max_eval_batches=None,
    plot_pca=True,
    seed=DEFAULT_SEED,
):
    text_all, vision_all, labels_all = collect_paired_embeddings(
        test_loader,
        device="cpu",
        max_batches=max_eval_batches,
    )
    labels_all = labels_all.long()

    text_norm, vision_norm, text_i0t, vision_i0t = apply_i0t_from_statistics(
        text_all.to(device),
        vision_all.to(device),
        statistics,
    )
    text_norm = text_norm.cpu()
    vision_norm = vision_norm.cpu()
    text_i0t = text_i0t.cpu()
    vision_i0t = vision_i0t.cpu()

    retrieval_orig = {
        1: float(_to_scalar(compute_retrieval("msrvtt", (text_norm, vision_norm), top_k=1))),
        5: float(_to_scalar(compute_retrieval("msrvtt", (text_norm, vision_norm), top_k=5))),
        10: float(_to_scalar(compute_retrieval("msrvtt", (text_norm, vision_norm), top_k=10))),
    }
    retrieval_i0t = {
        1: float(_to_scalar(compute_retrieval("msrvtt", (text_i0t, vision_i0t), top_k=1))),
        5: float(_to_scalar(compute_retrieval("msrvtt", (text_i0t, vision_i0t), top_k=5))),
        10: float(_to_scalar(compute_retrieval("msrvtt", (text_i0t, vision_i0t), top_k=10))),
    }

    gaps_orig = _compute_gap_bundle(text_norm, vision_norm)
    gaps_i0t = _compute_gap_bundle(text_i0t, vision_i0t)
    mean_rmg = mean_rmg_over_batches(
        test_loader,
        statistics,
        device=device,
        max_eval_batches=max_eval_batches,
    )
    gaps_orig["RMG"] = mean_rmg["orig"]
    gaps_i0t["RMG"] = mean_rmg["i0t"]

    if max_cluster_samples is not None and text_norm.shape[0] > max_cluster_samples:
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(text_norm.shape[0], generator=rng)[:max_cluster_samples]
        text_norm_cl = text_norm[idx]
        vision_norm_cl = vision_norm[idx]
        text_i0t_cl = text_i0t[idx]
        vision_i0t_cl = vision_i0t[idx]
        labels_cl = labels_all[idx]
    else:
        text_norm_cl = text_norm
        vision_norm_cl = vision_norm
        text_i0t_cl = text_i0t
        vision_i0t_cl = vision_i0t
        labels_cl = labels_all

    clustering_orig = clustering_metrics_two_modalities_msrvtt(
        text_norm_cl,
        vision_norm_cl,
        labels_cl,
        n_clusters=len(torch.unique(labels_all)),
        random_state=seed,
    )
    clustering_i0t = clustering_metrics_two_modalities_msrvtt(
        text_i0t_cl,
        vision_i0t_cl,
        labels_cl,
        n_clusters=len(torch.unique(labels_all)),
        random_state=seed,
    )

    if plot_pca:
        labels_np = labels_cl.numpy()
        labels_2n = np.concatenate([labels_np, labels_np], axis=0)
        emb_orig = torch.cat([text_norm_cl, vision_norm_cl], dim=0).numpy()
        emb_i0t = torch.cat([text_i0t_cl, vision_i0t_cl], dim=0).numpy()
        plot_pca_modalities(emb_orig, labels_2n, "MSRVTT v2 PCA 2D (original)", seed=seed)
        plot_pca_modalities(emb_i0t, labels_2n, "MSRVTT v2 PCA 2D (I0T)", seed=seed)

    return {
        "method": "I0T",
        "dataset": "msrvtt_v2",
        "retrieval_orig": retrieval_orig,
        "retrieval_i0t": retrieval_i0t,
        "gaps_orig": gaps_orig,
        "gaps_i0t": gaps_i0t,
        "mean_rmg_over_batches": mean_rmg,
        "statistics": statistics,
        "clustering_orig": clustering_orig,
        "clustering_i0t": clustering_i0t,
    }


def _to_float_dict(values):
    if values is None:
        return None
    return {str(key): float(value) for key, value in values.items()}


def summarize_i0t_result(result):
    summary = {
        "method": result.get("method"),
        "dataset": result.get("dataset"),
        "retrieval_orig": _to_float_dict(result.get("retrieval_orig")),
        "retrieval_i0t": _to_float_dict(result.get("retrieval_i0t")),
        "gaps_orig": _to_float_dict(result.get("gaps_orig")),
        "gaps_i0t": _to_float_dict(result.get("gaps_i0t")),
        "mean_rmg_over_batches": _to_float_dict(result.get("mean_rmg_over_batches")),
    }

    statistics = result.get("statistics")
    if statistics is not None:
        summary["statistics"] = {
            "n_fit": int(statistics.get("n_fit")) if statistics.get("n_fit") is not None else None,
            "embedding_dim": int(statistics.get("embedding_dim")) if statistics.get("embedding_dim") is not None else None,
            "mean_x_norm": float(statistics.get("mean_x_norm")) if statistics.get("mean_x_norm") is not None else None,
            "mean_y_norm": float(statistics.get("mean_y_norm")) if statistics.get("mean_y_norm") is not None else None,
        }

    for prefix in ("orig", "i0t"):
        clustering = result.get(f"clustering_{prefix}")
        if clustering is not None:
            summary[f"clustering_{prefix}"] = {
                "ARI": float(clustering["ARI"]),
                "NMI": float(clustering["NMI"]),
                "Homogeneity": float(clustering["Homogeneity"]),
                "V-measure": float(clustering["V-measure"]),
            }

    return summary


def _normalize_model_name(value):
    if value is None:
        return None

    value = str(value).strip().rstrip("/")
    if not value:
        return None

    base = os.path.basename(value)
    if base.startswith("precomputed_") or base in {"precomputed_train", "precomputed_test"}:
        parent = os.path.basename(os.path.dirname(value))
        if parent:
            base = parent

    base = re.sub(r"^cifar10_(?:train|test)_", "", base)
    base = base.replace("clip_vit_b_32", "ViT-B-32")
    base = base.replace("clip_vit_b_16", "ViT-B-16")
    base = re.sub(r"_v2$", "", base)
    base = re.sub(r"_{3,}", "__", base)
    base = re.sub(r"^(ViT-[A-Za-z0-9-]+)_(.+)$", r"\1__\2", base)
    return base if "ViT-" in base else None


def resolve_embedding_model_name(model_name=None, path_hints=None):
    candidates = []
    if model_name is not None:
        candidates.append(model_name)
    if path_hints is not None:
        candidates.extend(path_hints)
    candidates.extend(DEFAULT_PATHS.values())

    for candidate in candidates:
        normalized = _normalize_model_name(candidate)
        if normalized is not None:
            return normalized

    raise ValueError(
        "Could not resolve model name for export. Pass model_name explicitly or provide path hints."
    )


def export_i0t_results(results_by_dataset, export_path, model_name=None, path_hints=None):
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    compact = {}
    for dataset_name, result in results_by_dataset.items():
        if result is None:
            continue
        summary = summarize_i0t_result(result)
        summary["dataset"] = dataset_name
        compact[dataset_name] = summary

    resolved_model_name = resolve_embedding_model_name(model_name=model_name, path_hints=path_hints)

    if export_path.exists():
        with export_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {}

    model_payload = payload.get(resolved_model_name, {})
    model_payload.update(compact)
    payload[resolved_model_name] = model_payload

    with export_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return export_path
