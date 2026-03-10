# Inspired by 'The Surprising Effectiveness of Deep Orthogonal Procrustes Alignment in Unsupervised Domain Adaptation"
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, v_measure_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
from metrics.retrieval import compute_retrieval
from metrics.clustering import (
    clustering_metrics_from_two_modalities,
    clustering_metrics_two_modalities_multilabel_mscoco,
)
from analysis.modality_gap import compute_gap
from analysis.viz import plot_pca_2d_mscoco_multilabel_blend, plot_pca_2d_mscoco


def _clustering_metrics_two_modalities(X, Y, labels, n_clusters=10, random_state=0):
    """
    KMeans su [X; Y] (2N, D) e metriche vs labels duplicate (2N,).
    """
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
    Y_np = Y.detach().cpu().numpy() if torch.is_tensor(Y) else Y
    L_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels

    emb = np.vstack([X_np, Y_np])                       # (2N, D)
    true2 = np.concatenate([L_np, L_np], axis=0)        # (2N,)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    pred = km.fit_predict(emb)

    return {
        "ARI": adjusted_rand_score(true2, pred),
        "NMI": normalized_mutual_info_score(true2, pred),
        "Homogeneity": homogeneity_score(true2, pred),
        "V-measure": v_measure_score(true2, pred),
        "cluster_labels": pred,
        "true_labels_2N": true2,
        "emb_2N": emb
    }


def _plot_pca_2d(emb_2N, labels_2N, title, max_points=6000):
    """
    PCA 2D veloce per visualizzare cluster/label (2N punti).
    Marker:
      - text: 'x'
      - vision: 'o'
    """
    n2 = emb_2N.shape[0]
    n = n2 // 2
    modality = np.concatenate([
        np.zeros(n, dtype=np.int32),          # text (first N)
        np.ones(n2 - n, dtype=np.int32)       # vision (second N)
    ])

    if n2 > max_points:
        idx = np.random.RandomState(0).choice(n2, size=max_points, replace=False)
        emb_2N = emb_2N[idx]
        labels_2N = labels_2N[idx]
        modality = modality[idx]

    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(emb_2N)

    plt.figure(figsize=(6, 5))

    # same colormap for labels, different marker for modality
    sc_vis = plt.scatter(
        z[modality == 1, 0], z[modality == 1, 1],
        c=labels_2N[modality == 1], s=20, marker='o', alpha=0.75, label='vision'
    )
    plt.scatter(
        z[modality == 0, 0], z[modality == 0, 1],
        c=labels_2N[modality == 0], s=20, marker='x', alpha=0.75,
        cmap=sc_vis.cmap, norm=sc_vis.norm, label='text'
    )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best")
    plt.colorbar(sc_vis, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def collect_embeddings(loader, max_samples=10_000, device="cuda"):
    """
    Collects (text, vision) pairs from loader into torch tensors.
    Each batch is (text_emb, vision_emb).
    """
    Xs, Ys = [], []
    n_samples = 0
    
    with torch.no_grad():
        for text_b, vis_b, _ in tqdm(loader, desc=f"Collecting samples"):
            text_b = F.normalize(text_b.to(device), dim=-1)
            vis_b  = F.normalize(vis_b.to(device), dim=-1)
            Xs.append(text_b); Ys.append(vis_b)
            n_samples += text_b.shape[0]
            if n_samples >= max_samples:
                break
    X = torch.cat(Xs, dim=0)[:max_samples]
    Y = torch.cat(Ys, dim=0)[:max_samples]
    return X, Y

def fit_subspace_alignment(loader, n_fit=10_000, d_sub = 256, device="cuda"):
    # 1. Collect embeddings
    X, Y = collect_embeddings(loader, max_samples=n_fit, device=device)
    print(f"Collected {X.shape[0]} samples of dimension {X.shape[1]}, these will be used to fit the subspace alignment model with d_sub={d_sub}.")

    # center (PCA-style)
    muX = X.mean(axis=0, keepdims=True)
    muY = Y.mean(axis=0, keepdims=True)
    
    Xc = X - muX    
    Yc = Y - muY
    
    # SVD to get bases Ws, Wt (top d_sub right singular vectors)
    # Xc = U S V^T => Ws = V[:, :d_sub]
    # Yc = U S V^T => Wt = V[:, :d_sub]
    Xc = Xc.cpu().numpy()
    Yc = Yc.cpu().numpy()
    _, _, VtX = np.linalg.svd(Xc, full_matrices=False)
    _, _, VtY = np.linalg.svd(Yc, full_matrices=False)

    Ws = VtX[:d_sub].T   # (D, d_sub)
    Wt = VtY[:d_sub].T   # (D, d_sub)

    # closed-form subspace alignment
    Phi = Wt.T @ Ws      # (d_sub, d_sub)   Eq (7) in the paper :contentReference[oaicite:4]{index=4}

    return {"muX": muX, "muY": muY, "Ws": Ws, "Wt": Wt, "Phi": Phi, "d_sub": d_sub}
        
def apply_subspace_alignment(X, Y, model, renorm=True):
    device = X.device
    eps = 1e-12
    muX, muY = model["muX"], model["muY"]
    Ws, Wt, Phi = model["Ws"], model["Wt"], model["Phi"]
    muX = muX.to(device)
    muY = muY.to(device)
    Ws = torch.as_tensor(Ws).float().to(device)
    Wt = torch.as_tensor(Wt).float().to(device)
    Phi = torch.as_tensor(Phi).float().to(device)
    # center
    Xc = X - muX
    Yc = Y - muY

    # align Y into X-space (Eq. 10 style) :contentReference[oaicite:6]{index=6}
    Y_al = Yc @ Wt @ Phi @ Ws.T

    # optionally bring back to X mean
    Y_al = Y_al + muX

    if renorm:
        Xn = X / (torch.norm(X, dim=1, keepdim=True) + eps)
        Yn = Y / (torch.norm(Y, dim=1, keepdim=True) + eps)
        Yaln = Y_al / (torch.norm(Y_al, dim=1, keepdim=True) + eps)
        return Xn, Yn, Yaln
    
    return X, Y, Y_al

def analyze_subspace_dimensions(model, device):
    """
    sub_model must contain:
      - Ws: (D, d_sub) text basis
      - Wt: (D, d_sub) vision basis
    """
    Ws = model["Ws"]
    Wt = model["Wt"]

    #subspace importance per original dim
    imp_X = np.sum(Ws**2, axis=1)  # (D,)
    imp_Y = np.sum(Wt**2, axis=1)  # (D,)
    imp_joint = 0.5 * (imp_X + imp_Y)
    
    top_impX = np.argsort(imp_X)[::-1]
    top_impY = np.argsort(imp_Y)[::-1]
    top_imp_joint = np.argsort(imp_joint)[::-1]
    
    return top_impX, top_impY, top_imp_joint

def eval_subspace_alignment_mscoco(
    test_loader,
    alignment_model,
    device="cuda",
    gaps=("RMG", "L2M", "L2I", "cosineTP"),
    do_clustering=True,
    n_clusters=None,
    max_cluster_samples=5000,
    plot_pca=True,
    max_pca_points=6000,
    max_classes_to_color=20,
    label_mode="primary",
):
    r_orig = {1: [], 5: [], 10: []}
    r_al   = {1: [], 5: [], 10: []}
    gaps_orig_batches = {g: [] for g in gaps}
    gaps_al_batches = {g: [] for g in gaps}

    X_all, Y_all, Yal_all = [], [], []
    y_all = []
    Xt_buf, Xv_buf, Xv_al_buf, y_buf = [], [], [], []
    seen = 0
    def _to_scalar(v):
        if isinstance(v, dict):
            v = v.get("text_vision", next(iter(v.values())))
        if torch.is_tensor(v):
            v = v.item()
        return float(v)

    def _labels_batch_to_list(labels_batch):
        if torch.is_tensor(labels_batch):
            return [labels_batch[i] for i in range(labels_batch.shape[0])]
        if isinstance(labels_batch, np.ndarray):
            return [labels_batch[i] for i in range(labels_batch.shape[0])]
        if isinstance(labels_batch, (list, tuple)):
            return list(labels_batch)
        return [labels_batch]

    def _to_label_list(lbl):
        if lbl is None:
            return []
        if hasattr(lbl, "detach") and hasattr(lbl, "cpu"):
            lbl = lbl.detach().cpu().numpy()
        if isinstance(lbl, np.ndarray):
            if lbl.ndim == 0:
                return [int(lbl.item())]
            return [int(v) for v in lbl.reshape(-1).tolist()]
        if isinstance(lbl, (list, tuple)):
            return [int(v) for v in lbl]
        return [int(lbl)]

    def _primary_label(lbl):
        lbl_list = _to_label_list(lbl)
        return int(lbl_list[0]) if len(lbl_list) > 0 else -1

    with torch.no_grad():
        for text_b, vis_b, labels in tqdm(test_loader, desc="Eval subspace alignment"):
            X = F.normalize(text_b.to(device), dim=-1)
            Y = F.normalize(vis_b.to(device),  dim=-1)
            
            # original retrieval
            r_orig[1].append(compute_retrieval("mscoco", (X, Y), top_k=1))
            r_orig[5].append(compute_retrieval("mscoco", (X, Y), top_k=5))
            r_orig[10].append(compute_retrieval("mscoco", (X, Y), top_k=10))

            # aligned vision -> text-space
            _, _, Yaln = apply_subspace_alignment(X, Y, alignment_model, renorm=True)

            r_al[1].append(compute_retrieval("mscoco", (X, Yaln), top_k=1))
            r_al[5].append(compute_retrieval("mscoco", (X, Yaln), top_k=5))
            r_al[10].append(compute_retrieval("mscoco", (X, Yaln), top_k=10))

            for g in gaps:
                go = compute_gap(g, X, Y, iterations=None)
                ga = compute_gap(g, X, Yaln, iterations=None)
                gaps_orig_batches[g].append(_to_scalar(go))
                gaps_al_batches[g].append(_to_scalar(ga))

            X_all.append(X)
            Y_all.append(Y)
            Yal_all.append(Yaln)
            y_all.extend(_labels_batch_to_list(labels))

            if do_clustering and seen < max_cluster_samples:
                b = min(X.shape[0], max_cluster_samples - seen)
                Xt_buf.append(X[:b].detach().cpu())
                Xv_buf.append(Y[:b].detach().cpu())
                Xv_al_buf.append(Yaln[:b].detach().cpu())
                y_buf.extend(_labels_batch_to_list(labels)[:b])
                seen += b

    X_all = torch.cat(X_all, dim=0)
    Y_all = torch.cat(Y_all, dim=0)
    Yal_all = torch.cat(Yal_all, dim=0)
    gaps_orig, gaps_al = {}, {}

    for g in gaps:
        gaps_orig[g] = float(np.mean(gaps_orig_batches[g]))
        gaps_al[g] = float(np.mean(gaps_al_batches[g]))

    print("\n=== SUBSPACE ALIGNMENT TEST RESULTS ===")
    print(f"d_sub = {alignment_model['d_sub']}")
    print(f"Retrieval@1  orig: {np.mean(r_orig[1]):.4f} | aligned: {np.mean(r_al[1]):.4f}")
    print(f"Retrieval@5  orig: {np.mean(r_orig[5]):.4f} | aligned: {np.mean(r_al[5]):.4f}")
    print(f"Retrieval@10 orig: {np.mean(r_orig[10]):.4f} | aligned: {np.mean(r_al[10]):.4f}")
    print("Gaps original:", gaps_orig)
    print("Gaps aligned :", gaps_al)

    clustering_out = {}
    if do_clustering and len(Xt_buf) > 0:
        Xt_all = torch.cat(Xt_buf, dim=0)
        Xv_all = torch.cat(Xv_buf, dim=0)
        Xv_al_all = torch.cat(Xv_al_buf, dim=0)

        cluster_orig = clustering_metrics_two_modalities_multilabel_mscoco(
            Xt_all,
            Xv_all,
            y_buf,
            n_clusters=n_clusters,
            random_state=0,
            label_mode=label_mode,
        )
        cluster_al = clustering_metrics_two_modalities_multilabel_mscoco(
            Xt_all,
            Xv_al_all,
            y_buf,
            n_clusters=n_clusters,
            random_state=0,
            label_mode=label_mode,
        )

        print(
            f"[Clustering | mode={label_mode}] ORIG    "
            f"ARI={cluster_orig['ARI']:.4f} | NMI={cluster_orig['NMI']:.4f} | "
            f"Hom={cluster_orig['Homogeneity']:.4f} | V={cluster_orig['V-measure']:.4f} | "
            f"k={cluster_orig['n_clusters']} | classes={cluster_orig['n_classes']}"
        )
        print(
            f"[Clustering | mode={label_mode}] ALIGNED "
            f"ARI={cluster_al['ARI']:.4f} | NMI={cluster_al['NMI']:.4f} | "
            f"Hom={cluster_al['Homogeneity']:.4f} | V={cluster_al['V-measure']:.4f} | "
            f"k={cluster_al['n_clusters']} | classes={cluster_al['n_classes']}"
        )

        clustering_out["clustering_orig"] = {
            k: cluster_orig[k]
            for k in ["ARI", "NMI", "Homogeneity", "V-measure", "n_clusters", "n_classes", "label_mode"]
        }
        clustering_out["clustering_aligned"] = {
            k: cluster_al[k]
            for k in ["ARI", "NMI", "Homogeneity", "V-measure", "n_clusters", "n_classes", "label_mode"]
        }

    if plot_pca:
        n_samples = X_all.shape[0]
        if len(y_all) != n_samples:
            print(
                "Warning: labels size does not match embeddings; skipping PCA plots. "
                f"labels={len(y_all)} embeddings={n_samples}"
            )
        else:
            labels_primary = np.array([_primary_label(lbl) for lbl in y_all], dtype=np.int64)
            labels_2n = np.concatenate([labels_primary, labels_primary], axis=0)
            emb_orig = torch.cat([X_all, Y_all], dim=0).detach().cpu().numpy()
            emb_al = torch.cat([X_all, Yal_all], dim=0).detach().cpu().numpy()

            plot_pca_2d_mscoco(
                emb_2N=emb_orig,
                labels_2N=labels_2n,
                title=f"PCA 2D (orig, primary label) | subspace d={alignment_model['d_sub']}",
                max_points=max_pca_points,
                max_classes_to_color=max_classes_to_color,
            )
            plot_pca_2d_mscoco(
                emb_2N=emb_al,
                labels_2N=labels_2n,
                title=f"PCA 2D (aligned, primary label) | subspace d={alignment_model['d_sub']}",
                max_points=max_pca_points,
                max_classes_to_color=max_classes_to_color,
            )

            plot_pca_2d_mscoco_multilabel_blend(
                emb_2N=emb_orig,
                labels_per_sample=y_all,
                title=f"PCA 2D (orig, multi-label blend) | subspace d={alignment_model['d_sub']}",
                max_points=max_pca_points,
                max_classes_to_color=max_classes_to_color,
            )
            plot_pca_2d_mscoco_multilabel_blend(
                emb_2N=emb_al,
                labels_per_sample=y_all,
                title=f"PCA 2D (aligned, multi-label blend) | subspace d={alignment_model['d_sub']}",
                max_points=max_pca_points,
                max_classes_to_color=max_classes_to_color,
            )

    return {
        "retrieval_orig": {k: float(np.mean(v)) for k, v in r_orig.items()},
        "retrieval_aligned": {k: float(np.mean(v)) for k, v in r_al.items()},
        "gaps_orig": gaps_orig,
        "gaps_aligned": gaps_al,
        **clustering_out,
    }

def eval_subspace_alignment_cifar10(
    test_loader,
    alignment_model,
    device="cuda",
    labels_to_emb=None,
    do_clustering=True,
    n_clusters=10,
    max_cluster_samples=5000,
    plot_pca=True,
):

    def _to_scalar(v):
        if isinstance(v, dict):
            v = v.get("text_vision", next(iter(v.values())))
        if torch.is_tensor(v):
            v = v.item()
        return float(v)


    if labels_to_emb is None:
        raise ValueError("labels_to_emb must be provided for CIFAR-10 retrieval evaluation.")
    
    gaps=("RMG","L2M","L2I","cosineTP")
    r_orig = {1: [], 5: [], 10: []}
    r_al   = {1: [], 5: [], 10: []}
    gaps_orig_batches = {g: [] for g in gaps}
    gaps_al_batches = {g: [] for g in gaps}

    # buffers per clustering
    X_buf, Y_buf, L_buf = [], [], []
    Xa_buf, Ya_buf = [], []
    seen = 0

    with torch.no_grad():
        for text_b, vis_b, labels_b in tqdm(test_loader, desc="Eval subspace alignment"):
            X = F.normalize(text_b.to(device), dim=-1)
            Y = F.normalize(vis_b.to(device),  dim=-1)
            labels = labels_b.to(device)
            
        
            # original retrieval
            r_orig[1].append(_to_scalar(compute_retrieval("cifar10", (Y, X, labels), top_k=1, labels_to_emb=labels_to_emb)))
            r_orig[5].append(_to_scalar(compute_retrieval("cifar10", (Y, X, labels), top_k=5, labels_to_emb=labels_to_emb)))
            r_orig[10].append(_to_scalar(compute_retrieval("cifar10", (Y, X, labels), top_k=10, labels_to_emb=labels_to_emb)))

            # aligned vision -> text-space
            Xn, _, Yaln = apply_subspace_alignment(X, Y, alignment_model, renorm=True)

            r_al[1].append(_to_scalar(compute_retrieval("cifar10", (Yaln, Xn, labels), top_k=1, labels_to_emb=labels_to_emb)))
            r_al[5].append(_to_scalar(compute_retrieval("cifar10", (Yaln, Xn, labels), top_k=5, labels_to_emb=labels_to_emb)))
            r_al[10].append(_to_scalar(compute_retrieval("cifar10", (Yaln, Xn, labels), top_k=10, labels_to_emb=labels_to_emb)))

            for g in gaps:
                go = compute_gap(g, X, Y, iterations=None)
                ga = compute_gap(g, X, Yaln, iterations=None)
                gaps_orig_batches[g].append(_to_scalar(go))
                gaps_al_batches[g].append(_to_scalar(ga))

            # --- CLUSTERING BUFFERS ---
            if do_clustering and seen < max_cluster_samples:
                b = min(X.shape[0], max_cluster_samples - seen)
                X_buf.append(X[:b].detach().cpu())
                Y_buf.append(Y[:b].detach().cpu())
                Xa_buf.append(Xn[:b].detach().cpu())
                Ya_buf.append(Yaln[:b].detach().cpu())
                L_buf.append(labels[:b].detach().cpu())
                seen += b
    
    gaps_orig = {g: float(np.mean(v)) for g, v in gaps_orig_batches.items()}
    gaps_al = {g: float(np.mean(v)) for g, v in gaps_al_batches.items()}
    
    print("\n=== SUBSPACE ALIGNMENT TEST RESULTS ===")
    print(f"d_sub = {alignment_model['d_sub']}")
    print(f"Retrieval@1  orig: {np.mean(r_orig[1]):.4f} | aligned: {np.mean(r_al[1]):.4f}")
    print(f"Retrieval@5  orig: {np.mean(r_orig[5]):.4f} | aligned: {np.mean(r_al[5]):.4f}")
    print(f"Retrieval@10 orig: {np.mean(r_orig[10]):.4f} | aligned: {np.mean(r_al[10]):.4f}")
    print("Gaps original:", gaps_orig)
    print("Gaps aligned :", gaps_al)

    # --- CLUSTERING E VISUALIZZAZIONE ---
    if do_clustering and len(X_buf) > 0:
        X_all  = torch.cat(X_buf, dim=0)
        Y_all  = torch.cat(Y_buf, dim=0)
        Xa_all = torch.cat(Xa_buf, dim=0)
        Ya_all = torch.cat(Ya_buf, dim=0)
        L_all  = torch.cat(L_buf, dim=0)

        print(f"\n[Clustering] using N={X_all.shape[0]} samples (then 2N points for KMeans).")

        m_orig = _clustering_metrics_two_modalities(X_all, Y_all, L_all, n_clusters=n_clusters, random_state=0)
        m_al   = _clustering_metrics_two_modalities(Xa_all, Ya_all, L_all, n_clusters=n_clusters, random_state=0)

        print(f"[Clustering KMeans k={n_clusters}]")
        print(f"  ORIG   ARI={m_orig['ARI']:.4f} | NMI={m_orig['NMI']:.4f} | Hom={m_orig['Homogeneity']:.4f} | V={m_orig['V-measure']:.4f}")
        print(f"  ALIGNED ARI={m_al['ARI']:.4f} | NMI={m_al['NMI']:.4f} | Hom={m_al['Homogeneity']:.4f} | V={m_al['V-measure']:.4f}")

        if plot_pca:
            _plot_pca_2d(m_orig["emb_2N"], m_orig["true_labels_2N"], title=f"PCA 2D (orig) Subspace d={alignment_model['d_sub']}")
            _plot_pca_2d(m_al["emb_2N"],   m_al["true_labels_2N"], title=f"PCA 2D (aligned) Subspace d={alignment_model['d_sub']}")

        return {
            "retrieval_orig": {k: float(np.mean(v)) for k, v in r_orig.items()},
            "retrieval_aligned": {k: float(np.mean(v)) for k, v in r_al.items()},
            "gaps_orig": gaps_orig,
            "gaps_aligned": gaps_al,
            "clustering_orig": {k: m_orig[k] for k in ["ARI","NMI","Homogeneity","V-measure"]},
            "clustering_aligned": {k: m_al[k] for k in ["ARI","NMI","Homogeneity","V-measure"]},
        }

    return {
        "retrieval_orig": {k: float(np.mean(v)) for k,v in r_orig.items()},
        "retrieval_aligned": {k: float(np.mean(v)) for k,v in r_al.items()},
        "gaps_orig": gaps_orig,
        "gaps_aligned": gaps_al
    }