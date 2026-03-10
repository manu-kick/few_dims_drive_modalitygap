import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, v_measure_score

def clustering_metrics_from_two_modalities(feat_t: torch.Tensor, feat_v: torch.Tensor, labels: torch.Tensor,
                                          n_clusters=10, random_state=0):
    """
    KMeans su stack([text, vision]) e metriche vs labels duplicate.
    """
    assert feat_t.shape == feat_v.shape, "text e vision devono avere stessa shape (N, D)"
    assert feat_t.shape[0] == labels.shape[0], "labels devono avere N elementi"

    embeddings = torch.vstack([feat_t, feat_v]).cpu().numpy()     # (2N, D)
    true_labels = labels.cpu().numpy()
    true_labels_2 = np.concatenate([true_labels, true_labels], axis=0)  # (2N,)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(true_labels_2, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels_2, cluster_labels)
    hom = homogeneity_score(true_labels_2, cluster_labels)
    v   = v_measure_score(true_labels_2, cluster_labels)

    print(f"[KMeans k={n_clusters}] ARI={ari:.4f} | NMI={nmi:.4f} | Hom={hom:.4f} | V={v:.4f}")

    return {"ARI": ari, "NMI": nmi, "Homogeneity": hom, "V-measure": v}



def _label_tensor_to_list_mscoco(lbl):
    if torch.is_tensor(lbl):
        return [int(x) for x in lbl.detach().cpu().view(-1).tolist()]
    if isinstance(lbl, np.ndarray):
        return [int(x) for x in lbl.reshape(-1).tolist()]
    if isinstance(lbl, (list, tuple)):
        return [int(x) for x in lbl]
    return [int(lbl)]


def collapse_multilabels_mscoco(labels_batch, mode="primary"):
    """
    Converts a batch/list of variable-length COCO label arrays into a single target per sample.

    mode="primary": first label of the sorted label list. This is the closest analogue to CIFAR.
    mode="combo":   exact tuple of labels becomes a pseudo-class. More faithful, but many classes.
    """
    labels_list = [_label_tensor_to_list_mscoco(lbl) for lbl in labels_batch]

    if mode == "primary":
        y = np.array([lbl[0] if len(lbl) > 0 else -1 for lbl in labels_list], dtype=np.int64)
        meta = {"mode": mode, "class_names": None}
        return y, meta

    if mode == "combo":
        combos = [tuple(lbl) if len(lbl) > 0 else (-1,) for lbl in labels_list]
        uniq = {c: i for i, c in enumerate(sorted(set(combos)))}
        y = np.array([uniq[c] for c in combos], dtype=np.int64)
        meta = {"mode": mode, "combo_to_id": uniq}
        return y, meta

    raise ValueError(f"Unknown label collapse mode: {mode}")


def clustering_metrics_two_modalities_multilabel_mscoco(
    feat_t,
    feat_v,
    labels_batch,
    n_clusters=None,
    random_state=0,
    label_mode="primary",
):
    y, meta = collapse_multilabels_mscoco(labels_batch, mode=label_mode)
    valid = y >= 0

    feat_t = feat_t[valid]
    feat_v = feat_v[valid]
    y = y[valid]

    emb = torch.vstack([feat_t, feat_v]).cpu().numpy()
    true2 = np.concatenate([y, y], axis=0)

    if n_clusters is None:
        n_clusters = len(np.unique(y))
    n_clusters = int(max(2, min(n_clusters, len(np.unique(y)), emb.shape[0])))

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    pred = km.fit_predict(emb)

    return {
        "ARI": adjusted_rand_score(true2, pred),
        "NMI": normalized_mutual_info_score(true2, pred),
        "Homogeneity": homogeneity_score(true2, pred),
        "V-measure": v_measure_score(true2, pred),
        "cluster_labels": pred,
        "true_labels_2N": true2,
        "emb_2N": emb,
        "n_clusters": n_clusters,
        "n_classes": int(len(np.unique(y))),
        "label_mode": label_mode,
        "label_meta": meta,
    }
