import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import umap


# pca visualization
def visualize2d(cf, text_embeddings, vision_embeddings, iterations, save=True, title=None):
    # --- fit PCA on ALL embeddings together ---
    X_all = np.vstack([text_embeddings, vision_embeddings])
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=2, random_state=getattr(cf, "seed", None))
    )
    X_all_pca = pipe.fit_transform(X_all)
    n_text = text_embeddings.shape[0]
    text_pca = X_all_pca[:n_text]
    vision_pca = X_all_pca[n_text:]
    # explained variance (sum of the 2 PCs)
    pca_obj = pipe.named_steps["pca"]
    explained_variance_ratio = float(np.sum(pca_obj.explained_variance_ratio_))
    # --- plot ---
    plt.figure(figsize=(8, 6))
    plt.scatter(text_pca[:, 0], text_pca[:, 1], marker="*", s=60, label="Text")
    plt.scatter(vision_pca[:, 0], vision_pca[:, 1], marker="s", s=60, label="Vision")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(f"PCA 2D (shared space) - Explained Variance: {explained_variance_ratio*100:.2f}%" if title is None else title)
    plt.legend()
    # --- save ---
    if save:
        path = os.path.join(cf.plot_path, "latent_space_visualizations")
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"PCA_space_at_{iterations}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # --- wandb ---
        if (iterations % cf.eval_every == 0) and getattr(cf, "wandb", False):
            wandb.log({
                "pca_2d": wandb.Image(save_path),
                "pca_explained_variance": explained_variance_ratio
            })
            
    return text_pca, vision_pca, explained_variance_ratio


def visualize_3d(cf, text_embeddings, vision_embeddings, iterations, save=True, title=None):
    """
    Fit ONE PCA (3 components) on the concatenation of text+vision embeddings,
    then project both modalities into the SAME PCA space and plot together.
    """

    # --- to numpy ---
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        try:
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    text_embeddings = to_numpy(text_embeddings)
    vision_embeddings = to_numpy(vision_embeddings)

    assert text_embeddings.ndim == 2 and vision_embeddings.ndim == 2
    assert text_embeddings.shape[1] == vision_embeddings.shape[1], \
        "Text and vision embeddings must have same feature dim to share a PCA space."

    # --- fit PCA on ALL embeddings together ---
    X_all = np.vstack([text_embeddings, vision_embeddings])

    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=3, random_state=getattr(cf, "seed", None))
    )
    X_all_pca = pipe.fit_transform(X_all)

    n_text = text_embeddings.shape[0]
    text_pca = X_all_pca[:n_text]
    vision_pca = X_all_pca[n_text:]

    # explained variance (sum of the 3 PCs)
    pca_obj = pipe.named_steps["pca"]
    explained_variance_ratio = float(np.sum(pca_obj.explained_variance_ratio_))

    # --- plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(text_pca[:, 0], text_pca[:, 1], text_pca[:, 2],
               marker="*", s=60, label="Text")
    ax.scatter(vision_pca[:, 0], vision_pca[:, 1], vision_pca[:, 2],
               marker="s", s=60, label="Vision")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(f"PCA 3D (shared space) - Explained Variance: {explained_variance_ratio*100:.2f}%" if title is None else title)
    ax.legend()

    # optional: set limits based on data (more meaningful than fixed [-1,1])
    all_min = X_all_pca.min(axis=0)
    all_max = X_all_pca.max(axis=0)
    pad = 0.05 * (all_max - all_min + 1e-9)
    ax.set_xlim(all_min[0] - pad[0], all_max[0] + pad[0])
    ax.set_ylim(all_min[1] - pad[1], all_max[1] + pad[1])
    ax.set_zlim(all_min[2] - pad[2], all_max[2] + pad[2])

    # --- save ---
    if save:
        path = os.path.join(cf.plot_path, "latent_space_visualizations")
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"PCA_space_at_{iterations}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- wandb ---
        if (iterations % cf.eval_every == 0) and getattr(cf, "wandb", False):
            wandb.log({
                "pca_3d": wandb.Image(save_path),
                "pca_explained_variance": explained_variance_ratio
            })

    return text_pca, vision_pca, explained_variance_ratio 

def visualize_3d_interatively():
    pass


def tsne_3d(text_embeddings, vision_embeddings, iterations=None, perplexity=30, seed=0):
    """
    text_embeddings: (N, D) numpy
    vision_embeddings: (N, D) numpy
    """

    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    text_embeddings = to_numpy(text_embeddings)
    vision_embeddings = to_numpy(vision_embeddings)
    X = np.vstack([text_embeddings, vision_embeddings])
    labels = np.array([0]*len(text_embeddings) + [1]*len(vision_embeddings))  # 0=text, 1=vision

    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed
    )
    Z = tsne.fit_transform(X)  # (2N, 3)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Z[labels==0,0], Z[labels==0,1], Z[labels==0,2], s=8, alpha=0.6, label="text")
    ax.scatter(Z[labels==1,0], Z[labels==1,1], Z[labels==1,2], s=8, alpha=0.6, label="vision")
    ax.set_title(f"t-SNE 3D (iter={iterations})")
    ax.legend()
    plt.show()

    Z_txt = Z[labels==0]
    Z_vis = Z[labels==1]
    return Z_txt, Z_vis, labels


def umap_3d(text_embeddings, vision_embeddings, iterations=None, n_neighbors=15, min_dist=0.1, seed=0, title=""):
    """
    text_embeddings: (N, D) numpy
    vision_embeddings: (N, D) numpy
    """


    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    text_embeddings = to_numpy(text_embeddings)
    vision_embeddings = to_numpy(vision_embeddings)

    X = np.vstack([text_embeddings, vision_embeddings])
    labels = np.array([0] * len(text_embeddings) + [1] * len(vision_embeddings))  # 0=text, 1=vision

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    Z = reducer.fit_transform(X)  # (2N, 3)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Z[labels == 0, 0], Z[labels == 0, 1], Z[labels == 0, 2], s=8, alpha=0.6, label="text")
    ax.scatter(Z[labels == 1, 0], Z[labels == 1, 1], Z[labels == 1, 2], s=8, alpha=0.6, label="vision")
    ax.set_title(f"UMAP 3D (iter={iterations})" if title == "" else title)

    ax.legend()
    plt.show()

    Z_text = Z[labels == 0]
    Z_vision = Z[labels == 1]
    return Z_text, Z_vision, labels


def umap_3d_v2(text_embeddings, vision_embeddings, iterations=None, n_neighbors=15, min_dist=0.1, seed=0, title="", cmap="viridis"):
    """
    UMAP 3D visualization with matching pairs sharing the same color.
    
    Args:
        text_embeddings: (N, D) array or tensor
        vision_embeddings: (N, D) array or tensor
        iterations: iteration number for title
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        seed: random seed
        title: plot title
        cmap: colormap name for gradient palette (e.g., 'viridis', 'plasma', 'coolwarm')
    
    Returns:
        Z_text: (N, 3) UMAP coordinates for text
        Z_vision: (N, 3) UMAP coordinates for vision
        labels: (2N,) array of modality labels
    """
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    text_embeddings = to_numpy(text_embeddings)
    vision_embeddings = to_numpy(vision_embeddings)

    n_pairs = len(text_embeddings)
    assert len(text_embeddings) == len(vision_embeddings), "Text and vision must have same number of samples for pair coloring"

    X = np.vstack([text_embeddings, vision_embeddings])
    labels = np.array([0] * n_pairs + [1] * n_pairs)  # 0=text, 1=vision

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    Z = reducer.fit_transform(X)  # (2N, 3)

    Z_text = Z[:n_pairs]
    Z_vision = Z[n_pairs:]

    # Create color array: same color for matching pairs
    colormap = plt.cm.get_cmap(cmap)
    colors = colormap(np.linspace(0, 1, n_pairs))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot text embeddings with square markers
    ax.scatter(
        Z_text[:, 0], Z_text[:, 1], Z_text[:, 2],
        c=colors, marker="s", s=30, alpha=0.7, label="Text"
    )
    # Plot vision embeddings with x markers
    ax.scatter(
        Z_vision[:, 0], Z_vision[:, 1], Z_vision[:, 2],
        c=colors, marker="x", s=30, alpha=0.7, label="Vision"
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title(f"UMAP 3D (iter={iterations})" if title == "" else title)
    ax.legend()

    # Add colorbar to show gradient
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_pairs - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Pair index")

    plt.show()

    return Z_text, Z_vision, labels












###------------------ MSCOCO VIZ ----------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D



def plot_pca_2d_mscoco_multilabel_blend(
    emb_2N,
    labels_per_sample,
    title="PCA projection (multi-label color blending)",
    max_points=6000,
    max_classes_to_color=20,
    seed=0,
    cmap_name="tab20",
    unknown_color=(0.7, 0.7, 0.7, 1.0),
):
    """
    Joint PCA on [text; vision] with multi-label color blending.

    Parameters
    ----------
    emb_2N : np.ndarray
        Shape (2N, D), built as [text_embeddings; vision_embeddings]
        or any equivalent ordering as long as it is first-half / second-half.
    labels_per_sample : list
        Length N.
        Each element can be:
          - np.ndarray of shape (L_i,)
          - torch tensor of shape (L_i,)
          - list[int]
        representing all labels associated with that sample.
    title : str
        Plot title.
    max_points : int
        Max total number of plotted points over the 2N embeddings.
    max_classes_to_color : int
        Only the most frequent classes get their own base colors.
        Labels outside this set are ignored in the blending. If a sample has
        no kept labels, it gets unknown_color.
    seed : int
        Random seed for subsampling and PCA.
    cmap_name : str
        Matplotlib categorical colormap name.
    unknown_color : tuple
        RGBA color for samples with no valid/kept labels.
    """
    emb_2N = np.asarray(emb_2N)
    n2 = emb_2N.shape[0]
    n = n2 // 2

    if n2 % 2 != 0:
        raise ValueError(f"emb_2N must have even first dimension, got {n2}")

    if len(labels_per_sample) != n:
        raise ValueError(
            f"labels_per_sample must have length N={n}, got {len(labels_per_sample)}"
        )

    modality = np.concatenate([
        np.zeros(n, dtype=np.int32),   # text
        np.ones(n, dtype=np.int32),    # vision
    ])

    def to_label_list(x):
        if x is None:
            return []
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return [int(x.item())]
            return [int(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [int(v) for v in x]
        return [int(x)]

    labels_per_sample = [to_label_list(x) for x in labels_per_sample]

    # Count class frequency over samples
    freq = {}
    for lbls in labels_per_sample:
        for lbl in set(lbls):
            if lbl >= 0:
                freq[lbl] = freq.get(lbl, 0) + 1

    sorted_labels = sorted(freq.keys(), key=lambda k: -freq[k])
    keep_labels = sorted_labels[:max_classes_to_color]

    cmap = plt.get_cmap(cmap_name, max(max_classes_to_color, 1))
    label_to_color = {
        lbl: cmap(i % max_classes_to_color)
        for i, lbl in enumerate(keep_labels)
    }

    def blended_color(lbls):
        cols = [label_to_color[lbl] for lbl in lbls if lbl in label_to_color]
        if len(cols) == 0:
            return unknown_color
        cols = np.array(cols, dtype=np.float64)
        return tuple(cols.mean(axis=0))

    sample_colors = np.array([blended_color(lbls) for lbls in labels_per_sample], dtype=float)
    colors_2N = np.concatenate([sample_colors, sample_colors], axis=0)

    # Optional subsampling at sample level, so text/vision pairs stay aligned
    if n2 > max_points:
        max_samples = max_points // 2
        max_samples = max(1, max_samples)

        rng = np.random.RandomState(seed)
        keep_idx = rng.choice(n, size=min(n, max_samples), replace=False)

        emb_2N = np.concatenate([emb_2N[:n][keep_idx], emb_2N[n:][keep_idx]], axis=0)
        colors_2N = np.concatenate([sample_colors[keep_idx], sample_colors[keep_idx]], axis=0)
        modality = np.concatenate([
            np.zeros(len(keep_idx), dtype=np.int32),
            np.ones(len(keep_idx), dtype=np.int32),
        ])

    z = PCA(n_components=2, random_state=seed).fit_transform(emb_2N)

    txt_mask = (modality == 0)
    vis_mask = (modality == 1)

    plt.figure(figsize=(7, 6))

    plt.scatter(
        z[vis_mask, 0],
        z[vis_mask, 1],
        c=colors_2N[vis_mask],
        s=20,
        marker="o",
        alpha=0.72,
    )

    plt.scatter(
        z[txt_mask, 0],
        z[txt_mask, 1],
        c=colors_2N[txt_mask],
        s=20,
        marker="x",
        alpha=0.72,
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='vision',
               markerfacecolor='gray', markeredgecolor='gray', markersize=7),
        Line2D([0], [0], marker='x', color='gray', label='text',
               linestyle='None', markersize=7),
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()




def plot_pca_2d_mscoco(
    emb_2N,
    labels_2N,
    title="PCA projection",
    max_points=6000,
    max_classes_to_color=20,
    seed=0,
):
    """
    PCA joint on [text; vision].

    Colors are assigned by class label (for MSCOCO, typically the FIRST label
    chosen for each sample and replicated for both text and vision embeddings).
    Markers distinguish modality:
      - 'x' -> text
      - 'o' -> vision

    Parameters
    ----------
    emb_2N : np.ndarray
        Shape (2N, D), concatenation of [text_embeddings; vision_embeddings]
        or [vision_embeddings; text_embeddings], as long as labels_2N matches.
    labels_2N : np.ndarray
        Shape (2N,), class labels already expanded to both modalities.
        Example:
            labels_2N = np.concatenate([labels_primary, labels_primary], axis=0)
    title : str
        Plot title.
    max_points : int
        Maximum number of total points to display.
    max_classes_to_color : int
        Number of most frequent classes to color distinctly.
        Remaining classes are grouped into -1 and shown as a gray cloud.
    seed : int
        Random seed for subsampling.
    """
    emb_2N = np.asarray(emb_2N)
    labels_2N = np.asarray(labels_2N)

    if emb_2N.shape[0] != labels_2N.shape[0]:
        raise ValueError(
            f"emb_2N and labels_2N must have same first dimension, got "
            f"{emb_2N.shape[0]} and {labels_2N.shape[0]}"
        )

    n2 = emb_2N.shape[0]
    n = n2 // 2

    # assumes first half modality 0, second half modality 1
    modality = np.concatenate([
        np.zeros(n, dtype=np.int32),         # text
        np.ones(n2 - n, dtype=np.int32),     # vision
    ])

    # optional subsampling
    if n2 > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n2, size=max_points, replace=False)
        emb_2N = emb_2N[idx]
        labels_2N = labels_2N[idx]
        modality = modality[idx]

    # keep only the most frequent classes with unique colors
    uniq, counts = np.unique(labels_2N, return_counts=True)
    order = uniq[np.argsort(-counts)]
    keep = set(order[:max_classes_to_color].tolist())

    labels_vis = np.array(
        [lbl if lbl in keep else -1 for lbl in labels_2N],
        dtype=np.int64
    )

    # PCA on all displayed points jointly
    z = PCA(n_components=2, random_state=seed).fit_transform(emb_2N)

    plt.figure(figsize=(7, 6))

    # vision: circles
    vis_mask = (modality == 1)
    txt_mask = (modality == 0)

    sc_vis = plt.scatter(
        z[vis_mask, 0],
        z[vis_mask, 1],
        c=labels_vis[vis_mask],
        s=18,
        marker="o",
        alpha=0.70,
        cmap="tab20",
    )

    # text: crosses, same color mapping
    plt.scatter(
        z[txt_mask, 0],
        z[txt_mask, 1],
        c=labels_vis[txt_mask],
        s=18,
        marker="x",
        alpha=0.70,
        cmap="tab20",
        vmin=sc_vis.get_clim()[0],
        vmax=sc_vis.get_clim()[1],
    )

    # manual legend for modality only
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='vision',
               markerfacecolor='gray', markeredgecolor='gray', markersize=7),
        Line2D([0], [0], marker='x', color='gray', label='text',
               linestyle='None', markersize=7),
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()