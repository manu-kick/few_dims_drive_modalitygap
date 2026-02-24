import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE



def visualize_3d(cf, text_embeddings, vision_embeddings, iterations, save=True):
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
    ax.set_title(f"PCA 3D (shared space) - Explained Variance: {explained_variance_ratio*100:.2f}%")
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

    return Z, labels
    