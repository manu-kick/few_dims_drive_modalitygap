# equation 3/4 from "Semantic Compression via multimodal representation learning"
import torch 
import wandb

## varianza nella coppia i,i per le 2 modalità (la coppia i i è una classe)


def fisher_ratio(cf, text_embeddings, vision_embeddings, iterations, eps=1e-12):
    """
    Fisher ratio with NO labels for Flickr30k:
    treat modalities as the 2 classes: {text, vision}.

    text_embeddings:  [N, D]
    vision_embeddings: [N, D]

    Returns:
      fisher_score (scalar),
      SB (DxD),
      SW (DxD)
    """
    X_t = torch.as_tensor(text_embeddings).float()    # [N, D]
    X_v = torch.as_tensor(vision_embeddings).float()  # [N, D]

    assert X_t.ndim == 2 and X_v.ndim == 2, "Embeddings must be 2D [N, D]"
    assert X_t.shape[1] == X_v.shape[1], "Text and vision embeddings must have same D"

    D = X_t.shape[1]

    # Means per "class" (modality)
    mu_t = X_t.mean(dim=0)  # [D]
    mu_v = X_v.mean(dim=0)  # [D]

    # Global mean across all 2N samples
    global_mean = torch.cat([X_t, X_v], dim=0).mean(dim=0)  # [D]

    # --- Between-class scatter ---
    # Standard LDA-style: sum_c N_c (mu_c - mu)(mu_c - mu)^T
    Nt = torch.tensor(X_t.shape[0], dtype=torch.float32, device=X_t.device)
    Nv = torch.tensor(X_v.shape[0], dtype=torch.float32, device=X_v.device)

    diff_t = (mu_t - global_mean).unsqueeze(1)  # [D,1]
    diff_v = (mu_v - global_mean).unsqueeze(1)  # [D,1]

    SB = Nt * (diff_t @ diff_t.t()) + Nv * (diff_v @ diff_v.t())  # [D,D]

    # --- Within-class scatter ---
    centered_t = X_t - mu_t.unsqueeze(0)         # [N,D]
    centered_v = X_v - mu_v.unsqueeze(0)         # [N,D]
    SW = centered_t.t() @ centered_t + centered_v.t() @ centered_v  # [D,D]

    TR_SB = torch.trace(SB)
    TR_SW = torch.trace(SW)

    fisher_score = TR_SB / (TR_SW + eps)

    if iterations % cf.eval_every == 0 and getattr(cf, "wandb", False):
        wandb.log({"fisher_score": fisher_score.item()})

    return fisher_score, SB, SW


def cumulative_explained_variance(cf, SB, SW, iterations,  eps: float = 1e-12):
    """
    Compute cumulative explained variance (CEV) from scatter matrices.

    We follow the standard PCA-style definition:
        - Build a "total scatter" matrix S_T = S_B + S_W
        - Compute its eigenvalues (variance along each principal direction)
        - Sort eigenvalues in descending order
        - CEV(k) = sum_{i=1..k} λ_i / sum_{i=1..D} λ_i

    Args:
        SB: [D, D] between-class scatter matrix
        SW: [D, D] within-class scatter matrix
        eps: small constant to avoid division by zero

    Returns:
        cev: [D] cumulative explained variance curve
        eigvals_sorted: [D] eigenvalues sorted descending
        ST: [D, D] total scatter matrix used for decomposition
    """
    # 1) Ensure we are working in floating point (needed for eigendecomposition).
    SB = SB.float()
    SW = SW.float()
    
    # 2) Total scatter: captures overall variability.
    #    In classical LDA/PCA relations, S_T = S_B + S_W.
    ST = SB + SW  # [D, D]
    
    
    # 3) Numerical sanity: make sure ST is symmetric.
    #    In theory it is symmetric, but tiny floating errors can appear.
    #    Symmetrizing improves stability for eigenvalue routines.
    ST = 0.5 * (ST + ST.t())
    
    
    # 4) Compute eigenvalues of a symmetric matrix.
    #    eigvalsh is specialized for Hermitian/symmetric matrices and returns real eigenvalues.
    eigvals = torch.linalg.eigvalsh(ST)  # [D], ascending order by default
    
    # 5) Clamp very small negative eigenvalues caused by numerical noise.
    #    Scatter matrices should be PSD, so negatives are usually just floating error.
    eigvals = torch.clamp(eigvals, min=0.0)

    # 6) Reverse to descending order so "component 1" is the direction of largest variance.
    eigvals_sorted = torch.flip(eigvals, dims=[0])  # [D], descending

    # 7) Compute the cumulative sum of eigenvalues: numerator for each k.
    cumsum = torch.cumsum(eigvals_sorted, dim=0)  # [D]

    # 8) Total variance is the sum of all eigenvalues (same as trace(ST)).
    total = eigvals_sorted.sum()  # scalar

    # 9) Normalize cumulative sums to get CEV(k) in [0, 1].
    #    eps avoids division-by-zero in degenerate cases.
    cev = cumsum / (total + eps)  # [D]

    # 10) Return the curve plus extras useful for plotting/inspection.
    
    if iterations % cf.eval_every == 0 and cf.wandb:
        wandb.log({f"cumulative_explained_variance": cev.numpy()})
        
    return cev, eigvals_sorted, ST

def fisher_and_cumulative_explained_variance(cf,text_embeddings,vision_embeddings,iterations):
    fisher_score, SB, SW = fisher_ratio(cf, text_embeddings, vision_embeddings, iterations)
    cev, eigvals_sorted, ST = cumulative_explained_variance(cf, SB, SW, iterations=iterations)
    return fisher_score, cev, eigvals_sorted