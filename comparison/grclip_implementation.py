import torch
import torch.nn.functional as F
from tqdm import tqdm

from analysis.modality_gap import compute_gap


def _validate_2d_tensor(name, x):
    if not torch.is_tensor(x):
        raise TypeError(f"`{name}` must be a torch.Tensor.")
    if x.ndim != 2:
        raise ValueError(f"`{name}` must be 2D with shape (N, D).")


def _validate_same_dim(*pairs):
    dims = [x.shape[1] for x in pairs if x is not None]
    if not dims:
        raise ValueError("At least one tensor is required.")
    if len(set(dims)) != 1:
        raise ValueError("All tensors must share the same embedding dimension.")


def _to_scalar(value):
    if isinstance(value, dict):
        value = value.get("text_vision", next(iter(value.values())))
    if torch.is_tensor(value):
        value = value.item()
    return float(value)


def _normalize_rows(x, eps=1e-12):
    _validate_2d_tensor("x", x)
    return F.normalize(x, p=2, dim=1, eps=eps)


def fit_grclip_statistics(
    query_text_embeddings,
    document_text_embeddings=None,
    document_image_embeddings=None,
    eps=1e-12,
):
    """
    Fit GR-CLIP mean statistics.

    GR-CLIP removes the modality gap by subtracting role-specific global means:
    - query text mean
    - document text mean
    - document image mean

    If `document_text_embeddings` is not provided, the query text embeddings are reused.

    Returns a dictionary with normalized means stored as (1, D) tensors.
    """
    _validate_2d_tensor("query_text_embeddings", query_text_embeddings)

    if document_text_embeddings is None:
        document_text_embeddings = query_text_embeddings
    else:
        _validate_2d_tensor("document_text_embeddings", document_text_embeddings)

    if document_image_embeddings is not None:
        _validate_2d_tensor("document_image_embeddings", document_image_embeddings)

    _validate_same_dim(
        query_text_embeddings,
        document_text_embeddings,
        document_image_embeddings,
    )

    query_text_norm = _normalize_rows(query_text_embeddings, eps=eps)
    document_text_norm = _normalize_rows(document_text_embeddings, eps=eps)

    stats = {
        "query_text_mean": query_text_norm.mean(dim=0, keepdim=True),
        "document_text_mean": document_text_norm.mean(dim=0, keepdim=True),
        "embedding_dim": int(query_text_embeddings.shape[1]),
    }

    if document_image_embeddings is not None:
        document_image_norm = _normalize_rows(document_image_embeddings, eps=eps)
        stats["document_image_mean"] = document_image_norm.mean(dim=0, keepdim=True)
    else:
        stats["document_image_mean"] = None

    return stats


def apply_grclip_query(query_embeddings, statistics, eps=1e-12):
    """
    Apply GR-CLIP to text queries.
    """
    _validate_2d_tensor("query_embeddings", query_embeddings)

    query_norm = _normalize_rows(query_embeddings, eps=eps)
    query_mean = statistics["query_text_mean"].to(
        device=query_embeddings.device,
        dtype=query_embeddings.dtype,
    )
    query_gr = query_norm - query_mean
    query_gr = _normalize_rows(query_gr, eps=eps)
    return query_gr


def apply_grclip_document_text(document_text_embeddings, statistics, eps=1e-12):
    """
    Apply GR-CLIP to text documents.
    """
    _validate_2d_tensor("document_text_embeddings", document_text_embeddings)

    doc_text_norm = _normalize_rows(document_text_embeddings, eps=eps)
    doc_text_mean = statistics["document_text_mean"].to(
        device=document_text_embeddings.device,
        dtype=document_text_embeddings.dtype,
    )
    doc_text_gr = doc_text_norm - doc_text_mean
    doc_text_gr = _normalize_rows(doc_text_gr, eps=eps)
    return doc_text_gr


def apply_grclip_document_image(document_image_embeddings, statistics, eps=1e-12):
    """
    Apply GR-CLIP to image documents.
    """
    _validate_2d_tensor("document_image_embeddings", document_image_embeddings)

    if statistics.get("document_image_mean") is None:
        raise ValueError("`document_image_mean` is missing from statistics.")

    doc_image_norm = _normalize_rows(document_image_embeddings, eps=eps)
    doc_image_mean = statistics["document_image_mean"].to(
        device=document_image_embeddings.device,
        dtype=document_image_embeddings.dtype,
    )
    doc_image_gr = doc_image_norm - doc_image_mean
    doc_image_gr = _normalize_rows(doc_image_gr, eps=eps)
    return doc_image_gr


def fuse_grclip_document(
    document_text_embeddings,
    document_image_embeddings,
    statistics,
    alpha=0.5,
    eps=1e-12,
):
    """
    Build a multimodal GR-CLIP document embedding.

    Following the paper:
        e_doc = alpha * f_T(d_text) + (1 - alpha) * f_I(d_image)
                - [alpha * mean_doc_text + (1 - alpha) * mean_doc_image]

    The fused representation is then L2-normalized.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("`alpha` must be in [0, 1].")

    _validate_2d_tensor("document_text_embeddings", document_text_embeddings)
    _validate_2d_tensor("document_image_embeddings", document_image_embeddings)
    _validate_same_dim(document_text_embeddings, document_image_embeddings)

    if statistics.get("document_image_mean") is None:
        raise ValueError("`document_image_mean` is missing from statistics.")

    doc_text_norm = _normalize_rows(document_text_embeddings, eps=eps)
    doc_image_norm = _normalize_rows(document_image_embeddings, eps=eps)

    doc_text_mean = statistics["document_text_mean"].to(
        device=document_text_embeddings.device,
        dtype=document_text_embeddings.dtype,
    )
    doc_image_mean = statistics["document_image_mean"].to(
        device=document_image_embeddings.device,
        dtype=document_image_embeddings.dtype,
    )

    fused = (alpha * doc_text_norm) + ((1.0 - alpha) * doc_image_norm)
    fused_mean = (alpha * doc_text_mean) + ((1.0 - alpha) * doc_image_mean)
    fused_gr = fused - fused_mean
    fused_gr = _normalize_rows(fused_gr, eps=eps)
    return fused_gr


def apply_grclip_pair(
    text_embeddings,
    image_embeddings,
    statistics,
    text_role="query",
    eps=1e-12,
    return_stats=False,
):
    """
    Convenience wrapper for paired text/image embeddings.

    `text_role` controls which mean is subtracted from the text side:
    - "query": use query text mean
    - "document": use document text mean
    """
    _validate_2d_tensor("text_embeddings", text_embeddings)
    _validate_2d_tensor("image_embeddings", image_embeddings)
    _validate_same_dim(text_embeddings, image_embeddings)

    text_norm = _normalize_rows(text_embeddings, eps=eps)
    image_norm = _normalize_rows(image_embeddings, eps=eps)

    if text_role == "query":
        text_gr = apply_grclip_query(text_embeddings, statistics, eps=eps)
    elif text_role == "document":
        text_gr = apply_grclip_document_text(text_embeddings, statistics, eps=eps)
    else:
        raise ValueError("`text_role` must be either 'query' or 'document'.")

    image_gr = apply_grclip_document_image(image_embeddings, statistics, eps=eps)

    if not return_stats:
        return text_gr, image_gr

    stats = {
        "text_norm": text_norm,
        "image_norm": image_norm,
        "text_role": text_role,
    }
    return text_gr, image_gr, stats


def fit_grclip_statistics_from_loader(
    loader,
    n_fit=None,
    device="cpu",
    query_text_index=0,
    document_text_index=0,
    document_image_index=1,
):
    """
    Fit GR-CLIP statistics from a paired dataloader.

    For standard paired datasets, this uses:
    - batch[query_text_index] as query text
    - batch[document_text_index] as document text
    - batch[document_image_index] as document image

    In common text-image retrieval setups, `query_text_index == document_text_index == 0`.
    """
    query_buf, doc_text_buf, doc_image_buf = [], [], []
    seen = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Collect GR-CLIP calibration embeddings"):
            if not isinstance(batch, (list, tuple)):
                raise ValueError("Expected batches in the form (text, image, ...).")

            query_text = batch[query_text_index].to(device)
            doc_text = batch[document_text_index].to(device)
            doc_image = batch[document_image_index].to(device)

            if n_fit is not None:
                remaining = n_fit - seen
                if remaining <= 0:
                    break
                take = min(query_text.shape[0], remaining)
                query_text = query_text[:take]
                doc_text = doc_text[:take]
                doc_image = doc_image[:take]
                seen += take

            query_buf.append(query_text.detach().cpu())
            doc_text_buf.append(doc_text.detach().cpu())
            doc_image_buf.append(doc_image.detach().cpu())

    if not query_buf:
        raise ValueError("No calibration batches were collected.")

    query_all = torch.cat(query_buf, dim=0)
    doc_text_all = torch.cat(doc_text_buf, dim=0)
    doc_image_all = torch.cat(doc_image_buf, dim=0)

    statistics = fit_grclip_statistics(
        query_all,
        document_text_embeddings=doc_text_all,
        document_image_embeddings=doc_image_all,
    )
    statistics["n_fit"] = int(query_all.shape[0])
    statistics["query_text_mean_norm"] = float(torch.norm(statistics["query_text_mean"], p=2).item())
    statistics["document_text_mean_norm"] = float(torch.norm(statistics["document_text_mean"], p=2).item())
    statistics["document_image_mean_norm"] = float(torch.norm(statistics["document_image_mean"], p=2).item())
    return statistics


def mean_rmg_over_batches(
    loader,
    statistics,
    device="cpu",
    max_eval_batches=None,
    text_role="query",
    eps=1e-12,
):
    """
    Compute mean per-batch RMG before and after GR-CLIP.

    This is useful when full-dataset RMG is too memory-heavy.
    """
    rmg_orig_batches = []
    rmg_gr_batches = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="RMG over batches")):
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Expected loader batches in the form (text, image, ...).")

            text_b = batch[0].to(device)
            image_b = batch[1].to(device)

            text_norm = _normalize_rows(text_b, eps=eps)
            image_norm = _normalize_rows(image_b, eps=eps)
            text_gr, image_gr = apply_grclip_pair(
                text_b,
                image_b,
                statistics,
                text_role=text_role,
                eps=eps,
                return_stats=False,
            )

            rmg_orig_batches.append(
                float(_to_scalar(compute_gap("RMG", text_norm, image_norm, iterations=None)))
            )
            rmg_gr_batches.append(
                float(_to_scalar(compute_gap("RMG", text_gr, image_gr, iterations=None)))
            )

            if max_eval_batches is not None and batch_idx + 1 >= max_eval_batches:
                break

    if not rmg_orig_batches:
        raise ValueError("No batches were processed for RMG computation.")

    return {
        "orig": float(sum(rmg_orig_batches) / len(rmg_orig_batches)),
        "gr_clip": float(sum(rmg_gr_batches) / len(rmg_gr_batches)),
    }
