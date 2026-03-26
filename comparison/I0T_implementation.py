import torch
import torch.nn.functional as F
from tqdm import tqdm

from analysis.modality_gap import compute_gap


def _validate_pair(x, y):
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("`x` and `y` must be 2D tensors of shape (N, D).")
    if x.shape[1] != y.shape[1]:
        raise ValueError("`x` and `y` must have the same embedding dimension.")


def _to_scalar(value):
    if isinstance(value, dict):
        value = value.get("text_vision", next(iter(value.values())))
    if torch.is_tensor(value):
        value = value.item()
    return float(value)


def fit_i0t_statistics(x, y, eps=1e-12):
    """
    Compute modality-specific means after row-wise L2 normalization.
    """
    _validate_pair(x, y)

    x_norm = F.normalize(x, p=2, dim=1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=1, eps=eps)

    mean_x = x_norm.mean(dim=0, keepdim=True)
    mean_y = y_norm.mean(dim=0, keepdim=True)

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
    }


def apply_i0t_with_statistics(x, y, statistics, eps=1e-12, return_stats=False):
    """
    Apply I0T using pre-computed modality means.
    """
    _validate_pair(x, y)

    x_norm = F.normalize(x, p=2, dim=1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=1, eps=eps)

    mean_x = statistics["mean_x"].to(device=x.device, dtype=x.dtype)
    mean_y = statistics["mean_y"].to(device=y.device, dtype=y.dtype)

    x_centered = x_norm - mean_x
    y_centered = y_norm - mean_y

    x_aligned = F.normalize(x_centered, p=2, dim=1, eps=eps)
    y_aligned = F.normalize(y_centered, p=2, dim=1, eps=eps)

    if not return_stats:
        return x_aligned, y_aligned

    stats = {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "x_centered": x_centered,
        "y_centered": y_centered,
    }
    return x_aligned, y_aligned, stats


def apply_i0t(x, y, eps=1e-12):
    """
    Apply the I0T centering procedure described in the original comment.

    Steps:
    1. L2-normalize each sample in both modalities.
    2. Compute one mean vector per modality.
    3. Subtract the modality-specific mean.
    4. L2-normalize the centered samples again.

    Args:
        x: Tensor of shape (N, D) for the first modality.
        y: Tensor of shape (N, D) for the second modality.
        eps: Numerical stability term for normalization.

    Returns:
        x_aligned: Centered and re-normalized `x`.
        y_aligned: Centered and re-normalized `y`.
        stats: Dictionary with modality means and centered tensors.
    """
    statistics = fit_i0t_statistics(x, y, eps=eps)
    x_aligned, y_aligned, stats = apply_i0t_with_statistics(
        x,
        y,
        statistics,
        eps=eps,
        return_stats=True,
    )
    return x_aligned, y_aligned, stats


def mean_rmg_over_batches(loader, statistics, device="cpu", max_eval_batches=None, eps=1e-12):
    """
    Compute RMG by evaluating each batch independently and averaging the scores.

    This mirrors the intended evaluation protocol: RMG is not computed once on the
    entire test set, but per batch and then averaged across batches.
    """
    rmg_orig_batches = []
    rmg_i0t_batches = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="RMG over batches")):
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Expected loader batches in the form (text, vision, ...).")

            text_b = batch[0].to(device)
            vision_b = batch[1].to(device)

            text_norm = F.normalize(text_b, p=2, dim=1, eps=eps)
            vision_norm = F.normalize(vision_b, p=2, dim=1, eps=eps)
            text_i0t, vision_i0t = apply_i0t_with_statistics(
                text_b,
                vision_b,
                statistics,
                eps=eps,
                return_stats=False,
            )

            rmg_orig_batches.append(
                float(_to_scalar(compute_gap("RMG", text_norm, vision_norm, iterations=None)))
            )
            rmg_i0t_batches.append(
                float(_to_scalar(compute_gap("RMG", text_i0t, vision_i0t, iterations=None)))
            )

            if max_eval_batches is not None and batch_idx + 1 >= max_eval_batches:
                break

    if not rmg_orig_batches:
        raise ValueError("No batches were processed for RMG computation.")

    return {
        "orig": float(sum(rmg_orig_batches) / len(rmg_orig_batches)),
        "i0t": float(sum(rmg_i0t_batches) / len(rmg_i0t_batches)),
    }
