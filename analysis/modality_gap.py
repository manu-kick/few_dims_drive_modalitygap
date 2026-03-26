# This files implements several computation of the modality gap as described in several papers

# Modality gap as in the original paper
import torch
import torch.nn.functional as F
import numpy as np



def L2M(metric, text_embeddings, vision_embeddings, iterations):
    mean_text = text_embeddings.mean(dim=0)
    mean_vision = vision_embeddings.mean(dim=0)

    l2_text_vision = torch.norm(mean_text - mean_vision, p=2).item()

    return {'text_vision': l2_text_vision}

# Relative modality gap as in https://openreview.net/pdf?id=uAFHCZRmXk
def rmg_numerator(mod1, mod2):
    mod1 = torch.Tensor(mod1) if not isinstance(mod1, torch.Tensor) else mod1
    mod2 = torch.Tensor(mod2) if not isinstance(mod2, torch.Tensor) else mod2
    return torch.mean(1 - F.cosine_similarity(mod1, mod2)).item()

def rmg_denominator(mod1, mod2, numerator):
    mod1 = torch.Tensor(mod1) if not isinstance(mod1, torch.Tensor) else mod1
    mod2 = torch.Tensor(mod2) if not isinstance(mod2, torch.Tensor) else mod2
    mod1 = mod1.float() if not mod1.dtype == torch.float32 else mod1
    mod2 = mod2.float() if not mod2.dtype == torch.float32 else mod2
    N = mod1.shape[0]
    factor_multiplier = 1 / ((2 * N) * (N - 1))

    # Compute pairwise cosine similarities via normalized Gram matrices.
    # This avoids the previous N x N x D broadcasted tensor, which can
    # explode GPU memory for large batches.
    mod1 = F.normalize(mod1, dim=-1)
    mod2 = F.normalize(mod2, dim=-1)

    sim_mod1 = mod1 @ mod1.T
    intra_mod1 = ((1 - sim_mod1) / 2).triu(diagonal=1).sum().item()
    del sim_mod1

    sim_mod2 = mod2 @ mod2.T
    intra_mod2 = ((1 - sim_mod2) / 2).triu(diagonal=1).sum().item()
    del sim_mod2

    return (factor_multiplier * (intra_mod1 + intra_mod2)) + numerator

def RMG(metric, text_embeddings, vision_embeddings, iterations):
    numerator = rmg_numerator(text_embeddings, vision_embeddings)
    denominator = rmg_denominator(text_embeddings, vision_embeddings, numerator)

    return {'text_vision': numerator / denominator}

def L2I(metric, text_embeddings, vision_embeddings, iterations):
    l2i_text_vision = torch.norm(text_embeddings - vision_embeddings, p=2, dim=-1).mean().item()
    return {'text_vision': l2i_text_vision}

def cosineTP(text_embeddings, vision_embeddings):
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.detach().cpu().numpy()
    if isinstance(vision_embeddings, torch.Tensor):
        vision_embeddings = vision_embeddings.detach().cpu().numpy()
    if isinstance(text_embeddings, np.ndarray) and isinstance(vision_embeddings, np.ndarray):
         text_embeddings = torch.as_tensor(text_embeddings).float()
         vision_embeddings = torch.as_tensor(vision_embeddings).float()
    # cosine similarity between true pairs (text_embeddings[i], vision_embeddings[i]) averaged over the batch
    cosine_sim = F.cosine_similarity(text_embeddings, vision_embeddings, dim=-1).mean().item()
    return {'text_vision': cosine_sim}

def compute_gap(metric, text_embeddings, vision_embeddings, iterations):
    if metric == 'L2M':
        return L2M(metric, text_embeddings, vision_embeddings, iterations)
    elif metric == 'RMG':
        return RMG(metric, text_embeddings, vision_embeddings, iterations)
    elif metric == 'L2I':
        return L2I(metric, text_embeddings, vision_embeddings, iterations)
    elif metric == 'cosineTP':
        return cosineTP(text_embeddings, vision_embeddings)
    else:
        raise ValueError(f'Unknown metric {metric}')
