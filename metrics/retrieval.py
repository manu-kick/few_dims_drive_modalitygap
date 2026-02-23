import torch
import torch.nn.functional as F
import wandb

def retrieval(text_embeddings, vision_embeddings, top_k=1):
    """
    Compute retrieval metrics (e.g., Recall@K) between text and vision embeddings.

    Args:
      text_embeddings: [N, D]
      vision_embeddings: [N, D]
      top_k: int, which Recall@K to compute (e.g., 1 for Recall@1)
    Returns:
      A dict with retrieval metrics (e.g., {"recall_at_1": 0.75})
    """
    text_embeddings = torch.as_tensor(text_embeddings).detach().float()  # [N, D]
    vision_embeddings = torch.as_tensor(vision_embeddings).detach().float()  # [N, D]

    # Compute similarity matrix (N x N)
    similarity = text_embeddings @ vision_embeddings.t()  # [N, N]

    # For each text query, get the indices of the top-k most similar vision embeddings
    top_k_indices = torch.topk(similarity, k=top_k, dim=1).indices  # [N, top_k]

    # Check if the correct match (diagonal) is in the top-k indices
    correct_matches = (top_k_indices == torch.arange(text_embeddings.size(0)).unsqueeze(1).to(text_embeddings.device)).any(dim=1)  # [N]

    recall_at_k = correct_matches.float().mean().item()  # scalar

    return  recall_at_k
