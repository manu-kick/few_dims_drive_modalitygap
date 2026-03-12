import torch
import torch.nn.functional as F
import wandb

# retrieval flickr30k
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


def retrieval_cifar10(X, Y, labels, top_k=1, labels_to_emb=None):
    """
    Cross-modal Recall@K for CIFAR-10.
    X: image embeddings (N, D)
    Y: text embeddings (unused but kept for API consistency)
    labels: tensor (N,)
    labels_to_emb: dict {class_id: embedding}
    """

    # stack label embeddings -> (10, D)
    text_embs = torch.stack([labels_to_emb[i] for i in sorted(labels_to_emb.keys())]).to(X.device)

    # normalize for cosine similarity
    X = F.normalize(X, dim=1)
    text_embs = F.normalize(text_embs, dim=1)

    # similarity (N, 10)
    sim = X @ text_embs.T # image vs text similarity

    # top-k predicted classes
    topk = sim.topk(top_k, dim=1).indices

    # check if correct class is in top-k
    correct = (topk == labels.unsqueeze(1)).any(dim=1).float()

    recall = correct.mean().item()

    return recall
    
    
    
    
    
def compute_paired_retrieval_mscoco(inputs, top_k=1):
    """
    inputs = (queries, targets)
    Positive pair is the diagonal match inside the batch.
    """
    Q, T = inputs
    sims = Q @ T.T
    ranks = torch.argsort(sims, dim=1, descending=True)
    gt = torch.arange(Q.shape[0], device=Q.device).unsqueeze(1)
    hits = (ranks[:, :top_k] == gt).any(dim=1).float()
    return hits.mean().item()

  
    
def compute_retrieval(dataset_name, inputs, top_k=1, labels_to_emb=None):
  if dataset_name == "cifar10":
    x = inputs[0]
    y = inputs[1]
    labels = inputs[2]
    return retrieval_cifar10(x, y, labels, top_k=top_k, labels_to_emb=labels_to_emb)
  if dataset_name == 'mscoco':
    return compute_paired_retrieval_mscoco(inputs, top_k=top_k)
  if dataset_name == 'flickr30k':
    return retrieval(inputs[0], inputs[1], top_k=top_k)
  