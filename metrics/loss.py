import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss_anchor(text_embeddings, vision_embeddings, bs, contra_temp):   
    """
    Compute the contrastive loss using the "anchor" method.
    
    Args:
        text_embeddings: Tensor of shape (batch_size, embedding_dim) - text embeddings
        vision_embeddings: Tensor of shape (batch_size, embedding_dim) - vision embeddings
        contra_temp: Scalar temperature parameter for scaling similarities
        
    Returns:
        loss: Scalar tensor representing the contrastive loss
    """
    batch_size = text_embeddings.size(0)
    
    # Normalize embeddings
    text_norm = F.normalize(text_embeddings, p=2, dim=1)
    vision_norm = F.normalize(vision_embeddings, p=2, dim=1)
    
    # Compute similarity matrix (batch_size x batch_size)
    tv = torch.matmul(text_norm, vision_norm.t()) / contra_temp
    vt = torch.matmul(vision_norm, text_norm.t()) / contra_temp
    
    # Create labels for contrastive loss (positive pairs on the diagonal)
    labels = torch.arange(batch_size).to(text_embeddings.device)
    
    # Compute cross-entropy loss
    loss = (F.cross_entropy(tv, labels) + F.cross_entropy(vt, labels)) / 2
    
    return loss