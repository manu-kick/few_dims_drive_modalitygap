import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionMLPClassifier(nn.Module):
    def __init__(self, d=512, num_classes=1000, hidden=1024, dropout=0.2):
        super().__init__()
        in_dim = 4 * d  # [t, v, |t-v|, t*v]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, text_emb, vision_emb):
        # text_emb, vision_emb: (B, 512)
        x = torch.cat([text_emb, vision_emb,
                       torch.abs(text_emb - vision_emb),
                       text_emb * vision_emb], dim=-1)
        return self.net(x)  # (B, num_classes)
    
class LinearProbing(nn.Module):
    def __init__(self, d=512, num_classes=1000):
        super().__init__()
        self.vision_proj = nn.Linear(d*2, num_classes)

    def forward(self, text_emb, vision_emb):
        x = torch.cat([text_emb, vision_emb], dim=-1)  # (B, 1024)
        return self.vision_proj(x)  # simple late fusion
    
class LinearProbingIndependentModalities(nn.Module):
    def __init__(self, d=512, num_classes=1000):
        super().__init__()
        self.embedding_projected = nn.Linear(d, num_classes)

    def forward(self, embedding):
        return self.embedding_projected(embedding)  # (B, num_classes)
    
class NonLinearProbingIndependentModalities(nn.Module):
    def __init__(self, d=512, num_classes=1000, hidden=1024, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, embedding):
        return self.net(embedding)  # (B, num_classes)