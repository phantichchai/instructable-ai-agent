import torch
from torch import nn

class PolicyFromMineCLIP(nn.Module):
    def __init__(self, mineclip, embed_dim):
        super().__init__()
        self.mineclip = mineclip
        self.policy_head = nn.Sequential(
            nn.Linear(mineclip.output_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, video):
        with torch.no_grad():
            video_feat, _ = self.mineclip(video, None)
        return self.policy_head(video_feat)

