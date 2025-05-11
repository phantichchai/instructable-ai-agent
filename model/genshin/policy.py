import torch
from torch import nn
from model.mineclip.mineclip import MineCLIP

class PolicyFromMineCLIP(nn.Module):
    def __init__(self, mineclip: MineCLIP, embed_dim, num_actions):
        super().__init__()
        self.mineclip = mineclip
        self.policy_head = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_actions)
        )

    def forward(self, video):
        with torch.no_grad():
            video_feat = self.mineclip.encode_video(video)
        return self.policy_head(video_feat)

