import torch
from torch import nn
from model.mineclip.mineclip import MineCLIP

class PolicyFromMineCLIP(nn.Module):
    def __init__(self, mineclip: MineCLIP, input_dim, embed_dim, num_actions):
        super().__init__()
        self.mineclip = mineclip
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_actions)
        )

    def forward(self, obs_embed):
        return self.policy_head(obs_embed)
    
    def get_actions(self, obs_embed):
        pred = torch.sigmoid(self.forward(obs_embed))
        pred = (pred > 0.5).float()
        return pred



