import torch
from torch import nn

class PolicyFromMineCLIP(nn.Module):
    def __init__(self, input_dim, embed_dim, num_actions, dropout=0.3):
        super().__init__()
        self.output_dim = num_actions
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_actions)
        )

    def forward(self, obs_embed):
        return self.policy_head(obs_embed)
    
    def get_actions(self, obs_embed):
        pred = torch.sigmoid(self.forward(obs_embed))
        pred = (pred > 0.5).float()
        return pred



