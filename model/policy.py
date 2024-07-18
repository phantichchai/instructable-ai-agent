import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyConfig:
    def __init__(
        self,
        d_model=768,
        hidden_size=2048,
        num_actions=12,
        seq_len=None
    ):
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.seq_len = seq_len

class Policy(nn.Module):
    def __init__(self, policy_config=PolicyConfig()):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(policy_config.d_model, policy_config.hidden_size),
            nn.Dropout(0.1, False),
            nn.LayerNorm(policy_config.hidden_size),
            nn.Linear(policy_config.hidden_size, policy_config.num_actions),
        )
        
    def forward(self, input):
        return self.policy(input)