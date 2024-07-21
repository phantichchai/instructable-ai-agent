import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyConfig:
    def __init__(
        self,
        d_model=768,
        hidden_size=2048,
        keyboard_actions=8,
        mouse_actions=3,
        mouse_events=4,
        mouse_positions=2,
        seq_len=None
    ):
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.keyboard_actions = keyboard_actions
        self.mouse_actions = mouse_actions
        self.mouse_events = mouse_events
        self.mouse_positions = mouse_positions
        self.seq_len = seq_len

class Policy(nn.Module):
    def __init__(self, policy_config=PolicyConfig()):
        super(Policy, self).__init__()
        self.keyboard_policy = nn.Linear(policy_config.d_model, policy_config.keyboard_actions)
        self.mouse_action_policy = nn.Linear(policy_config.d_model, policy_config.mouse_actions)
        self.mouse_event_policy = nn.Linear(policy_config.d_model, policy_config.mouse_events)
        self.mouse_position_policy = nn.Linear(policy_config.d_model, policy_config.mouse_positions)
        
    def forward(self, input):
        keyboard_actions_probs = torch.sigmoid(self.keyboard_policy(input))
        keyboard_actions = (keyboard_actions_probs > 0.5).float()

        mouse_action_probs = torch.softmax(self.mouse_action_policy(input), dim=1)
        mouse_actions = torch.argmax(mouse_action_probs, dim=2, keepdim=True).float()

        mouse_event_probs = torch.softmax(self.mouse_event_policy(input), dim=1)
        mouse_events = torch.argmax(mouse_event_probs, dim=2, keepdim=True).float()

        mouse_position = self.mouse_position_policy(input)

        output = torch.cat([keyboard_actions, mouse_actions, mouse_events, mouse_position], dim=2)
        return output