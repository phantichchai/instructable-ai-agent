import torch.nn as nn

class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.ce_loss = nn.CrossEntropyLoss()  # Cross-Entropy Loss
        self.mse_loss = nn.MSELoss()  # Mean Squared Error Loss

    def forward(self, outputs, targets):
        # Separate the different types of outputs
        keyboard_actions = outputs[:, :, :8]
        mouse_actions = outputs[:, :, 8:9]
        mouse_events = outputs[:, :, 9:10]
        mouse_position = outputs[:, :, 10:]

        # Separate the different types of targets
        targets_keyboard_actions = targets[:, :, :8]
        targets_mouse_actions = targets[:, :, 8:9]
        targets_mouse_events = targets[:, :, 9:10]
        targets_mouse_position = targets[:, :, 10:]

        keyboard_loss = self.bce_loss(keyboard_actions, targets_keyboard_actions)
        mouse_action_loss = self.ce_loss(mouse_actions, targets_mouse_actions)
        mouse_event_loss = self.ce_loss(mouse_events, targets_mouse_events)
        mouse_position_loss = self.mse_loss(mouse_position, targets_mouse_position)

        # Combine the losses (you can adjust the weights if needed)
        total_loss = keyboard_loss + mouse_action_loss + mouse_event_loss + mouse_position_loss

        return total_loss