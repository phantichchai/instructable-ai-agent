import torch
import time
from model.base.multi_model_policy_network import MultiModalModel
from tools.genshin_impact_controller import GenshinImpactController

# Example usage
if __name__ == '__main__':
    # Initialize the model (replace with actual model initialization)
    model = MultiModalModel(num_actions=17)  # Assuming 17 actions including switch characters
    controller = GenshinImpactController(model)

    # Sample data for the model
    sample_image = torch.randn(1, 3, 224, 224)  # Batch size of 1
    sample_text = ["Move forward now!"]  # Example text input

    # Get the action logits from the model
    action_logits = model(sample_image, sample_text)

    # Select the action for the current input
    action_index = torch.argmax(action_logits, dim=1).item()

    # Execute the corresponding action using the controller
    controller.execute_action(action_index)

    # Example: Execute a series of actions
    time.sleep(1)
    for i in range(6):
        controller.execute_action(6)  # Normal attack
        time.sleep(0.5)