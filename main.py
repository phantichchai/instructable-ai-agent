import torch
import cv2
import time
import mss
import numpy as np
import torchvision.transforms as transforms
import keyboard
import random
from model.base.multi_model_policy_network import MultiModalModel
from tools.genshin_impact_controller import GenshinImpactController
from tools.window import get_window_coordinates
from tools.constants import KEY_BINDINGS


def capture_screen():
    with mss.mss() as sct:
        # Capture the screen using mss
        screen_shot = sct.grab(get_window_coordinates("Genshin Impact"))

        # Convert the screenshot to a NumPy array
        img = np.array(screen_shot)

        # Convert BGRA to RGB (since OpenCV uses BGR format and mss uses BGRA)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Resize the image to 224x224 (model input size)-+
        img_resized = cv2.resize(img, (224, 224))

        # Render the image using OpenCV
        # cv2.imshow("Resized Screen Capture", img_resized)
        # Display the image briefly; press any key to close the window 
        # cv2.waitKey(1)          

        # Convert the image to a PyTorch tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(img_resized).unsqueeze(0)  # Add batch dimension
        
        return img_tensor

def close_windows():
    cv2.destroyAllWindows()

def generate_text_prompt():
    # Ask the user for a text prompt from the terminal
    return random.choice(list(KEY_BINDINGS.keys()))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = MultiModalModel(num_actions=17, device=device, debug=False).to(device)
    controller = GenshinImpactController(model, debug=True)

    started = False  # Variable to track whether the agent has started
    paused = False  # Variable to track pause/resume state

    try:
        while True:
            if not started and keyboard.is_pressed('s'):  # Start when 's' is pressed
                started = True
                print("Agent started!")

            if started:
                if keyboard.is_pressed('p'):  # Pause when 'p' is pressed
                    paused = True
                    print("Agent paused. Press 'r' to resume.")
                    while paused:
                        if keyboard.is_pressed('r'):  # Resume when 'r' is pressed
                            paused = False
                            print("Agent resumed.")

                if keyboard.is_pressed('q'):  # Quit when 'q' is pressed
                    print("Agent terminated.")
                    break

                if not paused:
                    screen_image = capture_screen()
                    screen_image = screen_image.to(device)
                    text_prompt = generate_text_prompt()  # Get user input from terminal
                    print(f"Prompt text: {text_prompt}")

                    # Get the action logits from the model
                    action_logits = model(screen_image, [text_prompt])
                    print(f"Logit: {action_logits}")

                    # Select the action with the highest value
                    action_index = torch.argmax(action_logits, dim=1).item()
                    print(f"Index: {action_index}")

                    # Execute the corresponding action using the controller
                    controller.execute_action(action_index)

                    # Optional: Introduce a small delay to avoid too many rapid inputs
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("Program terminated by user.")
    close_windows()


if __name__ == '__main__':
    main()